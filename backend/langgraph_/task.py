from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import SystemMessage, HumanMessage

from sqlalchemy import create_engine, text
from sqlalchemy.sql.expression import Executable
from sqlalchemy.engine import Result


from .utils import (
    load_qwen_model,
    EmptyQueryResultError,
    NullQueryResultError,
    load_prompt,
)

from .utils import EmptyQueryResultError, NullQueryResultError, load_prompt
from typing import List, Any, Union, Sequence, Dict
from pydantic import BaseModel, Field
import os, re
import torch


if torch.cuda.is_available():
    model, tokenizer = load_qwen_model()
    print("Local Model is ready!")
else:
    print("Non-GPU env has detected.")


def evaluate_user_question(user_question: str) -> str:
    """사용자의 질문이 일상적인 대화문인지, 데이터 및 비즈니스와 관련된 질문인지를
    판단하는 역할을 담당하고 있는 함수입니다.
    현재는 LLM(gpt-4o-mini)을 통해 사용자의 질문을 판단하고 있습니다.

    Args:
        user_question (str): 사용자의 질문

    Returns:
        str: "1" : 데이터 또는 비즈니스와 관련된 질문, "0" : 일상적인 대화문
    """
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("prompts/question_evaluation/main_v1.prompt")
            ),
            (
                "human",
                "질문(user_question): {user_question}",
            ),
        ]
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini")  # type: ignore
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output


def simple_conversation(user_question: str) -> str:
    """사용자의 질문이 일상적인 대화문이라고 판단되었을 경우
    사용자와 일상적인 대화를 진행하는 함수입니다.
    현재는 LLM(gpt-4o-mini)으로 대응을 진행하고 있습니다.

    Args:
        user_question (str): 사용자의 일상적인 질문

    Returns:
        str: 사용자의 일상적인 질문에 대한 AI의 대답
    """
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("prompts/general_conversation/main_v1.prompt")
            ),
            (
                "human",
                "{user_question}",
            ),
        ]
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini")  # type: ignore
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output


def analyze_user_question(user_question: str) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("prompts/question_analysis/main_v1.prompt")
            ),
            (
                "human",
                "사용자 질문: {user_question}"
                + load_prompt("prompts/question_analysis/human_v1.prompt"),
            ),
        ]
    )

    analyze_chain = ANALYZE_PROMPT | llm | output_parser
    analyze_question = analyze_chain.invoke({"user_question": user_question})

    return analyze_question


def clarify_user_question(
    user_question: str, user_question_analyze: str, collected_questions: List[str]
) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("prompts/additional_question/main_v1.prompt")
            ),
            (
                "human",
                "원래 사용자 질문:\n{user_question}\n\n초기 질문 분석:\n{user_question_analyze}\n\n이전 질문 기록:\n{collected_questions}\n\n"
                + load_prompt("prompts/additional_question/human_postfix_v1.prompt"),
            ),
        ]
    )

    chain = prompt | llm | output_parser
    chat_history = "\n".join(f"{i+1}. {q}" for i, q in enumerate(collected_questions))

    leading_question = chain.invoke(
        {
            "user_question": user_question,
            "user_question_analyze": user_question_analyze,
            "collected_questions": chat_history,
        }
    )

    return leading_question


def check_leading_question(leading_question: str) -> int:
    if leading_question.startswith("종료") or leading_question.startswith('"종료'):
        return 0
    else:
        return 1


def refine_user_question(user_question: str, user_question_analyze: str) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    REFINE_PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("prompts/question_refinement/main_v1.prompt")
            ),
            (
                "human",
                """사용자 질문: 
                {user_question}
                
                사용자 질문 분석: 
                {user_question_analyze}
                
                구체화된 질문:""",
            ),
        ]
    )

    refine_chain = REFINE_PROMPT | llm | output_parser
    refine_question = refine_chain.invoke(
        {"user_question": user_question, "user_question_analyze": user_question_analyze}
    )

    return refine_question


def select_relevant_tables(
    user_question: str, context_cnt: int, vector_store: VectorStore
) -> List[str]:
    """user_question과 관련성이 가장 높은 k(context_cnt)개의 document에서 page_content만 추출하여 리스트의 형태로 반환하는 함수입니다.
    입력으로 들어오는 vector_store는 반드시 MySQL 서버 내 테이블에 대한 메타데이터가 임베딩 되어 있어야 정상적으로 작동 합니다.
    관련성은 vetor store에 기본으로 내장된 유사도 검색 알고리즘을 사용합니다.

    Args:
        user_question (str): 사용자의 질문
        context_cnt (int): 반환할 context의 개수
        vector_store (VectorStore): MySQL 서버 내 테이블에 대한 메타데이터가 임베딩 되어있는 벡터 스토어

    Returns:
        List[str]: context가 포함된 리스트
    """
    relevant_tables = vector_store.similarity_search(query=user_question, k=context_cnt)
    table_contexts = [doc.metadata["context"] for doc in relevant_tables]

    return table_contexts


def extract_context(
    user_question: str,
    table_contexts: List[str],
    flow_status: str = "KEEP",
    prev_list: List[int] = [],
    prev_query: str = "",
    error_msg: str = "",
) -> List[int]:
    """벡터 스토어에서 검색으로 얻어낸 context들을 대상으로 사용자의 질문(user_question)에 기반한 SQL문을 생성함에 있어 필요한지를 판단한 후,
    필요한 context의 인덱스를 담은 리스트를 반환하는 함수입니다.

    Args:
        user_question (str): 사용자의 질문
        table_contexts (List[str]): context들을 담은 리스트

    Returns:
        List[int]: 필요한 context의 index를 담은 리스트
    """
    if not table_contexts:
        # 평가할 context가 없다면 빈 리스트 반환
        return []

    system_instruction = load_prompt("prompts/table_selection/main_v2.prompt")

    if flow_status == "RESELECT":
        print("검색된 테이블 스키마 재검수")
        system_instruction += load_prompt(
            "prompts/table_selection/regen_postfix_v1.prompt"
        ).format(prev_list=prev_list, prev_query=prev_query, error_msg=error_msg)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_instruction),
            (
                "human",
                "user_question:\n{user_question}\n\ncontext:\n{context}",
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    class context_list(BaseModel):
        """Index list of the context which is necessary for answering user_question."""

        ids: List[int | None] = Field(description="Ids of contexts.")

    structured_llm = llm.with_structured_output(context_list)
    context = ""
    for idx, table_info in enumerate(table_contexts):
        context += f"{idx}.\n{table_info}\n\n"

    chain = prompt | structured_llm

    output = chain.invoke({"user_question": user_question, "context": context})
    return output.ids  # type: ignore


def create_query(
    user_question,
    table_contexts,
    table_contexts_ids,
    llm_api,
    flow_status="KEEP",
    prev_query="",
    error_msg="",
):
    try:
        # 컨텍스트 생성
        context = ""
        for idx, table_info in enumerate(table_contexts):
            if idx in set(table_contexts_ids):
                context += table_info + "\n\n"

        # 프롬프트 로드 및 구성
        prefix = load_prompt("prompts/query_creation/prefix_v1.prompt").format(
            context=context
        )
        postfix = load_prompt("prompts/query_creation/postfix_v1.prompt")

        # flow_status에 따른 프롬프트 생성
        if flow_status == "KEEP":
            main_prompt = load_prompt("prompts/query_creation/generate_v1.prompt")
            system_prompt = prefix + main_prompt + postfix
        else:
            regen_prompt = load_prompt(
                "prompts/query_creation/regenerate_v1.prompt"
            ).format(prev_query=prev_query, result_msg=error_msg)
            system_prompt = prefix + regen_prompt + postfix

        # GPU를 사용 가능하며, 사용자가 로컬 LLM 사용을 원할 경우
        if torch.cuda.is_available() and llm_api == "Local":
            full_prompt = system_prompt + f"\n\nuser_question: {user_question}"
            # 입력 토크나이징
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            # 모델 추론
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 결과 디코딩 및 SQL 추출
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            output_parser = StrOutputParser()
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    ("human", """user_question: {user_question}"""),
                ]
            )

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            chain = prompt | llm | output_parser

            output = chain.invoke({"user_question": user_question})

        try:
            sql_query = re.search(r"```sql\s*(.*?)\s*```", output, re.DOTALL).group(1)
        except:
            sql_query = re.search(r"SELECT.*?;", output, re.DOTALL).group(0)

        return sql_query.strip()

    except Exception as e:
        print("\n=== 에러 발생 ===")
        print(f"에러 타입: {type(e)}")
        print(f"에러 메시지: {str(e)}")
        raise


def check_query_result(result: Sequence[Dict[str, Any]]) -> Exception | None:

    if not result:
        # 쿼리문 결과가 빈 리스트이면, 에러 발생
        raise EmptyQueryResultError()

    for row in result:
        # 쿼리문 결과 row 중 하나라도 NULL이면, 통과
        if not all(value is None for value in row.values()):
            return None
    # 쿼리문 결과 row 중 모두 NULL 인 경우, 에러 발생
    raise NullQueryResultError()


def execute_query(command: str | Executable, fetch="all") -> Union[Sequence[Dict[str, Any]], Result]:  # type: ignore
    """
    Executes SQL command through underlying engine.

    If the statement returns no rows, an empty list is returned.
    """
    parameters = {}
    execution_options = {}
    db_path = os.path.join(os.getenv("URL"), "INFORMATION_SCHEMA")  # type: ignore
    engine = create_engine(db_path)

    with engine.begin() as connection:
        if isinstance(command, str):
            command = text(command)
        elif isinstance(command, Executable):
            pass
        else:
            raise TypeError(f"Query expression has unknown type: {type(command)}")

        cursor = connection.execute(
            command,
            parameters,
            execution_options=execution_options,
        )
        if cursor.returns_rows:
            if fetch == "all":
                result = [x._asdict() for x in cursor.fetchall()]
            elif fetch == "one":
                first_result = cursor.fetchone()
                result = [] if first_result is None else [first_result._asdict()]
            elif fetch == "cursor":
                return cursor
            else:
                raise ValueError(
                    "Fetch parameter must be either 'one', 'all', or 'cursor'"
                )
            check_query_result(result)
            return result


def truncate_word(content: Any, *, length: int = 300, suffix: str = "...") -> str:
    """
    Truncate a string to a certain number of words, based on the max string
    length.
    """

    if not isinstance(content, str) or length <= 0:
        return content

    if len(content) <= length:
        return content

    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix


def get_query_result(command, fetch, include_columns=False):
    result = execute_query(command, fetch)
    if fetch == "cursor":
        return result

    # 너무 긴 데이터는 잘라내기
    res = [
        {column: truncate_word(value, length=300) for column, value in r.items()}
        for r in result
    ]

    # column 이름을 제거해서 token 수 절약하기
    # SQL 쿼리문을 같이 입력으로 주면 column 이름이 없어도 된다.
    if not include_columns:
        res = [tuple(row.values()) for row in res]  # type: ignore[misc]

    if not res:
        return ""
    else:
        return str(res)


def business_conversation(user_question, sql_query, query_result) -> str:
    output_parser = StrOutputParser()

    instruction = load_prompt("prompts/sql_conversation/main_v1.prompt").format(
        sql_query=sql_query, query_result=query_result
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=instruction),
            (
                "human",
                """user_question: {user_question}""",
            ),
        ]
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
    )
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output
