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

from unsloth import FastLanguageModel
import torch


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
            (
                "system",
                """당신은 사용자의 질문을 분석하여 데이터베이스 쿼리에 필요한 요소들을 파악하는 전문가입니다.
                    다음과 같은 요소들을 체계적으로 분석해주세요:
                    - 질문의 핵심 의도
                    - 필요한 데이터 항목
                    - 시간적 범위
                    - 데이터 필터링 조건
                    - 데이터 정렬 및 그룹화 요구사항
                    
                    분석 과정에서 발견된 문제점을 다음 태그를 사용하여 설명해주세요:
                    - [불명확] - 질문이 모호하거나 여러 해석이 가능한 경우
                    예시: "최근", "적절한" 같이 기준이 불분명한 표현
                    - [확인필요] - 추가 정보나 맥락이 필요한 경우
                    예시: 특정 용어의 정의나 기준이 명시되지 않은 경우
                    - [에러] - 논리적 모순이 발생한 경우""",
            ),
            (
                "human",
                """사용자 질문: {user_question}
                    
                    아래 형식으로 상세하게 분석해주세요:
                    - 주요 의도: 사용자가 질문의 핵심적으로 묻고자 하는 바는 무엇인가요?
                    - 필요한 데이터 항목: 사용자가 원하는 구체적인 데이터나 정보는 무엇인가요?
                    - 시간적 범위: 사용자가 원하는 시간적 범위는 어떻게 되나요?
                    - 데이터 필터링 조건: 사용자가 요청한 데이터에는 어떤 조건이나 필터링이 필요한가요?
                    - 데이터 정렬 및 그룹화: 사용자가 원하는 데이터의 정렬 기준이나 그룹화 방식은 무엇인가요?
                    
                    분석 과정에서 발견한 문제점을 아래에 태그와 함께 설명해주세요:
                    - [불명확]:
                    - [확인필요]:
                    - [에러]:""",
            ),
        ]
    )

    analyze_chain = ANALYZE_PROMPT | llm | output_parser
    analyze_question = analyze_chain.invoke({"user_question": user_question})

    return analyze_question


def clarify_user_question(user_question: str, user_question_analyze: str) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    user_add_questions = []
    final_analysis = user_question_analyze
    CLARIFY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 데이터 분석가로서 [불명확], [확인필요], [에러] 태그 뒤에 명시된 문제를 파악하고 해결하기 위한 적절한 추가 질문을 합니다. 이를 통해 문제가 해결됬는지 판단하세요. 
                
                추가 질문 조건:
                1. 태그([불명확], [확인필요], [에러])로 표시된 문제만 질문하고 이외 추가적인 질문은 절대 금지
                2. 한 번에 하나의 질문만 제시
                3. 객관식 질문으로 구성
                4. 구체적이고 이해하기 쉬운 질문으로 작성
                5. 이전 질문 기록을 검토하여 중복 질문 방지
                
                종료 조건:
                1. [불명확], [확인필요], [에러] 태그에 해당하는 모든 문제가 해결된 경우
                2. "해당 사항 없음"과 같은 사용자가 더 이상의 추가 정보 제공을 원하지 않는 경우
                3. 이전 질문 기록에 중복된 질문이 반복될 경우

                응답 형식:
                1. 추가 질문이 필요한 경우:
                "질문:\n1) 옵션 1\n2) 옵션 2\n3) 옵션 3\n4) 직접 입력\n"
                
                2. 종료 조건을 만족한 경우:
                "종료\n최종분석:\n주요 의도:\n필요한 데이터 항목:\n시간적 범위:\n데이터 필터링 조건:\n데이터 정령 및 그룹화 요구사항:"
                """,
            ),
            (
                "human",
                """원래 사용자 질문:
                {user_question}
                
                초기 질문 분석:
                {user_question_analyze}
                
                이전 질문 기록:
                {collected_questions}
                
                지시사항:
                태그([불명확], [확인필요], [에러])가 표시된 모든 문제가 해결되면 분석을 종료하세요.
                """,
            ),
        ]
    )

    while True:
        clarify_chain = CLARIFY_PROMPT | llm | output_parser
        collected_questions = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(user_add_questions)
        )

        clarify_question = clarify_chain.invoke(
            {
                "user_question": user_question,
                "user_question_analyze": user_question_analyze,
                "collected_questions": collected_questions,
            }
        )

        if clarify_question.startswith("종료"):
            final_analysis = clarify_question.split("최종분석:")[1].strip()
            print(f"\n{final_analysis}")
            break

        print(f"\n{clarify_question}", flush=True)
        user_answer = input(
            "답변을 입력하세요 (종료하려면 '해당 사항 없음' 입력): "
        ).strip()
        print(f"사용자 답변: {user_answer}")

        user_add_questions.append(
            f"\n질문: \n{clarify_question}\n답변: {user_answer}\n"
        )

    return final_analysis


def refine_user_question(user_question: str, user_question_analyze: str) -> str:
    output_parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    REFINE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 사용자의 질문을 더 명확하고 구체적으로 다듬는 전문가입니다. 주어진 질문과 분석을 바탕으로 더 구체적인 질문으로 재구성하세요.""",
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
    relevant_tables = vector_store.similarity_search(user_question, k=context_cnt)
    table_contexts = [doc.page_content for doc in relevant_tables]

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

    system_instruction = load_prompt("prompts/table_selection/main_v1.prompt")

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
            full_prompt = (
                prefix + main_prompt + postfix + f"\n\nuser_question: {user_question}"
            )
        else:
            regen_prompt = load_prompt(
                "prompts/query_creation/regenerate_v1.prompt"
            ).format(prev_query=prev_query, result_msg=error_msg)
            full_prompt = (
                prefix + regen_prompt + postfix + f"\n\nuser_question: {user_question}"
            )

        # Qwen 모델 로드 및 추론 준비
        model, tokenizer = load_qwen_model()
        model = FastLanguageModel.for_inference(model)  # 추론을 위한 모델 준비

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
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output
