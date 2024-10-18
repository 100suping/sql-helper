from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
            (
                "system",
                "당신은 사용자의 입력을 SQL문으로 바꾸어주는 조직의 팀원입니다. 당신의 임무는 주어진 질문(user_question)이 데이터 또는 비즈니스와 관련된 일인지를 판단하는 것 입니다.",
            ),
            (
                "human",
                "주어진 질문(user_question)이 데이터 또는 비즈니스와 관련되어 있으면 1, 아니면 0을 출력하세요. `1`과 같이 결과만 작성하세요.\n\n#질문(user_question): {user_question}",
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
    # TODO
    # 동일한 함수에서 저장된 prompt의 경로만 교체해서 효율성을 높이는 것을 고려해보아야 한다
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 SQL 헬퍼 입니다.",
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


def do_embodiment(user_question: str) -> str:
    """데이터 및 비즈니스 관련 질문으로 판단된 사용자의 질문을 구체화하는 함수입니다.
    구체화를 어떤 식으로 진행할지는 아직 정해지지 않았으니, 일단은 사용자의 질문을 2배로 늘린 결과를 return 합니다.

    Args:
        user_question (str): 데이터 및 비즈니스 관련 사용자의 질문

    Returns:
        str: 사용자의 질문을 구체화한 결과
    """

    return user_question * 2


def do_extraction(user_question: str) -> list:
    """데이터 및 비즈니스 관련 질문으로 판단된 사용자의 질문에서 필요한 정보들을 추출하는 함수입니다.
    정보 추출을 어떤 식으로 진행할지는 아직 정해지지 않았으니, 일단은 사용자의 질문을 띄어쓰기 기준으로 split 한 결과를 return 합니다.

    Args:
        user_question (str): 데이터 및 비즈니스 관련 사용자의 질문

    Returns:
        list: 사용자의 질문에서 추출된 정보
    """
    return user_question.split()


def do_category_classification(
    user_question: str, embodied_question: str, extracted_data: list[str]
) -> str:
    """사용자의 질문에 대해서 우리가 정의한 기능 중 어떤 기능으로 대응해야 하는지 분류해주는 함수입니다.
    사용할 모델 또는 api 에 따라서 입력에 필요한 데이터가 달라질 수 있으며, 일단은 모두 입력 받아오도록 만들었습니다.
    아직 데이터 및 라벨에 대한 정의가 존재하지 않으니 임의로 생성하여 랜덤한 값을 return 합니다.

    Args:
        user_question (str): 데이터 및 비즈니스 관련 사용자의 질문
        embodied_question (str): 사용자의 질문을 구체화한 결과
        extracted_data (list[str]): 사용자의 질문에서 추출된 정보

    Returns:
        str: 기능 분류 결과
    """

    from random import choice

    labels = [
        "쿼리문 생성",
        "쿼리문 해설",
        "테이블 해설",
        "쿼리문 문법 검증",
        "테이블 및 컬럼 활용 안내",
    ]
    return choice(labels)
