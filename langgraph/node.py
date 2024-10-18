from typing import TypedDict
from task import (
    evaluate_user_question,
    simple_conversation,
    do_embodiment,
    do_extraction,
    do_category_classification,
)


# GrpahState 정의
class GraphState(TypedDict):
    # Warning!
    # 그래프 내에서 사용될 모든 key값을 정의해야 오류가 나지 않는다.
    user_question: str  # 사용자의 질문
    user_question_eval: str  # 사용자 질문 평가
    embodied_question: str  # 사용자의 구체화 된 질문
    extracted_data: list[str]  # 사용자의 질문에서 추출된 정보
    category: str  # 사용자의 질문에 대해 대응해야하는 카테고리
    final_answer: str  # 서비스 흐름에서 사용자에게 전달될 최종 답변


########################### 정의된 노드 ###########################
def question_evaluation(state: GraphState) -> GraphState:
    """사용자의 질문을 평가하는 작업을 진행하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 평가한 결과가 추가된 그래프 상태
    """
    user_question = state["user_question"]
    # 사용자 질문 평가
    user_question_eval = evaluate_user_question(user_question)

    return GraphState(user_question_eval=user_question_eval)  # type: ignore


def embodying_and_extracting(state: GraphState) -> GraphState:
    """사용자의 질문을 가지고 구체화하고 정보를 추출하는 작업을 진행하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 구체화한 결과와 사용자의 질문에서 추출된 정보 추가된 그래프 상태
    """
    user_question = state["user_question"]
    # 사용자 질문을 구체화
    embodied_question = do_embodiment(user_question)
    # 사용자 질문에서 필요 정보 추출
    extracted_data = do_extraction(user_question)

    return GraphState(
        embodied_question=embodied_question, extracted_data=extracted_data
    )  # type: ignore


def classify_question_category(state: GraphState) -> GraphState:
    """사용자의 질문 및 기타 정보(현재 버전: 구체화된 질문, 추출된 정보)를 가지고
    사용자의 질문을 기능적으로 분류하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 기능적으로 분류한 결과가 추가된 그래프 상태
    """
    user_question = state["user_question"]
    embodied_question = state["embodied_question"]
    extracted_data = state["extracted_data"]

    # 사용자 질문과 구체화 정보,필요 정보들을 고려하여 분류
    category = do_category_classification(
        user_question, embodied_question, extracted_data
    )

    return GraphState(category=category)  # type: ignore


def non_sql_conversation(state: GraphState) -> GraphState:
    """일상적인 대화를 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문에 대한 대답이 추가된 그래프 상태
    """
    user_question = state["user_question"]
    final_answer = simple_conversation(user_question)

    return GraphState(final_answer=final_answer)  # type: ignore


def crete_query(state: GraphState) -> GraphState:
    """쿼리문 생성 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def explain_query(state: GraphState) -> GraphState:
    """쿼리문 설명 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def explain_table(state: GraphState) -> GraphState:
    """테이블 설명 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def correct_query_grammar(state: GraphState) -> GraphState:
    """쿼리문 문법 검증의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def guide_utilization(state: GraphState) -> GraphState:
    """테이블 및 컬럼 활용 안내 기능의 시작점이 되는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: _description_
    """
    final_answer = state["category"]
    return GraphState(final_answer=final_answer)  # type: ignore


def user_question_checker(state: GraphState) -> str:
    """그래프 상태에서 사용자의 질문 분류 결과를 가져오는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 사용자의 질문 분류 결과 ("1" or "0")
    """
    return state["user_question_eval"]


def category_checker(state: GraphState) -> str:
    """사용자의 질문을 기능적으로 분류한 결과를 반환하는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 기능 분류 결과
    """
    return state["category"]
