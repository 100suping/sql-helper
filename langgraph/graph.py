from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from node import (
    GraphState,
    question_evaluation,
    embodying_and_extracting,
    classify_question_category,
    non_sql_conversation,
    crete_query,
    explain_query,
    explain_table,
    correct_query_grammar,
    guide_utilization,
    user_question_checker,
    category_checker,
)


def make_graph() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("질문 평가", question_evaluation)
    workflow.add_node("일반적인 대화", non_sql_conversation)
    workflow.add_node("질문 구체화 및 정보 추출", embodying_and_extracting)

    workflow.add_node("질문 카테고리 분류", classify_question_category)
    workflow.add_node("crete_query", crete_query)
    workflow.add_node("explain_query", explain_query)
    workflow.add_node("explain_table", explain_table)
    workflow.add_node("correct_query_grammar", correct_query_grammar)
    workflow.add_node("guide_utilization", guide_utilization)

    workflow.add_conditional_edges(
        "질문 평가",  # 사용자의 질문이 데이터 및 비즈니스와 관련 되어있는지 평가
        user_question_checker,  # check로 평가 데이터를 추출한 뒤, 아래의 조건에 따라 다른 노드에 전달
        {
            "0": "일반적인 대화",
            "1": "질문 구체화 및 정보 추출",
        },
    )

    workflow.add_edge("질문 구체화 및 정보 추출", "질문 카테고리 분류")

    workflow.add_conditional_edges(
        "질문 카테고리 분류",
        category_checker,
        {
            "쿼리문 생성": "crete_query",
            "쿼리문 해설": "explain_query",
            "테이블 해설": "explain_table",
            "쿼리문 문법 검증": "correct_query_grammar",
            "테이블 및 컬럼 활용 안내": "guide_utilization",
        },
    )

    # 시작점을 설정합니다.
    workflow.set_entry_point("질문 평가")

    # 기록을 위한 메모리 저장소를 설정합니다.
    memory = MemorySaver()

    # 그래프를 컴파일합니다.
    app = workflow.compile(checkpointer=memory)
    return app
