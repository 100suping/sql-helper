from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from .node import (
    GraphState,
    question_evaluation,
    table_selection,
    non_sql_conversation,
    user_question_checker,
    user_question_analyze_checker,
    query_creation,
    question_analyze,
    question_refine,
    question_clarify,
    query_validation,
    sql_conversation,
    user_question_analyze_checker,
    query_checker,
)


def make_graph() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("question_evaluation", question_evaluation)  # 질문 평가
    workflow.add_node("general_conversation", non_sql_conversation)  # 일반적인 대화
    workflow.add_node("question_analysis", question_analyze)  # 질문 분석
    workflow.add_node("additional_questions", question_clarify)  # 추가 질문
    workflow.add_node("question_refinement", question_refine)  # 질문 구체화
    workflow.add_node("table_selection", table_selection)  # 테이블 선택
    workflow.add_node("sql_query_generation", query_creation)  # SQL 쿼리 생성
    workflow.add_node("sql_query_validation", query_validation)  # SQL 쿼리 결과 확인
    workflow.add_node("response", sql_conversation)  # 답변

    workflow.add_conditional_edges(
        "question_evaluation",
        user_question_checker,
        {
            "0": "general_conversation",
            "1": "question_analysis",
        },
    )

    workflow.add_conditional_edges(
        "question_analysis",
        user_question_analyze_checker,
        {
            True: "additional_questions",
            False: "question_refinement",
        },
    )

    workflow.add_edge("additional_questions", "question_refinement")
    workflow.add_edge("question_refinement", "table_selection")
    workflow.add_edge("table_selection", "sql_query_generation")
    workflow.add_edge("sql_query_generation", "sql_query_validation")

    workflow.add_conditional_edges(
        "sql_query_validation",
        query_checker,
        {
            "KEEP": "response",
            "REGENERATE": "sql_query_generation",
            "RESELECT": "table_selection",
        },
    )

    workflow.add_edge("response", END)

    # Set the entry point
    workflow.set_entry_point("question_evaluation")

    # Set up memory storage for recording
    memory = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    return app


def make_graph_for_test() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("question_evaluation", question_evaluation)  # 질문 평가
    workflow.add_node("general_conversation", non_sql_conversation)  # 일반적인 대화
    # workflow.add_node("question_analysis", question_analyze)  # 질문 분석
    # workflow.add_node("additional_questions", question_clarify)  # 추가 질문
    # workflow.add_node("question_refinement", question_refine)  # 질문 구체화
    workflow.add_node("table_selection", table_selection)  # 테이블 선택
    workflow.add_node("sql_query_generation", query_creation)  # SQL 쿼리 생성
    workflow.add_node("sql_query_validation", query_validation)  # SQL 쿼리 결과 확인
    workflow.add_node("response", sql_conversation)  # 답변

    workflow.add_conditional_edges(
        "question_evaluation",
        user_question_checker,
        {
            "0": "general_conversation",
            "1": "table_selection",
        },
    )

    workflow.add_edge("table_selection", "sql_query_generation")
    workflow.add_edge("sql_query_generation", "sql_query_validation")

    workflow.add_conditional_edges(
        "sql_query_validation",
        query_checker,
        {
            "KEEP": "response",
            "REGENERATE": "sql_query_generation",
            "RESELECT": "table_selection",
        },
    )

    workflow.add_edge("response", END)

    # Set the entry point
    workflow.set_entry_point("question_evaluation")

    # Set up memory storage for recording
    memory = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    return app
