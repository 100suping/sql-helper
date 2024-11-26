from typing import TypedDict, List, Any
from .utils import EmptyQueryResultError, NullQueryResultError
from .task import (
    evaluate_user_question,
    simple_conversation,
    select_relevant_tables,
    extract_context,
    create_query,
    analyze_user_question,
    refine_user_question,
    clarify_user_question,
    check_leading_question,
    get_query_result,
    business_conversation,
)

# FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다.
from .faiss_init import get_vector_stores


# GrpahState 정의
class GraphState(TypedDict):
    # Warning!
    # 그래프 내에서 사용될 모든 key값을 정의해야 오류가 나지 않는다.
    user_question: str  # 사용자의 질문
    user_question_eval: str  # 사용자의 질문이 SQL 관련 질문인지 여부
    user_question_analyze: str  # 사용자 질문 분석
    collected_questions: List[str]  # 사용자의 질문에 대한 추가 질문-대답 기록
    ask_user: int  # leading question 질문 여부 [0, 1]
    final_answer: str
    # TODO
    # context_cnt가 동적으로 조절 되도록 알고리즘을 짜야 한다.
    context_cnt: int  # 사용자의 질문에 대답하기 위해서 정보를 가져올 context 갯수
    table_contexts: List[str]
    table_contexts_ids: List[int]
    need_clarification: bool  # 사용자 추가 질문(설명)이 필요한지 여부
    sample_info: int
    sql_query: str
    flow_status: str  # KEEP, REGENERATE, RE-RETRIEVE, RESELECT, RE-REQUEST
    max_query_fix: int
    query_fix_cnt: int
    query_result: List[Any]
    error_msg: str
    # TODO
    # 지금은 FAISS 벡터 DB를 쓰기에 아래와 같이 딕셔너리에 넣어놓지만, Redis DB 서버를 만들어서 이용할 경우에는 index가 들어가야 한다.
    # FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다. 노드 안에서 객체를 불러오는 것으로 한다.
    # vector_store_dict: Dict[str, VectorStore | FAISS]  # RAG를 위한 벡터 DB


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


def question_analyze(state: GraphState) -> GraphState:
    """질문 분석을 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 분석한 대답이 추가된 그래프 상태
    """
    user_question = state["user_question"]
    analyze_question = analyze_user_question(user_question)

    return GraphState(user_question_analyze=analyze_question)  # type: ignore


def question_clarify(state: GraphState) -> GraphState:
    """사용자 질문이 모호할 경우 추가 질문을 통해 질문 분석을 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 분석한 대답이 추가된 그래프 상태
    """
    user_question_analyze = state["user_question_analyze"]
    user_question = state["user_question"]
    collected_questions = state.get("collected_questions", [])

    leading_question = clarify_user_question(
        user_question, user_question_analyze, collected_questions
    )
    ask_user = check_leading_question(leading_question)
    collected_questions.append(leading_question)

    return GraphState(collected_questions=collected_questions, ask_user=ask_user)  # type: ignore


def human_feedback(state: GraphState) -> GraphState:

    return state


def question_refine(state: GraphState) -> GraphState:
    """질문 구체화를 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문에 대한 대답이 추가된 그래프 상태
    """
    collected_questions = state["collected_questions"]
    user_question_analyze = collected_questions[-1]
    user_question = state["user_question"]
    refine_question = refine_user_question(user_question, user_question_analyze)

    return GraphState(user_question=refine_question)  # type: ignore


def table_selection(state: GraphState) -> GraphState:
    """사용자의 질문과 연관된 테이블 메타 데이터를 검색하고 검수하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 검색된 context들과 검수를 통과한 context의 index list가 추가된 그래프 상태
    """
    user_question = state["user_question"]
    context_cnt = state["context_cnt"]
    flow_status = state.get("flow_status", "KEEP")
    sample_info = state["sample_info"]
    vector_store = get_vector_stores(sample_info)["db_info"]  # table_ddl + 실제 데이터
    # vector_store = get_vector_stores()["table_info"]  # table_ddl
    # FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다.
    # vector_store = state["vector_store_dict"]["table_info"] # table_ddl
    # 사용자 질문과 관련성이 있는 테이블+컬럼정보를 검색
    table_contexts = select_relevant_tables(
        user_question=user_question, context_cnt=context_cnt, vector_store=vector_store
    )
    # 검색된 context를 검수
    if flow_status == "RESELECT":
        prev_list = state["table_contexts_ids"]
        prev_query = state["sql_query"]
        error_msg = state["error_msg"]
        table_contexts_ids = extract_context(
            user_question=user_question,
            table_contexts=table_contexts,
            flow_status=flow_status,
            prev_list=prev_list,
            prev_query=prev_query,
            error_msg=error_msg,
        )
    else:
        table_contexts_ids = extract_context(
            user_question=user_question, table_contexts=table_contexts
        )
    return GraphState(
        table_contexts=table_contexts,
        table_contexts_ids=table_contexts_ids,
        flow_status=flow_status,
    )  # type: ignore


def query_creation(state: GraphState) -> GraphState:
    user_question = state["user_question"]
    table_contexts = state["table_contexts"]
    table_contexts_ids = state["table_contexts_ids"]

    # 디버깅을 위한 타입 체크
    print(f"Type of user_question in node.py: {type(user_question)}")
    print(f"Value of user_question in node.py: {user_question}")

    # user_question이 문자열이 아닌 경우 문자열로 변환
    if not isinstance(user_question, str):
        user_question = str(user_question)

    query_fix_cnt = state.get("query_fix_cnt")
    flow_status = state.get("flow_status", "KEEP")

    if flow_status == "REGENERATE":
        prev_query = state["sql_query"]
        error_msg = state["error_msg"]
        print("Do Query Fix!!!")
        sql_query = create_query(
            user_question,
            table_contexts,
            table_contexts_ids,
            flow_status=flow_status,
            prev_query=prev_query,
            error_msg=error_msg,
        )

    else:
        sql_query = create_query(user_question, table_contexts, table_contexts_ids)

    return GraphState(
        sql_query=sql_query,
        query_fix_cnt=query_fix_cnt + 1,
        flow_status=flow_status,
    )  # type: ignore


def query_validation(state: GraphState) -> GraphState:
    sql_query = state["sql_query"]
    query_fix_cnt = state["query_fix_cnt"]
    max_query_fix = state["max_query_fix"]
    try:
        query_result = get_query_result(command=sql_query, fetch="all")
        return GraphState(
            query_result=query_result,
            flow_status="KEEP",
        )  # type: ignore

    except Exception as e:
        # 지정된 최대 재생성 횟수를 넘어서면 사이클 중단
        if query_fix_cnt >= max_query_fix:

            return GraphState(
                flow_status="KEEP",
                query_result=e._message(),
            )  # type: ignore

        else:

            if isinstance(e, NullQueryResultError):
                # 쿼리문 결과가 모두 NULL인 경우 -> 테이블 재선택
                print("Null Query Result Error")
                return GraphState(
                    flow_status="RESELECT",
                    error_msg=e._message(),
                )  # type: ignore

            elif isinstance(e, EmptyQueryResultError):
                # 쿼리문 결과가 없는 경우 -> 테이블 재선택
                print("Empty Query Result Error")
                return GraphState(
                    flow_status="RESELECT",
                    error_msg=e._message(),
                )  # type: ignore
            else:
                # 쿼리문의 문법 오류 -> 쿼리문 재생성
                return GraphState(
                    flow_status="REGENERATE",
                    error_msg=e._message(),
                )  # type: ignore


def sql_conversation(state: GraphState) -> GraphState:
    user_question = state["user_question"]
    sql_query = state["sql_query"]
    query_result = state["query_result"]
    final_answer = business_conversation(
        user_question, sql_query=sql_query, query_result=query_result
    )

    return GraphState(final_answer=final_answer)  # type: ignore


################### ROUTERS ###################
def user_question_checker(state: GraphState) -> str:
    """그래프 상태에서 사용자의 질문 분류 결과를 가져오는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 사용자의 질문 분류 결과 ("1" or "0")
    """
    return state["user_question_eval"]


def user_question_analyze_checker(state: GraphState) -> str:
    user_question_analyze = state["user_question_analyze"]
    analyze_question = analyze_user_question(user_question_analyze)

    keywords = ["[불명확]", "[확인필요]", "[에러]"]

    if any(keyword in analyze_question for keyword in keywords):
        state["need_clarification"] = True
    else:
        state["need_clarification"] = False

    return state["need_clarification"]


def leading_question_checker(state: GraphState) -> str:
    ask_user = state["ask_user"]
    if ask_user == 0:
        return "ESCAPE"
    else:
        return "KEEP"


def query_checker(state: GraphState) -> str:
    flow_status = state["flow_status"]
    max_query_fix = state["max_query_fix"]
    query_fix_cnt = state["query_fix_cnt"]

    # 정해진 최대 재생성 횟수를 넘어서면 pass
    if flow_status == "KEEP" or query_fix_cnt >= max_query_fix:
        return "KEEP"
    else:
        return flow_status
