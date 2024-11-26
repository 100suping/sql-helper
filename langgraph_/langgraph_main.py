import argparse
from dotenv import load_dotenv

from .node import GraphState
from .graph import make_graph, make_graph_for_test
from .utils import get_runnable_config, str2bool
from .faiss_init import get_vector_stores


def get_config():
    # parser 생성
    parser = argparse.ArgumentParser(
        prog="Get Config",
    )

    parser.add_argument("--recursion-limit", default=50, type=int)

    parser.add_argument("--thread-id", default="TEST_RUN", type=str)
    # TODO
    # 현재 구조에서 argument로 model_name을 받아 해당 모델을 사용하도록 만들기 위해서는 대규모 공사가 필요
    # 추후에 진행 예정
    parser.add_argument("--model-name", default="gpt-4o-mini", type=str)

    parser.add_argument("--context-cnt", default=10, type=int)

    parser.add_argument(
        "--user-question",
        default="common_db의 전체 주문 중 환불된 주문의 비율은 얼마인가요?",
        type=str,
    )

    parser.add_argument(
        "--max-query-fix",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--query-fix-cnt",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "--sample-info",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--test",
        default=True,
        type=str2bool,
    )

    config = parser.parse_args()

    return config


def text2sql(user_input, test: bool = True):
    # config 설정
    config = get_config()
    runnable_config = get_runnable_config(
        recursion_limit=config.recursion_limit, thread_id=config.thread_id
    )

    # graph 생성
    if test:
        graph = make_graph_for_test()
    else:
        graph = make_graph()

    # 입력을 위해 그래프 상태 만들기
    inputs = GraphState(user_question=user_input, context_cnt=config.context_cnt, max_query_fix=config.max_query_fix, query_fix_cnt=config.query_fix_cnt, sample_info=config.sample_info)  # type: ignore
    outputs = graph.invoke(input=inputs, config=runnable_config)

    return outputs["final_answer"]


if __name__ == "__main__":
    load_dotenv()

    config = get_config()

    print(text2sql(user_input=config.user_question, test=config.test))
