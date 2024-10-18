import argparse

from node import GraphState
from graph import make_graph
from utils import get_runnable_config


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

    parser.add_argument(
        "--user-question",
        default="강호준이 최근 일주일간 몇캔의 닥터페퍼를 마셨는지 알려줘",
        type=str,
    )

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    # config 설정
    config = get_config()
    runnable_config = get_runnable_config(
        recursion_limit=config.recursion_limit, thread_id=config.thread_id
    )

    # graph 생성
    graph = make_graph()

    # 입력을 위해 그래프 상태 만들기
    inputs = GraphState(user_question=config.user_question)  # type: ignore
    outputs = graph.invoke(input=inputs, config=runnable_config)

    print(outputs)
