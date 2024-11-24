from langchain_core.runnables import RunnableConfig


def get_runnable_config(recursion_limit: int, thread_id: str) -> RunnableConfig:
    # set config
    # RunnableConfig에 thread id로 추적 기록을 추적 가능하게 할 수 있습니다.
    # recursion_limit은 최대 노드를 몇번 거치게 할 것인지에 대한 한계 값입니다.
    config = RunnableConfig(
        recursion_limit=recursion_limit, configurable={"thread_id": thread_id}
    )
    return config


class EmptyQueryResultError(Exception):
    def __init__(self):
        self.msg = "No rows returned by the SQL query."

    def __str__(self):
        return self.msg

    # SQLAlchemy 에서 에러메시지를 출력하기 위한 메서드
    def _message(self):
        return self.msg


class NullQueryResultError(Exception):
    def __init__(self):
        self.msg = "SQL query only returns NULL for every column."

    def __str__(self):
        return self.msg

    # SQLAlchemy 에서 에러메시지를 출력하기 위한 메서드
    def _message(self):
        return self.msg
