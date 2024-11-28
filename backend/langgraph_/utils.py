from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import argparse
from dotenv import load_dotenv
import os


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


def load_prompt(prompt_path: str) -> str:
    """
    입력된 경로에 존재하는 프롬프트 파일을 로드합니다.

    Args:
        prompt_path (str): 프롬프트 파일의 경로.

    Returns:
        str: 로드된 프롬프트 내용.
    """
    with open(f"./{prompt_path}", "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
