from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from unsloth import FastLanguageModel
from huggingface_hub import login


def str2bool(value: str) -> bool:
    """
    문자열을 부울 값으로 변환합니다.

    Args:
        value (str): 변환할 문자열.

    Returns:
        bool: 변환된 부울 값.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"Cannot convert {value} to a boolean.")


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


def load_qwen_model():
    # Hugging Face 토큰 설정
    huggingface_token = "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"
    login(token=huggingface_token)

    # 모델 이름 설정
    model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"

    # 8비트 양자화 설정
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # FastLanguageModel을 사용하여 모델과 토크나이저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, token=huggingface_token, quantization_config=bnb_config
    )

    # Unsloth 사용 시 inference 모드 전환
    model = FastLanguageModel.for_inference(model)

    return model, tokenizer
