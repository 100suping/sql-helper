from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import argparse, os, re, csv
from datetime import datetime


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


def load_qwen_model():
    from unsloth import FastLanguageModel

    global model, tokenizer
    # Hugging Face 토큰 설정

    # 환경 변수에서 Hugging Face 토큰 가져오기
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=huggingface_token)

    # 모델 이름 설정
    model_name = "unsloth/Qwen2.5-Coder-32B-Instruct"

    # 8비트 양자화 설정
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    home_path = os.path.expanduser("~")
    cache_dir = os.path.join(home_path, "sql-helper/.cache/unsloth")

    # FastLanguageModel을 사용하여 모델과 토크나이저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        token=huggingface_token,
        quantization_config=bnb_config,
        cache_dir=cache_dir,
        device_map="auto",
    )

    # Unsloth 사용 시 inference 모드 전환
    model = FastLanguageModel.for_inference(model)

    return model, tokenizer


def extract_context_tables(table_contexts, table_contexts_ids):
    context_table_list = []
    if not table_contexts_ids:
        return []
    table_pattern = r"CREATE TABLE\s+(.+?)\s*\("
    for idx in table_contexts_ids:
        table_name = re.findall(table_pattern, table_contexts[idx])[-1]
        context_table_list.append(table_name.strip("`"))

    return context_table_list


def save_conversation(snapshot, feedback):
    os.makedirs("logs", exist_ok=True)

    # CSV 파일명 생성 (현재 시간 포함)
    timestamp = datetime.now().strftime("%Y%m%d")
    csv_filename = f"logs/user_feedback_{timestamp}.csv"
    # 파일을 새로 만들지, 이어 쓸지 확인
    if os.path.isfile(f"./logs/user_feedback_{timestamp}.csv"):
        mode = "a"
    else:
        mode = "w"
    # CSV 파일 생성
    with open(csv_filename, mode=mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quotechar='"')
        # 첫 작성 시 헤더 작성
        if mode == "w":
            writer.writerow(
                [
                    "user_question",
                    "collected_questions",
                    # "table_contexts",
                    # "table_contexts_ids",
                    "table_names",
                    "query_result",
                    "final_answer",
                    "feedback",
                    "timestamp",
                ]
            )
        writer.writerow(
            [
                snapshot["user_question"],
                snapshot["collected_questions"],
                # snapshot["table_contexts"],
                # snapshot["table_contexts_ids"],
                extract_context_tables(
                    snapshot["table_contexts"], snapshot["table_contexts_ids"]
                ),
                snapshot["query_result"],
                snapshot["final_answer"],
                feedback,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 현재 시간 기록
            ]
        )
    print("Save Completed!")
