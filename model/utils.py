from unsloth import FastLanguageModel
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline, BitsAndBytesConfig
from huggingface_hub import login

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os


def initialize_model():
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

    adapter_path = "100suping/Qwen2.5-Coder-34B-Instruct-kosql-adapter"
    adapter_revision = "checkpoint-200"
    model.load_adapter(
        adapter_path,
        revision=adapter_revision,
    )
    print("Adapter is loaded!")

    # Unsloth 사용 시 inference 모드 전환
    model = FastLanguageModel.for_inference(model)
    print("Inference mode activated!")

    # return model, tokenizer
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512
    )
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    print("HuggingFacePipeline is ready!")

    return hf_llm


def do_text2sql(system_prompt, human_prompt, input_dict, hf_llm):
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            ("human", human_prompt),
        ]
    )

    chain = prompt | hf_llm | output_parser

    output = chain.invoke(input_dict)
    response = output.split(human_prompt.format(**input_dict))[-1]
    return response
