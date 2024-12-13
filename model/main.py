from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, os
from dotenv import load_dotenv

from utils import initialize_model, do_text2sql


load_dotenv(override=True)
hf_llm = initialize_model()

app = FastAPI()


class LLMInput(BaseModel):
    input_dict: dict
    system_prompt: str
    human_prompt: str


@app.post("/qwen")
def qwen(model_input: LLMInput):
    processed_input = model_input.model_dump()
    input_dict = processed_input["input_dict"]
    system_prompt = processed_input["system_prompt"]
    human_prompt = processed_input["human_prompt"]

    response = do_text2sql(system_prompt, human_prompt, input_dict, hf_llm)
    print("Text2SQL Done.")
    print(response)
    return {"response": response}
