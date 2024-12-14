from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langgraph_.graph import make_graph, multiturn_test, make_graph_for_test
from langgraph_.utils import (
    get_runnable_config,
    extract_context_tables,
    save_conversation,
)
from dotenv import load_dotenv

app = FastAPI()

# make_graph: 전체 과정, make_graph_for_test: 질문 구체화 생략, multiturn_test: 질문 구체화만 진행
workflow = make_graph()
print("LLM WORKFLOW STARTED.")


class LLMWorkflowInput(BaseModel):
    user_question: str
    initial_question: int
    thread_id: str
    last_snapshot_values: dict | None
    llm_api: str


class UserFeedbackInput(BaseModel):
    user_feedback: int  # 좋아요 1-chosen, 싫어요 0-rejected
    thread_id: str


@app.post("/llm_workflow")
def llm_workflow(workflow_input: LLMWorkflowInput):
    global workflow
    processed_input = workflow_input.model_dump()
    config = get_runnable_config(30, processed_input["thread_id"])
    inputs = {
        "user_question": processed_input["user_question"],
        "context_cnt": 10,
        "max_query_fix": 2,
        "query_fix_cnt": -1,
        "sample_info": 5,
        "llm_api": processed_input["llm_api"],
    }
    # 초기 질문이 아닌 경우
    if processed_input["initial_question"] == 0:
        values = processed_input["last_snapshot_values"]
        values["collected_questions"][
            -1
        ] += f"\n답변: {processed_input['user_question']}"
        values["llm_api"] = processed_input["llm_api"]
        workflow.update_state(
            config,
            values,
            "additional_questions",
        )
        outputs = workflow.invoke(
            input=None,
            config=config,
            interrupt_before=["human_feedback"],
        )
    else:  # 초기 질문인 경우
        # TODO
        workflow = make_graph()
        # 첫 번째 초기질문은 잘 작동하나 두번째 초기질문에서 GraphState가 초기화되지 않는 문제 발생
        outputs = workflow.invoke(
            input=inputs,
            config=config,
            interrupt_before=["human_feedback"],
        )
    # print(outputs)
    if "final_answer" in outputs and outputs["user_question_eval"] == "1":
        print(
            "RAG과정에서 사용된 context table:",
            extract_context_tables(
                outputs["table_contexts"], outputs["table_contexts_ids"]
            ),
        )
    return outputs


@app.post("/user_feedback")
def user_feedback(feedback_input: UserFeedbackInput):
    processed_input = feedback_input.model_dump()
    feedback = processed_input["user_feedback"]
    config = get_runnable_config(30, processed_input["thread_id"])
    snapshot = list(workflow.get_state_history(config))[0].values

    if snapshot["user_question_eval"] == "1":
        save_conversation(snapshot, feedback)
    else:
        print("simple conversation would not be saved.")


if __name__ == "__main__":
    load_dotenv(override=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
