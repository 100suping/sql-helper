import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os


load_dotenv()


# GPT 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# DB 연결 초기화
URL = os.getenv("URL")
db = SQLDatabase.from_uri(URL)


# 세션 상태 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# SQL agents 사용
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)


# Streamlit UI
st.title("GPT로 SQL 쿼리 생성기")
user_input = st.text_input("데이터에 대해 질문하세요요:")


if st.button("쿼리 생성 및 실행"):
    # SQL 에이전트를 통해 쿼리 생성 및 실행
    response = agent_executor.invoke({"input": user_input})

    # response의 전체 내용 확인 (디버깅 목적)
    st.write("Response 전체 내용:")
    st.write(response)

    # 대화 기록에 추가
    st.session_state.conversation_history.append(
        {"question": user_input, "response": response}
    )

# 대화 기록 표시
st.subheader("대화 기록록")
for entry in st.session_state.conversation_history:
    st.write(f"질문: {entry['question']}")
    st.write(f"Response 전체 내용: {entry['response']}")
