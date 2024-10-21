import os
import json
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentType
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

import ast
import re


# 환경 변수 로드 및 DB 연결 설정 함수
def load_env_db():
    # .env 파일에서 환경 변수 로드
    load_dotenv()

    # DB 연결
    URL = os.getenv("URL")
    db = SQLDatabase.from_uri(URL)
    return db


# 임베딩 및 검색기 생성 함수
def create_retriever_from_texts(texts):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(texts, embeddings)
    retriever = vector_db.as_retriever()
    return retriever


# Document 리스트를 사용하여 검색기 생성 함수
def create_retriever_from_documents(documents):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever()
    return retriever


# 에이전트 생성 함수
def create_agent(llm, db, tools, suffix):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        extra_tools=tools,
        suffix=suffix,
    )
    return agent


def few_shot(config):
    # DB 연결 설정
    db = load_env_db()

    # JSON 파일에서 few_shot 데이터 로드
    with open(config.few_shot_path, "r", encoding="utf-8") as file:
        few_shots = json.load(file)

    # Document 객체 리스트 생성
    documents = [
        Document(page_content=question, metadata={"sql_query": few_shots[question]})
        for question in few_shots
    ]

    # Document를 사용한 검색기 생성
    retriever = create_retriever_from_documents(documents)

    # 검색 도구 생성
    retriever_tool = create_retriever_tool(
        retriever,
        name="sql_get_similar_examples",
        description="이 도구는 유사한 예시를 이해하여 사용자 질문에 적용하는 데 도움이 됩니다.",
    )

    # LLM 설정
    llm = ChatOpenAI(model_name=config.model_name, temperature=0)

    # 커스텀 서픽스 정의
    custom_suffix = """
    먼저 제가 알고 있는 비슷한 예제를 가져와야 합니다.
    예제가 쿼리를 구성하기에 충분하다면 쿼리를 작성할 수 있습니다.
    그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
    그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
    """

    # 에이전트 생성 및 반환
    return create_agent(llm, db, [retriever_tool], custom_suffix)


def cardinality(config):
    # DB 연결 설정
    db = load_env_db()

    # 쿼리 결과를 리스트로 변환하는 함수
    def query_as_list(query):
        res = db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return res

    # "employees" 테이블에서 성 및 이름 목록 가져오기
    artists = query_as_list("SELECT DISTINCT lastName FROM employees")
    albums = query_as_list("SELECT DISTINCT firstName FROM employees")
    texts = artists + albums

    # 텍스트를 사용한 검색기 생성
    retriever = create_retriever_from_texts(texts)

    # 검색 도구 생성
    retriever_tool = create_retriever_tool(
        retriever,
        name="name_search",
        description="이름, 성 데이터가 실제로 어떻게 쓰여졌는지 알아내는 데 사용합니다.",
    )

    # LLM 설정
    llm = ChatOpenAI(model_name=config.model_name, temperature=0)

    # 커스텀 서픽스 정의
    custom_suffix = """
    사용자가 고유명사를 기준으로 필터링해 달라고 요청하는 경우, 먼저 name_search 도구를 사용하여 철자를 확인해야 합니다.
    그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
    그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
    """

    # 에이전트 생성 및 반환
    return create_agent(llm, db, [retriever_tool], custom_suffix)
