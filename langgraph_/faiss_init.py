import pandas as pd
import os
from typing import List, Dict, Tuple
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase

from sqlalchemy import create_engine, inspect, MetaData
from dotenv import load_dotenv
from sqlalchemy.schema import CreateTable


def get_db_names(engine) -> List[str]:
    db_names_query = """
    SELECT 
    DISTINCT
        TABLE_SCHEMA AS db
    FROM TABLES
    WHERE TABLE_SCHEMA NOT IN ('mysql', 'performance_schema', 'sys', 'information_schema');
    """

    print("접근 가능한 모든 데이터베이스를 가져오는 중...")
    db_names_df = pd.read_sql(db_names_query, engine)
    db_names = []
    for _, row in db_names_df.iterrows():
        db_names.append(row["db"])

    print(f"총 {len(db_names)}의 접근 가능한 모든 데이터베이스를 가져왔습니다.\n")

    return db_names


def embed_db_meta_data(db_names):
    db_meta_data = {}
    db_meta_data["accounting_db"] = (
        "accounting_db는 주로 회계 및 환불 관련 트랜잭션, 통계 및 용어 정의를 관리하기 위해 설계된 데이터베이스입니다. 이 데이터베이스는 고객의 구매 및 환불 내역을 추적하고, 환불 비용을 관리하며, 회계 용어 및 정의를 저장하는 여러 테이블로 구성되어 있습니다. 데이터베이스의 주요 목적은 회계 데이터의 정확한 분석과 보고를 통해 재무 결정을 지원하는 것입니다."
    )
    db_meta_data["common_db"] = (
        "common_db는 고객, 제품, 주문 및 주문 항목에 대한 정보를 관리하는 데이터베이스입니다. 이 데이터베이스는 전자상거래 환경에서 고객과 제품 간의 상호작용을 효율적으로 처리할 수 있도록 설계되었습니다."
    )
    db_meta_data["cs_db"] = (
        "cs_db는 고객 서비스(CS) 관련 데이터를 저장하고 관리하기 위한 데이터베이스입니다. 이 데이터베이스는 주로 환불 요청, 통계, 용어 정의, 환불 처리 및 고객 만족도 데이터를 포함하여 고객 서비스 운영의 효율성을 높이고 고객 경험을 개선하는 데 기여합니다."
    )

    db_meta_texts = []
    for db_name in db_names:
        text = f"""DB name: {db_name}\nBusniss meta data: {db_meta_data[db_name]}"""
        db_meta_texts.append(text)

    print("벡터 데이터베이스(DB 메타 데이터) 생성 중...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(db_meta_texts, embeddings)
    print("벡터 데이터베이스(DB 메타 데이터) 생성 완료!\n")

    return vector_store


def embed_table_info(db_names, engine):
    table_info_query = """SELECT
    t.TABLE_SCHEMA AS 데이터베이스,
    t.TABLE_NAME AS 테이블명,
    GROUP_CONCAT(
        CONCAT(
            c.COLUMN_NAME, ' (',
            c.COLUMN_TYPE,
            CASE WHEN c.IS_NULLABLE = 'YES' THEN ', nullable' ELSE '' END,
            CASE WHEN c.COLUMN_KEY = 'PRI' THEN ', primary key' ELSE '' END,
            ')'
            ) SEPARATOR '; '
        ) AS 컬럼정보
    FROM information_schema.TABLES t
    JOIN information_schema.COLUMNS c
        ON t.TABLE_SCHEMA = c.TABLE_SCHEMA
        AND t.TABLE_NAME = c.TABLE_NAME
    WHERE t.TABLE_SCHEMA NOT IN ('mysql', 'performance_schema', 'sys', 'information_schema')
    GROUP BY t.TABLE_SCHEMA, t.TABLE_NAME;"""

    print("모든 테이블 데이터를 가져오는 중...")
    table_info_df = pd.read_sql(table_info_query, engine)
    table_ddl_texts = []
    available_dbs = set(db_names)
    for _, row in table_info_df.iterrows():
        db_name = row["데이터베이스"]
        if db_name in available_dbs:
            text = f"""데이터베이스: {db_name}\n테이블명: {row['테이블명']}\n컬럼정보: {row['컬럼정보']}"""
            table_ddl_texts.append(text)
    print(f"총 {len(table_ddl_texts)}개의 테이블 테이터 확보")

    print("벡터 데이터베이스(테이블 데이터) 생성 중...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(table_ddl_texts, embeddings)
    print("벡터 데이터베이스(테이블 데이터) 생성 완료!\n")
    return vector_store


def get_table_ddl(db_name: str | None, DB_SERVER) -> Tuple[List[str], List[str]]:
    if not db_name:
        return None, None  # type: ignore
    print(f"Get Table DDL data from {db_name}")
    db_url = os.path.join(DB_SERVER, db_name)
    engine = create_engine(db_url)
    print("Engine Created!")
    # DB 내 정보 확인
    inspector = inspect(engine)
    # DB 내 모든 테이블 이름 가져오기
    all_tables = set(inspector.get_table_names() + inspector.get_view_names())
    print("Table inspection Completed!")

    # TODO
    # 이 단계에서 사용자에 따라 접근 가능한 테이블을 반영은 할 수 있다.
    # 다만, 이 함수가 chat model의 tool로서 쓰이지 않는다면 굳이 반영을 안해도 된다.
    to_reflect = set(all_tables)
    print(f"Tables to get DDL: {to_reflect}")
    # 테이블의 메타데이터(컬럼 정보 등)를 저장하기 위한 객체 생성
    meta_data = MetaData()
    # 지정된 db 로부터 메타데이터 추출
    if to_reflect:
        meta_data.reflect(
            views=True,
            bind=engine,
            only=list(to_reflect),
            schema=None,
        )
    # 모든 테이블 중 to_reflect에 존재하는(확인하기로 한) 테이블만 추출
    meta_tables = [tbl for tbl in meta_data.sorted_tables if tbl.name in to_reflect]
    # 각 테이블의 메타데이터를 보고 테이블 DDL 정보를 작성
    ddl_data, table_names = [], []
    for table in meta_tables:
        create_table = str(CreateTable(table).compile(engine))
        table_info = f"{create_table.rstrip()}"
        ddl_data.append(table_info)
        table_names.append(table.name)
    print(f"GETTING TABLE DDL from {db_name} has been completed!\n")
    return table_names, ddl_data


def embed_db_info(db_names: str | None, DB_SERVER: str, sample_info: int):
    # 데이터베이스와 테이블 정보를 저장할 리스트 초기화
    db_info = []

    # 주어진 모든 데이터베이스 이름에 대해 반복
    for db_name in db_names:

        # SQLDatabase 객체를 생성하여 데이터베이스에 연결
        sql_db = SQLDatabase.from_uri(
            os.path.join(DB_SERVER, db_name), sample_rows_in_table_info=sample_info
        )

        # 데이터베이스의 테이블 정보를 저장할 리스트 초기화
        information = []

        # 데이터베이스에서 사용할 수 있는 테이블 이름을 가져와 반복
        for tables in sql_db.get_usable_table_names():

            # 테이블의 스키마 정보를 가져옴
            table_schema = sql_db.get_table_info([tables])

            # 데이터베이스 이름과 테이블 정보를 가공하여 문자열로 저장
            processed_schema = f"데이터베이스:{db_name}\n테이블 정보:{table_schema}"

            # 가공된 정보를 information 리스트에 추가
            information.append(processed_schema)

        # information에 포함된 모든 테이블 정보를 db_info에 추가
        db_info.extend(information)

    print(f"총 {len(db_info)}개의 데이터 확보")

    print("벡터 데이터베이스 생성 중...")
    # OpenAI 임베딩을 사용하여 텍스트 정보를 벡터로 변환
    embeddings = OpenAIEmbeddings()

    # FAISS 벡터 스토어를 사용하여 텍스트 임베딩을 벡터로 변환하여 저장
    vector_store = FAISS.from_texts(db_info, embeddings)
    print("벡터 데이터베이스 생성 완료!\n")

    # 생성된 벡터 스토어 반환
    return vector_store


def get_vector_stores(sample_info: int) -> Dict[str, VectorStore]:
    load_dotenv()
    DB_SERVER = os.getenv("URL")
    information_schema_path = os.path.join(DB_SERVER, "INFORMATION_SCHEMA")  # type: ignore
    # create_engine으로 db에 접근 준비
    engine = create_engine(information_schema_path)

    vector_store_dict = {}
    # 접근 가능한 DB 이름 얻기
    db_names = get_db_names(engine)
    vector_store_dict["db_info"] = embed_db_info(db_names, DB_SERVER, sample_info)
    # vector_store_dict["db_meta_data"] = embed_db_meta_data(db_names)
    # vector_store_dict["table_info"] = embed_table_info(db_names, engine)
    # 오래 걸리기에 Redis가 완성되면 ddl 데이터를 임베딩 하는 것으로 한다.
    # vector_store_dict['table_ddl'] = embed_table_ddl(db_names, DB_SERVER)

    return vector_store_dict
