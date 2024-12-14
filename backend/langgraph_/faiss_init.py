import pandas as pd
import os
import time
import shutil
from typing import List, Dict, Tuple

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores.utils import DistanceStrategy

from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv


def get_db_names(engine) -> List[str]:
    """
    접근 가능한 모든 데이터베이스 이름을 조회합니다.

    Args:
        engine: SQLAlchemy 엔진 인스턴스

    Returns:
        List[str]: 데이터베이스 이름 목록
    """

    db_names_query = """
    SELECT DISTINCT TABLE_SCHEMA AS db
    FROM TABLES
    WHERE TABLE_SCHEMA NOT IN ('mysql', 'performance_schema', 'sys', 'information_schema', 'connect_db');
    """

    print("접근 가능한 모든 데이터베이스를 가져오는 중...")
    db_names_df = pd.read_sql(db_names_query, engine)
    db_names = [row["db"] for _, row in db_names_df.iterrows()]

    print(f"총 {len(db_names)}의 접근 가능한 모든 데이터베이스를 가져왔습니다.\n")

    return db_names


def embed_db_info(
    db_names: str | None, DB_SERVER: str, sample_info: int
) -> VectorStore:
    """
    FAISS 벡터 데이터베이스에 데이터베이스 정보를 임베딩합니다.

    Args:
        db_names: 데이터베이스 이름 목록
        DB_SERVER: 데이터베이스 서버 경로
        sample_info: 각 테이블에서 샘플링할 행 수

    Returns:
        FAISS 벡터 데이터베이스 인스턴스
    """

    def is_local_data_valid(local_path: str) -> bool:
        """
        로컬 데이터의 유효성과 최신성을 확인합니다.
        데이터가 생성된지 24시간 이내라면 TRUE, 아니면 FALSE를 반환합니다.
        """
        if not os.path.exists(local_path):
            return False

        # 예시: 파일 생성 후 24시간 이내인지 확인
        file_age = time.time() - os.path.getctime(local_path)
        return file_age < 24 * 60 * 60  # 24시간

    # OpenAI 임베딩을 사용하여 텍스트 정보를 벡터로 변환
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    local_path = "local_faiss"

    # 로컬 데이터가 유효한 경우 로드
    if is_local_data_valid(local_path):
        print("기존 FAISS 벡터 데이터베이스 불러오는 중...")
        vector_store = FAISS.load_local(
            local_path,
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        print("로컬 FAISS 벡터 데이터베이스 불러오기 완료!")
    else:
        # 데이터베이스와 테이블 정보를 저장할 리스트 초기화
        db_info = []
        db_metadata = []  # 실제 반환될 메타데이터 리스트

        print("데이터 확보 중...")
        # 주어진 모든 데이터베이스 이름에 대해 반복
        for db_name in db_names:
            # SQLDatabase 객체를 생성하여 데이터베이스에 연결
            sql_db_info = SQLDatabase.from_uri(
                os.path.join(DB_SERVER, db_name), sample_rows_in_table_info=0
            )

            sql_db_meta = SQLDatabase.from_uri(
                os.path.join(DB_SERVER, db_name), sample_rows_in_table_info=sample_info
            )

            # 데이터베이스에서 사용할 수 있는 테이블 이름을 가져와 반복
            for table_name in sql_db_info.get_usable_table_names():
                # 테이블의 스키마 정보를 가져옴
                table_schema = sql_db_info.get_table_info([table_name])

                # DDL 정보만 벡터 임베딩을 위한 데이터로 사용
                processed_schema = f"{table_schema}"
                db_info.append(processed_schema)

            # 데이터베이스에서 사용할 수 있는 테이블 이름을 가져와 반복
            for table_name in sql_db_meta.get_usable_table_names():
                # 테이블의 스키마 정보를 가져옴
                table_schema = sql_db_meta.get_table_info([table_name])
                # 메타데이터는 별도로 저장
                metadata = {"context": f"DB:{db_name}\nDDL:{table_schema}"}
                db_metadata.append(metadata)

        print(f"총 {len(db_info)}개의 데이터 확보")

        print("FAISS 벡터 데이터베이스 생성 중...")
        # FAISS 벡터 데이터베이스 생성 및 로컬 저장
        vector_store = FAISS.from_texts(
            texts=db_info,
            embedding=embeddings,
            metadatas=db_metadata,
            distance_strategy=DistanceStrategy.COSINE,
        )

        # .save_local에 경우 덮어씌우기가 되지 않아서 기존에 폴더가 존재할 경우 삭제 필요
        # FAISS 저장
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        vector_store.save_local(local_path)
        print("FAISS 벡터 데이터베이스 생성 완료!\n")

    return vector_store


def get_vector_stores(sample_info: int = 5) -> VectorStore:
    load_dotenv()
    DB_SERVER = os.getenv("URL")
    information_schema_path = os.path.join(DB_SERVER, "INFORMATION_SCHEMA")

    # create_engine으로 db에 접근 준비
    engine = create_engine(information_schema_path)

    # 접근 가능한 DB 이름 얻기
    db_names = get_db_names(engine)

    # FAISS 벡터 스토어 얻기
    vector_store = embed_db_info(db_names, DB_SERVER, sample_info)

    return vector_store
