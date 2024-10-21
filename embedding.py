import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import hashlib

# .env 파일 로드
load_dotenv()

# MySQL 연결
def connect_mysql():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        if connection.is_connected():
            print("MySQL에 성공적으로 연결되었습니다.")
        return connection
    except Error as e:
        print(f"MySQL 연결 오류: {e}")
        return None

# 테이블 메타데이터 추출
def get_table_metadata(connection):
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    metadata = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        metadata[table_name] = [
            {
                "Field": col[0],
                "Type": col[1],
                "Null": col[2],
                "Key": col[3],
                "Default": col[4],
                "Extra": col[5]
            } for col in columns
        ]
    
    return metadata

# Chroma DB 클라이언트 초기화 및 생성
def init_chroma():
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY")
    print(f"CHROMA_PERSIST_DIRECTORY from env: {persist_directory}")
    
    if not persist_directory:
        persist_directory = os.path.join(os.getcwd(), "chroma_persist")
        print(f"Using default persist directory: {persist_directory}")
    
    try:
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            print(f"ChromaDB 디렉토리가 생성되었습니다: {persist_directory}")
        else:
            print(f"ChromaDB 디렉토리가 이미 존재합니다: {persist_directory}")
        
        client = PersistentClient(path=persist_directory)
        print("ChromaDB 클라이언트가 성공적으로 초기화되었습니다.")
        return client
    except PermissionError as pe:
        print(f"ChromaDB 디렉토리 생성 또는 접근 권한 오류: {pe}")
    except Exception as e:
        print(f"ChromaDB 초기화 중 오류 발생: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
    return None
    
    
# 메타데이터를 문자열로 변환
def metadata_to_string(metadata):
    return json.dumps(metadata, indent=2)

# 메타데이터 해시 계산
def calculate_hash(metadata_string):
    return hashlib.sha256(metadata_string.encode()).hexdigest()

def main():
    try:
        # MySQL 연결
        connection = connect_mysql()
        if not connection:
            return

        # 메타데이터 추출
        metadata = get_table_metadata(connection)
        metadata_string = json.dumps(metadata, indent=2)

        # Chroma DB 초기화
        chroma_client = init_chroma()
        if not chroma_client:
            print("ChromaDB 초기화 실패")
            return

        # 기존 컬렉션 삭제 (있다면)
        try:
            chroma_client.delete_collection("mysql_metadata")
            print("기존 컬렉션 삭제됨")
        except ValueError as e:
            print(f"컬렉션 삭제 중 오류 발생 (무시해도 됨): {e}")

        # 새 컬렉션 생성
        collection = chroma_client.create_collection("mysql_metadata")
        print("새 컬렉션 생성됨")

        # 메타데이터 임베딩 및 저장
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(metadata_string)

        collection.add(
            ids=["metadata"],
            embeddings=[embedding.tolist()],
            documents=[metadata_string],
            metadatas=[{"source": "mysql"}]
        )
        print("메타데이터가 ChromaDB에 저장되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("MySQL 연결이 닫혔습니다.")

if __name__ == "__main__":
    main()