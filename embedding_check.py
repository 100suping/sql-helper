import os
from dotenv import load_dotenv
from chromadb import PersistentClient
import json

# .env 파일 로드
load_dotenv()

def init_chroma():
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY")
    if not persist_directory:
        persist_directory = os.path.join(os.getcwd(), "chroma_persist")
    
    print(f"ChromaDB 디렉토리: {persist_directory}")
    
    try:
        client = PersistentClient(path=persist_directory)
        print("ChromaDB 클라이언트가 성공적으로 초기화되었습니다.")
        return client
    except Exception as e:
        print(f"ChromaDB 초기화 중 오류 발생: {e}")
        return None

def check_embeddings():
    chroma_client = init_chroma()
    if not chroma_client:
        return

    try:
        collection = chroma_client.get_collection("mysql_metadata")
        print("'mysql_metadata' 컬렉션을 찾았습니다.")
        
        # 모든 데이터 조회
        results = collection.get()
        
        if results['ids']:
            print(f"\n총 {len(results['ids'])}개의 항목이 저장되어 있습니다.")
            
            for i, id in enumerate(results['ids']):
                print(f"\n항목 {i+1}:")
                print(f"ID: {id}")
                print(f"메타데이터: {results['metadatas'][i]}")
                
                # 문서(메타데이터 문자열) 출력
                print("문서 (처음 200자):")
                doc = results['documents'][i]
                print(doc[:200] + "..." if len(doc) > 200 else doc)
                
                # 임베딩 벡터 출력 (처음 5개 요소만)
                print("임베딩 벡터:")
                if 'embeddings' in results and results['embeddings'] is not None:
                    embedding = results['embeddings'][i]
                    if embedding is not None:
                        print(embedding[:5] if len(embedding) > 5 else embedding)
                    else:
                        print("임베딩 벡터가 None입니다.")
                else:
                    print("임베딩 데이터가 없습니다.")
                
        else:
            print("저장된 데이터가 없습니다.")
    
    except Exception as e:
        print(f"임베딩 확인 중 오류 발생: {e}")
        print(f"오류 타입: {type(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    check_embeddings()