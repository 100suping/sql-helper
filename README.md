# sql-helper

## 파이썬 환경 설정
```
pip install -r requirements.txt
```

## 현재 .env 양식
```
OPENAI_API_KEY="your-api-key"
LANGCHAIN_API_KEY="your-api-key"
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT="your-project-name"
URL="DB 주소"
```

## 실행
```
# RAG
python main.py
```

```
# Streamit
streamlit run streamlit_main.py
```

```
# LangGraph
## langgraph 디렉토리로 이동
cd langgraph
## 테스트 진행
sh langgraph.sh
```