import streamlit as st
import requests

# FastAPI 서버의 엔드포인트 URL
API_URL = "14.35.173.251:8000/chat"


def main():
    st.title("🤖 AI 텍스트-to-SQL 챗봇")

    # 세션 상태에 메시지 저장
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # SQL 쿼리 있으면 추가 표시
            if message.get("sql"):
                with st.expander("생성된 SQL 쿼리"):
                    st.code(message["sql"], language="sql")

    # 사용자 입력 처리
    if prompt := st.chat_input("데이터베이스에 대해 무엇을 알고 싶으신가요?"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)

        # 로딩 상태 표시
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("생각 중...")

        try:
            # FastAPI 백엔드로 요청 보내기
            response = requests.post(API_URL, json={"question": prompt})
            result = response.json()

            # 응답 메시지 추가 및 표시
            full_message = {
                "role": "assistant",
                "content": result["answer"],
                "sql": result.get("sql"),  # SQL 쿼리 포함
            }
            st.session_state.messages.append(full_message)
            message_placeholder.markdown(result["answer"])

            # SQL 쿼리 있으면 expander로 표시
            if result.get("sql"):
                with st.expander("생성된 SQL 쿼리"):
                    st.code(result["sql"], language="sql")

        except requests.exceptions.RequestException as e:
            message_placeholder.markdown(f"서버 연결 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
