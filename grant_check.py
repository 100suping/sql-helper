import streamlit as st
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import bcrypt
import pandas as pd
import time
from mysql.connector import pooling

# .env 파일에서 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="DB Access Control", page_icon="🔐", layout="wide")

# CSS 스타일
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .stButton>button {
        color: #4F8BF9;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 부서 목록 정의
DEPARTMENTS = ['Sales', 'HR', 'Finance', 'IT', 'Management']

# 데이터베이스 연결 함수
connection_pool = None

def create_connection_pool():
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME")
            )
            st.success("MySQL 연결 풀이 성공적으로 생성되었습니다.")
        except Error as e:
            st.error(f"연결 풀 생성 오류: {e}")
    return connection_pool  # 여기서 connection_pool을 반환

def get_connection():
    pool = create_connection_pool()
    if pool:
        return pool.get_connection()
    return None
    
def authenticate_user(username, password):
    connection = None
    cursor = None
    try:
        connection = get_connection()  # 연결 풀에서 연결 가져오기
        if connection:
            cursor = connection.cursor(dictionary=True)
            query = "SELECT * FROM connect_user WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                return user
            else:
                st.warning("사용자 이름 또는 비밀번호가 일치하지 않습니다.")
        else:
            st.error("데이터베이스 연결을 설정할 수 없습니다.")
    except Error as e:
        st.error(f"인증 과정 중 오류 발생: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()  # 연결 풀에 연결 반환
    return None

# 회원가입 함수
def register_user(username, password, department, role, max_retries=3):
    for attempt in range(max_retries):
        connection = None
        cursor = None
        try:
            connection = get_connection()  # 연결 풀에서 연결 가져오기
            if connection:
                cursor = connection.cursor()
                st.info(f"Attempting to register user: {username}, Department: {department}, Role: {role}")
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                query = "INSERT INTO connect_user (username, password, department, role) VALUES (%s, %s, %s, %s)"
                st.info(f"Executing query: {query}")
                cursor.execute(query, (username, hashed_password, department, role))
                connection.commit()
                st.success("User registered successfully!")
                return True
            else:
                st.error("데이터베이스 연결을 설정할 수 없습니다.")
        except mysql.connector.Error as e:
            st.error(f"MySQL 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            st.error(f"오류 코드: {e.errno}")
            st.error(f"SQL 상태: {e.sqlstate}")
            st.error(f"오류 메시지: {e.msg}")
            if attempt < max_retries - 1:
                st.warning("연결 재시도 중...")
                time.sleep(2)  # 2초 대기 후 재시도
            else:
                st.error("최대 재시도 횟수 초과. 회원가입 실패.")
                return False
        except Exception as e:
            st.error(f"예상치 못한 오류: {e}")
            st.error(f"오류 유형: {type(e)}")
            return False
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()  # 연결 풀에 연결 반환
                st.info("Database connection closed.")
    return False

# 테이블 접근 권한 확인 함수 (예시)
def check_table_permission(department, table_name):
    permissions = {
        'Sales': ['customers', 'orders', 'orderdetails'],
        'HR': ['employees', 'offices'],
        'Finance': ['payments', 'orders'],
        'IT': ['products', 'productlines'],
        'Management': ['customers', 'employees', 'orders', 'products', 'payments']
    }
    return table_name in permissions.get(department, [])

# 회원가입 폼
def show_signup_form():
    st.subheader("회원가입")
    new_username = st.text_input("새 사용자명", key="new_username")
    new_password = st.text_input("새 비밀번호", type="password", key="new_password")
    department = st.selectbox("부서", DEPARTMENTS)
    role = st.selectbox("역할", ['User', 'Manager', 'Admin'])
    if st.button("가입하기"):
        if register_user(new_username, new_password, department, role):
            st.success("회원가입이 완료되었습니다. 이제 로그인할 수 있습니다.")
        else:
            st.error("회원가입에 실패했습니다. 다시 시도해주세요.")

# 메인 애플리케이션
def main():
    st.markdown('<p class="big-font">데이터베이스 접근 제어 시스템</p>', unsafe_allow_html=True)

    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["로그인", "회원가입"])
        
        with tab1:
            username = st.text_input("사용자명")
            password = st.text_input("비밀번호", type="password")
            if st.button("로그인", key="login"):
                user = authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()  # 여기를 수정
                else:
                    st.error("인증 실패. 사용자명과 비밀번호를 확인하세요.")
        
        with tab2:
            show_signup_form()

    else:
        st.sidebar.success(f"로그인됨: {st.session_state.user['username']}")
        st.sidebar.info(f"부서: {st.session_state.user['department']}")
        st.sidebar.info(f"역할: {st.session_state.user['role']}")
        if st.sidebar.button("로그아웃"):
            st.session_state.user = None
            st.rerun()  # 여기도 수정

        st.subheader("테이블 접근 테스트")
        table_name = st.selectbox("테이블 선택", ['customers', 'employees', 'offices', 'orders', 'orderdetails', 'payments', 'products', 'productlines'])
        if st.button("접근 테스트"):
            if check_table_permission(st.session_state.user['department'], table_name):
                st.success(f"{table_name} 테이블에 접근 가능합니다.")
            else:
                st.error(f"{table_name} 테이블에 접근 권한이 없습니다.")

        # 데이터베이스 스키마 표시
        st.subheader("데이터베이스 스키마")
        schema_data = {
            "테이블": ["productlines", "products", "orderdetails", "orders", "customers", "employees", "offices", "payments"],
            "필드": [
                "productLine, textDescription, htmlDescription, image",
                "productCode, productName, productLine, productScale, productVendor, productDescription, quantityInStock, buyPrice, MSRP",
                "orderNumber, productCode, quantityOrdered, priceEach, orderLineNumber",
                "orderNumber, orderDate, requiredDate, shippedDate, status, comments, customerNumber",
                "customerNumber, customerName, contactLastName, contactFirstName, phone, addressLine1, addressLine2, city, state, postalCode, country, salesRepEmployeeNumber, creditLimit",
                "employeeNumber, lastName, firstName, extension, email, officeCode, reportsTo, jobTitle",
                "officeCode, city, phone, addressLine1, addressLine2, state, country, postalCode, territory",
                "customerNumber, checkNumber, paymentDate, amount"
            ]
        }
        schema_df = pd.DataFrame(schema_data)
        st.table(schema_df)

if __name__ == "__main__":
    main()