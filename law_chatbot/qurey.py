import os
import psycopg2
from dotenv import load_dotenv

# 1. .env 파일 로드
load_dotenv()

# 2. DB 설정 정보 (기본값 설정 및 환경변수 로드)
# 병행 실행 중이시라면 .env의 PG_PORT를 5433으로 설정하거나 아래 기본값을 수정하세요.
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")  # 5433으로 명시적 설정 권장
PG_USER = os.getenv("PG_USER", "airflow")
PG_DB = os.getenv("PG_DATABASE", "airflow")
PG_PW = os.getenv("PG_PASSWORD", "airflow") # .env에 없으면 'airflow' 사용

# 3. DB 연결 (변수 직접 대입)
conn = psycopg2.connect(
    host=PG_HOST,
    port=PG_PORT,
    database=PG_DB,
    user=PG_USER,
    password=PG_PW
)

# 4. 쿼리 실행
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS chatbot_logs;")
conn.commit()
cur.close()
conn.close()