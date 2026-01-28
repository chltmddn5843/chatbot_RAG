import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# DB 설정 정보
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER", "airflow")
PG_PASSWORD = os.getenv("PG_PASSWORD", "airflow")
PG_DATABASE = os.getenv("PG_DATABASE", "airflow")

def get_pg_conn():
    """PostgreSQL 연결 객체 반환"""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE
    )

def ensure_chatbot_logs_table():
    """챗봇 로그 저장을 위한 테이블 생성 (최초 1회 실행)"""
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS chatbot_logs (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            file_name TEXT,
            similarity_score FLOAT,
            chunk_id TEXT,
            parent_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.commit()
    cur.close()
    conn.close()

def save_chat_log(question, answer, retrieved_chunks=None):
    """대화 내용 및 검색 메타데이터 저장"""
    conn = get_pg_conn()
    cur = conn.cursor()
    try:
        if retrieved_chunks and len(retrieved_chunks) > 0:
            for chunk in retrieved_chunks:
                file_name = str(chunk.get('page', 'N/A')) + " page / " + str(chunk.get('row', 'N/A')) + " row"
                similarity_score = float(chunk.get('score', 0.0))
                chunk_id = str(chunk.get('chunk_id'))
                parent_id = str(chunk.get('parent_id'))
                cur.execute(
                    """
                    INSERT INTO chatbot_logs (question, answer, file_name, similarity_score, chunk_id, parent_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (question, answer, file_name, similarity_score, chunk_id, parent_id, datetime.now())
                )
        else:
            cur.execute(
                """
                INSERT INTO chatbot_logs (question, answer, file_name, similarity_score, chunk_id, parent_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (question, answer, None, None, None, None, datetime.now())
            )
        conn.commit()
    except Exception as e:
        print(f"❌ DB Insert 오류: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
        
def show_recent_logs(limit=10):
    """최근 대화 로그 출력 (디버깅용)"""
    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, question, answer, created_at FROM chatbot_logs ORDER BY id DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        print(f"\n최근 {limit}개 대화 로그:")
        for row in rows:
            print(f"[{row[0]}] {row[3]}\nQ: {row[1]}\nA: {row[2]}\n---")
        cur.close()
        conn.close()
    except Exception as e:
        print("DB 조회 오류:", e)