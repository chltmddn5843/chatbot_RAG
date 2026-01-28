import os
import psycopg2

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_USER = os.getenv("PG_USER", "airflow")
PG_PASSWORD = os.getenv("PG_PASSWORD", "airflow")
PG_DATABASE = os.getenv("PG_DATABASE", "airflow")

def run_query(sql):
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE
    )
    cur = conn.cursor()
    try:
        cur.execute(sql)
        if cur.description:
            rows = cur.fetchall()
            for row in rows:
                print(row)
        else:
            conn.commit()
            print("Query executed successfully.")
    except Exception as e:
        print("Error:", e)
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    print("SQL 입력 후 Enter를 누르세요. (종료: exit)")
    while True:
        sql = input("SQL> ")
        if sql.strip().lower() == "exit":
            break
        run_query(sql)
