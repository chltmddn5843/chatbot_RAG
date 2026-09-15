"""Local UI: python law_chatbot/web.py"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
from db_utils import ensure_chatbot_logs_table, get_recent_logs, save_chat_log

# ponytail: one inference at a time on CPU; use a job queue for multiple users.
chat_lock = Lock()


def status():
    from pymilvus import connections, utility, Collection
    result = {'api_key': bool(os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY'))}
    try:
        ensure_chatbot_logs_table()
        result['postgres'] = '연결됨'
    except Exception:
        result['postgres'] = '연결 실패: .env의 PG_* 설정과 Docker를 확인하세요.'
    name = os.getenv('MILVUS_COLLECTION', 'col_1')
    try:
        connections.connect(alias='web_status', host=os.getenv('MILVUS_HOST', '127.0.0.1'), port=19530, timeout=3)
        if utility.has_collection(name, using='web_status'):
            result['milvus'] = f'{name} · {Collection(name, using="web_status").num_entities:,}개 데이터'
        else:
            result['milvus'] = f'{name} 컬렉션 없음: 검색 데이터를 먼저 적재하세요.'
    except Exception:
        result['milvus'] = '연결 실패: Milvus 실행 상태를 확인하세요.'
    finally:
        connections.disconnect('web_status')
    return result


def answer(question):
    from chatbot import search_milvus, ask_legal_expert
    hits = search_milvus(question)
    if not hits:
        return {'answer': '검색 결과가 없습니다. 컬렉션의 데이터와 질문을 확인하세요.', 'sources': []}
    response = ask_legal_expert(question, hits)
    warning = None
    try:
        ensure_chatbot_logs_table()
        save_chat_log(question, response, hits)
    except Exception:
        warning = '답변은 생성했지만 로그를 저장하지 못했습니다. PostgreSQL 연결을 확인하세요.'
    return {'answer': response, 'sources': hits, 'warning': warning}


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, data, content_type='application/json; charset=utf-8'):
        payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def allowed(self):
        hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        return self.headers.get('Host') in hosts and self.headers.get('Origin') in {None, *(f'http://{h}' for h in hosts)}

    def do_GET(self):
        if not self.allowed():
            return self.reply(403, {'error': '로컬 웹에서만 접근할 수 있습니다.'})
        if self.path == '/':
            return self.reply(200, Path(__file__).with_name('web.html').read_bytes(), 'text/html; charset=utf-8')
        try:
            if self.path == '/api/status':
                return self.reply(200, status())
            if self.path == '/api/logs':
                ensure_chatbot_logs_table()
                return self.reply(200, {'logs': get_recent_logs(20)})
            self.reply(404, {'error': '페이지가 없습니다.'})
        except Exception:
            self.reply(503, {'error': '로그를 조회하지 못했습니다. PostgreSQL 연결을 확인하세요.'})

    def do_POST(self):
        if not self.allowed():
            return self.reply(403, {'error': '로컬 웹에서만 요청할 수 있습니다.'})
        if self.path != '/api/chat':
            return self.reply(404, {'error': '페이지가 없습니다.'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.reply(415, {'error': 'JSON 요청이 필요합니다.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 20000:
                raise ValueError()
            data = json.loads(self.rfile.read(length))
            question = data.get('question') if isinstance(data, dict) else None
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
                raise ValueError()
        except (ValueError, UnicodeError):
            return self.reply(400, {'error': '질문을 1~4,000자로 입력하세요.'})
        if not chat_lock.acquire(blocking=False):
            return self.reply(429, {'error': '다른 답변을 생성하고 있습니다. 잠시 후 다시 시도하세요.'})
        try:
            result = answer(question.strip())
        except Exception as exc:
            print(f'Chat failed: {type(exc).__name__}', flush=True)
            self.reply(503, {'error': '답변 생성 실패. DB 상태, 검색 컬렉션, API 키와 모델 다운로드 연결을 확인하세요.'})
        else:
            self.reply(200, result)
        finally:
            chat_lock.release()


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', int(os.getenv('PORT', '8000'))), Handler)
    print(f'법률 RAG 웹: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
