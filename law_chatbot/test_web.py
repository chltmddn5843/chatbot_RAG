"""Run: .venv/bin/python law_chatbot/test_web.py (no external services)."""
import json
import threading
from http.client import HTTPConnection
from unittest.mock import patch

import web

server = web.ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()


def request(method, path, body=None, headers=None):
    conn = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    conn.request(method, path, body, headers or {})
    response = conn.getresponse()
    result = response.status, response.read()
    conn.close()
    return result


try:
    assert request('GET', '/')[0] == 200
    assert request('GET', '/.env')[0] == 404
    headers = {'Content-Type': 'application/json'}
    for value in ['', ' '*3, 123, 'x'*4001]:
        assert request('POST', '/api/chat', json.dumps({'question': value}), headers)[0] == 400
    assert request('POST', '/api/chat', '{bad', headers)[0] == 400
    assert request('POST', '/api/chat', '{}', {**headers, 'Origin': 'https://example.com'})[0] == 403
    assert request('GET', '/', headers={'Host': 'evil.example'})[0] == 403
    with patch.object(web, 'answer', return_value={'answer': '답변', 'sources': []}) as answer:
        code, body = request('POST', '/api/chat', json.dumps({'question': ' 질문 '}), headers)
        assert code == 200 and json.loads(body)['answer'] == '답변'
        answer.assert_called_once_with('질문')
    with patch.object(web, 'answer', side_effect=RuntimeError('private-key')):
        code, body = request('POST', '/api/chat', json.dumps({'question': '질문'}), headers)
        assert code == 503 and b'private-key' not in body
    assert not web.chat_lock.locked()
    web.chat_lock.acquire()
    assert request('POST', '/api/chat', json.dumps({'question': '질문'}), headers)[0] == 429
    web.chat_lock.release()
    print('PASS: UI, validation, origin/host protection, chat dispatch, error handling, concurrency')
finally:
    server.shutdown()
    server.server_close()
