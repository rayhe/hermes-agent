"""Exercise the real public HTTPS API without printing credentials."""
import argparse
import json
import os
from pathlib import Path
import socket
import urllib.request
import urllib.error
from datetime import datetime, timezone
from dotenv import dotenv_values

parser = argparse.ArgumentParser()
parser.add_argument('--home', required=True)
parser.add_argument('--base-url', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--dns-override', action='store_true')
args = parser.parse_args()
from acceptance_http import validate_api_origin, build_api_opener
BASE = validate_api_origin(args.base_url)
OPENER = build_api_opener(BASE)
PUBLIC = Path(args.output_dir)
PUBLIC.mkdir(parents=True, exist_ok=True)
from urllib.parse import urlparse
HOST = urlparse(BASE).hostname
KEY = os.environ.get('API_SERVER_KEY') or dotenv_values(Path(args.home) / '.env')['API_SERVER_KEY']
ips = []
if args.dns_override:
    dns = json.load(urllib.request.urlopen('https://dns.google/resolve?name=' + HOST + '&type=A', timeout=15))
    ips = [a['data'] for a in dns.get('Answer', []) if a['type'] == 1]
    assert ips, 'No public A records'
    original_getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, *a, **kw: original_getaddrinfo(ips[0] if host == HOST else host, port, *a, **kw)
results = {'checked_at': datetime.now(timezone.utc).isoformat(), 'base_url': BASE + '/v1',
    'dns_resolution': 'Google public DNS (explicit override)' if args.dns_override else 'system resolver', 'public_ips': ips, 'checks': {}}


def call(path, token=None, payload=None, stream=False):
    headers = {'Accept': 'application/json'}
    if token is not None:
        headers['Authorization'] = 'Bearer ' + token
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    try:
        response = OPENER.open(req, timeout=180)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    with response:
        if stream:
            lines = [line.decode().rstrip() for line in response]
            return response.status, lines
        return response.status, json.load(response)


def save(name, data):
    results['checks'][name] = data
    (PUBLIC / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({name: data}), flush=True)

status, health = call('/health')
assert status == 200 and health.get('status') == 'ok', json.dumps({'status': status, 'body': health})
save('https_health', {'status': status, 'body': health})
status, _ = call('/v1/models')
assert status == 401, f'Unauthenticated model request returned {status}'
save('no_token_rejected', status)
status, _ = call('/v1/models', 'invalid-verification-token')
assert status == 401, f'Invalid-token request returned {status}'
save('wrong_token_rejected', status)
status, _ = call('/v1/chat/completions', payload={'model': 'hermes-agent', 'messages': [{'role': 'user', 'content': 'Hello'}]})
assert status == 401, f'Unauthenticated chat returned {status}'
save('unauthenticated_chat_rejected', status)
status, models = call('/v1/models', KEY)
assert status == 200 and any(m['id'] == 'hermes-agent' for m in models['data'])
save('authenticated_models', {'status': status, 'models': [m['id'] for m in models['data']]})
status, capabilities = call('/v1/capabilities', KEY)
assert status == 200
save('authenticated_capabilities', {'status': status, 'auth': capabilities.get('auth'), 'features': capabilities.get('features')})
status, chat = call('/v1/chat/completions', KEY, {'model': 'hermes-agent', 'stream': False,
    'messages': [{'role': 'user', 'content': 'Reply with exactly HERMES_API_OK. Do not use any tools.'}]})
assert status == 200, json.dumps(chat)
answer = chat['choices'][0]['message']['content']
assert 'HERMES_API_OK' in answer, f'Unexpected answer: {answer}'
save('real_chat_completion', {'status': status, 'answer': answer, 'runtime': chat.get('runtime'), 'usage': chat.get('usage')})
status, frames = call('/v1/chat/completions', KEY, {'model': 'hermes-agent', 'stream': True,
    'messages': [{'role': 'user', 'content': 'Reply with exactly HERMES_STREAM_OK. Do not use any tools.'}]}, stream=True)
assert status == 200 and any(line == 'data: [DONE]' for line in frames)
text = ''
for line in frames:
    if line.startswith('data: ') and line != 'data: [DONE]':
        chunk = json.loads(line[6:])
        for choice in chunk.get('choices', []):
            text += choice.get('delta', {}).get('content') or ''
assert 'HERMES_STREAM_OK' in text, text
save('real_stream_completion', {'status': status, 'answer': text, 'done_received': True})
status, response = call('/v1/responses', KEY, {'model': 'hermes-agent', 'store': True,
    'input': 'Use the terminal tool to run python -c "print(\'HERMES_API_TOOL_OK\')". Do not read or modify any files. Then reply exactly HERMES_API_TOOL_OK.'})
assert status == 200, json.dumps(response)
items = response.get('output', [])
tools = [item.get('name') for item in items if item.get('type') == 'function_call']
assert 'terminal' in tools, f'Terminal tool was not called: {tools}'
assert any('HERMES_API_TOOL_OK' in str(item.get('output', '')) for item in items if item.get('type') == 'function_call_output'), 'No real terminal output'
status, replay = call('/v1/responses/' + response['id'], KEY)
assert status == 200 and replay['id'] == response['id'] and replay['status'] == 'completed'
save('responses_api_with_real_terminal', {'status': status, 'response_id': response['id'], 'tools_called': tools,
    'terminal_stdout_verified': True, 'stored_response_retrieval_verified': True})
results['all_passed'] = True
(PUBLIC / 'verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('All real public API checks passed', flush=True)
