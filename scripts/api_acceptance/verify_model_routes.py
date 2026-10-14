"""Verify every exposed model runs the real Hermes agent/tool loop over HTTPS."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dotenv import dotenv_values

parser = argparse.ArgumentParser()
parser.add_argument('--home', required=True)
parser.add_argument('--preflight', action='store_true', help='Check direct cloud request routes before listener restart')
parser.add_argument('--base-url', required=True, help='HTTPS origin without /v1')
parser.add_argument('--output-dir', required=True)
parser.add_argument('--dns-override', action='store_true')
args = parser.parse_args()
from acceptance_http import validate_api_origin, build_api_opener
BASE = validate_api_origin(args.base_url)
OPENER = build_api_opener(BASE)
HOME = Path(args.home)
KEY = os.environ.get('API_SERVER_KEY') or dotenv_values(HOME / '.env')['API_SERVER_KEY']
PUBLIC = Path(args.output_dir)
PUBLIC.mkdir(parents=True, exist_ok=True)
from urllib.parse import urlparse
HOST = urlparse(BASE).hostname
EXPECTED = ['hermes-agent', 'gpt-6.1-sol', 'opus-5.5', 'swift-1.5-qwen3.8-27b-orcarouter']
BACKENDS = {'hermes-agent': ('openai-codex', 'gpt-6.1-sol'), 'gpt-6.1-sol': ('openai-codex', 'gpt-6.1-sol'),
    'opus-5.5': ('nous', 'anthropic/claude-opus-5.5'),
    'swift-1.5-qwen3.8-27b-orcarouter': ('custom', 'swift-1.5-qwen3.8-27b-orcarouter')}
if args.dns_override:
    dns = json.load(urllib.request.urlopen('https://dns.google/resolve?name=' + HOST + '&type=A', timeout=15))
    ip = next(a['data'] for a in dns['Answer'] if a['type'] == 1)
    original_resolver = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, *a, **kw: original_resolver(ip if host == HOST else host, port, *a, **kw)
report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'base_url': BASE + '/v1',
    'model_discovery': None, 'checks': {}, 'uses_real_Hermes_runs': True, 'public_resolver_override': args.dns_override}
REPORT = PUBLIC / ('model-route-preflight.json' if args.preflight else 'model-route-verification.json')


def call(path, payload=None, headers=None):
    h = {'Authorization': 'Bearer ' + KEY, 'Accept': 'application/json', **(headers or {})}
    if payload is not None:
        h['Content-Type'] = 'application/json'
    req = urllib.request.Request(BASE + path, headers=h, data=None if payload is None else json.dumps(payload).encode())
    try:
        with OPENER.open(req, timeout=40) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except ValueError:
            body = {'error': 'Non-JSON HTTP failure'}
        return e.code, body


def save():
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')


if not args.preflight:
    status, models = call('/v1/models')
    ids = [m['id'] for m in models.get('data', [])]
    report['model_discovery'] = {'status': status, 'ids': ids, 'passed': status == 200 and sorted(EXPECTED) == sorted(ids)}
    save()
    print(json.dumps({'model_discovery': report['model_discovery']}), flush=True)
    assert report['model_discovery']['passed'], 'Discovered aliases must exactly match the expected deployment aliases'

for alias in (['gpt-6.1-sol', 'opus-5.5'] if args.preflight else EXPECTED):
    marker = 'HERMES_ROUTE_' + secrets.token_hex(5).upper()
    # The marker is absent from the prompt. Correct output requires this real tool.
    private_file = HOME / 'cache' / 'scratch' / ('route-probe-' + secrets.token_hex(6) + '.txt')
    private_file.parent.mkdir(parents=True, exist_ok=True)
    private_file.write_text(marker, encoding='utf-8')
    command = 'python -c "from pathlib import Path; print(Path(\'' + str(private_file).replace('\\', '/') + '\').read_text())"'
    prompt = ('Use the terminal tool to run exactly this read-only command: ' + command +
        '. It reads a synthetic verification marker only. Reply only with the actual command stdout. Do not invent output. Do not modify any files.')
    provider, target = BACKENDS[alias]
    payload = {'model': alias, 'input': prompt,
        'model_options': {'reasoning': {'enabled': True, 'effort': 'low'}}}
    if args.preflight:
        payload.update({'model': target, 'provider': provider})
    idem = 'model-route-test-' + secrets.token_hex(12)
    start = time.monotonic()
    status, submitted = call('/v1/runs', payload, {'Idempotency-Key': idem})
    check = {'submitted_status': status, 'requested_alias': alias, 'expected_backend': {'provider': provider, 'model': target}}
    if status != 202 or not submitted.get('run_id'):
        check.update({'passed': False, 'error': submitted.get('error')})
        report['checks'][alias] = check
        save()
        print(json.dumps({alias: check}), flush=True)
        continue
    run_id = submitted['run_id']
    check['run_id'] = run_id
    deadline = time.monotonic() + 300
    while True:
        status, state = call('/v1/runs/' + run_id)
        if status != 200 or state.get('status') in {'completed', 'failed', 'cancelled'}:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded model verification deadline exceeded for ' + alias + '; run ' + run_id)
        time.sleep(2)
    output = state.get('output') or ''
    runtime = state.get('runtime') or {}
    # Fetch the exact persisted session to independently prove the terminal call.
    session_id = state.get('session_id')
    tool_called = False
    tool_marker = False
    if session_id:
        _, transcript = call('/api/sessions/' + session_id + '/messages?limit=100')
        for message in transcript.get('data', transcript.get('messages', [])):
            if message.get('role') == 'tool' and message.get('tool_name') == 'terminal':
                tool_called = True
                tool_marker |= marker in str(message.get('content', ''))
    backend_matches = runtime.get('provider') == provider and runtime.get('model') == target
    # Some packaged versions omit runtime metadata: DB session model config can
    # confirm routing only when BOTH provider and model are proven.
    if not runtime and session_id:
        metadata_status, session = call('/api/sessions/' + session_id)
        metadata = session.get('data', session)
        model, resolved_provider = metadata.get('model'), metadata.get('provider')
        backend_matches = metadata_status == 200 and model == target and resolved_provider == provider
        runtime = {'model': model, 'provider': resolved_provider, 'source': 'authenticated_session_metadata'}
    check.update({'run_state': state.get('status'), 'poll_status': status, 'runtime': runtime,
        'real_terminal_called': tool_called, 'tool_marker_matches': tool_marker,
        'output_matches': marker in output, 'elapsed_seconds': round(time.monotonic() - start, 2),
        'passed': status == 200 and state.get('status') == 'completed' and marker in output and tool_called and tool_marker and backend_matches})
    if state.get('error'):
        check['error'] = state['error']
    report['checks'][alias] = check
    save()
    print(json.dumps({alias: check}), flush=True)
    private_file.unlink(missing_ok=True)
if not args.preflight:
    # Also require ordinary OpenAI clients to work without any model_options.
    report['openai_compatibility'] = {}
    for alias in EXPECTED:
        marker = 'COMPAT_' + secrets.token_hex(5).upper()
        prompt = 'Reply exactly ' + marker + '. Do not use any tools.'
        status, chat = call('/v1/chat/completions', {'model': alias, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False})
        answer = chat.get('choices', [{}])[0].get('message', {}).get('content', '')
        chat_ok = status == 200 and marker in answer
        status, response = call('/v1/responses', {'model': alias, 'input': prompt, 'store': True, 'stream': False})
        output_text = ''.join(part.get('text', '') for item in response.get('output', []) if item.get('type') == 'message' for part in item.get('content', []) if part.get('type') == 'output_text')
        response_ok = status == 200 and response.get('status') == 'completed' and marker in output_text
        retrieval_ok = False
        if response_ok:
            retrieval_status, replay = call('/v1/responses/' + response['id'])
            retrieval_ok = retrieval_status == 200 and replay == response
        result = {'chat_passed': chat_ok, 'responses_passed': response_ok, 'stored_response_matches': retrieval_ok, 'passed': chat_ok and response_ok and retrieval_ok}
        if not chat_ok:
            result['chat_failure'] = answer or chat.get('error')
        if not response_ok:
            result['response_error'] = response.get('error')
        report['openai_compatibility'][alias] = result
        save()
        print(json.dumps({'openai_compatibility': {alias: result}}), flush=True)
report['all_passed'] = all(c['passed'] for c in report['checks'].values()) and all(c['passed'] for c in report.get('openai_compatibility', {}).values())
save()
assert report['all_passed'], 'Model route verification failed; see report'
print('ALL MODEL ROUTES END-TO-END VERIFIED', flush=True)
