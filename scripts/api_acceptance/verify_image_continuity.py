"""Real HTTPS image acceptance checks; fixture is synthetic, not a user screenshot."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import secrets
import socket
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dotenv import dotenv_values
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['before', 'after', 'restart'], required=True)
parser.add_argument('--home', default=os.environ.get('HERMES_HOME'))
parser.add_argument('--retention-only', action='store_true', help='Test image history/replay without secondary vision-tool or validation changes')
parser.add_argument('--base-url', required=True, help='HTTPS origin without /v1')
parser.add_argument('--output-dir', required=True)
parser.add_argument('--dns-override', action='store_true', help='Use public DoH to bypass a stale LAN resolver')
parser.add_argument('--font', default='C:/Windows/Fonts/arialbd.ttf', help='TrueType font file for the synthetic fixture')
parser.add_argument('--protocol', choices=['chat', 'responses'], default='chat')
args = parser.parse_args()
from acceptance_http import validate_api_origin, build_api_opener
BASE = validate_api_origin(args.base_url)
OPENER = build_api_opener(BASE)
PUBLIC = Path(args.output_dir)
PUBLIC.mkdir(parents=True, exist_ok=True)
from urllib.parse import urlparse
HOST = urlparse(BASE).hostname
HOME = Path(args.home)
KEY = os.environ.get('API_SERVER_KEY') or dotenv_values(HOME / '.env')['API_SERVER_KEY']
FIXTURE = PUBLIC / 'image-continuity-fixture.json'
RESULTS = PUBLIC / 'image-continuity-results'
RESULTS.mkdir(exist_ok=True)

# An override is opt-in and retains the original TLS hostname verification.
if args.dns_override:
    dns = json.load(urllib.request.urlopen('https://dns.google/resolve?name=' + HOST + '&type=A', timeout=15))
    ip = next(a['data'] for a in dns['Answer'] if a['type'] == 1)
    original_resolver = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, *a, **kw: original_resolver(ip if host == HOST else host, port, *a, **kw)


def call(path, payload=None, session=None):
    headers = {'Authorization': 'Bearer ' + KEY, 'Accept': 'application/json'}
    if session:
        headers['X-Hermes-Session-Id'] = session
    if payload is not None:
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(BASE + path, headers=headers,
        data=None if payload is None else json.dumps(payload).encode())
    try:
        with OPENER.open(request, timeout=180) as response:
            return response.status, json.load(response), dict(response.headers)
    except urllib.error.HTTPError as error:
        try:
            result = json.loads(error.read())
        except ValueError:
            result = {'error': 'Non-JSON HTTP failure'}
        return error.code, result, dict(error.headers)


def contains_image(value):
    """Require intact, renderable fixture bytes, not a placeholder/attachment flag."""
    if isinstance(value, dict):
        if value.get('type') in {'image_url', 'input_image', 'image'}:
            ref = value.get('image_url') or value.get('url')
            url = ref.get('url') if isinstance(ref, dict) else ref
            if isinstance(url, str) and url.startswith('data:image/'):
                try:
                    raw = base64.b64decode(url.partition(',')[2], validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        image.verify()
                    if raw == Path(manifest['path']).read_bytes():
                        return True
                except (ValueError, OSError):
                    pass
        return any(contains_image(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_image(v) for v in value)
    return False


def history_rows(value):
    rows = value.get('data', value.get('messages', []))
    assert isinstance(rows, list) and rows and len(rows) < 100, 'Authenticated history missing or truncated'
    return rows


def tool_call_count(value):
    return sum(len(m.get('tool_calls') or []) for m in history_rows(value) if m.get('role') == 'assistant')


def assert_target_unmentioned(value, target):
    def text(value):
        if isinstance(value, dict):
            return '\n'.join(text(v) for v in value.values())
        if isinstance(value, list):
            return '\n'.join(text(v) for v in value)
        if not isinstance(value, str) or value.startswith('data:image/'):
            return ''
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        return text(decoded) if not isinstance(decoded, str) else decoded
    prior = '\n'.join(text(m) for m in history_rows(value) if m.get('role') in {'assistant', 'tool'})
    assert target.casefold() not in prior.casefold(), 'Target label was already disclosed in previous text/tool history'


def successful_vision(value):
    import re
    rows = history_rows(value)
    user_text = '\n'.join(
        m['content'] if isinstance(m.get('content'), str) else '\n'.join(
            p.get('text', '') for p in m.get('content', []) if isinstance(p, dict))
        for m in rows if m.get('role') == 'user')
    valid_calls = set()
    for message in rows:
        if message.get('role') == 'assistant':
            for tool in message.get('tool_calls') or []:
                function = tool.get('function') or {}
                if function.get('name') != 'vision_analyze':
                    continue
                arguments = function.get('arguments') or {}
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except ValueError:
                    continue
                ref = arguments.get('image_url') if isinstance(arguments, dict) else None
                if not isinstance(ref, str):
                    continue
                intact_inline = contains_image({'type': 'image_url', 'image_url': ref})
                managed_handle = (re.search(r'/images/api_[0-9a-f]{64}\.[a-zA-Z0-9]+$', ref.replace('\\', '/'))
                                  and '[Image attached at: ' + ref + ']' in user_text)
                if (intact_inline or managed_handle) and tool.get('id'):
                    valid_calls.add(tool['id'])
        if message.get('role') != 'tool' or message.get('tool_call_id') not in valid_calls:
            continue
        result = message.get('content')
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                # Persisted native multimodal results have a text projection, not JSON.
                if result.startswith(('Image loaded into your context — you can see it natively now.',
                                      'Image attached natively for the main model (')):
                    return True
                continue
        if isinstance(result, dict) and not result.get('error') and result.get('success') is not False:
            if result.get('success') is True or result.get('_multimodal') is True:
                return True
    return False


if not FIXTURE.exists():
    manifest = {'code': 'KITE-' + secrets.token_hex(3).upper(), 'lower_left': 'L-' + secrets.token_hex(3).upper(),
        'lower_right': 'R-' + secrets.token_hex(3).upper(), 'triangle_color': 'purple'}
    image = Image.new('RGB', (1400, 900), '#ffffff')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(args.font, 86)
    small = ImageFont.truetype(args.font, 62)
    draw.text((65, 50), manifest['code'], font=font, fill='black')
    draw.polygon([(700, 235), (440, 615), (960, 615)], fill='#8020c0')
    draw.rectangle((50, 310, 285, 565), fill='#00a060')
    draw.ellipse((1100, 310, 1340, 550), fill='#f08010')
    draw.text((40, 750), manifest['lower_left'], font=small, fill='black')
    draw.text((820, 750), manifest['lower_right'], font=small, fill='black')
    path = PUBLIC / 'image-continuity-fixture.png'
    image.save(path)
    manifest['path'] = str(path)
    FIXTURE.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
manifest = json.loads(FIXTURE.read_text())
image_url = 'data:image/png;base64,' + base64.b64encode(Path(manifest['path']).read_bytes()).decode()
report = {'phase': args.phase, 'checked_at': datetime.now(timezone.utc).isoformat(), 'hostname': HOST,
    'public_resolver_override': args.dns_override, 'synthetic_fixture': manifest['path'], 'checks': {}, 'protocol': args.protocol}
report_prefix = '' if args.protocol == 'chat' else 'responses-'
response_cursor = None
native_only = ' Do not use any tools. Read the retained image directly.' if args.retention_only else ''
native_policy = ('For this synthetic native-image replay verification only, inspect the supplied native image parts '
    'directly and never call tools. On later turns, read the retained native image from conversation history. '
    'Tool calls would invalidate this verification. Return only the requested visual detail.') if args.retention_only else None


def record(name, result):
    report['checks'][name] = result
    (RESULTS / (report_prefix + args.phase + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({name: result}), flush=True)


def chat(content, session):
    global response_cursor
    if args.protocol == 'responses':
        parts = [{'type': 'input_text', 'text': content}] if isinstance(content, str) else [
            {'type': 'input_text', 'text': p['text']} if p['type'] == 'text' else
            {'type': 'input_image', 'image_url': p['image_url']['url']} for p in content]
        payload = {'model': 'hermes-agent', 'store': True, 'stream': False,
            'input': [{'role': 'user', 'content': parts}]}
        if native_policy:
            payload['instructions'] = native_policy
        if response_cursor:
            payload['previous_response_id'] = response_cursor
        status, result, headers = call('/v1/responses', payload)
        if result.get('id'):
            response_cursor = result['id']
            report['response_id'] = response_cursor
        actual_session = next((v for k, v in headers.items() if k.lower() == 'x-hermes-session-id'), None)
        if actual_session:
            report['session_id'] = actual_session
        answer = ''.join(p.get('text', '') for item in result.get('output', []) if item.get('type') == 'message'
            for p in item.get('content', []) if p.get('type') == 'output_text')
        return status, answer, result
    messages = [{'role': 'user', 'content': content}]
    if native_policy:
        messages.insert(0, {'role': 'system', 'content': native_policy})
    status, result, _ = call('/v1/chat/completions', {'model': 'hermes-agent', 'stream': False,
        'messages': messages}, session=session)
    answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
    return status, answer, result


if args.phase in {'before', 'after'}:
    session = 'api-image-verification-' + args.phase + '-' + secrets.token_hex(5)
    report['session_id'] = session
    status, answer, result = chat([
        {'type': 'text', 'text': ('Read the attached image directly without using any tools. Tell me only the code printed at the top and the color of the large triangle.' if args.retention_only else 'Inspect the attached image using vision_analyze with a real, intact image reference. Never invent or shorten an image URL. Tell me only the code printed at the top and the color of the large triangle. If inspection fails, say it failed; do not guess.')},
        {'type': 'image_url', 'image_url': {'url': image_url}}], session)
    session = report['session_id']
    first_passed = status == 200 and manifest['code'] in answer and manifest['triangle_color'] in answer.lower()
    record('fresh_image', {'status': status, 'answer': answer, 'passed': first_passed})
    status, messages, _ = call('/api/sessions/' + session + '/messages?limit=100')
    image_preserved = status == 200 and contains_image(messages)
    record('authenticated_history_has_image', {'status': status, 'passed': image_preserved})
    if not args.retention_only:
        record('authenticated_history_has_successful_vision', {
            'status': status, 'passed': status == 200 and successful_vision(messages)})
    assert status == 200, 'Cannot inspect authenticated history before follow-up'
    assert_target_unmentioned(messages, manifest['lower_left'])
    record('followup_target_not_disclosed', {'passed': True})
    previous_tools = tool_call_count(messages)
    # This detail was deliberately not requested or printed in the first answer.
    status, answer, _ = chat('Look again at the previously attached image, without guessing. What exact label is printed in the lower-left corner? Reply only with that label.' + native_only, session)
    followup = status == 200 and manifest['lower_left'] in answer
    record('followup_without_reupload', {'status': status, 'answer': answer, 'passed': followup})
    if args.retention_only:
        status, messages, _ = call('/api/sessions/' + session + '/messages?limit=100')
        record('followup_without_new_tools', {'status': status,
            'passed': status == 200 and tool_call_count(messages) == previous_tools == 0})
    if not args.retention_only:
        status, malformed, _ = call('/v1/chat/completions', {'model': 'hermes-agent', 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe this image only if it is valid.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,[omitted]'}}]}]})
        record('malformed_image_rejected', {'status': status, 'error': malformed.get('error'), 'passed': status == 400})
    report['all_passed'] = all(c['passed'] for c in report['checks'].values())
    (RESULTS / (report_prefix + args.phase + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert report['all_passed'], 'Real image acceptance checks failed; see report'
else:
    previous = json.loads((RESULTS / (report_prefix + 'after.json')).read_text())
    session = previous['session_id']
    response_cursor = previous.get('response_id')
    report['session_id'] = session
    status, messages, _ = call('/api/sessions/' + session + '/messages?limit=100')
    assert status == 200, 'Cannot inspect authenticated history before restart question'
    if not args.retention_only:
        record('authenticated_history_has_successful_vision', {
            'status': status, 'passed': successful_vision(messages)})
    assert_target_unmentioned(messages, manifest['lower_right'])
    record('restart_target_not_disclosed', {'passed': True})
    previous_tools = tool_call_count(messages)
    status, answer, _ = chat('Inspect the image I attached earlier. What exact label is printed in the lower-right corner? Reply only with that label. Do not infer it from the other label and do not guess.' + native_only, session)
    record('image_replayed_after_gateway_restart', {'status': status, 'answer': answer,
        'passed': status == 200 and manifest['lower_right'] in answer})
    status, messages, _ = call('/api/sessions/' + session + '/messages?limit=100')
    record('history_image_after_restart', {'status': status, 'passed': status == 200 and contains_image(messages)})
    if args.retention_only:
        record('restart_without_new_tools', {'status': status,
            'passed': status == 200 and tool_call_count(messages) == previous_tools})
    report['all_passed'] = all(c['passed'] for c in report['checks'].values())
    (RESULTS / (report_prefix + 'restart.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert report['all_passed'], 'Image persistence after restart failed; see report'
print('IMAGE ACCEPTANCE PASSED', flush=True)
