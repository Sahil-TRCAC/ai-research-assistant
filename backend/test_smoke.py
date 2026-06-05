import sys, os, json
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['FLASK_ENV'] = 'development'
os.environ['SECRET_KEY'] = 'test-key'
os.environ['CORS_ORIGINS'] = 'http://localhost:3000'

from app import create_app
app = create_app()
client = app.test_client()

tests = [
    ('GET',  '/api/health'),
    ('GET',  '/api/health/db'),
    ('GET',  '/api/health/detailed'),
    ('GET',  '/api/documents'),
    ('POST', '/api/research/sessions'),       # missing body -> 400
    ('GET',  '/api/documents/not-a-uuid'),    # bad UUID -> 400
    ('GET',  '/api/documents/00000000-0000-0000-0000-000000000000'),  # not found -> 404
]

all_passed = True
for method, path in tests:
    if method == 'GET':
        r = client.get(path)
    else:
        r = client.post(path, json={})
    body = json.loads(r.data)
    ok = r.status_code < 500
    if not ok:
        all_passed = False
    tag = 'PASS' if ok else 'FAIL'
    print(f"{tag}  {method:6} {path}  ->  {r.status_code}  success={body.get('success')}")

print()
print('ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED')
