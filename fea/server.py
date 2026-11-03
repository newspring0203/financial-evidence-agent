"""Local BYOK web workspace. No keys in files, URLs, logs or localStorage."""
import json
import asyncio
from contextlib import asynccontextmanager
import secrets
import threading
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from .agent import answer
from .domain import ROOT, Tools, DomainError
from .live import load_public
from .knowledge import parse_document, retrieve, public_metadata

@asynccontextmanager
async def lifespan(app):
    async def expire_sessions():
        while True:
            await asyncio.sleep(30)
            with LOCK:
                prune()
    sweeper = asyncio.create_task(expire_sessions())
    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except asyncio.CancelledError:
            pass
        with LOCK:
            SESSIONS.clear()


app = FastAPI(title='Financial Evidence Agent', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
SESSIONS = {}
LOCK = threading.Lock()
COLLECT_LOCK = threading.BoundedSemaphore(2)
TTL = 3600


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return JSONResponse({'detail': '입력 형식·길이·범위를 확인하세요. 키와 상세 입력은 반환하지 않습니다.'}, status_code=422)


@app.middleware('http')
async def local_boundary(request: Request, call_next):
    if request.method in ('POST', 'DELETE'):
        origin = request.headers.get('origin')
        if origin and urlparse(origin).netloc != request.headers.get('host'):
            return JSONResponse({'detail': '같은 사이트에서 요청하세요.'}, status_code=403)
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            return JSONResponse({'detail': '잘못된 요청입니다.'}, status_code=400)
        if length > (1_100_000 if request.url.path == '/api/knowledge' else 16000):
            return JSONResponse({'detail': '요청이 너무 큽니다.'}, status_code=413)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    return response


def prune():
    now = time.monotonic()
    for token in list(SESSIONS):
        if SESSIONS[token]['expires'] < now:
            del SESSIONS[token]


def session(request, public=True):
    with LOCK:
        prune()
        value = SESSIONS.get(request.cookies.get('fea_session', ''))
        if not value:
            raise HTTPException(401, 'OpenDART 키를 입력해 공시를 먼저 조회하세요. 세션은 1시간 후 만료됩니다.')
        if public and not value.get('data'):
            raise HTTPException(401, '공시 분석을 사용하려면 OpenDART 키로 기업 공시를 먼저 조회하세요.')
        return value


class Connect(BaseModel):
    dart_key: SecretStr
    company: str = Field(min_length=1, max_length=100)
    year: int = Field(ge=2016, le=2100)
    basis: str = Field(pattern='^(CFS|OFS)$')
    openai_key: SecretStr = SecretStr('')
    model: str = Field(default='', max_length=100)


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: str = Field(default='rules', pattern='^(rules|llm)$')
    strategy: str = Field(default='bm25', pattern='^(keyword|expanded|bm25)$')


class Scenario(BaseModel):
    loss_rate: Decimal = Field(ge=0, le=100, max_digits=6, decimal_places=3)


def metadata(value):
    data = value['data']
    if not data:
        return {'connected':False, 'knowledge_ready':True}
    tool = Tools('dart', data=data)
    cards = []
    for metric, label in [('revenue', '매출'), ('operating_profit', '영업이익'), ('inventory', '재고자산')]:
        try:
            fact = tool.get_financials(value['company'], value['year'], value['basis'], metric)
            prior = tool.get_financials(value['company'], value['year'] - 1, value['basis'], metric)
            change = tool.calculate_change(fact['id'], prior['id'])
            cards.append({'label': label, 'fact': fact, 'prior': prior, 'change': change})
        except DomainError:
            cards.append({'label': label, 'unavailable': True})
    return {'connected': True, 'knowledge_ready':True, 'company': data['companies'][0], 'year': value['year'], 'basis': value['basis'],
            'collected_at': data['collected_at'], 'warnings': data.get('warnings', []), 'cards': cards,
            'documents': [{k: v for k, v in d.items() if k != 'text'} for d in data['documents']],
            'llm_ready': bool(value['credentials'][0] and value['credentials'][1]),
            'model': value['credentials'][1], 'expires_in_seconds': max(0, int(value['expires'] - time.monotonic()))}


@app.get('/api/status')
def status(request: Request):
    try:
        return metadata(session(request, public=False))
    except HTTPException:
        return {'connected': False}


@app.post('/api/connect')
def connect(body: Connect, request: Request, response: Response):
    if not COLLECT_LOCK.acquire(blocking=False):
        raise HTTPException(429, '다른 조회가 진행 중입니다. 잠시 후 다시 시도하세요.')
    try:
        with LOCK:
            prune()
            if len(SESSIONS) >= 8 and request.cookies.get('fea_session') not in SESSIONS:
                raise HTTPException(429, '로컬 세션 한도에 도달했습니다. 기존 세션을 종료하세요.')
        data = load_public(body.dart_key.get_secret_value(), body.company, body.year, body.basis)
        token = secrets.token_urlsafe(32)
        value = {'data': data, 'company': data['companies'][0]['corp_code'], 'year': body.year,
                 'basis': body.basis, 'expires': time.monotonic() + TTL,
                 'credentials': (body.openai_key.get_secret_value(), body.model.strip()),
                 'knowledge': [], 'folders': default_folders()}
        with LOCK:
            previous = SESSIONS.get(request.cookies.get('fea_session',''))
            if previous:
                value['knowledge'] = previous.get('knowledge', [])
                value['folders'] = previous.get('folders', default_folders())
        with LOCK:
            SESSIONS.pop(request.cookies.get('fea_session', ''), None)
            SESSIONS[token] = value
        response.set_cookie('fea_session', token, httponly=True, samesite='strict', max_age=TTL, secure=request.url.scheme == 'https')
        return metadata(value)
    except DomainError as error:
        raise HTTPException(400, str(error)) from None
    except (ValueError, OSError, KeyError):
        raise HTTPException(502, '공시 수집 응답을 처리하지 못했습니다. 잠시 후 다시 시도하세요.') from None
    finally:
        COLLECT_LOCK.release()


@app.delete('/api/session')
def disconnect(request: Request, response: Response):
    with LOCK:
        SESSIONS.pop(request.cookies.get('fea_session', ''), None)
    response.delete_cookie('fea_session')
    return {'disconnected': True}


@app.post('/api/ask')
def ask(body: Ask, request: Request):
    value = session(request)
    try:
        return answer(body.question, value['company'], value['year'], value['basis'],
                      source='dart', mode=body.mode, strategy=body.strategy,
                      data=value['data'], credentials=value['credentials'])
    except DomainError as error:
        raise HTTPException(400, str(error)) from None


@app.post('/api/scenario')
def scenario(body: Scenario, request: Request):
    value = session(request)
    tool = Tools('dart', data=value['data'])
    try:
        inventory = tool.get_financials(value['company'], value['year'], value['basis'], 'inventory')
        profit = tool.get_financials(value['company'], value['year'], value['basis'], 'operating_profit')
        if (inventory['currency'], inventory['unit']) != (profit['currency'], profit['unit']):
            raise DomainError('서로 다른 통화·단위를 비교할 수 없습니다.')
        loss = (Decimal(inventory['amount']) * body.loss_rate / 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        return {'rate': str(body.loss_rate), 'additional_loss': str(loss),
                'pro_forma_operating_profit': str(Decimal(profit['amount']) - loss),
                'inventory': inventory, 'profit': profit,
                'formula': '공시 재고자산 장부금액 × 사용자 가정률 / 100',
                'note': '입력한 손실률만큼 재고에서 추가 손실이 생겼을 때의 영업이익을 계산합니다.\n\n계산에서는 추가 손실 전액을 영업이익에서 차감합니다. 이미 반영된 평가손실과 세금 효과는 추가로 계산하지 않습니다.\n\n실제 손실금액은 재고 상태와 판매 가능금액을 확인해 별도로 판단해 주세요.'}
    except DomainError as error:
        raise HTTPException(400, str(error)) from None


class PlanRequest(BaseModel):
    area: str = Field(pattern='^(revenue|inventory|operating_profit)$')
    due_date: str = Field(default='', max_length=10)


@app.post('/api/review-plan')
def review_plan(body: PlanRequest, request: Request):
    from .planning import make_plan
    value = session(request)
    try:
        return make_plan(value['data'], value['company'], value['year'], value['basis'], body.area, body.due_date)
    except DomainError as error:
        raise HTTPException(400, str(error)) from None


@app.get('/api/footing')
def footing(request: Request):
    value = session(request)
    tools = Tools('dart', data=value['data'])
    rows = []
    for metric in ('revenue','operating_profit','inventory'):
        try:
            current = tools.get_financials(value['company'],value['year'],value['basis'],metric)
            prior = tools.get_financials(value['company'],value['year']-1,value['basis'],metric)
            result = tools.reconcile_prior(current['id'],prior['id'])
            rows.append({'metric':metric,'current':current,'prior':prior,'result':result})
        except DomainError as error:
            rows.append({'metric':metric,'unavailable':str(error)})
    return {'rows':rows,'note':'서로 다른 연간 보고서에 표시된 같은 전기 금액을 비교합니다. 차이는 재작성·표시 변경·정정 여부를 확인할 대상으로 남깁니다. 주석 간 대사나 전체 재무제표 풋팅이 완료되었다는 뜻은 아닙니다.'}


DEFAULT_FOLDERS = [
    {'code':'C','name':'현금및현금성자산'}, {'code':'D','name':'금융자산'},
    {'code':'E','name':'매출채권'}, {'code':'F','name':'재고자산'},
    {'code':'VC','name':'판관비'}, {'code':'UB&VD&UC&VE','name':'기타손익'},
    {'code':'COMMON','name':'공통 자료'},
]


def default_folders():
    return [dict(f) for f in DEFAULT_FOLDERS]


@app.post('/api/knowledge/session')
def start_knowledge_session(request: Request, response: Response):
    with LOCK:
        prune()
        token = request.cookies.get('fea_session','')
        if token not in SESSIONS:
            if len(SESSIONS) >= 8:
                raise HTTPException(429, '작업 공간이 가득 찼습니다. 기존 세션을 종료하세요.')
            token = secrets.token_urlsafe(32)
            SESSIONS[token] = {'data':None, 'expires':time.monotonic()+TTL,
                               'credentials':('',''), 'knowledge':[], 'folders':default_folders()}
    response.set_cookie('fea_session',token,httponly=True,samesite='strict',max_age=TTL,secure=request.url.scheme=='https')
    return {'knowledge_ready':True}


class FolderRequest(BaseModel):
    code: str = Field(min_length=1,max_length=30,pattern=r'^[A-Za-z0-9&_-]+$')
    name: str = Field(min_length=1,max_length=60)


@app.post('/api/knowledge/folders')
def add_folder(body: FolderRequest, request: Request):
    value = session(request,public=False)
    folders = value.setdefault('folders',default_folders())
    code = body.code.strip().upper()
    if code=='ALL' or any(f['code']==code for f in folders):
        raise HTTPException(400,'이미 사용 중이거나 예약된 계정 코드입니다.')
    if len(folders)>=40:
        raise HTTPException(400,'계정 폴더는 40개까지 만들 수 있습니다.')
    folders.append({'code':code,'name':body.name.strip()})
    return {'folders':folders}


@app.post('/api/knowledge')
async def upload_knowledge(request: Request, file: UploadFile = File(...), title: str = Form(''),
                           version: str = Form(''), effective_date: str = Form(''), account_code: str = Form('COMMON')):
    value = session(request,public=False)
    if account_code not in {f['code'] for f in value.get('folders',default_folders())}:
        raise HTTPException(400,'계정 폴더를 선택하세요.')
    if len(value.setdefault('knowledge', [])) >= 30:
        raise HTTPException(400, '세션당 문서는 30개까지 등록할 수 있습니다.')
    try:
        raw = await file.read(1_000_001)
        doc = parse_document(file.filename or '', raw, title, version, effective_date)
        doc['account_code'] = account_code
        if any(d['id'] == doc['id'] for d in value['knowledge']):
            raise DomainError('같은 내용의 문서가 이미 등록되어 있습니다.')
        value['knowledge'].append(doc)
        return public_metadata(doc)
    except DomainError as error:
        raise HTTPException(400, str(error)) from None
    finally:
        await file.close()


@app.get('/api/knowledge')
def list_knowledge(request: Request):
    value = session(request,public=False)
    return {'documents': [public_metadata(d) for d in value.get('knowledge', [])], 'folders':value.get('folders',default_folders())}


class KnowledgeSearch(Ask):
    account_code: str = Field(default='ALL',max_length=30)


@app.post('/api/knowledge/search')
def search_knowledge(body: KnowledgeSearch, request: Request):
    value = session(request,public=False)
    documents = value.get('knowledge', [])
    if not documents:
        raise HTTPException(400, '등록된 문서가 없습니다. 파일을 먼저 등록한 뒤 검색하세요.')
    if body.account_code != 'ALL':
        documents = [d for d in documents if d.get('account_code','COMMON') == body.account_code]
        if not documents:
            raise HTTPException(400,'선택한 계정 폴더에 문서가 없습니다. 문서를 등록하거나 다른 계정을 선택하세요.')
    found = retrieve(documents, body.question)
    return {'question': body.question, 'results': found, 'external_llm_transmission': False,
            'status': 'found' if found else 'insufficient_evidence',
            'note': '등록한 문서에서 찾은 관련 내용입니다. 문서에 없는 내용은 답변하지 않습니다. 적용일과 버전이 현재 업무에 맞는지 확인해 주세요.'}


@app.delete('/api/knowledge/{document_id}')
def remove_knowledge(document_id: str, request: Request):
    value = session(request,public=False)
    value['knowledge'] = [d for d in value.get('knowledge', []) if d['id'] != document_id]
    return {'removed': True}


@app.get('/api/experiments')
def experiments():
    path = ROOT / 'experiments' / 'results.json'
    result = json.loads(path.read_text()) if path.exists() else {'runs': [], 'note': '이 환경에는 로컬 평가 기록이 없습니다. 결과 파일은 배포에 포함하지 않습니다.'}
    frozen = ROOT / 'experiments' / 'retrieval-frozen-results.json'
    if frozen.exists():
        report = json.loads(frozen.read_text())
        result['runs'].extend(report['runs'])
        result['note'] += '\n' + report['note']
    return result


@app.get('/')
def index():
    return FileResponse(ROOT / 'web' / 'index.html')


app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')
