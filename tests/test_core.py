import copy
import json
import time
from decimal import Decimal
import httpx
import pytest
from fastapi.testclient import TestClient
from fea.agent import Run, answer
from fea.domain import DomainError, Tools
from fea.server import app, SESSIONS


@pytest.fixture
def tools():
    return Tools('demo')


def test_financial_basis_and_currency_are_part_of_fact_identity(tools):
    consolidated = tools.get_financials('DEMO0001', 2024, 'CFS', 'revenue')
    separate = tools.get_financials('DEMO0001', 2024, 'OFS', 'revenue')
    assert consolidated['amount'] == '125000000000'
    assert separate['amount'] == '100000000000'
    assert consolidated['id'] != separate['id']
    with pytest.raises(DomainError):
        tools.calculate_change(consolidated['id'], separate['id'])


def test_calculation_zero_and_missing_prior_are_not_silently_estimated(tools):
    a = tools.get_financials('DEMO0001', 2024, 'CFS', 'revenue')
    b = tools.get_financials('DEMO0001', 2023, 'CFS', 'revenue')
    assert tools.calculate_change(a['id'], b['id'])['value'] == '25.00'
    for row in tools.data['facts']:
        if row['id'] == b['id']:
            row['amount'] = '0'
    with pytest.raises(DomainError, match='0'):
        tools.calculate_change(a['id'], b['id'])
    with pytest.raises(DomainError):
        tools.get_financials('DEMO0001', 2022, 'CFS', 'revenue')


def test_unknown_company_and_tool_are_rejected(tools):
    with pytest.raises(DomainError):
        tools.get_financials('삼성', 2024, 'CFS', 'revenue')
    with pytest.raises(DomainError):
        tools.call('__getattribute__', {'name': 'data'})


def test_tampered_quote_offset_and_nonretrieved_fact_rejected(tools):
    found = tools.search_disclosures('DEMO0001', 2024, 'CFS', '재고자산 평가정책')
    original = found[0]
    assert tools.verify([original], found)[0]['valid']
    for field, value in [('quote', '허위 인용'), ('start', 999), ('sha256', 'bad')]:
        altered = {**original, field: value}
        assert not tools.verify([altered], found)[0]['valid']
    fact = tools.get_financials('DEMO0001', 2024, 'CFS', 'revenue')
    assert not tools.verify([fact], found)[0]['valid']


def test_company_and_scope_escape_blocked_before_execution():
    run = Run('DEMO0001', 2024, 'CFS', 'demo', 'expanded')
    assert 'error' in run.call('get_financials', {'company':'DEMO0002','year':2024,'basis':'CFS','metric':'revenue'})
    assert 'error' in run.call('get_financials', {'company':'DEMO0001','year':2024,'basis':'OFS','metric':'revenue'})
    assert 'error' in run.call('calculate_change', {'current_id':'invented','prior_id':'invented'})
    final = run.finish({'claims':[{'text':'허위 주장','evidence_ids':['invented']}], 'follow_up':[], 'limitations':[]}, 'rules', time.perf_counter())
    assert final['status'] == 'insufficient_evidence'
    assert not final['claims']


def test_rules_comparison_returns_retrieved_verified_evidence():
    result = answer('매출과 재고 증감률 비교', 'DEMO0001', 2024, source='demo')
    assert result['mode'] == 'rules'
    assert any('25.00%' in c['text'] for c in result['claims'])
    assert all(c['valid'] for c in result['checks'])


def test_missing_llm_key_does_not_silently_fall_back(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_MODEL', raising=False)
    with pytest.raises(DomainError, match='설정'):
        answer('매출 금액', 'DEMO0001', 2024, source='demo', mode='llm')


def test_responses_loop_executes_real_function_protocol_with_mock_transport(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_MODEL', 'test-model')
    requests = []
    fact_id = 'F-DEMO0001-2024-CFS-revenue'
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['store'] is False
        if len(requests) == 1:
            return httpx.Response(200,json={'output':[{'type':'reasoning','id':'r1','summary':[]},{'type':'function_call','call_id':'c1','name':'get_financials','arguments':json.dumps({'company':'DEMO0001','year':2024,'basis':'CFS','metric':'revenue'})}], 'usage':{'input_tokens':10,'output_tokens':20}})
        assert any(x.get('type')=='function_call_output' and x.get('call_id')=='c1' for x in body['input'])
        assert any(x.get('type')=='reasoning' for x in body['input'])
        final={'claims':[{'text':'매출 125,000,000,000원','evidence_ids':[fact_id]}], 'follow_up':[], 'limitations':[]}
        return httpx.Response(200,json={'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(final)}]}], 'usage':{'input_tokens':30,'output_tokens':40}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result=answer('매출 금액', 'DEMO0001',2024,source='demo',mode='llm',client=client)
    assert result['usage']=={'input_tokens':40,'output_tokens':60,'requests':2}
    assert result['status']=='answered'
    assert result['trace'][0]['tool']=='get_financials'


def test_local_api_requires_session_and_masks_validation_input():
    with TestClient(app) as client:
        assert client.post('/api/ask',json={'question':'매출'}).status_code==401
        response=client.post('/api/connect',json={'dart_key':'TOP-SECRET','company':'','year':1,'basis':'bad'})
        assert response.status_code==422
        assert 'TOP-SECRET' not in response.text
        assert client.post('/api/connect',json={},headers={'origin':'https://evil.example'}).status_code==403
        assert client.get('/').status_code==200
        assert "frame-ancestors 'none'" in client.get('/').headers['content-security-policy']


def test_byok_session_isolation_scenario_and_logout(monkeypatch, tools):
    import fea.server as server
    data=copy.deepcopy(tools.data)
    data['collected_at']='2026-11-03T00:00:00Z'
    data['warnings']=[]
    def fake_load(key,company,year,basis):
        assert key=='A'*40
        return data
    monkeypatch.setattr(server,'load_public',fake_load)
    with TestClient(app) as client, TestClient(app) as outsider:
        body={'dart_key':'A'*40,'company':'DEMO0001','year':2024,'basis':'CFS'}
        r=client.post('/api/connect',json=body)
        assert r.status_code==200
        assert 'A'*40 not in r.text
        assert 'HttpOnly' in r.headers['set-cookie']
        token=client.cookies.get('fea_session')
        assert 'dart_key' not in SESSIONS[token]
        assert outsider.post('/api/ask',json={'question':'매출'}).status_code==401
        r=client.post('/api/scenario',json={'loss_rate':'5'})
        assert r.status_code==200 and r.json()['additional_loss']=='750000000'
        assert r.json()['pro_forma_operating_profit']=='10250000000'
        assert client.post('/api/scenario',json={'loss_rate':'101'}).status_code==422
        assert client.post('/api/ask',json={'question':'매출액 얼마'}).json()['status']=='answered'
        assert client.delete('/api/session').status_code==200
        assert token not in SESSIONS
        assert client.get('/api/status').json()=={'connected':False}


def test_expired_sessions_rejected(tools):
    with TestClient(app) as client:
        SESSIONS['expired']={'expires':time.monotonic()-1,'data':tools.data}
        client.cookies.set('fea_session','expired')
        assert client.post('/api/ask',json={'question':'매출'}).status_code==401
        assert 'expired' not in SESSIONS
