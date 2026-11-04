import httpx
import pytest
from fastapi.testclient import TestClient
from fea.agent import answer, llm_error_message
from fea.domain import DomainError
from fea.server import app


@pytest.mark.parametrize('code,expected',[
 ('credit_balance_exhausted','크레딧 잔액'),('insufficient_quota','크레딧 또는 사용 한도'),
 ('rate_limit_exceeded','분당 요청'),('slow_down','짧은 시간'),
 ('organization_spend_limit_exceeded','조직의 API 지출'),('project_spend_limit_exceeded','프로젝트의 API 지출'),
 ('organization_usage_limit_exceeded','조직에 적용된 API 사용'),
])
def test_error_codes_are_specific_and_redacted(code,expected):
 response=httpx.Response(429,json={'error':{'code':code,'message':'SECRET-DO-NOT-RETURN'}})
 message=llm_error_message(response)
 assert expected in message and code in message
 assert 'SECRET' not in message


def test_unknown_or_html_errors_are_not_echoed():
 for response in [httpx.Response(429,text='SECRET'),httpx.Response(429,json={'error':{'code':['SECRET']}}),httpx.Response(429,json={'error':'SECRET'})]:
  message=llm_error_message(response)
  assert 'SECRET' not in message and '429' in message


def test_billing_failure_stops_without_retry_or_fallback():
 calls=[]
 def fail(request):
  calls.append(request)
  return httpx.Response(429,json={'error':{'code':'credit_balance_exhausted','type':'insufficient_quota'}})
 with httpx.Client(transport=httpx.MockTransport(fail)) as client:
  with pytest.raises(DomainError,match='credit_balance_exhausted'):
   answer('매출액','DEMO0001',2024,source='demo',mode='llm',client=client,credentials=('test-key','test-model'))
 assert len(calls)==1


def test_pbc_without_dart_and_folder_isolation():
 with TestClient(app) as client:
  assert client.post('/api/knowledge/session',json={}).status_code==200
  assert client.get('/api/status').json()=={'connected':False,'knowledge_ready':True}
  assert client.post('/api/ask',json={'question':'매출'}).status_code==401
  assert client.post('/api/knowledge/search',json={'question':'재고'}).status_code==400
  assert client.post('/api/knowledge/folders',json={'code':'G','name':'유형자산'}).status_code==200
  text='재고자산 평가 시 장기체화 내역과 판매실적을 확인한다.'
  r=client.post('/api/knowledge',files={'file':('check.md',text.encode())},data={'account_code':'F'})
  assert r.status_code==200
  found=client.post('/api/knowledge/search',json={'question':'재고자산 평가','account_code':'F'})
  assert found.status_code==200 and found.json()['results']
  assert client.post('/api/knowledge/search',json={'question':'재고자산 평가','account_code':'C'}).status_code==400
  client.delete('/api/session')
