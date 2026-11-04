import io,json,zipfile,time
import httpx,pytest
from fastapi.testclient import TestClient
from fea.agent import Run,answer
from fea.domain import DomainError
from fea.dart_archive import safe_xml,archive_members
import fea.server as server

@pytest.mark.parametrize('args',[
 {'company':'DEMO0001','year':True,'basis':'CFS','metric':'revenue'},
 {'company':'DEMO0001','year':2024,'basis':'CFS','metric':'revenue','unexpected':'secret'},
 {'company':['DEMO0001'],'year':2024,'basis':'CFS','metric':'revenue'},
 {'company':'DEMO0001','year':2024,'basis':'CFS','metric':'__class__'},
])
def test_tool_schema_rejects_before_execution(args):
 run=Run('DEMO0001',2024,'CFS','demo','bm25')
 assert 'error' in run.call('get_financials',args)
 assert not run.retrieved

@pytest.mark.parametrize('payload',[
 {'status':'incomplete','output':[]},
 {'status':'failed','output':[]},
 {'output':{}},[],
 {'output':[{'type':'function_call','name':'get_financials','arguments':'{}'}]},
 {'output':[{'type':'function_call','call_id':'x','name':'get_financials','arguments':'{}'}]*9},
])
def test_provider_bad_responses_fail_closed(payload):
 with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=payload))) as client:
  with pytest.raises(DomainError):answer('매출','DEMO0001',2024,source='demo',mode='llm',client=client,credentials=('test','test'))

def test_provider_non_json_response_is_sanitized():
 with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,text='secret-provider-message'))) as client:
  with pytest.raises(DomainError) as caught:answer('매출','DEMO0001',2024,source='demo',mode='llm',client=client,credentials=('test','test'))
  assert 'secret-provider-message' not in str(caught.value)

@pytest.mark.parametrize('origin',['https://evil.example','null','https://testserver','http://testserver/path'])
def test_exact_origin_check(origin):
 with TestClient(server.app) as c:
  assert c.post('/api/knowledge/session',headers={'Origin':origin}).status_code==403

def test_cross_site_header_and_chunked_body_limits():
 with TestClient(server.app) as c:
  assert c.post('/api/knowledge/session',headers={'sec-fetch-site':'cross-site'}).status_code==403
  r=c.post('/api/valuation',content=iter([b'x'*9000,b'y'*9000]),headers={'Content-Type':'application/json'})
  assert r.status_code==413

def test_doc_ownership_and_logout():
 with TestClient(server.app) as a,TestClient(server.app) as b:
  a.post('/api/knowledge/session');b.post('/api/knowledge/session')
  result=a.post('/api/knowledge',files={'file':('../../owned.md',b'private secret audit evidence','text/plain')},data={'title':'<script>alert(1)</script>'})
  assert result.status_code==200
  identifier=result.json()['id']
  b.delete('/api/knowledge/'+identifier)
  assert len(a.get('/api/knowledge').json()['documents'])==1
  assert b.get('/api/knowledge').json()['documents']==[]
  a.delete('/api/session')
  assert a.get('/api/knowledge').status_code==401

@pytest.mark.parametrize('raw',[b'<!DOCTYPE x [<!ENTITY y "secret">]><x>&y;</x>',b'<broken>'])
def test_xml_entity_and_malformed_xml_rejected(raw):
 with pytest.raises(DomainError):safe_xml(raw)

def test_same_session_cannot_start_concurrent_llm_runs(monkeypatch):
 from fea.domain import Tools
 d=Tools('demo').data;d.update(collected_at='now',warnings=[])
 monkeypatch.setattr(server,'load_public',lambda *args:d)
 with TestClient(server.app) as c:
  c.post('/api/connect',json={'dart_key':'A'*40,'company':'DEMO0001','year':2024,'basis':'CFS'})
  server.SESSIONS[c.cookies.get('fea_session')]['analysis_running']=True
  assert c.post('/api/ask',json={'question':'매출'}).status_code==429
