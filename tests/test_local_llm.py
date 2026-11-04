import json
import httpx,pytest
from fea import local_llm as llm
from fea.agent import Run,answer
from fea.domain import DomainError

def test_local_backend_requires_operator_opt_in(monkeypatch):
 monkeypatch.delenv('FEA_ENABLE_OLLAMA',raising=False)
 with pytest.raises(DomainError):answer('재고 평가','DEMO0001',2024,source='demo',mode='local')

def test_selector_enum_and_one_retry(monkeypatch):
 calls=[]
 def respond(req):
  body=json.loads(req.content);calls.append(body)
  assert body['format']['properties']['candidate_ids']['items']['enum']==['E1']
  value={'candidate_ids':['bad']} if len(calls)==1 else {'candidate_ids':['E1']}
  return httpx.Response(200,json={'done':True,'done_reason':'stop','message':{'content':json.dumps(value)}})
 with httpx.Client(transport=httpx.MockTransport(respond)) as c:
  chosen,retries=llm.select(c,'질문',[{'candidate_id':'E1','quote':'근거','section':'주석','evidence':{'id':'known'}}])
 assert chosen['id']=='known' and retries==1 and len(calls)==2

def test_abstention_is_not_retried():
 count=[]
 def respond(r):
  count.append(1);return httpx.Response(200,json={'done':True,'message':{'content':'{"candidate_ids":[]}'}})
 with httpx.Client(transport=httpx.MockTransport(respond)) as c:
  result,retries=llm.select(c,'질문',[{'candidate_id':'E1','quote':'근거','section':'주석','evidence':{'id':'known'}}])
 assert result is None and retries==0 and len(count)==1

def test_truncated_model_output_rejected():
 with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'done':True,'done_reason':'length','message':{'content':'partial'}}))) as c:
  with pytest.raises(DomainError):llm.chat(c,[])

@pytest.mark.parametrize('vector',[[float('nan')],[0,0],[True],['x'],[]])
def test_invalid_embedding_rejected(vector):
 with pytest.raises(DomainError):llm.normalize(vector)

def test_live_data_and_original_evidence_shared_with_web(monkeypatch):
 monkeypatch.setenv('FEA_ENABLE_OLLAMA','1')
 run=Run('DEMO0001',2024,'CFS','demo','bm25')
 original=run.tools.search_disclosures('DEMO0001',2024,'CFS','재고자산 평가정책')[0]
 monkeypatch.setattr(llm,'hybrid',lambda *a:[original])
 response=iter(['{"candidate_ids":["E1"]}','{"candidate_ids":["S1"]}'])
 with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'done':True,'done_reason':'stop','message':{'content':next(response)}}))) as c:
  final=llm.answer_local(run,'재고 평가',c)
 assert final['claims'][0]['evidence_ids']==[original['id']]
 assert all(x['valid'] for x in run.tools.verify([original],run.retrieved))
 assert [x['tool'] for x in run.trace]==['search_disclosures_hybrid','select_evidence','compose_source_answer']
