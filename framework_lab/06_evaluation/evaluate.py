"""Bounded local Ragas evaluation. No retrieval or answer generation."""
import os
os.environ['RAGAS_DO_NOT_TRACK']='true'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['LANGSMITH_TRACING']='false'
import argparse,asyncio,inspect,importlib.metadata as md,json,math,time,urllib.request,sys
from pathlib import Path
import httpx
from openai import AsyncOpenAI
from ragas import EvaluationDataset,SingleTurnSample
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import Faithfulness,ContextPrecision
from data import fixtures,deterministic,evaluation_dataset

BUDGET={'rows':4,'metrics':['faithfulness','context_precision'],'expected_http_calls':15,'max_http_calls':24,'temperature':0,'seed':42,'max_tokens':1024,'top_p':0.1,'reasoning_effort':'none','openai_retries':0,'instructor_max_attempts':1,'request_timeout_seconds':60,'metric_timeout_seconds':180,'repetitions':1}
def dump(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n');temp.replace(path)

class BudgetExceeded(RuntimeError):pass
class LocalOnlyTransport(httpx.AsyncBaseTransport):
    def __init__(self,out=None,limit=24,inner=None):
        self.inner=inner or httpx.AsyncHTTPTransport(retries=0);self.out=Path(out) if out else None;self.limit=limit;self.records=[];self.active={}
    def persist(self):
        if self.out:dump(self.out,self.records)
    async def handle_async_request(self,request):
        if request.url.host!='127.0.0.1' or request.url.port!=11434 or request.url.path!='/v1/chat/completions':raise RuntimeError('only local Ollama judge endpoint allowed')
        if len(self.records)>=self.limit:raise BudgetExceeded('judge HTTP call budget exhausted before transport')
        payload=json.loads(request.content);record={'sequence':len(self.records)+1,**self.active,'request':payload,'endpoint':str(request.url),'status':'started'}
        self.records.append(record);self.persist();started=time.perf_counter()
        try:
            response=await self.inner.handle_async_request(request);await response.aread()
            record.update(status='http_response',status_code=response.status_code)
            try:record['response']=response.json()
            except ValueError:record['raw_response']=response.text
            return response
        except BaseException as exc:record.update(status='transport_error',error_type=type(exc).__name__,error=str(exc));raise
        finally:record['wall_seconds']=time.perf_counter()-started;self.persist()
    async def aclose(self):await self.inner.aclose()

class RecordingLLM(InstructorBaseRagasLLM):
    def __init__(self,delegate,path=None):self.delegate=delegate;self.records=[];self.path=path;self.active={}
    def generate(self,prompt,response_model):raise RuntimeError('this PoC uses async evaluation')
    async def agenerate(self,prompt,response_model):
        r={'sequence':len(self.records)+1,**self.active,'response_model':response_model.__name__,'prompt':prompt};self.records.append(r);started=time.perf_counter()
        try:
            result=await self.delegate.agenerate(prompt,response_model);r.update(status='parsed',output=result.model_dump());return result
        except Exception as exc:r.update(status='structured_output_error',error_type=type(exc).__name__,error=str(exc));raise
        finally:
            r['wall_seconds']=time.perf_counter()-started
            if self.path:dump(self.path,self.records)

def metric_kwargs(name,sample):
    if name=='faithfulness':return dict(user_input=sample.user_input,response=sample.response,retrieved_contexts=sample.retrieved_contexts)
    return dict(user_input=sample.user_input,reference=sample.reference,retrieved_contexts=sample.retrieved_contexts)

async def score_one(metric,name,sample):
    started=time.perf_counter()
    try:
        score=await asyncio.wait_for(metric.ascore(**metric_kwargs(name,sample)),timeout=180)
        value=float(score.value)
        if not math.isfinite(value):return {'status':'nonfinite_metric','value':None,'raw_value':str(value),'wall_seconds':time.perf_counter()-started}
        return {'status':'scored','value':value,'wall_seconds':time.perf_counter()-started}
    except Exception as exc:return {'status':'evaluation_error','value':None,'error_type':type(exc).__name__,'error':str(exc),'wall_seconds':time.perf_counter()-started}

def local_metadata():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(path):
        with opener.open('http://127.0.0.1:11434/api/'+path,timeout=5) as f:return json.load(f)
    return {'version':get('version'),'model':next(m for m in get('tags')['models'] if m['name']=='qwen3:1.7b')}

async def run(out):
    out.mkdir(parents=True,exist_ok=False);rows=fixtures();dataset=evaluation_dataset(rows)
    dump(out/'synthetic_dataset.json',rows);dump(out/'evaluation_dataset.json',dataset.to_list());dump(out/'deterministic_results.json',[deterministic(r) for r in rows]);dump(out/'budget.json',BUDGET)
    env={'python':sys.version,'versions':{n:md.version(n) for n in ['ragas','langchain-community','openai','instructor','httpx']},'api_signatures':{getattr(x,'__name__',str(x)):str(inspect.signature(x)) for x in [EvaluationDataset.from_list,SingleTurnSample,llm_factory,Faithfulness,ContextPrecision,Faithfulness.ascore,ContextPrecision.ascore]},'dependency_resolution':'Ragas import failed with langchain-community 0.4.2 removed vertexai module; E06-only pin 0.3.31 imports successfully','metric_API':'ragas.metrics.collections .ascore -> MetricResult.value','factory':'Instructor JSON mode as implemented by Ragas; not legacy LangchainLLMWrapper','telemetry_disabled':True}
    dump(out/'environment.json',env)
    transport=LocalOnlyTransport(out/'model_calls.json',24);results=[];started=time.perf_counter()
    try:
        dump(out/'ollama.json',local_metadata())
        async with httpx.AsyncClient(transport=transport,timeout=60,trust_env=False) as http:
            async with AsyncOpenAI(base_url='http://127.0.0.1:11434/v1',api_key='ollama',max_retries=0,http_client=http) as client:
                judge=llm_factory('qwen3:1.7b',provider='openai',client=client,adapter='instructor',cache=None,temperature=0,top_p=.1,seed=42,max_tokens=1024,reasoning_effort='none',max_retries=1)
                recorded=RecordingLLM(judge,out/'structured_judge_calls.json');metrics={'faithfulness':Faithfulness(llm=recorded),'context_precision':ContextPrecision(llm=recorded)}
                for row,sample in zip(rows,dataset.samples):
                    for name,metric in metrics.items():
                        active={'case_id':row['id'],'metric':name};recorded.active=active;transport.active=active
                        before=len(transport.records);r=await score_one(metric,name,sample);r.update(**active,http_call_sequences=list(range(before+1,len(transport.records)+1)))
                        expected=deterministic(row)['supported_claim_fraction' if name=='faithfulness' else 'context_average_precision']
                        r.update(deterministic_comparison=expected,absolute_difference=abs(r['value']-expected) if r['value'] is not None else None)
                        results.append(r);dump(out/'ragas_results.json',results)
                        print(row['id'],name,r['status'],r['value'],flush=True)
                dump(out/'completion.json',{'status':'completed_with_failures' if any(r['status']!='scored' for r in results) else 'complete','metric_attempts':len(results),'successful_scores':sum(r['status']=='scored' for r in results),'model_http_calls':len(transport.records),'wall_seconds':time.perf_counter()-started,'answer_generation_calls':0,'retrieval_calls':0,'embedding_calls':0})
    except Exception as exc:
        dump(out/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),'model_http_calls':len(transport.records),'metric_attempts':len(results)})
    finally:transport.persist()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();asyncio.run(run(args.out))
