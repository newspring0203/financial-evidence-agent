import os
os.environ['RAGAS_DO_NOT_TRACK']='true'
os.environ['HF_HUB_OFFLINE']='1'
import asyncio,unittest,math
import httpx
from ragas import SingleTurnSample
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import Faithfulness,ContextPrecision
from data import fixtures,deterministic,evaluation_dataset,average_precision
from evaluate import LocalOnlyTransport,BudgetExceeded,score_one,metric_kwargs

class ScriptedJudge(InstructorBaseRagasLLM):
    def __init__(self,outputs):self.outputs=iter(outputs);self.calls=0
    def generate(self,prompt,response_model):raise RuntimeError('async only')
    async def agenerate(self,prompt,response_model):
        self.calls+=1;return response_model.model_validate(next(self.outputs))

class Tests(unittest.TestCase):
    def test_dataset_roundtrip_and_metadata_exclusion(self):
        rows=fixtures();ds=evaluation_dataset(rows);self.assertEqual(len(ds.samples),4)
        self.assertTrue(all(isinstance(s,SingleTurnSample) for s in ds.samples))
        self.assertEqual(ds.to_list()[0]['response'],rows[0]['response'])
        self.assertNotIn('claims',ds.to_list()[0]);self.assertNotIn('context_fact_keys',ds.to_list()[0])
        self.assertEqual(rows,fixtures())
    def test_deterministic_cases(self):
        checks={r['id']:deterministic(r) for r in fixtures()}
        self.assertTrue(checks['correct']['task_correct']);self.assertEqual(checks['correct']['supported_claim_fraction'],1)
        self.assertTrue(checks['unsupported']['task_correct']);self.assertEqual(checks['unsupported']['supported_claim_fraction'],0)
        self.assertFalse(checks['unrelated']['task_correct']);self.assertEqual(checks['unrelated']['supported_claim_fraction'],1)
        self.assertFalse(checks['partly_wrong']['task_correct']);self.assertEqual(checks['partly_wrong']['supported_claim_fraction'],.5)
        self.assertEqual(checks['partly_wrong']['context_average_precision'],.5)
    def test_context_ap_order(self):
        self.assertEqual(average_precision([True,False]),1);self.assertEqual(average_precision([False,True]),.5);self.assertEqual(average_precision([False,False]),0)
    def test_real_faithfulness_code_with_scripted_judge(self):
        fake=ScriptedJudge([{'statements':['Revenue is 1500.','Currency is KRW.']},{'statements':[{'statement':'Revenue is 1500.','reason':'wrong amount','verdict':0},{'statement':'Currency is KRW.','reason':'supported','verdict':1}]}])
        sample=evaluation_dataset(fixtures()).samples[3]
        result=asyncio.run(score_one(Faithfulness(llm=fake),'faithfulness',sample))
        self.assertEqual(result['value'],.5);self.assertEqual(fake.calls,2)
    def test_real_context_precision_code_with_scripted_judge(self):
        fake=ScriptedJudge([{'reason':'irrelevant','verdict':0},{'reason':'relevant','verdict':1}])
        result=asyncio.run(score_one(ContextPrecision(llm=fake),'context_precision',evaluation_dataset(fixtures()).samples[3]))
        self.assertAlmostEqual(result['value'],.5);self.assertEqual(fake.calls,2)
    def test_nonfinite_is_not_zero_or_success(self):
        fake=ScriptedJudge([{'statements':[]}])
        result=asyncio.run(score_one(Faithfulness(llm=fake),'faithfulness',evaluation_dataset(fixtures()).samples[0]))
        self.assertEqual(result['status'],'nonfinite_metric');self.assertIsNone(result['value'])
    def test_parse_error_preserved(self):
        fake=ScriptedJudge([{'bad':'schema'}]);result=asyncio.run(score_one(Faithfulness(llm=fake),'faithfulness',evaluation_dataset(fixtures()).samples[0]))
        self.assertEqual(result['status'],'evaluation_error');self.assertIsNone(result['value']);self.assertEqual(result['error_type'],'ValidationError')
    def test_http_budget_and_external_block(self):
        async def scenario():
            invoked=[]
            async def backend(req):invoked.append(1);return httpx.Response(200,json={'usage':{'prompt_tokens':1,'completion_tokens':2}})
            tr=LocalOnlyTransport(limit=1,inner=httpx.MockTransport(backend))
            async with httpx.AsyncClient(transport=tr) as client:
                await client.post('http://127.0.0.1:11434/v1/chat/completions',json={})
                with self.assertRaises(BudgetExceeded):await client.post('http://127.0.0.1:11434/v1/chat/completions',json={})
                with self.assertRaises(RuntimeError):await client.post('https://example.com/v1/chat/completions',json={})
            self.assertEqual(len(invoked),1);self.assertEqual(len(tr.records),1)
        asyncio.run(scenario())
    def test_context_precision_does_not_use_response(self):
        sample=evaluation_dataset(fixtures()).samples[2];args=metric_kwargs('context_precision',sample)
        self.assertNotIn('response',args);self.assertEqual(args['reference'],sample.reference)
if __name__=='__main__':unittest.main()
