import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_gold import CATEGORIES, validate
from grade_e09 import evaluate

class TestE09(unittest.TestCase):
    def test_48_drafts_and_holdout(self):
        rows = [dict(self.make_gold(f'TEST-{category}-{i}'), category=category,
                     split='new_eval' if i < 6 else 'sealed_holdout')
                for category in sorted(CATEGORIES) for i in range(8)]
        self.assertEqual(len(rows),48)
        self.assertEqual(sum(r['split']=='new_eval' for r in rows),36)
        self.assertEqual(sum(r['split']=='sealed_holdout' for r in rows),12)
        self.assertEqual(len({r['question_id'] for r in rows}),48)
        self.assertEqual(validate(rows,Path('.'))[0],[])

    def make_gold(self, qid='E09-01'):
        quote='연결회사는 재고자산을 원가와 순실현가능가치 중 낮은 금액으로 측정합니다.'
        return {'question_id':qid,'category':'accounting_policy','split':'new_eval','qa_scope':'note_qa',
                'company':'가상기업','fiscal_year':'2024','basis':'CFS','filing_id':'TEST-001',
                'question':'재고자산 평가정책은?','expected_answer':'원가와 순실현가능가치 중 낮은 금액',
                'required_claims_json':'["저가법"]','answerability':'answerable','evidence_json':json.dumps([{
                    'chunk_id':'S1','source_id':'D1','start':0,'end':len(quote),'quote':quote,
                    'sha256':hashlib.sha256(quote.encode()).hexdigest()
                }],ensure_ascii=False), 'source_url':'https://example.com/fictional',
                'negative_scope_review':'','review_status':'verified','gold_verified_by':'test','verified_at':'2026-11-11','notes':'Synthetic test only'}

    def test_bad_hash_rejected(self):
        g=self.make_gold();e=json.loads(g['evidence_json']);e[0]['sha256']='bad';g['evidence_json']=json.dumps(e)
        err,_=validate([g],Path('.'),require_verified=True)
        self.assertTrue(any('SHA-256' in x for x in err))

    def test_metrics_with_independent_manual_grade(self):
        g=self.make_gold()
        attempts=[{'question_id':g['question_id'],'condition':'hybrid_explicit_cfs',
                   'status':'answered','retrieved_ids':['S1','S2','S3'],'citation_chunk_ids':['S1']}]
        review={g['question_id']:{'question_id':g['question_id'],'answer_correctness':'partial',
               'selection_relevance':'1','citation_support':'partial','reviewer':'test'}}
        result=evaluate([g],attempts,review)
        summary=result['groups']['all']
        self.assertEqual(summary['hit_at_3'],1.0)
        self.assertEqual(summary['complete_correct'],0)
        self.assertEqual(summary['partial'],1)
        self.assertEqual(summary['human_reviewed'],1)

    def test_unreviewed_excluded_from_accuracy(self):
        g=self.make_gold()
        attempts=[{'question_id':g['question_id'],'status':'answered','retrieved_ids':['S1']}]
        s=evaluate([g],attempts,{})['groups']['all']
        self.assertIsNone(s['full_correct_rate_reviewed'])
        self.assertEqual(s['unreviewed'],1)

    def test_holdout_not_scored_by_default(self):
        g=self.make_gold();g['split']='sealed_holdout'
        self.assertEqual(evaluate([g],[],{})['groups']['all']['n_gold'],0)

    def test_all_gold_requires_every_evidence_in_top_three(self):
        g = self.make_gold()
        evidence = json.loads(g['evidence_json'])
        evidence.append(dict(evidence[0], chunk_id='S4'))
        g['evidence_json'] = json.dumps(evidence)
        attempt = {'question_id': g['question_id'], 'status': 'answered',
                   'retrieved_ids': ['S1', 'S2', 'S3', 'S4']}
        summary = evaluate([g], [attempt], {})['groups']['all']
        self.assertEqual(summary['hit_at_3'], 1.0)
        self.assertEqual(summary['all_gold_hit_at_3'], 0.0)
        attempt['retrieved_ids'] = ['S4', 'S1', 'S2']
        summary = evaluate([g], [attempt], {})['groups']['all']
        self.assertEqual(summary['all_gold_hit_at_3'], 1.0)

    def test_pending_gold_and_regression_excluded(self):
        pending = dict(self.make_gold('TEST-pending'), review_status='pending')
        regression = dict(self.make_gold('TEST-regression'), split='regression')
        summary = evaluate([pending, regression], [], {})['groups']['all']
        self.assertEqual(summary['n_gold'], 0)
        summary = evaluate([pending, regression], [], {}, include_regression=True)['groups']['all']
        self.assertEqual(summary['n_gold'], 1)

if __name__=='__main__':unittest.main()
