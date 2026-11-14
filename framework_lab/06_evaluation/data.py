"""Four synthetic RAG examples, rendered from explicit fictional facts."""
import hashlib,json
from ragas import EvaluationDataset

FACTS={'revenue':1200,'currency':'KRW','office_color':'blue'}
TEXT={'revenue':'Fictional Aster reported 2024 revenue of 1200 million KRW.', 'currency':'The reporting currency of Fictional Aster is KRW.', 'office_color':'The office color of Fictional Aster is blue.'}

def fixtures():
    question='What was Fictional Aster revenue in 2024, in million KRW?'
    reference=TEXT['revenue']
    definitions=[
        ('correct','correct_answer',[['revenue','currency'],['office_color']],[('revenue',1200),('currency','KRW')]),
        ('unsupported','answer_without_evidence',[['office_color']],[('revenue',1200)]),
        ('unrelated','question_irrelevant_answer',[['revenue','currency'],['office_color']],[('office_color','blue')]),
        ('partly_wrong','partially_incorrect_answer',[['office_color'],['revenue','currency']],[('revenue',1500),('currency','KRW')]),
    ]
    rows=[]
    for name,kind,context_keys,claims in definitions:
        response=' '.join(TEXT[k] if FACTS[k]==v else f'Fictional Aster reported 2024 revenue of {v} million KRW.' for k,v in claims)
        row={'id':name,'case':kind,'user_input':question,'response':response,'reference':reference,'retrieved_contexts':[' '.join(TEXT[k] for k in keys) for keys in context_keys],'context_fact_keys':context_keys,'claims':[{'key':k,'value':v} for k,v in claims]}
        row['sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest();rows.append(row)
    return rows

def average_precision(relevant):
    n=sum(relevant)
    if not n:return 0.
    return sum(sum(relevant[:i])/i for i,flag in enumerate(relevant,1) if flag)/n

def deterministic(row):
    available={k for group in row['context_fact_keys'] for k in group}
    checks=[{'key':c['key'],'value_correct':FACTS.get(c['key'])==c['value'],'context_support':c['key'] in available and FACTS.get(c['key'])==c['value']} for c in row['claims']]
    relevance=any(c['key']=='revenue' for c in row['claims'])
    correct=any(c['key']=='revenue' and c['value_correct'] for c in checks)
    return {'id':row['id'],'claim_checks':checks,'all_values_correct':all(c['value_correct'] for c in checks),'answers_requested_fact':relevance,'task_correct':relevance and correct and all(c['value_correct'] for c in checks),'supported_claim_fraction':sum(c['context_support'] for c in checks)/len(checks),'context_relevance_flags':['revenue' in group for group in row['context_fact_keys']],'context_average_precision':average_precision(['revenue' in group for group in row['context_fact_keys']])}

def evaluation_dataset(rows):
    # Keep oracle/metadata out of the evaluation model's inputs.
    return EvaluationDataset.from_list([{k:r[k] for k in ['user_input','retrieved_contexts','response','reference']} for r in rows])
