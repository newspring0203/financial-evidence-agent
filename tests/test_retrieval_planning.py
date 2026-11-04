import copy
import pytest
from fea.domain import Tools,DomainError
from fea.planning import make_plan
from fea.retrieval import structured_chunks,retrieve


def test_structured_chunks_keep_exact_offsets_and_cover_tail():
    text='3. 연결재무제표 주석\n2.8 재고자산\n'+('재고자산은 원가와 순실현가능가치 중 낮은 금액으로 측정합니다.\n'*60)
    doc={'id':'D-test','text':text,'title':'공시','url':'https://dart.fss.or.kr/','company':'00000001','year':2025,'basis':'ALL'}
    parts=list(structured_chunks(doc))
    assert parts[-1]['end']==len(text)
    assert all(text[p['start']:p['end']]==p['quote'] for p in parts)
    assert any(p['basis_context']=='CFS' for p in parts)
    result=retrieve([doc],'재고 평가정책','CFS')
    assert result and all(r['retrieval_method']=='bm25_structure' for r in result)
    assert len(result)<=3


def test_bm25_unknown_terms_do_not_return_random_text():
    tools=Tools('demo','bm25')
    assert not tools.search_disclosures('DEMO0001',2024,'CFS','초전도체양자통신')
    found=tools.search_disclosures('DEMO0001',2024,'CFS','재고자산 평가정책')
    assert found and all(c['valid'] for c in tools.verify(found,found))


def test_prior_report_tieout_detects_difference_without_claiming_error():
    tools=Tools('demo')
    current=tools.get_financials('DEMO0001',2024,'CFS','revenue')
    prior=tools.get_financials('DEMO0001',2023,'CFS','revenue')
    with pytest.raises(DomainError):tools.reconcile_prior(current['id'],prior['id'])
    for row in tools.data['facts']:
        if row['id']==current['id']:row['comparative_amount']='99999999999'
    tie=tools.reconcile_prior(current['id'],prior['id'])
    assert tie['difference']=='-1' and tie['status']=='difference_requires_review'
    assert '오류로 단정하지' in tie['note']
    assert tools.verify([tie],[tie])[0]['valid']


def test_review_plan_uses_actual_facts_and_keeps_request_a_draft():
    tools=Tools('demo')
    plan=make_plan(tools.data,'DEMO0001',2024,'CFS','inventory','2026-11-10')
    assert plan['delta']=='3000000000'
    assert len(plan['requests'])==3
    assert '15,000,000,000' in plan['draft']
    assert '메일을 발송하지' in plan['draft']
    assert plan['facts'][0]['id'].startswith('F-DEMO')
    with pytest.raises(DomainError):make_plan(tools.data,'DEMO0001',2024,'CFS','unsupported')


def test_explicit_prior_year_question_uses_that_year_not_selected_current_year():
    from fea.agent import answer
    r=answer('2023년 매출액 금액은 얼마인가요?','DEMO0001',2024,source='demo')
    assert any('100,000,000,000' in c['text'] for c in r['claims'])
    assert not any('125,000,000,000' in c['text'] for c in r['claims'])


def test_web_footing_and_planner_use_session_facts(monkeypatch):
    from fastapi.testclient import TestClient
    import fea.server as server
    data=Tools('demo').data
    data.update(collected_at='2026-11-03T00:00:00Z',warnings=[])
    prior={f['metric']:f for f in data['facts'] if f['company']=='DEMO0001' and f['year']==2023 and f['basis']=='CFS'}
    for f in data['facts']:
        if f['company']=='DEMO0001' and f['year']==2024 and f['basis']=='CFS':
            f['comparative_amount']=prior[f['metric']]['amount']
    monkeypatch.setattr(server,'load_public',lambda *args:data)
    with TestClient(server.app) as client:
        assert client.get('/api/footing').status_code==401
        client.post('/api/connect',json={'dart_key':'A'*40,'company':'DEMO0001','year':2024,'basis':'CFS'})
        r=client.get('/api/footing')
        assert r.status_code==200
        assert all(row['result']['status']=='matched' for row in r.json()['rows'])
        plan=client.post('/api/review-plan',json={'area':'inventory'}).json()
        assert plan['delta']=='3000000000'
        client.delete('/api/session')
