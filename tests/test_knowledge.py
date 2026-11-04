import pytest
from fastapi.testclient import TestClient
from fea.domain import DomainError, Tools
from fea.knowledge import parse_document, retrieve
from fea.server import app


def test_document_metadata_offsets_unicode_and_integrity():
    text='# 내부 검토 예시\n재고자산 평가 검토 시 장기체화 내역과 판매실적을 확인한다.\n'
    doc=parse_document('manual.md',text.encode(),'검토 예시','v1','2026-01-01')
    found=retrieve([doc],'재고 평가 검토')
    assert found and found[0]['quote']==text[found[0]['start']:found[0]['end']]
    assert found[0]['verified'] and found[0]['effective_date']=='2026-01-01'
    assert doc['external_llm_transmission'] is False
    assert not retrieve([doc],'초전도체')


@pytest.mark.parametrize('name,raw,date',[('a.pdf',b'test',''),('a.md',b'\xff',''),('a.txt',b'\x00',''),('a.md',b'x','not-a-date')])
def test_invalid_private_documents_rejected(name,raw,date):
    with pytest.raises(DomainError):parse_document(name,raw,'test',effective_date=date)


def test_private_documents_are_session_scoped_and_removed(monkeypatch):
    import fea.server as server
    data=Tools('demo').data
    data.update(collected_at='2026-11-03T00:00:00Z',warnings=[])
    monkeypatch.setattr(server,'load_public',lambda *args:data)
    with TestClient(app) as owner,TestClient(app) as outsider:
        assert outsider.get('/api/knowledge').status_code==401
        assert owner.post('/api/connect',json={'dart_key':'A'*40,'company':'DEMO0001','year':2024,'basis':'CFS'}).status_code==200
        text='재고자산 평가 검토: 장기체화와 판매실적을 확인한다.'
        response=owner.post('/api/knowledge',files={'file':('manual.md',text.encode(),'text/markdown')},data={'title':'내부 예시','version':'v1','effective_date':'2026-01-01'})
        assert response.status_code==200
        doc=response.json()
        assert 'text' not in doc
        assert outsider.post('/api/knowledge/search',json={'question':'재고'}).status_code==401
        found=owner.post('/api/knowledge/search',json={'question':'재고 평가'}).json()
        assert found['results'][0]['quote']==text
        assert found['external_llm_transmission'] is False
        assert owner.post('/api/knowledge',files={'file':('same.txt',text.encode())}).status_code==400
        owner.delete('/api/knowledge/'+doc['id'])
        assert owner.get('/api/knowledge').json()['documents']==[]
        assert owner.post('/api/knowledge/search',json={'question':'재고'}).status_code==400
        owner.delete('/api/session')
