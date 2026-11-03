"""Private in-session text documents. Never forwarded to external LLMs."""
import hashlib
import math
import re
from datetime import date
from .domain import DomainError, search_terms

MAX_BYTES=1_000_000


def parse_document(filename, raw, title, version='', effective_date=''):
    suffix=filename.rsplit('.',1)[-1].lower() if '.' in filename else ''
    if suffix not in ('txt','md'):
        raise DomainError('첫 버전은 UTF-8 TXT·Markdown 파일을 지원합니다.')
    if not raw or len(raw)>MAX_BYTES:
        raise DomainError('문서는 1바이트~1MB 범위여야 합니다.')
    if effective_date:
        try:date.fromisoformat(effective_date)
        except ValueError:raise DomainError('적용일은 YYYY-MM-DD 형식이어야 합니다.') from None
    try:text=raw.decode('utf-8-sig')
    except UnicodeError:raise DomainError('UTF-8 인코딩 문서를 사용하세요.') from None
    if '\x00' in text or not text.strip():
        raise DomainError('텍스트가 비어 있거나 지원하지 않는 문서입니다.')
    digest=hashlib.sha256(text.encode()).hexdigest()
    return {'id':'K-'+digest[:20], 'title':title.strip()[:150] or filename[:150],
            'filename':filename[:150], 'version':version.strip()[:80],
            'effective_date':effective_date, 'sha256':digest, 'text':text,
            'source_type':'user_document', 'external_llm_transmission':False}


def retrieve(documents, query):
    if not 1<=len(query.strip())<=1000:
        raise DomainError('검색 질문은 1~1000자여야 합니다.')
    terms=search_terms(query)
    # Common question suffixes should not retrieve unrelated internal guidance.
    terms-={'찾아줘','알려줘','무엇인가요','어떻게','내용','관련','문서','설명','해주세요','확인'}
    ranked=[]
    for doc in documents:
        text=doc['text']
        for start in range(0,len(text),1050):
            end=min(len(text),start+1200)
            quote=text[start:end]
            score=sum(1+math.log(1+quote.lower().count(t)) for t in terms if t in quote.lower())
            if score>0:
                line=text.count('\n',0,start)+1
                end_line=text.count('\n',0,end)+1
                ranked.append({'id':f"{doc['id']}:{start}:{end}", 'document_id':doc['id'],
                               'title':doc['title'],'filename':doc['filename'],'version':doc['version'],
                               'effective_date':doc['effective_date'], 'start':start,'end':end,
                               'start_line':line,'end_line':end_line,'quote':quote,'sha256':doc['sha256'],
                               'score':round(score,3),'source_type':'user_document','account_code':doc.get('account_code','COMMON'),
                               'verified':text[start:end]==quote and hashlib.sha256(text.encode()).hexdigest()==doc['sha256']})
            if end==len(text):break
    return sorted(ranked,key=lambda r:(-r['score'],r['id']))[:5]


def public_metadata(doc):
    return {k:v for k,v in doc.items() if k!='text'}
