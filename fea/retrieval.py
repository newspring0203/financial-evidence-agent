"""BM25 + Korean character bigrams, boundary-aware chunks and explicit context.
No paid model or embedding API. Source offsets always reference exact saved text.
"""
import bisect
import hashlib
import math
import re
from collections import Counter

STOP={'찾아줘','알려줘','확인','설명','관련','어떻게','무엇','정리','해주세요','보여줘','정도','있습니다'}


def tokenize(text):
    terms=[]
    for word in re.findall(r'[가-힣]+|[a-zA-Z]{2,}',text.lower()):
        if word in STOP:continue
        terms.append(word)
        if re.fullmatch('[가-힣]+',word) and len(word)>2:
            terms.extend('~'+word[i:i+2] for i in range(len(word)-1))
    return terms


def structured_chunks(doc):
    text=doc['text']
    digest=hashlib.sha256(text.encode()).hexdigest()
    headings=[]
    contexts=[]
    for m in re.finditer(r'(?m)^[^\n]+',text):
        line=m.group().strip()
        if len(line)<=90 and re.match(r'^(?:\d+(?:\.\d+)?[.．) ]|[IVXⅠⅡⅢⅣⅤ]+[.． ])',line) and len(re.findall('[가-힣A-Za-z]',line))>=3:
            headings.append((m.start(),line))
        context=re.fullmatch(r'(?:\d+[.．]\s*)?(연결)?재무제표\s*주석',line)
        if context:
            contexts.append((m.start(),'CFS' if context.group(1) else 'OFS'))
    heading_positions=[x[0] for x in headings]
    context_positions=[x[0] for x in contexts]
    start=0
    while start<len(text):
        end=min(len(text),start+1100)
        if end<len(text):
            boundary=text.rfind('\n',start+650,end)
            if boundary>start:end=boundary+1
        section_index=bisect.bisect_right(heading_positions,start)-1
        context_index=bisect.bisect_right(context_positions,start)-1
        quote=text[start:end]
        lines=[line.strip() for line in quote.splitlines() if line.strip()]
        yield {'id':f"{doc['id']}:{start}:{end}",'source_id':doc['id'],'start':start,'end':end,
               'quote':quote,'title':doc['title'],'url':doc.get('url'),'sha256':digest,
               'company':doc['company'],'year':doc['year'],'basis':doc['basis'],
               'synthetic':doc.get('synthetic',False),
               'section':headings[section_index][1] if section_index>=0 else '미식별',
               'basis_context':contexts[context_index][1] if context_index>=0 else 'unknown',
               '_numeric_share':sum(bool(re.fullmatch(r'[\d,().% −-]+',line)) for line in lines)/max(1,len(lines))}
        if end==len(text):break
        start=max(start+1,end-150)


def query_terms(query):
    inventory_policy=('재고' in query and any(w in query for w in ('정책','평가','저가법','측정','순실현')))
    if inventory_policy:
        # Query decomposition: financial comparisons are handled by numeric tools.
        query='재고자산 평가정책 원가 순실현가능가치 저가법'+(' 평가손실' if '손실' in query else '')
    extra=[]
    if '계열사' in query or '특수관계' in query:extra.extend(['특수관계자','관계기업','거래'])
    if '유동성' in query:extra.extend(['유동성위험','자금','조달','위험관리'])
    if '수익인식' in query or '수익 인식' in query:extra.extend(['수익인식','수익','통제','수행의무'])
    terms=Counter(tokenize(query+' '+' '.join(extra)))
    return {term:(0.25 if term.startswith('~') else 1.0) for term in terms},inventory_policy


def retrieve(documents, query, basis='CFS', k=3, *, structure=True, expansion=True, rerank=True, diversity=True):
    if structure:
        corpus=[c for doc in documents for c in structured_chunks(doc)]
    else:
        from .domain import chunks
        corpus=[c for doc in documents for c in chunks(doc)]
    if not corpus:return []
    counters=[Counter(tokenize(c['quote'])) for c in corpus]
    lengths=[sum(c.values()) for c in counters]
    avg=sum(lengths)/max(1,len(lengths))
    df=Counter(term for counter in counters for term in counter)
    terms,inventory_policy=query_terms(query)
    if not expansion:
        terms={term:(.25 if term.startswith('~') else 1.0) for term in set(tokenize(query))}
    candidates=[]
    n=len(corpus)
    for chunk,counts,length in zip(corpus,counters,lengths):
        score=0.0
        for term,weight in terms.items():
            tf=counts.get(term,0)
            if not tf:continue
            idf=math.log(1+(n-df[term]+0.5)/(df[term]+0.5))
            score+=weight*idf*(tf*2.5)/(tf+1.5*(0.25+0.75*length/max(1,avg)))
        if score<=0:continue
        text=chunk['quote']
        if rerank and inventory_policy:
            if '재고' not in text:continue
            if '순실현가능가치' in text and ('낮은 금액' in text or '저가법' in text):score*=1.8
            if chunk.get('_numeric_share',0)>.35:score*=.4
        if rerank and ('수익인식' in query or '수익 인식' in query) and any(w in text for w in ('반품','수행의무','통제가 이전')):score*=1.25
        if rerank and ('특수관계' in query or '계열사' in query):
            if re.search(r'특수관계자.{0,45}(거래|내역)',text):score*=1.25
        # This is a weak context hint, never proof of statement scope.
        if rerank and chunk.get('basis_context')==basis:score*=1.03
        elif rerank and chunk.get('basis_context','unknown')!='unknown':score*=.97
        candidates.append({**{key:value for key,value in chunk.items() if not key.startswith('_')},
                           'score':round(score,4),'retrieval_method':'bm25_structure'})
    candidates.sort(key=lambda c:(-c['score'],c['id']))
    if not diversity:return candidates[:k]
    selected=[]
    for candidate in candidates:
        duplicate=False
        for chosen in selected:
            if candidate['source_id']!=chosen['source_id']:continue
            overlap=max(0,min(candidate['end'],chosen['end'])-max(candidate['start'],chosen['start']))
            if overlap/min(candidate['end']-candidate['start'],chosen['end']-chosen['start'])>.45:
                duplicate=True
                break
            if re.sub(r'\s+','',candidate['quote'])[:250]==re.sub(r'\s+','',chosen['quote'])[:250]:
                duplicate=True
                break
        if not duplicate:selected.append(candidate)
        if len(selected)==k:break
    return selected
