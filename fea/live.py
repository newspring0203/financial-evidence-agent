"""User-supplied DART key is used only during collection; never persisted."""
import hashlib
import json
from datetime import datetime, timezone
from .financials import extract_facts
from .dart_archive import OpenDartClient, document_text
from .domain import DomainError


def load_public(key, company_query, year, basis):
    if not 2016 <= year <= 2100 or basis not in ('CFS', 'OFS'):
        raise DomainError('비교 가능한 연도·연결구분을 지정하세요.')
    client = OpenDartClient(key)
    company = client.resolve_company(company_query)
    data = {'companies': [{'name': company['corp_name'], 'corp_code': company['corp_code']}],
            'facts': [], 'documents': [], 'warnings': [],
            'collected_at': datetime.now(timezone.utc).isoformat()}
    seen = set()
    for y in (year - 1, year):
        raw = client._get('fnlttSinglAcntAll.json', corp_code=company['corp_code'],
                          bsns_year=y, reprt_code='11011', fs_div=basis)
        payload = json.loads(raw)
        if payload.get('status') == '013':
            data['warnings'].append(f'{y}년 {basis} 연간 재무정보 미확보')
            continue
        rows = extract_facts(payload, company, y, basis, hashlib.sha256(raw).hexdigest())
        data['facts'].extend(rows)
        for fact in rows:
            receipt = fact['receipt']
            if receipt in seen:
                continue
            seen.add(receipt)
            archive = client._get('document.xml', rcept_no=receipt)
            text = document_text(archive)
            if len(text) > 12_000_000:
                raise DomainError('공시 원문이 세션 처리 한도를 초과했습니다.')
            data['documents'].append({'id': 'D-' + receipt, 'company': company['corp_code'],
                                      'year': y, 'basis': 'ALL', 'title': f"{company['corp_name']} {y} 사업보고서",
                                      'text': text, 'url': fact['url'], 'receipt': receipt,
                                      'archive_sha256': hashlib.sha256(archive).hexdigest(), 'synthetic': False})
    if not data['facts']:
        raise DomainError('선택한 회사·연도·연결구분의 재무 수치를 확보하지 못했습니다.')
    return data
