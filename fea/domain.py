"""Shared tool implementation for web, function calling and MCP."""
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


class DomainError(ValueError):
    pass


def dataset(source='demo'):
    if source not in ('demo', 'dart'):
        raise DomainError('지원하지 않는 데이터 모드입니다.')
    path = DATA / ('demo.json' if source == 'demo' else 'dart.json')
    if not path.exists():
        raise DomainError('OpenDART 데이터를 먼저 수집하세요. README의 수집 명령을 확인하세요.')
    return json.loads(path.read_text())


def scope(data, company, year, basis):
    if basis not in ('CFS', 'OFS') or not isinstance(year, int) or isinstance(year, bool) or not 2015 <= year <= 2100:
        raise DomainError('연도 또는 연결·별도 조건이 올바르지 않습니다.')
    companies = [c for c in data['companies'] if company in (c['name'], c['corp_code'])]
    if len(companies) != 1:
        raise DomainError('회사를 정확하게 지정하세요. 지원 회사 목록을 확인하세요.')
    return companies[0]


def search_terms(query, expanded=True):
    terms = set(re.findall(r'[가-힣A-Za-z0-9]+', query.lower()))
    if expanded:
        for topic, words in {
            '재고': ['재고', '재고자산', '평가손실', '순실현가능가치'],
            '매출': ['매출', '수익', '수익인식', '고객'],
            '차입': ['차입', '차입금', '유동성', '만기'],
            '관계': ['특수관계자', '관계자'],
            '소송': ['소송', '우발부채', '약정'],
        }.items():
            if topic in query or any(w in query for w in words):
                terms.update(words)
    return terms


def chunks(doc, size=700):
    text = doc['text']
    digest = hashlib.sha256(text.encode()).hexdigest()
    # Stable offsets into the exact saved extracted text, with small overlap.
    for start in range(0, len(text), size - 100):
        end = min(len(text), start + size)
        yield {'id': f"{doc['id']}:{start}:{end}", 'source_id': doc['id'],
               'start': start, 'end': end, 'quote': text[start:end],
               'title': doc['title'], 'url': doc.get('url'),
               'sha256': digest,
               'company': doc['company'], 'year': doc['year'], 'basis': doc['basis'],
               'synthetic': doc.get('synthetic', False)}
        if end == len(text):
            break


class Tools:
    def __init__(self, source='demo', strategy='expanded', data=None):
        self.source = source
        self.data = data if data is not None else dataset(source)
        if strategy not in ('keyword', 'expanded', 'bm25'):
            raise DomainError('지원하지 않는 검색 방식입니다.')
        self.strategy = strategy

    def company(self, company, year, basis):
        return scope(self.data, company, year, basis)

    def get_financials(self, company: str, year: int, basis: str, metric: str):
        c = self.company(company, year, basis)
        if metric not in ('revenue', 'operating_profit', 'inventory'):
            raise DomainError('지원 지표: revenue, operating_profit, inventory')
        rows = [r for r in self.data['facts'] if r['company'] == c['corp_code'] and
                r['year'] == year and r['basis'] == basis and r['metric'] == metric]
        if len(rows) != 1:
            raise DomainError('해당 조건의 수치를 유일하게 확인할 수 없습니다. 추정하지 않습니다.')
        return dict(rows[0])

    def calculate_change(self, current_id: str, prior_id: str):
        facts = {r['id']: r for r in self.data['facts']}
        if current_id not in facts or prior_id not in facts:
            raise DomainError('조회된 재무 수치 ID를 사용하세요.')
        a, b = facts[current_id], facts[prior_id]
        if any(a[k] != b[k] for k in ('company', 'basis', 'metric', 'currency', 'unit', 'report_code')) or a['year'] != b['year'] + 1:
            raise DomainError('동일 회사·지표·단위·연결구분의 연속 연간 수치만 비교합니다.')
        current, prior = Decimal(a['amount']), Decimal(b['amount'])
        if prior == 0:
            raise DomainError('전기 금액이 0이므로 증감률을 계산하지 않습니다.')
        value = ((current - prior) / abs(prior) * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return {'id': f"calc:{a['id']}:{b['id']}", 'kind': 'calculation',
                'current_id': current_id, 'prior_id': prior_id, 'value': str(value), 'unit': '%',
                'formula': '(current - prior) / abs(prior) × 100', 'rounding': '0.01, ROUND_HALF_UP'}

    def reconcile_prior(self, current_id: str, prior_id: str):
        facts = {r['id']: r for r in self.data['facts']}
        if current_id not in facts or prior_id not in facts:
            raise DomainError('공시 수치 ID가 필요합니다.')
        a, b = facts[current_id], facts[prior_id]
        if any(a[k] != b[k] for k in ('company','basis','metric','currency','unit','report_code')) or a['year'] != b['year'] + 1:
            raise DomainError('같은 회사·연결구분·지표의 연속 연간 공시만 대사합니다.')
        comparative = a.get('comparative_amount')
        if comparative is None:
            raise DomainError('당기 보고서의 전기 비교금액을 확보하지 못했습니다.')
        difference = Decimal(comparative) - Decimal(b['amount'])
        return {'id': f"tieout:{current_id}:{prior_id}", 'kind':'tieout',
                'current_id': current_id, 'prior_id': prior_id,
                'comparative_amount': comparative, 'prior_report_amount': b['amount'],
                'difference': str(difference), 'currency': a['currency'], 'unit': a['unit'],
                'status': 'matched' if difference == 0 else 'difference_requires_review',
                'formula':'당기 보고서의 전기 비교금액 - 전기 보고서의 당기금액',
                'note':'차이는 재작성·표시 변경·정정·조회 버전 차이일 수 있습니다. 오류로 단정하지 않습니다.'}

    def search_disclosures(self, company: str, year: int, basis: str, query: str):
        c = self.company(company, year, basis)
        if not query.strip() or len(query) > 1000:
            raise DomainError('검색어는 1~1000자여야 합니다.')
        if self.strategy == 'bm25':
            from .retrieval import retrieve
            docs = [d for d in self.data['documents'] if (d['company'],d['year']) == (c['corp_code'],year) and d['basis'] in (basis,'ALL')]
            return retrieve(docs, query, basis)
        terms = search_terms(query, self.strategy == 'expanded')
        policy = any(w in query for w in ('평가정책', '순실현가능', '평가방법', '수익인식', '수익 인식'))
        inventory_policy = policy and ('재고' in query or '순실현' in query)
        ranked = []
        for doc in self.data['documents']:
            if (doc['company'], doc['year']) != (c['corp_code'], year) or doc['basis'] not in (basis, 'ALL'):
                continue
            for chunk in chunks(doc):
                text = chunk['quote'].lower()
                score = sum((1 + math.log(1 + text.count(t))) for t in terms if t in text)
                if self.strategy == 'expanded':
                    # Prefer explanatory prose over repeated numeric table labels for policy questions.
                    if inventory_policy:
                        score += 9 * sum(phrase in text for phrase in ('순실현가능가치', '낮은 금액', '가중평균', '평균법'))
                    if '수익인식' in query or '수익 인식' in query:
                        score += 7 * sum(phrase in text for phrase in ('통제가 이전', '수행의무', '반품'))
                    if '유동성' in query:
                        score += 7 * sum(phrase in text for phrase in ('유동성위험은', '유동성 위험은', '유동성위험 관리'))
                    lines = [line for line in text.splitlines() if line.strip()]
                    numeric_share = sum(bool(re.fullmatch(r'[\d,().% −-]+', line.strip())) for line in lines) / max(1, len(lines))
                    if policy and numeric_share > 0.35:
                        score *= 0.35
                if score > 0:
                    ranked.append({**chunk, 'score': round(score, 4)})
        return sorted(ranked, key=lambda r: (-r['score'], r['id']))[:3]

    def call(self, name, args):
        if name not in ('get_financials', 'calculate_change', 'search_disclosures', 'reconcile_prior'):
            raise DomainError('허용되지 않은 도구입니다.')
        try:
            return getattr(self, name)(**args)
        except (TypeError, InvalidOperation, KeyError):
            raise DomainError('도구 인자를 확인하세요.') from None

    def verify(self, evidence, retrieved):
        """Identity checks only. Does NOT prove semantic entailment."""
        known = {r['id']: r for r in retrieved}
        docs = {d['id']: d for d in self.data['documents']}
        facts = {f['id']: f for f in self.data['facts']}
        checks = []
        for item in evidence:
            eid = item.get('id', '')
            ref = known.get(eid)
            valid = ref is not None and item == ref
            kind = 'fact'
            if valid and 'quote' in item:
                kind = 'passage'
                doc = docs.get(item['source_id'])
                valid = bool(doc and doc['text'][item['start']:item['end']] == item['quote']
                             and hashlib.sha256(doc['text'].encode()).hexdigest() == item['sha256'])
            elif valid and item.get('kind') == 'tieout':
                kind = 'tieout'
                valid = item == self.reconcile_prior(item['current_id'], item['prior_id'])
            elif valid and item.get('kind') == 'calculation':
                kind = 'calculation'
                valid = item == self.calculate_change(item['current_id'], item['prior_id'])
            elif valid:
                valid = item == facts.get(eid)
            checks.append({'id': eid, 'kind': kind, 'valid': bool(valid)})
        return checks


TOOL_DEFINITIONS = [
    {'type': 'function', 'name': 'get_financials', 'description': '명시된 회사·연도·연결/별도의 연간 재무 수치 조회. 금액은 원래 통화·단위를 유지한다.',
     'strict': True, 'parameters': {'type': 'object', 'properties': {
         'company': {'type': 'string'}, 'year': {'type': 'integer'},
         'basis': {'type': 'string', 'enum': ['CFS', 'OFS']},
         'metric': {'type': 'string', 'enum': ['revenue', 'operating_profit', 'inventory']}},
         'required': ['company', 'year', 'basis', 'metric'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'calculate_change', 'description': 'get_financials에서 조회한 현재·전기 수치 ID로 증감률을 계산한다.',
     'strict': True, 'parameters': {'type': 'object', 'properties': {
         'current_id': {'type': 'string'}, 'prior_id': {'type': 'string'}},
         'required': ['current_id', 'prior_id'], 'additionalProperties': False}},
    {'type': 'function', 'name': 'search_disclosures', 'description': '선택한 회사·연도·연결/별도의 공시 본문 검색. 문단 ID·추출 텍스트 위치·원문 반환.',
     'strict': True, 'parameters': {'type': 'object', 'properties': {
         'company': {'type': 'string'}, 'year': {'type': 'integer'},
         'basis': {'type': 'string', 'enum': ['CFS', 'OFS']}, 'query': {'type': 'string'}},
         'required': ['company', 'year', 'basis', 'query'], 'additionalProperties': False}},
]

TOOL_DEFINITIONS.append({
    'type':'function','name':'reconcile_prior',
    'description':'당기 보고서의 전기 비교금액과 전기 보고서의 당기금액을 대사한다. 차이는 정정·재작성 여부를 검토해야 하며 오류로 단정하지 않는다.',
    'strict':True,'parameters':{'type':'object','properties':{'current_id':{'type':'string'},'prior_id':{'type':'string'}},
        'required':['current_id','prior_id'],'additionalProperties':False}
})
