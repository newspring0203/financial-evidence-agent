"""Normalize OpenDART annual statement facts with explicit source identity."""
import re
from decimal import Decimal, InvalidOperation
from .domain import DomainError
from .dart_archive import check_status

METRICS = {
    'revenue': (('ifrs-full_Revenue', 'ifrs_Revenue'), ('매출액', '수익(매출액)', '매출', '영업수익')),
    'operating_profit': (('dart_OperatingIncomeLoss',), ('영업이익', '영업이익(손실)')),
    'inventory': (('ifrs-full_Inventories', 'ifrs_Inventories'), ('재고자산',)),
}


def extract_facts(payload, company, year, basis, raw_hash):
    check_status(payload)
    result = []
    for metric, (ids, names) in METRICS.items():
        statement = ('BS',) if metric == 'inventory' else ('IS', 'CIS')
        candidates = [r for r in payload['list'] if r.get('account_id') in ids and r.get('sj_div') in statement]
        if not candidates:
            candidates = [r for r in payload['list'] if r.get('account_nm') in names and r.get('sj_div') in statement]
        # IS and CIS duplicates may be identical. Reject differing amounts/receipts.
        signatures = {(r.get('thstrm_amount'), r.get('rcept_no'), r.get('currency', 'KRW')) for r in candidates}
        if len(signatures) != 1:
            continue
        row = candidates[0]
        if row.get('corp_code') != company['corp_code'] or row.get('bsns_year') != str(year) or row.get('reprt_code') != '11011':
            raise DomainError('재무 응답의 회사·사업연도·보고서가 요청과 다릅니다.')
        receipt = row.get('rcept_no', '')
        if not re.fullmatch(r'\d{14}', receipt):
            raise DomainError('공시 접수번호가 올바르지 않습니다.')
        try:
            amount = Decimal(row['thstrm_amount'].replace(',', ''))
            if not amount.is_finite():
                raise InvalidOperation
        except (InvalidOperation, KeyError):
            continue
        try:
            comparative = Decimal((row.get('frmtrm_amount') or '').replace(',',''))
            comparative = str(comparative) if comparative.is_finite() else None
        except InvalidOperation:
            comparative = None
        result.append({'id': f"F-{company['corp_code']}-{year}-{basis}-{metric}-{receipt}",
                       'company': company['corp_code'], 'year': year, 'basis': basis,
                       'metric': metric, 'amount': str(amount), 'currency': row.get('currency', 'KRW'),
                       'unit': '원', 'report_code': '11011', 'period_label': row.get('thstrm_dt', ''),
                       'comparative_amount': comparative, 'comparative_year': year - 1, 'comparative_period_label': row.get('frmtrm_dt',''),
                       'account_id': row.get('account_id'), 'account_name': row.get('account_nm'),
                       'receipt': receipt, 'url': f'https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}',
                       'response_sha256': raw_hash, 'source': 'OpenDART 단일회사 전체 재무제표', 'synthetic': False})
    return result

