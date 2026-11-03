"""Item-by-item inventory valuation with exact decimal arithmetic."""
from decimal import Decimal, InvalidOperation

def evaluate_inventory(items):
    if not 1 <= len(items) <= 30:
        raise ValueError('품목은 1~30개를 입력하세요.')
    rows=[]
    for item in items:
        values={}
        for key in ('quantity','cost','price','completion','selling'):
            try:
                value=Decimal(str(item[key]))
            except (InvalidOperation, KeyError, ValueError):
                raise ValueError('수량과 단위 금액을 숫자로 입력하세요.') from None
            if not value.is_finite() or value<0 or value>Decimal('1000000000000') or value.as_tuple().exponent < -2:
                raise ValueError('0 이상, 소수 둘째 자리 이하의 수량·금액을 입력하세요.')
            values[key]=value
        q,c,p,f,s=(values[k] for k in ('quantity','cost','price','completion','selling'))
        nrv=p-f-s
        if nrv<0:
            raise ValueError('판매가격보다 완성원가와 판매원가의 합계가 큰 품목은 별도 검토가 필요합니다. 이 계산 범위에 포함하지 않습니다.')
        amount=min(c,nrv)*q
        original=c*q
        rows.append({'name':str(item.get('name','품목'))[:80],**{k:str(v) for k,v in values.items()},
                     'nrv':str(nrv),'unit_value':str(min(c,nrv)), 'original':str(original),
                     'carrying':str(amount),'loss':str(original-amount)})
    return {'rows':rows,**{key:str(sum((Decimal(r[key]) for r in rows),Decimal(0))) for key in ('original','carrying','loss')},
            'basis':'품목별 원가와 순실현가능가치 비교',
            'source_url':'https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/'}
