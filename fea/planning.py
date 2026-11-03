"""Author-created review prompts, not accounting rules or audit conclusions."""
from .domain import DomainError, Tools

TEMPLATES={
 'inventory':{'label':'재고자산','objective':'재고 증감과 평가에 대한 설명을 필요한 증빙과 연결합니다.',
  'requests':[
   {'item':'기말 재고 명세 및 총계정원장 대사표','purpose':'공시 잔액과 명세의 합계·범위를 확인','format':'품목·수량·단가·금액·사업장·기준일 포함'},
   {'item':'재고 연령분석 및 장기 미이동 재고 내역','purpose':'체화·진부화 가능성을 추가 검토','format':'입출고일·연령구간·담당자 설명 포함'},
   {'item':'평가손실 산정 내역 및 근거자료','purpose':'판단에 사용한 가격·원가·처분 가정을 확인','format':'품목별 계산식·가정·증빙 참조 포함'}],
  'questions':['판매수량·제품구성·생산계획 변화가 재고 증감액을 각각 얼마나 설명하나요?',
               '장기체화 내역과 평가손실 산정 범위가 일치하나요?',
               '관련 판단에 사용한 가격·판매실적의 기준일과 출처는 무엇인가요?'],
  'search_query':'재고자산 평가정책과 평가손실'},
 'revenue':{'label':'매출','objective':'매출 변동의 설명과 정량적 기여를 구분해 확인합니다.',
  'requests':[
   {'item':'제품·지역·고객별 전기 및 당기 매출 집계','purpose':'변동을 구성요소별로 분해','format':'비교 기준·통화·상계 및 연결조정 여부 포함'},
   {'item':'판매수량·단가 및 주요 계약조건 변화 내역','purpose':'금액 변동과 영업활동을 연결','format':'산출근거·기간·신규/기존 고객 구분 포함'},
   {'item':'주요 계약서·출하 및 검수 증빙','purpose':'주요 거래의 사실관계를 추가 확인','format':'거래번호·계약조항·납품/검수일 참조 포함'}],
  'questions':['물량·단가·환율·고객구성 요인의 기여액 합계가 전체 증감액을 얼마나 설명하나요?',
               '설명되지 않은 잔여 증감액과 추가 확인할 자료는 무엇인가요?',
               '회사의 설명을 독립적으로 확인할 증빙을 확보했나요?'],
  'search_query':'수익인식 수행의무 통제 이전 반품'},
 'operating_profit':{'label':'영업이익','objective':'손익 변동을 매출과 비용 요인의 설명·근거로 연결합니다.',
  'requests':[
   {'item':'전기·당기 손익 변동 분석표','purpose':'매출·원가·판관비 요인별 기여 확인','format':'비교 범위·금액·계산식·원본 참조 포함'},
   {'item':'주요 비용 계정의 월별 명세','purpose':'일회성 항목과 지속적 변화의 설명을 확인','format':'계정·부서·거래유형·근거자료 포함'},
   {'item':'생산량·인원·사업장 등 관련 운영 지표','purpose':'재무 수치 변화와 비재무 요인 연결','format':'기간·집계 기준·원천 시스템 포함'}],
  'questions':['이익 변동을 설명하는 요인별 기여액과 미설명 잔액은 얼마인가요?',
               '일회성으로 설명한 항목의 발생 근거와 반복 가능성을 확인했나요?',
               '집계 자료의 정확성·완전성을 어떻게 확인했나요?'],
  'search_query':'영업이익 비용 성격별 분류'},
}


def make_plan(data,company,year,basis,area,due_date=''):
    if area not in TEMPLATES:raise DomainError('지원 검토 영역을 선택하세요.')
    if due_date:
        from datetime import date
        try:date.fromisoformat(due_date)
        except ValueError:raise DomainError('기한은 YYYY-MM-DD 형식이어야 합니다.') from None
    template=TEMPLATES[area]
    tools=Tools('dart',data=data)
    current=tools.get_financials(company,year,basis,area)
    prior=tools.get_financials(company,year-1,basis,area)
    change=tools.calculate_change(current['id'],prior['id'])
    from decimal import Decimal
    delta=Decimal(current['amount'])-Decimal(prior['amount'])
    name=tools.company(company,year,basis)['name']
    heading=f"[자료 요청 초안] {name} {year}년 {template['label']} 변동 검토"
    draft=[heading,'',f"{year}년 {template['label']} 공시 금액은 {Decimal(current['amount']):,} {current['unit']}이며, 전기 대비 증감액은 {delta:,} {current['unit']}입니다.",
           '변동 내용을 확인하기 위해 다음 자료와 설명을 요청드립니다.','',
           '1. 자료 요청',*[f"- {r['item']}\n  목적: {r['purpose']}\n  요청 형식: {r['format']}" for r in template['requests']],
           '', '2. 확인 질문',*[f'- {q}' for q in template['questions']],
           '',f"희망 회신일: {due_date or '미지정 · 발송 전 지정'}",'수신자·사실관계·기한을 검토한 후 사용하세요. 이 앱은 메일을 발송하지 않습니다.']
    return {'area':area,'label':template['label'],'objective':template['objective'],
            'facts':[current,prior,change],'delta':str(delta),'requests':template['requests'],
            'questions':template['questions'],'search_query':template['search_query'],
            'draft':'\n'.join(draft),'due_date':due_date,
            'note':'작성자가 구성한 검토 질문 템플릿입니다. 블로그 글은 업무 아이디어의 참고이며 회계·감사기준이나 실제 법인의 내부 매뉴얼이 아닙니다. 검토 항목의 완전성·감사절차의 충분성을 보장하지 않습니다.'}
