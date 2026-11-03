"""OpenDART company resolution and bounded ZIP/XML extraction."""
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import zipfile
from .transport import request_bytes,provider_error

API='https://opendart.fss.or.kr/api/'
STATUS={'010':'등록되지 않은 OpenDART 키입니다.','011':'사용 중지된 OpenDART 키입니다.',
        '012':'OpenDART에 허용되지 않은 IP입니다.','013':'조회된 공시가 없습니다.',
        '014':'공시 원문 파일이 없습니다.','020':'OpenDART 요청 한도를 초과했습니다.',
        '100':'OpenDART 요청 인자를 확인하세요.','800':'OpenDART 시스템 점검 중입니다.',
        '901':'OpenDART 계정 정보 보유기간 만료로 키를 사용할 수 없습니다.'}


def safe_xml(raw):
    if b'<!DOCTYPE' in raw.replace(b'\x00',b'').upper() or b'<!ENTITY' in raw.replace(b'\x00',b'').upper():
        provider_error('UNSAFE_XML','허용되지 않는 XML 선언이 있습니다.','OpenDART')
    try:return ET.fromstring(raw)
    except ET.ParseError:provider_error('INVALID_XML','OpenDART XML 응답을 읽을 수 없습니다.','OpenDART')


def check_status(payload,allow_empty=False):
    if not isinstance(payload,dict):provider_error('INVALID_DART_RESPONSE','OpenDART 응답 형식이 잘못됐습니다.','OpenDART')
    status=str(payload.get('status',''))
    if status=='000':return True
    if status=='013' and allow_empty:return False
    provider_error('DART_'+(status if re.fullmatch(r'\d{3}',status) else 'INVALID_STATUS'),
                   STATUS.get(status,'OpenDART 응답 상태를 확인하세요.'),'OpenDART')


def archive_members(raw):
    if not zipfile.is_zipfile(BytesIO(raw)):
        # 오류일 때 ZIP 대신 XML 또는 JSON이 반환될 수 있다.
        try:payload=json.loads(raw)
        except (ValueError,UnicodeError):
            xml=safe_xml(raw);payload={'status':xml.findtext('status')}
        check_status(payload)
        provider_error('INVALID_ARCHIVE','공시 ZIP 응답이 아닙니다.','OpenDART')
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            members=[i for i in archive.infolist() if not i.is_dir()]
            if len(members)>200 or sum(i.file_size for i in members)>128*1024*1024:
                provider_error('ARCHIVE_TOO_LARGE','공시 압축파일이 처리 용량을 초과했습니다.','OpenDART')
            return [(i.filename,archive.read(i)) for i in members if i.filename.lower().endswith(('.xml','.html','.htm'))]
    except (zipfile.BadZipFile,RuntimeError):provider_error('INVALID_ARCHIVE','손상되거나 암호화된 공시 ZIP입니다.','OpenDART')


class TextParser(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.parts=[];self.hidden=0
    def handle_starttag(self,tag,attrs):
        tag=tag.lower()
        if tag in ('script','style'):self.hidden+=1
        if tag in ('p','tr','div','br','title','section','table'):self.parts.append('\n')
        if tag in ('td','th'):self.parts.append('\t')
    def handle_endtag(self,tag):
        if tag.lower() in ('script','style'):self.hidden=max(0,self.hidden-1)
        if tag.lower() in ('p','tr','div','title','section','table'):self.parts.append('\n')
    def handle_data(self,data):
        if not self.hidden:self.parts.append(data)


def document_text(raw):
    texts=[]
    for name,data in archive_members(raw):
        declaration=re.search(br'encoding\s*=\s*[\'"]([^\'"]+)',data[:300],re.I)
        encodings=[declaration.group(1).decode('ascii','ignore')] if declaration else []
        encodings+=['utf-8-sig','cp949','euc-kr','utf-16']
        decoded=None
        for encoding in encodings:
            try:decoded=data.decode(encoding);break
            except (UnicodeError,LookupError):continue
        if decoded is None:provider_error('DOCUMENT_ENCODING_ERROR','공시 원문의 문자 인코딩을 읽을 수 없습니다.','OpenDART')
        parser=TextParser()
        try:parser.feed(decoded);parser.close()
        except Exception:provider_error('DOCUMENT_PARSE_ERROR','공시 원문을 텍스트로 변환하지 못했습니다.','OpenDART')
        lines=[' '.join(line.split()) for line in ''.join(parser.parts).splitlines()]
        text='\n'.join(line for line in lines if line)
        if text:texts.append(f'[원문 파일: {name}]\n{text}')
    combined='\n\n'.join(texts)
    if not combined:provider_error('EMPTY_DOCUMENT','공시 원문에서 읽을 수 있는 텍스트를 찾지 못했습니다.','OpenDART')
    return combined


class OpenDartClient:
    def __init__(self,api_key,*,transport=request_bytes,archive_dir=None):
        if not re.fullmatch(r'[A-Za-z0-9]{40}',api_key):provider_error('DART_KEY_REQUIRED','40자리 OpenDART API 키가 필요합니다.','Config')
        self._key=api_key;self._transport=transport;self.archive_dir=Path(archive_dir) if archive_dir else None
        self._companies=None

    def _get(self,endpoint,**params):
        # URL은 로그·예외·보고서에 저장하지 않는다(쿼리에 인증키가 있음).
        return self._transport(API+endpoint+'?'+urlencode({'crtfc_key':self._key,**params}))

    def companies(self):
        if self._companies is None:
            members=archive_members(self._get('corpCode.xml'))
            if not members:provider_error('EMPTY_COMPANY_LIST','회사 목록 XML이 없습니다.','OpenDART')
            root=safe_xml(members[0][1])
            self._companies=[{k:(item.findtext(k) or '').strip() for k in ['corp_code','corp_name','stock_code']} for item in root.findall('list')]
        return [dict(c) for c in self._companies]

    def find_companies(self,query):
        query=str(query).strip()
        if not query:provider_error('COMPANY_REQUIRED','회사명 또는 DART 고유번호를 입력하세요.','Config')
        return [c for c in self.companies() if query in c['corp_name'] or query==c['corp_code'] or query==c['stock_code']]

    def resolve_company(self,query):
        query=str(query).strip()
        exact=[c for c in self.companies() if query in (c['corp_name'],c['corp_code'],c['stock_code'])]
        if len(exact)!=1:provider_error('COMPANY_NOT_UNIQUE','회사명을 정확히 입력하거나 find_companies 결과의 DART 고유번호를 선택하세요.','Config')
        return exact[0]

    def disclosures(self,corp_code,*,start_date,end_date,max_reports=1,report_type="A001"):
        if report_type not in ('A001','A002','A003'):provider_error('INVALID_REPORT_TYPE','사업·반기·분기보고서 중 하나를 선택하세요.','Config')
        if not re.fullmatch(r'\d{8}',corp_code):provider_error('INVALID_CORP_CODE','DART 고유번호는 8자리여야 합니다.','Config')
        try:
            start=datetime.strptime(start_date,'%Y%m%d');end=datetime.strptime(end_date,'%Y%m%d')
            valid=bool(re.fullmatch(r'\d{8}',start_date) and re.fullmatch(r'\d{8}',end_date)) and start<=end
        except (ValueError,TypeError):valid=False
        if not valid:provider_error('INVALID_DATE_RANGE','조회 시작일·종료일을 YYYYMMDD로 확인하세요.','Config')
        if not isinstance(max_reports,int) or not 1<=max_reports<=5:raise ValueError('최초 버전은 사업보고서 1~5건만 요청할 수 있습니다.')
        reports=[];page=1
        while len(reports)<max_reports:
            raw=self._get('list.json',corp_code=corp_code,bgn_de=start_date,end_de=end_date,
                pblntf_ty='A',pblntf_detail_ty=report_type,last_reprt_at='Y',sort='date',sort_mth='desc',page_no=page,page_count=100)
            try:payload=json.loads(raw)
            except (ValueError,UnicodeError):provider_error('INVALID_JSON','OpenDART 목록 응답을 읽지 못했습니다.','OpenDART')
            if not check_status(payload,allow_empty=True):break
            rows=payload.get('list',[])
            if not isinstance(rows,list):provider_error('INVALID_DISCLOSURE_LIST','공시 목록 형식이 잘못됐습니다.','OpenDART')
            for row in rows:
                if not isinstance(row,dict):provider_error('INVALID_DISCLOSURE_LIST','공시 목록 항목 형식이 잘못됐습니다.','OpenDART')
                if row.get('corp_code')!=corp_code:provider_error('COMPANY_MISMATCH','요청한 회사와 다른 공시가 반환됐습니다.','OpenDART')
                if not re.fullmatch(r'\d{14}',str(row.get('rcept_no',''))):provider_error('INVALID_RECEIPT','공시 접수번호를 확인하세요.','OpenDART')
            reports.extend(rows)
            try:total_page=int(payload.get('total_page',1))
            except (ValueError,TypeError):provider_error('INVALID_PAGINATION','공시 목록 페이지 정보를 확인하세요.','OpenDART')
            if not rows or page>=total_page:break
            page+=1
        return reports[:max_reports]

    def fetch_documents(self,company,*,start_date,end_date,max_reports=1,report_type="A001"):
        selected=self.resolve_company(company)
        reports=self.disclosures(selected['corp_code'],start_date=start_date,end_date=end_date,max_reports=max_reports,report_type=report_type)
        documents=[]
        for report in reports:
            receipt=report['rcept_no'];raw=self._get('document.xml',rcept_no=receipt)
            text=document_text(raw)
            digest=hashlib.sha256(raw).hexdigest()
            if self.archive_dir:
                # 파일명에 사용자 입력을 사용하지 않는다. 원문은 공개 공시 데이터만 저장한다.
                self.archive_dir.mkdir(parents=True,exist_ok=True)
                (self.archive_dir/f'{receipt}.zip').write_bytes(raw)
                (self.archive_dir/f'{receipt}.txt').write_text(text,encoding='utf-8')
            documents.append({'title':report.get('report_nm','사업보고서'),'url':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+receipt,
                'published_at':report.get('rcept_dt',''),'text':text,'rcept_no':receipt,
                'corp_code':selected['corp_code'],'corp_name':selected['corp_name'],'document_sha256':digest})
        return documents
