'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let workspace = null, lastRun = null, lastScenario = null, lastPlan = null, busy = false, knowledgeCount = 0, currentTab = 'review', knowledgeReady = false, selectedAccount = 'ALL', accountFolders = [];
const names = {valuation:['Inventory Valuation','재고자산 평가금액을 확인하세요.','입력 조건에서 결과까지, 품목별 산출 과정을 함께 확인합니다.'],review:['Disclosure Review','숫자 뒤의 근거를 확인하세요.','실제 기업 공시를 조회하고, 재무 변동과 관련 주석을 하나의 검토 흐름으로.'],planning:['Review Planner','검토 결과를 다음 질문으로 이어가세요.','재무 변동·필요 증빙·확인 질문을 연결하고 자료 요청 초안을 만듭니다.'],footing:['Footing','같은 전기 금액이, 같은 숫자인지 확인하세요.','당기 비교금액과 전기 공시를 대사하고 차이가 생긴 항목을 추가 검토합니다.'],scenario:['Scenario Lab','재고 손실에 따른 손익 변화를 확인하세요.','재고에서 추가 손실이 생겼을 때 영업이익이 얼마나 줄어드는지 계산합니다.'],evaluation:['Evaluation','측정 가능한 결과를 남깁니다.','테스트 조건, 평가 범위와 실패 사례를 함께 확인하세요.'],knowledge:['PBC Workspace','계정별 자료를 등록하고 찾아보세요.','현금·매출채권·재고 등 계정별로 자료를 정리하고 필요한 내용을 찾습니다.'],references:['Reference Desk','검토에 필요한 자료를 이어서 확인하세요.','공시에서 확인한 사실과 회계기준 적용 판단을 구분합니다.']};
async function api(path, body, method='POST') {
 const response = await fetch(path, {method, credentials:'same-origin', headers:body ? {'Content-Type':'application/json'} : {}, body:body ? JSON.stringify(body) : undefined});
 let data;
 try { data = await response.json(); } catch { throw new Error('서버 응답을 확인하지 못했습니다.'); }
 if (!response.ok) { if (response.status === 401) { workspace = null; setEnabled(); } throw new Error(typeof data.detail === 'string' ? data.detail : '요청을 처리하지 못했습니다.'); }
 return data;
}
function feedback(selector, text='', kind='') { const el=$(selector); el.textContent=text; el.className='feedback'+(kind?' '+kind:''); }
function setEnabled() { $('#connection-pill').textContent=workspace?'공시 연결됨':knowledgeReady?'자료 작업 중':'연결 전'; $('#ask-submit').disabled = busy || !workspace; $('#connect-submit').disabled=busy; $('#export').disabled=!lastRun; $('#disconnect').hidden=!workspace&&!knowledgeReady; $('#plan-submit').disabled=busy||!workspace; $('#footing-refresh').disabled=busy||!workspace; $('#knowledge-submit').disabled=busy||!knowledgeReady; $('#knowledge-search-submit').disabled=busy||!knowledgeReady||knowledgeCount===0; $$('.prompt').forEach(b=>b.disabled=busy); }
function setBusy(value) { busy=value; setEnabled(); }
function setPageContext(tab) { $('#connect-panel').hidden=!['review','footing'].includes(tab); $('#scope-summary').hidden=!workspace||['references','evaluation','knowledge','valuation'].includes(tab); $('#export').hidden=tab!=='review'; $('.review-subnav').hidden=!['review','footing','valuation'].includes(tab); }
function chooseTab(tab) {
 currentTab=tab;setPageContext(tab);
 const primary=['review','footing','valuation'].includes(tab)?'review':tab;
 $$('.nav').forEach(b=>b.classList.toggle('active',b.dataset.tab===primary));
 $$('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===tab));
 $$('.tab-content').forEach(p=>p.hidden=p.id!=='tab-'+tab);
 $('#breadcrumb').textContent=names[tab][0];$('#page-title').textContent=names[tab][1];$('#page-subtitle').textContent=names[tab][2];
 if(tab==='knowledge')ensureKnowledge();
}
$$('[data-view]').forEach(b=>b.addEventListener('click',()=>chooseTab(b.dataset.view)));
function resultTab(name) {
$$('.result-tab').forEach(b=>{b.classList.toggle('active',b.dataset.result===name);b.setAttribute('aria-selected',String(b.dataset.result===name));}); $$('.result-panel').forEach(p=>p.hidden=p.id!=='result-'+name); }
$$('.nav').forEach(b=>b.addEventListener('click',()=>chooseTab(b.dataset.tab)));
$$('.result-tab').forEach(b=>b.addEventListener('click',()=>resultTab(b.dataset.result)));
$$('.prompt').forEach(b=>b.addEventListener('click',()=>{ $('#question').value=b.dataset.question; $('#question').focus(); }));
$('#mode').addEventListener('change',()=>{$('#mode-note').textContent=$('#mode').value==='llm'?'LLM이 도구를 선택합니다. 질문·검색 공시가 OpenAI로 전송되며 API 요금이 발생합니다.':'기본 분석은 지원하는 질문 유형에 맞춰 재무 수치를 조회·계산하고 관련 공시를 찾습니다.';});
function won(value) { return new Intl.NumberFormat('ko-KR', {maximumFractionDigits:2}).format(Number(value)/100000000); }
function full(value) { return new Intl.NumberFormat('ko-KR', {maximumFractionDigits:2}).format(Number(value)); }
function safeURL(url) { try {const u=new URL(url);return u.protocol==='https:' && u.hostname==='dart.fss.or.kr' ? u.href : null;}catch{return null;} }
function sourceLink(url,label='DART 원문 확인 ↗') { const safe=safeURL(url);return safe?`<a href="${esc(safe)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`:'<span class="muted">테스트 데이터 · 외부 원문 없음</span>'; }
function applyWorkspace(data) {
 workspace=data.connected?data:null;knowledgeReady=Boolean(data.knowledge_ready||workspace);
 $('#connection-pill').textContent=workspace?'공시 연결됨':'연결 전'; $('#connection-pill').className='pill '+(workspace?'success':'neutral');
 const llm=$('#mode option[value="llm"]'); llm.disabled=!workspace?.llm_ready; llm.textContent=workspace?.llm_ready?'LLM 도구 호출':'LLM 도구 호출 · 키 필요';
 if(knowledgeReady) loadKnowledge(); else {knowledgeCount=0;$('#knowledge-list').textContent='등록된 문서가 없습니다.';$('#knowledge-results').textContent='공시를 연결하고 문서를 등록하면 검색할 수 있습니다.';}
 if (!workspace?.llm_ready) $('#mode').value='rules';
 $('#scope-summary').hidden=!workspace;
 if(workspace) {
  const date=new Date(data.collected_at).toLocaleString('ko-KR',{timeZone:'Asia/Seoul'});
  $('#scope-summary').innerHTML=`<div><strong>${esc(data.company.name)}</strong> &nbsp; ${data.year-1}–${data.year} 사업보고서 · ${data.basis==='CFS'?'연결':'별도'} · 조회 ${esc(date)} KST</div><div class="scope-links">${data.documents.map(d=>sourceLink(d.url,`${d.year} 원문 ↗`)).join('')}</div>`;
  $('#metrics').innerHTML=data.cards.map(c=>c.unavailable?`<article class="metric"><span>${esc(c.label)}</span><strong>미확보</strong><p>해당 조건의 수치를 확인하세요.</p></article>`:`<article class="metric"><span>${esc(c.label)} · ${esc(c.fact.currency)}</span><strong>${won(c.fact.amount)}<small>억원</small></strong><p><span class="delta">${Number(c.change.value)>0?'+':''}${esc(c.change.value)}%</span> 전기 대비 · ${data.year}년</p></article>`).join('');
  if(data.warnings.length) feedback('#connect-feedback',data.warnings.join('\n'),'error');
 }
 setPageContext(currentTab);setEnabled();
}
function clearResults() { lastRun=null;lastScenario=null;lastPlan=null; $('#plan-result').textContent='공시 연결 후 검토 계획을 만들 수 있습니다.';$('#footing-results').textContent='공시 연결 후 대사를 실행하세요.';$('#export').disabled=true;$('#evidence-count').textContent='0';$('#result-status').textContent='실행 대기';$('#result-status').className='pill neutral';$('#result-answer').innerHTML='<div class="empty-state"><div class="empty-icon">⌕</div><h3>질문에서 시작해, 근거까지.</h3><p>공시를 연결하고 검토 질문을 입력하세요.</p></div>';$('#result-evidence').textContent='실행 후 근거 원문을 확인할 수 있습니다.';$('#result-trace').textContent='실행 후 호출 도구를 확인할 수 있습니다.';$('#review-followup').hidden=true;feedback('#ask-feedback'); }
$('#connect-form').addEventListener('submit',async e=>{
 e.preventDefault();if(busy)return;setBusy(true);feedback('#connect-feedback','OpenDART에서 회사 정보·재무 수치·사업보고서 원문을 조회하고 있습니다. 공시 크기에 따라 1~3분이 걸릴 수 있습니다.','loading');
 const payload={dart_key:$('#dart-key').value,company:$('#company').value.trim(),year:Number($('#year').value),basis:$('#basis').value,openai_key:$('#openai-key').value,model:$('#model').value.trim()};
 $('#dart-key').value='';$('#openai-key').value='';
 try{const data=await api('/api/connect',payload);clearResults();feedback('#connect-feedback');applyWorkspace(data);$('#connect-feedback').textContent=data.warnings.length?data.warnings.join('\n'):'실제 공시 조회 완료. 분석 범위는 다음 조회 전까지 고정됩니다.';}
 catch(error){feedback('#connect-feedback',error.message,'error');}
 finally{payload.dart_key='';payload.openai_key='';setBusy(false);}
});
$('#disconnect').addEventListener('click',async()=>{if(busy)return;try{await api('/api/session',null,'DELETE');applyWorkspace({connected:false});clearResults();$('#metrics').innerHTML=['매출','영업이익','재고자산'].map(n=>`<article class="metric"><span>${n}</span><strong>—</strong><p>공시 연결 후 표시</p></article>`).join('');feedback('#connect-feedback','세션을 종료했습니다. 다시 조회하려면 키를 입력하세요.');}catch(error){feedback('#connect-feedback',error.message,'error');}});
const metricLabels = {revenue:'매출',operating_profit:'영업이익',inventory:'재고자산'};
function evidenceLabel(e, evidence=[]) {
 if(!e) return '근거 확인';
 if(e.kind==='calculation'||e.kind==='tieout') {
  const input=evidence.find(r=>r.id===e.current_id);
  return `${metricLabels[input?.metric]||'재무 수치'} · ${e.kind==='tieout'?'전기 금액 대사':'전기 대비 증감률'}`;
 }
 if(e.quote!==undefined) return `${e.title||'사업보고서'} · ${e.section&&e.section!=='미식별'?e.section:'관련 본문'}`;
 return `${e.year}년 ${e.basis==='CFS'?'연결':'별도'} · ${e.account_name||metricLabels[e.metric]||'재무 수치'} · 사업보고서`;
}
function excerptHTML(text) {
 const lines=String(text).split(/\r?\n/);
 return lines.map(line=>{
  if(!line.trim())return '<div class="excerpt-gap"></div>';
  const heading=line.length<90 && /^(?:\d+(?:\.\d+)*[.．) ]|[IVXⅠⅡⅢⅣⅤ]+[.． ])/.test(line) && /[가-힣A-Za-z]{2}/.test(line);
  return `<p class="${heading?'excerpt-heading':'excerpt-paragraph'}">${esc(line)}</p>`;
 }).join('');
}
function evidenceHTML(e) {
 let body;
 if(e.quote!==undefined) body=`<blockquote class="source-excerpt">${excerptHTML(e.quote)}</blockquote><small>추출 텍스트 문자 범위 [${e.start}, ${e.end}) · 보고서 전체 검색 결과${e.section?`<br>섹션 ${esc(e.section)} · 구분 힌트 ${esc(e.basis_context)} (원문 확인 필요)`:""}</small><small>텍스트 SHA-256 ${esc(e.sha256)}</small>`;
 else if(e.kind==='tieout') body=`<div class="fact-grid"><span>당기 보고서 전기 비교금액</span><strong>${full(e.comparative_amount)} ${esc(e.unit)}</strong><span>전기 보고서 금액</span><strong>${full(e.prior_report_amount)} ${esc(e.unit)}</strong><span>차이</span><strong>${full(e.difference)} ${esc(e.unit)}</strong></div><p class="muted">${esc(e.note)}</p><small>현재: ${esc(e.current_id)}<br>전기: ${esc(e.prior_id)}</small>`;
 else if(e.kind==='calculation') body=`<div class="scenario-value">${esc(e.value)}<small>%</small></div><p class="muted">${esc(e.formula)}</p><small>현재: ${esc(e.current_id)}<br>전기: ${esc(e.prior_id)}<br>반올림: ${esc(e.rounding)}</small>`;
 else body=`<div class="fact-grid"><span>금액</span><strong>${full(e.amount)} ${esc(e.unit)}</strong><span>연도 · 구분</span><strong>${e.year} · ${esc(e.basis)}</strong><span>계정</span><strong>${esc(e.account_name||e.metric)}</strong><span>공시 기간</span><strong>${esc(e.period_label||'연간')}</strong><span>통화 · 접수번호</span><strong>${esc(e.currency)} · ${esc(e.receipt||'테스트')}</strong></div><small>응답 SHA-256 ${esc(e.response_sha256||'합성 테스트 데이터')}</small>`;
 return `<article class="evidence-item" data-evidence-id="${esc(e.id)}"><span class="pill success">출처·무결성 확인</span><h3>${esc(evidenceLabel(e,lastRun?.evidence||[]))}</h3><details class="evidence-technical"><summary>출처 식별 정보</summary><small>${esc(e.id)}</small></details>${body}${e.kind==='calculation'?'':sourceLink(e.url)}</article>`;
}
function renderRun(run) {
 const analyzed=new Set(run.evidence.filter(e=>e.kind!=='calculation'&&e.amount!==undefined).map(e=>e.metric));
 const options=[...$('#plan-area').options];
 options.forEach(o=>{o.disabled=!analyzed.has(o.value);});
 const available=options.find(o=>!o.disabled);
 $('#review-followup').hidden=!available;
 if(available)$('#plan-area').value=available.value;
 $('#plan-result').textContent='공시만으로 변동 원인과 증빙의 충분성을 판단하기 어렵습니다. 필요한 자료를 확인해 보세요.';
 lastPlan=null;

 lastRun=run;$('#result-status').textContent=run.status==='answered'?'검토 필요':'근거 부족';$('#result-status').className='pill '+(run.status==='answered'?'success':'amber');$('#evidence-count').textContent=run.evidence.length;
 $('#result-answer').innerHTML=`<div class="run-meta"><span>${run.mode==='llm'?'LLM 도구 호출':'기본 분석'}</span><span>${run.strategy==='bm25'?'문맥·관련도 검색':run.strategy==='expanded'?'검색어 확장':'기본 키워드'}</span><span>${(run.duration_ms/1000).toFixed(2)}초</span>${run.usage?`<span>API ${run.usage.requests}회 · 입력 ${run.usage.input_tokens} / 출력 ${run.usage.output_tokens} 토큰</span>`:''}</div>${run.claims.length?run.claims.map(c=>`<article class="finding"><div class="finding-text">${excerptHTML(c.text)}</div>${c.evidence_ids.map(id=>`<button class="source-chip" data-source="${esc(id)}">↗ ${esc(evidenceLabel(run.evidence.find(e=>e.id===id),run.evidence))}</button>`).join('')}</article>`).join(''):'<div class="followup"><h3>확인된 근거가 충분하지 않습니다.</h3><p>추정한 답변 대신 추가 확인사항을 표시합니다.</p></div>'}<div class="followup"><h3>추가 검토 질문</h3>${run.follow_up.map(x=>`<p>□ ${esc(x)}</p>`).join('')}</div><div class="limitations"><h3>검토 시 참고사항</h3>${run.limitations.map(x=>`<p>· ${esc(x)}</p>`).join('')}</div>`;
 $('#result-evidence').innerHTML=`<p class="muted">${run.checks.filter(c=>c.valid).length}/${run.checks.length}개 근거의 출처와 계산값을 확인했습니다. 답변과 연결된 원문을 함께 검토해 주세요.</p>`+run.evidence.map(evidenceHTML).join('');
 $('#result-trace').innerHTML=run.trace.map((t,i)=>`<details class="trace-item"><summary><span class="trace-number">${String(i+1).padStart(2,'0')}</span>${esc(t.tool)} <span class="pill ${t.status==='ok'?'success':'amber'}">${t.status==='ok'?'완료':'오류'}</span></summary><p class="muted">입력 인자</p><pre>${esc(JSON.stringify(t.arguments,null,2))}</pre><p class="muted">실행 결과</p><pre>${esc(JSON.stringify(t.result||{error:t.error},null,2))}</pre></details>`).join('')||'<p class="muted">호출한 도구가 없습니다.</p>';
 $$('.source-chip').forEach(b=>b.addEventListener('click',()=>{resultTab('evidence');const target=$$('.evidence-item').find(e=>e.dataset.evidenceId===b.dataset.source);target?.scrollIntoView({behavior:'smooth',block:'center'});}));
 resultTab('answer');setEnabled();
}
$('#ask-form').addEventListener('submit',async e=>{e.preventDefault();if(busy||!workspace)return;if(!$('#question').value.trim()){feedback('#ask-feedback','검토 질문을 입력하세요.','error');return;}setBusy(true);feedback('#ask-feedback',$('#mode').value==='llm'?'LLM이 도구를 호출하고 답변 근거를 확인하고 있습니다.':'재무 수치와 공시 근거를 확인하고 있습니다.','loading');try{const result=await api('/api/ask',{question:$('#question').value.trim(),mode:$('#mode').value,strategy:$('#strategy').value});feedback('#ask-feedback');renderRun(result);}catch(error){feedback('#ask-feedback',error.message,'error');}finally{setBusy(false);}});
$('#export').addEventListener('click',()=>{if(!lastRun)return;const r=lastRun;const text=[`# Financial Evidence Agent 검토 메모`,`\n회사: ${workspace?.company.name||r.company}\n연도: ${r.year} · ${r.basis}\n질문: ${r.question||''}\n실행: ${r.mode} · 검색: ${r.strategy}\n작성: ${new Date().toISOString()}`,'\n## 확인된 내용',...r.claims.map(c=>`- ${c.text}\n  근거: ${c.evidence_ids.join(', ')}`),'\n## 추가 검토 질문',...r.follow_up.map(x=>`- [ ] ${x}`),'\n## 근거',...r.evidence.map(e=>`\n### ${e.id}\n${e.quote!==undefined?`${e.quote}\n문자 위치 [${e.start}, ${e.end})\nSHA-256 ${e.sha256}`:JSON.stringify(e,null,2)}\n${e.url||''}`),'\n## 확인 범위',...r.limitations.map(x=>`- ${x}`),...(lastScenario?['\n## 사용자 가정 시나리오',JSON.stringify(lastScenario,null,2)]:[])].join('\n');const u=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download=`financial-review-${r.year}.md`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
Promise.all([api('/api/status',null,'GET').then(applyWorkspace).catch(()=>feedback('#connect-feedback','서버에 연결하지 못했습니다.','error'))]);

async function ensureKnowledge(){
 if(!knowledgeReady){try{await api('/api/knowledge/session',{});knowledgeReady=true;}catch(error){feedback('#knowledge-feedback',error.message,'error');return;}}
 setEnabled();await loadKnowledge();
}
function renderFolders(documents){
 const count=code=>documents.filter(d=>code==='ALL'||(d.account_code||'COMMON')===code).length;
 $('#folder-tree').innerHTML=[{code:'ALL',name:'전체 자료'},...accountFolders].map(f=>`<button class="folder-button ${selectedAccount===f.code?'active':''}" data-folder="${esc(f.code)}">▱ ${f.code==='ALL'||f.code==='COMMON'?'':esc(f.code)+'. '}${esc(f.name)} <span>${count(f.code)}</span></button>`).join('');
 $('#knowledge-account').innerHTML=accountFolders.map(f=>`<option value="${esc(f.code)}">${f.code==='COMMON'?'':esc(f.code)+'. '}${esc(f.name)}</option>`).join('');
 $('#knowledge-account').value=selectedAccount==='ALL'?'COMMON':selectedAccount;
 $$('[data-folder]').forEach(button=>button.addEventListener('click',()=>{selectedAccount=button.dataset.folder;$('#knowledge-results').textContent='선택한 계정의 문서에서 검색합니다.';loadKnowledge();}));
}
async function loadKnowledge(){try{
 const data=await api('/api/knowledge',null,'GET');accountFolders=data.folders||[];renderFolders(data.documents);
 const documents=data.documents.filter(d=>selectedAccount==='ALL'||(d.account_code||'COMMON')===selectedAccount);
 knowledgeCount=documents.length;setEnabled();
 $('#knowledge-list').innerHTML=documents.length?documents.map(d=>`<article class="knowledge-file"><div><strong>${esc(d.title)}</strong><small>${esc(d.account_code||'COMMON')} · ${esc(d.version||'버전 미입력')} · ${esc(d.effective_date||'적용일 미입력')}</small></div><button class="text-button" data-remove-doc="${esc(d.id)}">제거</button></article>`).join(''):'<p class="muted">이 폴더에 등록된 자료가 없습니다. 파일을 먼저 등록하세요.</p>';
 if(!knowledgeCount)$('#knowledge-results').innerHTML='<p class="muted">이 폴더에 등록된 자료가 없습니다. 파일을 먼저 등록하세요.</p>';
 $$('[data-remove-doc]').forEach(b=>b.addEventListener('click',async()=>{await api('/api/knowledge/'+encodeURIComponent(b.dataset.removeDoc),null,'DELETE');$('#knowledge-results').textContent='문서가 제거되었습니다. 다시 검색하세요.';loadKnowledge();}));
}catch(error){feedback('#knowledge-feedback',error.message,'error');}}
$('#folder-form').addEventListener('submit',async e=>{e.preventDefault();try{await ensureKnowledge();await api('/api/knowledge/folders',{code:$('#folder-code').value.trim(),name:$('#folder-name').value.trim()});$('#folder-form').reset();await loadKnowledge();}catch(error){feedback('#knowledge-feedback',error.message,'error');}});
$('#knowledge-upload').addEventListener('submit',async e=>{e.preventDefault();if(busy||!knowledgeReady)return;const file=$('#knowledge-file').files[0];if(!file)return;if(file.size>1000000){feedback('#knowledge-feedback','1MB 이하 파일을 선택하세요.','error');return;}const form=new FormData();form.append('file',file);form.append('account_code',$('#knowledge-account').value);form.append('title',$('#knowledge-title').value);form.append('version',$('#knowledge-version').value);form.append('effective_date',$('#knowledge-date').value);setBusy(true);try{const response=await fetch('/api/knowledge',{method:'POST',body:form,credentials:'same-origin'});const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'문서 등록 실패');feedback('#knowledge-feedback','문서를 등록했습니다. 외부 LLM으로 전송하지 않았습니다.');$('#knowledge-upload').reset();await loadKnowledge();}catch(error){feedback('#knowledge-feedback',error.message,'error');}finally{setBusy(false);}});
$('#knowledge-search').addEventListener('submit',async e=>{e.preventDefault();if(busy||!knowledgeReady)return;setBusy(true);try{const r=await api('/api/knowledge/search',{question:$('#knowledge-query').value,account_code:selectedAccount});$('#knowledge-results').innerHTML=`<p class="mode-note">${esc(r.note)}</p>`+(r.results.length?r.results.map(d=>`<article class="evidence-item"><span class="pill success">User document · 외부 전송 없음</span><h3>${esc(d.title)}</h3><small>${esc(d.version||'버전 미입력')} · 적용일 ${esc(d.effective_date||'미입력')} · 줄 ${d.start_line}–${d.end_line}</small><blockquote class="source-excerpt">${excerptHTML(d.quote)}</blockquote><small>추출 텍스트 [${d.start}, ${d.end}) · ${esc(d.id)}<br>SHA-256 ${esc(d.sha256)}</small></article>`).join(''):'<p class="muted">관련 근거를 확보하지 못했습니다. 문서 등록 여부와 검색어를 확인하세요.</p>');}catch(error){$('#knowledge-results').textContent=error.message;}finally{setBusy(false);}});

$('#plan-form').addEventListener('submit',async e=>{e.preventDefault();if(busy||!workspace||!lastRun)return;setBusy(true);try{const p=await api('/api/review-plan',{area:$('#plan-area').value,due_date:$('#plan-due').value});lastPlan=p;$('#plan-result').innerHTML=`<div class="section-kicker">${esc(p.label)} · REVIEW PLAN</div><h2>${esc(p.objective)}</h2><p class="muted">전기 대비 증감액 ${full(p.delta)}원 · ${workspace.basis}</p><div class="followup"><h3>필요 자료 · 확인 목적</h3>${p.requests.map(r=>`<p><strong>${esc(r.item)}</strong><br>${esc(r.purpose)}<br>요청 형식: ${esc(r.format)}</p>`).join('')}</div><div class="followup"><h3>확인 질문</h3>${p.questions.map(q=>`<p>□ ${esc(q)}</p>`).join('')}</div><details class="trace-item"><summary>RFI · 정보 요청 초안 확인</summary><pre>${esc(p.draft)}</pre></details><button class="button secondary" id="plan-export">↓ RFI 초안 다운로드</button><p class="mode-note">조회한 공시 금액을 바탕으로 작성한 자료 요청 예시입니다. 실제 변동 원인이 확인되었다는 뜻은 아니며, 확보한 증빙과 회사 상황에 따라 조정하세요.</p>${p.facts.filter(f=>f.url).map(f=>sourceLink(f.url,f.year+'년 공시 ↗')).join(' · ')}`;$('#plan-export').addEventListener('click',()=>{const blob=URL.createObjectURL(new Blob([p.draft+'\n\n근거 수치\n'+JSON.stringify(p.facts,null,2)],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=blob;a.download='review-plan-'+p.area+'.md';a.click();setTimeout(()=>URL.revokeObjectURL(blob),1000);});feedback('#plan-feedback');}catch(error){feedback('#plan-feedback',error.message,'error');}finally{setBusy(false);}});
$('#footing-refresh').addEventListener('click',async()=>{if(busy||!workspace)return;setBusy(true);try{const r=await api('/api/footing',null,'GET');$('#footing-results').innerHTML=`<p class="mode-note">${esc(r.note)}</p>`+r.rows.map(row=>row.unavailable?`<article class="evidence-item"><h3>${esc(row.metric)}</h3><p>${esc(row.unavailable)}</p></article>`:`<article class="evidence-item"><span class="pill ${row.result.status==='matched'?'success':'amber'}">${row.result.status==='matched'?'금액 일치':'차이 확인 필요'}</span><h3>${esc(row.current.account_name||row.metric)} · ${row.current.year-1}년</h3><div class="fact-grid"><span>당기 보고서의 전기 비교금액</span><strong>${full(row.result.comparative_amount)}원</strong><span>전기 보고서의 당기금액</span><strong>${full(row.result.prior_report_amount)}원</strong><span>차이</span><strong>${full(row.result.difference)}원</strong></div><p class="muted">${esc(row.result.note)}</p>${sourceLink(row.current.url,'당기 보고서 ↗')} · ${sourceLink(row.prior.url,'전기 보고서 ↗')}</article>`).join('');}catch(error){$('#footing-results').textContent=error.message;}finally{setBusy(false);}});

function valuationNumber(value) { return new Intl.NumberFormat('ko-KR',{maximumFractionDigits:4}).format(Number(value)); }
function addValuationItem() {
 if($$('.valuation-item').length>=30)return;
 const row=document.createElement('fieldset');row.className='valuation-item';
 row.innerHTML=`<legend>품목 정보</legend><div class="valuation-fields"><div class="field"><label>품목명<input data-value="name" maxlength="80" placeholder="예: 제품 A" required></label></div>${[['quantity','수량'],['cost','단위원가'],['price','단위 판매가격'],['completion','단위 추가 완성원가'],['selling','단위 판매원가']].map(([key,label])=>`<div class="field"><label>${label}<input data-value="${key}" type="number" min="0" max="1000000000000" step="0.01" placeholder="0" required></label></div>`).join('')}</div><button type="button" class="button secondary valuation-remove">품목 삭제</button>`;
 row.querySelector('.valuation-remove').addEventListener('click',()=>{row.remove();invalidateValuation();});
 row.addEventListener('input',invalidateValuation);
 $('#valuation-items').append(row);invalidateValuation();
}
function invalidateValuation() {
 $('#valuation-result').innerHTML='<p class="muted">입력 내용이 변경되었습니다. 평가 결과 확인을 눌러 다시 계산하세요.</p>';
 feedback('#valuation-feedback');
}
$('#valuation-add').addEventListener('click',addValuationItem);
$('#valuation-form').addEventListener('submit',async event=>{
 event.preventDefault();
 const items=$$('.valuation-item').map(row=>Object.fromEntries([...row.querySelectorAll('[data-value]')].map(input=>[input.dataset.value,input.value])));
 const buttons=[...$('#valuation-form').querySelectorAll('button')];buttons.forEach(b=>b.disabled=true);
 try {
  const r=await api('/api/valuation',{items});
  $('#valuation-result').innerHTML=`<h3>평가 결과 · 사용자 입력 기준</h3><div class="fact-grid"><span>평가 전 원가 합계</span><strong>${valuationNumber(r.original)}원</strong><span>평가 후 장부금액</span><strong>${valuationNumber(r.carrying)}원</strong><span>평가손실 합계</span><strong>${valuationNumber(r.loss)}원</strong></div>${r.rows.map(row=>`<details class="trace-item" open><summary>${esc(row.name)} · 평가손실 ${valuationNumber(row.loss)}원</summary><ol><li>단위당 순실현가능가치 = 판매가격 ${valuationNumber(row.price)} − 완성원가 ${valuationNumber(row.completion)} − 판매원가 ${valuationNumber(row.selling)} = <strong>${valuationNumber(row.nrv)}원</strong></li><li>단위당 평가금액 = 원가 ${valuationNumber(row.cost)}와 순실현가능가치 ${valuationNumber(row.nrv)} 중 낮은 금액 = <strong>${valuationNumber(row.unit_value)}원</strong></li><li>평가 후 장부금액 = ${valuationNumber(row.unit_value)} × 수량 ${valuationNumber(row.quantity)} = <strong>${valuationNumber(row.carrying)}원</strong></li><li>평가손실 = 원가 총액 ${valuationNumber(row.original)} − 평가 후 금액 ${valuationNumber(row.carrying)} = <strong>${valuationNumber(row.loss)}원</strong></li></ol></details>`).join('')}<p class="mode-note">품목별로 평가한 뒤 합산했습니다. 다른 품목의 가치 상승으로 평가손실을 상계하지 않습니다. 입력한 가격·원가의 타당성이나 실제 분개·세금 효과는 이 계산에서 판단하지 않습니다.</p>`;
  feedback('#valuation-feedback','계산을 완료했습니다.');
 }catch(error){$('#valuation-result').textContent='입력 조건을 확인한 뒤 다시 계산하세요.';feedback('#valuation-feedback',error.message,'error');}
 finally{buttons.forEach(b=>b.disabled=false);}
});
addValuationItem();
