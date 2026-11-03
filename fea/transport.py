"""API 인증정보를 예외 메시지에 포함하지 않는 제한된 HTTP 통신."""
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
import socket
from .domain import DomainError


def provider_error(code,message,provider):
    raise DomainError(f'{provider}: {code} · {message}') from None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


def request_bytes(url, *, method='GET', headers=None, body=None, timeout=45, max_bytes=64*1024*1024):
    request=Request(url,data=body,headers=headers or {},method=method)
    provider='OpenAI' if url.startswith('https://api.openai.com/') else 'OpenDART'
    try:
        with build_opener(NoRedirect()).open(request,timeout=timeout) as response:
            raw=response.read(max_bytes+1)
    except HTTPError as error:
        messages={401:'API 인증정보를 확인하세요.',403:'API 접근 권한을 확인하세요.',
                  429:'요청 한도 또는 API 잔액을 확인하세요.'}
        provider_error(f'{provider.upper()}_HTTP_{error.code}',messages.get(error.code,'외부 API 요청이 실패했습니다.'),provider)
    except (URLError,TimeoutError,socket.timeout,OSError):
        provider_error('API_CONNECTION_ERROR','API 연결에 실패했습니다. 네트워크와 제한 시간을 확인하세요.',provider)
    if len(raw)>max_bytes:provider_error('RESPONSE_TOO_LARGE','API 응답이 처리 용량을 초과했습니다.',provider)
    return raw
