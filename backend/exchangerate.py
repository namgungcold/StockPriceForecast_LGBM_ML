import requests
from bs4 import BeautifulSoup


def exchangerate():
    # 네이버 환율 페이지 URL
    url = "https://finance.naver.com/marketindex/"

    # HTTP 요청
    response = requests.get(url)
    response.raise_for_status()

    # BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 하나은행 기준 환율 데이터 추출 (예: 달러-원 환율)
    usd_krw = soup.select_one('div.market1 div.head_info > span.value').text
    
    return usd_krw