import requests
from bs4 import BeautifulSoup
import pandas as pd


def dailyVolume():
    # TradingView 미국 활발한 주식 페이지 URL
    url = "https://kr.tradingview.com/markets/stocks-usa/market-movers-active/"

    # HTTP 요청
    response = requests.get(url)
    response.raise_for_status()  # 요청 성공 여부 확인

    # BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 필요한 데이터 추출 (예: 티커, 회사명, 가격, 변동률 등)
    ticker = []
    volume = []

    # 테이블 행을 순회하며 정보 추출
    for row in soup.select('table tr'):
        cols = row.find_all('td')
        if len(cols) > 4:  # 데이터가 있는 행인지 확인
            ticker.append(cols[0].text.strip())         # 회사명
            volume.append(cols[1].text.strip())       # 거래량

    # 데이터프레임 생성
    data = pd.DataFrame({
        'Company': ticker,
        'Volume': volume
    })

    return data

def dailyChange():

    # TradingView 미국 활발한 주식 페이지 URL
    url = "https://kr.tradingview.com/markets/stocks-usa/market-movers-gainers/"

    # HTTP 요청
    response = requests.get(url)
    response.raise_for_status()  # 요청 성공 여부 확인

    # BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 필요한 데이터 추출 (예: 티커, 회사명, 가격, 변동률 등)
    ticker = []
    changePer = []

    # 테이블 행을 순회하며 정보 추출
    for row in soup.select('table tr'):
        cols = row.find_all('td')
        if len(cols) > 4:  # 데이터가 있는 행인지 확인
            ticker.append(cols[0].text.strip())         # 회사명
            changePer.append(cols[1].text.strip())       # 변화율

    # 데이터프레임 생성
    data = pd.DataFrame({
        'Company': ticker,
        'changePercents': changePer
    })

    # 결과 확인
    return data

