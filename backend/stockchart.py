import requests
from bs4 import BeautifulSoup
import pandas as pd


def dailyChart():
    # TradingView 미국 활발한 주식 페이지 URL
    url_volume = "https://kr.tradingview.com/markets/stocks-usa/market-movers-active/"
    url_change = "https://kr.tradingview.com/markets/stocks-usa/market-movers-gainers/"

    # HTTP 요청
    response_volume = requests.get(url_volume)
    response_volume.raise_for_status()  # 요청 성공 여부 확인

    response_change = requests.get(url_change)
    response_change.raise_for_status()  # 요청 성공 여부 확인

    # BeautifulSoup으로 HTML 파싱
    soup_volume = BeautifulSoup(response_volume.text, 'html.parser')
    soup_change = BeautifulSoup(response_change.text, 'html.parser')

    # 필요한 데이터 추출 (예: 티커, 회사명, 가격, 변동률 등)
    ticker_volume = []
    ticker_change = []
    volume = []
    changePer = []
    rank_volume=[]
    rank_change=[]

    # 테이블 행을 순회하며 정보 추출
    for i,row in enumerate(soup_volume.select('table tr')):
        cols = row.find_all('td')
        if len(cols) > 4:  # 데이터가 있는 행인지 확인
            rank_volume.append(i) 
            ticker_volume.append(cols[0].text.strip())       # 회사명
            volume.append(cols[1].text.strip())       # 거래량
    
    for i,row in enumerate(soup_change.select('table tr')):
        cols = row.find_all('td')
        if len(cols) > 4:  # 데이터가 있는 행인지 확인
            rank_change.append(i)
            ticker_change.append(cols[0].text.strip())       # 회사명
            changePer.append(cols[1].text.strip())    # 변화율

    # 데이터프레임 생성
    data_volume = pd.DataFrame({
        'Rank': rank_volume,
        'Company': ticker_volume,
        'Volume': volume
    })

    data_change = pd.DataFrame({
        'Rank': rank_change,
        'Company': ticker_change,
        'changePercents': changePer
    })
    return data_volume,data_change

