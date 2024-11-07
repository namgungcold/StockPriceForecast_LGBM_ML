import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import requests

# 주식 데이터를 가져오고 차트를 생성하는 함수
def get_yfinance_stock_data(ticker, start=None, end=None):
    # 기본값 설정
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    if start is None:
        start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # yfinance를 이용해 주식 데이터 가져오기
    data = yf.download(ticker, start=start, end=end)
    
    # 결과가 없으면 안내 문구 반환
    if data.empty:
        return "데이터가 없습니다. 유효한 티커를 입력하세요.", None
    
    # 데이터프레임을 정리하여 표시할 정보 선택
    data_display = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # 열 이름을 문자열로 설정하여 문제 해결
    data_display.columns = [col[0] if isinstance(col, tuple) else col for col in data_display.columns]
    
    # 인덱스를 열로 변환
    data_display.reset_index(inplace=True)
    
    # 날짜 정보를 년월일 정보만 출력
    data_display['Date'] = data_display['Date'].dt.strftime('%Y-%m-%d')
    
    # 나머지 값들은 소수점 둘째자리까지 출력하고 천단위 컴마(,)를 붙여줌
    # for col in ['Open', 'High', 'Low', 'Close']:
    #     data_display[col] = data_display[col].apply(lambda x: f"{x:,.2f}")
    # data_display['Volume'] = data_display['Volume'].apply(lambda x: f"{x:,}")
    
    # 차트 생성
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label="Close Price", color='blue')
    # plt.plot(data['High'], label="High Price", color='red')
    # plt.plot(data['Low'], label="Low Price", color='green')
    plt.title(f"{ticker} - Daily Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    
    # 이미지를 메모리에 저장
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)  # 버퍼의 시작으로 이동
    img = Image.open(buf)  # PIL 이미지로 변환
    
    data_display = data_display.iloc[::-1].reset_index(drop=True)
    return data_display, img

def get_prediction(ticker):
    # 요청 URL
    url = "http://121.162.37.155:5000/predict"

    # 요청 헤더 설정
    headers = {
        "Content-Type": "application/json"
    }

    # 요청 본문 (JSON 형식)
    payload = {
        "ticker": ticker
    }

    json_response = {}
    try:
        # POST 요청 보내기
        response = requests.post(url, headers=headers, json=payload)
        
        # 요청이 성공했는지 확인
        response.raise_for_status()
        
        # JSON 응답 받기
        json_response = response.json()
        print("응답 받은 JSON:", json_response)
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 에러 발생: {http_err}")
    except Exception as err:
        print(f"기타 에러 발생: {err}")

    return json_response

# 백엔드 서비스 호출 더미 함수
def predict_stock_price(ticker):
    data_display, img = get_yfinance_stock_data(ticker)
    
    # data_display 처음 행을 맨 처음에 추가
    first_row = data_display.iloc[[0]]
    data_display = pd.concat([first_row, data_display], ignore_index=True)

    return data_display, img

# 백엔드 서비스 호출 더미 함수
def predict_stock_price2(ticker):
    data_display, img = get_yfinance_stock_data(ticker)
    
    date = "2024-11-07"
    open = 221.54
    close = 223.11

    if close >= open:
        isUp = True

    # data_display 처음 행을 맨 처음에 추가
    first_row = data_display.iloc[[0]].copy()

    for col in ['High', 'Low', 'Volume']:
        first_row[col] = "-"
        
    first_row['Date'] = date
    first_row['Open'] = f"{open:,.2f}"
    first_row['Close'] = f"{close:,.2f}"
    first_row['Volume'] = "상승 예상" if isUp else "하락 예상"

    data_display = pd.concat([first_row, data_display], ignore_index=True)
    
    return data_display, img, isUp

# 백엔드 서비스 호출 더미 함수
def predict_stock_price3(ticker):
    data_display, img = get_yfinance_stock_data(ticker)

    result = get_prediction(ticker)
    
    date = "2024-11-07"
    open = result['Open']
    close = result['Predicted Close']
    prediction = result['Prediction']
    max_diff = result['Max_diff']
    min_diff = result['Min_diff']

    isUp = False

    if close >= open:
        isUp = True

    # data_display 처음 행을 맨 처음에 추가
    # first_row = data_display.iloc[[0]].copy()

    # for col in ['High', 'Low', 'Volume']:
    #     first_row[col] = "-"
        
    # first_row['Date'] = date
    # first_row['Open'] = f"{open:,.2f}"
    # first_row['Close'] = f"{close:,.2f}"
    # first_row['Volume'] = "상승 예상" if isUp else "하락 예상"

    # data_display = pd.concat([first_row, data_display], ignore_index=True)
    
    return data_display, img, isUp, open, close, prediction, max_diff, min_diff