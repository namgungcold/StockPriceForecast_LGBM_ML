import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

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
    data_display['Open'] = data_display['Open'].apply(lambda x: f"{x:,.2f}")
    data_display['High'] = data_display['High'].apply(lambda x: f"{x:,.2f}")
    data_display['Low'] = data_display['Low'].apply(lambda x: f"{x:,.2f}")
    data_display['Close'] = data_display['Close'].apply(lambda x: f"{x:,.2f}")
    data_display['Volume'] = data_display['Volume'].apply(lambda x: f"{x:,}")
    
    # 차트 생성
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label="Close Price", color='blue')
    plt.title(f"{ticker} - 일별 종가 차트")
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