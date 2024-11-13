# Import necessary libraries
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from flask import Flask, request, jsonify
import seaborn as sns
import pandas_datareader as pdr
import datetime
import warnings

warnings.filterwarnings('ignore')

# Flask 애플리케이션 생성
app = Flask(__name__)


sns.set(style="whitegrid")



def stock_predict(ticker):
    
    # start_date부터 당일까지 출력가능
    end_date = datetime.datetime.now()  
    # past_date = end_date - datetime.timedelta(days=7300)
    start_date = '2010-01-01'
    

    # yfinance data 불러오기
    stock_data = yf.download(ticker, start=start_date, end=end_date) 
    stock_df = pd.DataFrame(stock_data)

    # yfinance data 전처리
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = stock_df['Date'].dt.strftime('%Y-%m-%d')
    stock_df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    stock_df.drop(columns=['Adj Close', "High", "Low", "Volume"], inplace=True)

    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    #FRED에서 가져올 컬럼들을 딕셔너리화 함
    indicators = {
    'InterestRate': 'FEDFUNDS',  # 미국 단기 이자율 (FRED)
    'VIX': 'VIXCLS',  # VIX (변동성 지수) (FRED)
    'TEDSpread': 'TEDRATE',  # TED 스프레드 (FRED)
    'EFFR': 'EFFR',  # 유효 연방 기금 금리 (FRED)
    }

    #위에서 설정한 딕셔너리 목록을 'pdr.get_data_fred'를 통해 불러옴
    macro_data = {}
    for name, code in indicators.items():
        try:
            macro_data[name] = pdr.get_data_fred(code, start_date, end_date)
            print(f"{name} 데이터 가져오기 성공!")
        except Exception as e:
            print(f"{name} 데이터 가져오기 실패: {e}")

    for name, data in macro_data.items():
        if data is not None and not data.empty:
            print(f"\n{name} 데이터:\n", data.head())
        else:
            print(f"{name} 데이터가 없습니다.")

    # 불러온 FRED데이터는 분기별로 있어 결측치 처리가 필요. 그래서 결측치 처리를 함
    macro_df = pd.concat(macro_data, axis=1)
    macro_df.columns = [col[1] for col in macro_df.columns]  # MultiIndex 열 이름 정리
    macro_df.fillna(method='ffill', inplace=True)
    macro_df.reset_index(inplace=True)
    macro_df.columns = ['Date','InterestRate','VIX','TEDSpread','EFFR']
    macro_df = macro_df.drop(index=0).reset_index(drop=True)
    macro_df['TEDSpread'] = macro_df['TEDSpread'].fillna(0.09)

    #환율,Gold,oil 컬럼을 yfinance를 통해 가져옴
    df_oil = yf.download('CL=F', start=start_date, end=end_date)
    df_usdkrw = yf.download('EURUSD=X', start=start_date, end=end_date)
    df_gold = yf.download('GC=F', start=start_date, end=end_date)

    df_oil.reset_index(inplace=True)
    df_oil['Date'] = df_oil['Date'].dt.strftime('%Y-%m-%d')
    df_oil.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df_oil.drop(columns=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)

    df_usdkrw.reset_index(inplace=True)
    df_usdkrw['Date'] = df_usdkrw['Date'].dt.strftime('%Y-%m-%d')
    df_usdkrw.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df_usdkrw.drop(columns=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)

    df_gold.reset_index(inplace=True)
    df_gold['Date'] = df_gold['Date'].dt.strftime('%Y-%m-%d')
    df_gold.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    df_gold.drop(columns=['Adj Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)


    # Date 열을 기준으로 미국 거시지표 데이터프레임 병합 
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    df_oil['Date'] = pd.to_datetime(df_oil['Date'])
    df_usdkrw['Date'] = pd.to_datetime(df_usdkrw['Date'])
    df_gold['Date'] = pd.to_datetime(df_gold['Date'])

    merged_df = macro_df.merge(df_oil, on='Date', how='inner') \
                        .merge(df_usdkrw, on='Date', how='inner') \
                        .merge(df_gold, on='Date', how='inner')
    merged_df = merged_df.rename(columns={'Close_x': 'Oil', 'Close_y': 'ExchangeRate','Close': 'Gold'})

    #거시지표와 경제지표를 합쳐서 최종적인 데이터프레임으로 병합
    merged_df_end = pd.merge(stock_df, merged_df, on='Date', how='outer')

    #우리가 사용할 LGMBRegressor는 분류, 회귀모델 둘 다 사용가능. 그러나 시계열성은 가지고 있지않음.  
    #다음날 종가를 예측해야하며, 추세를 따라가야하기 때문에 시계열성을 띄는 모델이여야 하나 그건 사용불가
    #강제로 시계열성을 부여한 것이 아래 5줄의 과정, 이때 rolling(252)는 1년치가 주말 공휴일 등을 제외한 날이 약 252일이라 252로 정함
    merged_df_end['Close_InterestRate_Corr'] = merged_df_end['Close'].rolling(252).corr(merged_df_end['InterestRate'])
    merged_df_end['Close_VIX_Corr'] = merged_df_end['Close'].rolling(252).corr(merged_df_end['VIX'])
    merged_df_end['Rolling_Volatility'] = merged_df_end['Close'].rolling(window=30).std()
    merged_df_end['Daily_Return'] = merged_df_end['Close'].pct_change() # 현재와 이전 데이터와 차이의 분수값
    merged_df_end['Rolling_Mean_Close'] = merged_df_end['Close'].rolling(window=30).mean()

    #이제 모든 data의 전처리과정
    merged_df_end['Date'] = pd.to_datetime(merged_df_end['Date'])
    merged_df_end.set_index('Date', inplace=True)
    merged_df_end.columns = merged_df_end.columns.str.replace(' ', '')
    merged_df_end.fillna(method='ffill', inplace=True)
    merged_df_end = merged_df_end.fillna(0)
    


    # Feature와 Target 설정
    X = merged_df_end.drop(['Close'], axis=1)
    y = merged_df_end['Close']

    # 결측치 및 무한 값 처리
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method='ffill', inplace=True)

    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 표준화
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 예측 데이터 #당일의 open값과 전날의 거시지표 + 상관계수 사용 -2인 이유는 -1이 마지막 지표이나 가끔 늦게나오는 현상 대비해 -2로 설정,
    # 원래는 -1로 설정하는게 맞음
    predict_data = merged_df_end.iloc[[-2]]  
    

    # 당일의 시초가를 불러오는 과정
    data = yf.download(ticker, period="1d")

    predict_open = data['Open'].values[0]

    predict_data['Open'] = predict_open

    #당일의 시초가와 거시지표가 합쳐진 data를 scale 
    predict_data.drop(['Close'], axis=1, inplace=True)
    predict_scaled = scaler.transform(predict_data)

    # LGBMRegressor모델 학습
    lgbm_model = LGBMRegressor(learning_rate= 0.1, n_estimators=500, num_leaves= 31)

    # 모델 학습
    lgbm_model.fit(X_train_scaled, y_train)

    # 학습된 모델로 예측
    y_pred_test_lgbm = lgbm_model.predict(predict_scaled)


    # 예측 결과 출력
    print(f'Predicted Close price for {ticker} on Next day: {y_pred_test_lgbm[0]}')

    #아래부턴 예측의 범주이나 실은 실제론 우리가 테스트해봐야 하는 경우이긴 함. data의 부족도 한몫. 
    #추세데이터는 하루차이로 값의 변동이 미약하게라도 반영이 됨. 그래서 조금 다른 부분으로 수정해도 좋을 것 같음. 
    #아래 코드들은 위에 코드랑 다를게 없음. 그저 head와 tail로 나눈 것에 불과함. 학습하는 과정은 같음
    stock_merge_data_head = merged_df_end.iloc[:-10] # 앞쪽 
    stock_merge_data_tail = merged_df_end.iloc[-10:] # 뒷쪽 10개
    
    X = stock_merge_data_head.drop(['Close'], axis=1)  
    y = stock_merge_data_head['Close'] 

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method='ffill', inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Final_X = stock_merge_data_tail.drop(['Close'], axis=1)  
    Final_y = stock_merge_data_tail['Close']  
    Final_X.replace([np.inf, -np.inf], np.nan, inplace=True)
    Final_X.fillna(method='ffill', inplace=True)

    X_Final_scaled = scaler.transform(Final_X)

    lgbm_model = LGBMRegressor(learning_rate= 0.1, n_estimators=500, num_leaves= 31)

    # 모델 학습
    lgbm_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_rf = lgbm_model.predict(X_test_scaled)
    y_pred_Final = lgbm_model.predict(X_Final_scaled)
    
    Final_data = stock_merge_data_tail[['Open','Close']]
    Final_data['Pred_Close'] = y_pred_Final
    Final_data['Close_diff'] = Final_data['Pred_Close'] - Final_data['Open']
    Final_data['diff_per'] = ((Final_data['Close']-Final_data['Open']) / Final_data['Open']) * 100.0
   
    diff_lt = []
    for i in range(len(Final_data)):
        if Final_data['Close_diff'][i] > 0:
            diff_lt.append(Final_data['diff_per'][i])
    max_diff = max(diff_lt)
    min_diff = min(diff_lt)
    print(max_diff)
    print(min_diff)    
                
    
    print(Final_data)
    predict =''
    if (y_pred_test_lgbm[0] - predict_data['Open'].values[0])>0:
        predict = 'UP'
    else:
        predict = 'Down'
    
    return predict_data['Open'].values[0], float(y_pred_test_lgbm[0]),predict, max_diff,min_diff

# flask부분은 저도 많이 몰라서 설명이 미흡합니다..
# API 경로 설정 및 POST 메소드 생성
# 외부에서 요청하면 위에 만들어진 stock_predict가 호출되어 값을 return해주는 과정
@app.route('/predict', methods=['POST'])
def predict():

    # json형식으로 ticker값을 전달받음
    data = request.get_json()
    ticker = data.get('ticker')
    
    # ticker가 잘못되면 오류메세지를 전달
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    
    #stock_predict함수를 호출하고 return 받는 값들을 각각의 변수에 저장
    try:
        open_price, predicted_close, prediction,max_diff,min_diff = stock_predict(ticker)
        result = {
            "Open": open_price,
            "Predicted Close": predicted_close,
            "Prediction": prediction,
            "Max_diff" : max_diff,
            "Min_diff" : min_diff
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#아래는 host를 저리 설정하면 외부에서 제한없이 사용가능
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
