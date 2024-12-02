# Stock Price Forecasting with LGBM Model

본 프로젝트는 Regressor Model과 Finance Data를 활용하여 주식 시장 추세를 분석하고 예측하는 시스템을 제공합니다. `Flask`를 사용한 RESTful API, `Gradio` 기반의 사용자 인터페이스, 그리고 `LGBM` 모델을 활용하여 당일 종가 예측 및 데이터 시각화를 수행합니다.   
(본 프로젝트는 MS AI SCHOOL 5기 1차 프로젝트를 진행하며 개발되었습니다. 프로젝트 기간: 10/31 ~ 11/8)
![발표사진](https://github.com/user-attachments/assets/bff4527d-d89b-44e0-92a6-02ee91fa1666)

---

## 주요 기능

### 1. 데이터 수집 및 전처리
- **주식 데이터**:
  - 해외 주식을 대상으로 데이터 수집
  - `yfinance`라이브러리를 활용하여 Open, High, Low, Close, Volume 데이터를 수집.
  - 추가적으로 환율, 금, 유가 데이터를 수집하여 분석에 활용.
- **거시 경제 지표**:
  - `FRED`라이브러리를 통해 Interest Rate, VIX, TEDSpread 등의 데이터를 수집.
  - 주식, 거시경제지표 데이터를 결측치 처리 및 전처리하여 학습 데이터로 병합.
  - 사용한 거시 지표는 대부분 변동이 급변하지 않으므로 결측치는 앞의 값을 따라가도록 하게 함.  
- **실시간 데이터**:
  - `TradingView`에서 실시간 상승률 상위 종목 및 거래량 상위 종목 데이터를 수집.
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/c792308c-e7fb-43c6-9150-dc0824d1630f" alt="Image 1" width="49%"/>
  <img src="https://github.com/user-attachments/assets/cc9f38b1-62f3-4e9e-91c5-2b3164593dff" alt="Image 2" width="49%"/>
</div>

### 2. 머신러닝 기반 예측
- **모델 사용**:
  - `LGBM` 모델을 활용하여 주식 종가를 예측.
  - 다른 머신러닝 Regressor Model들을 활용했으나 평가지표로 사용한 RMSE값이 가장 작은 `LGBM`을 선택함. 
- **분석 지표 추가**:
  - 본 프로젝트에서 사용할 수 있는 머신러닝 Regressor Model은 시계열 특성이 없어 지표를 추가하기로 함.    
  - Daily_Return, Rolling_Mean_Close, 상관계수 등 특성 추가.
  - 이동평균과 이동표준편차를 사용해 시계열적 특성을 반영.<br>
![web_2](https://github.com/user-attachments/assets/54bb30e3-e870-42e3-896b-4b8174a0705c)

### 3. API 제공
- **Flask RESTful API**:
  - `backend/stock_backend.py`에서 Flask 서버를 실행하여 `/predict` 엔드포인트를 제공합니다.
  - JSON 응답 예시:
    ```json
    {
      "Open": 135.67, ##시초가
      "Predicted Close": 140.45, ##예측 당일 종가
      "Prediction": "UP", ##예측값
      "Max_diff": 2.5, ##예측한 최대 수익률
      "Min_diff": -1.3 ##예측한 최소 수익
    }
    ```

### 4. 시각화 및 사용자 인터페이스
- **Gradio 기반 웹 UI**:
  - `frontend/index.py`를 통해 종목을 검색하여 예측 결과와 차트를 확인 가능.
  - 상승/하락 여부에 따라 시각적 표시.
- **Plotly 및 Matplotlib 차트**:
  - 주식 종가 및 변동성 차트를 생성하여 사용자에게 제공.

---

## 프로젝트 구조

```plaintext
STOCKPRICEFORECAST_REGRESSION_ML/
│
├── .gradio/                    # Gradio 설정 파일
├── .vscode/                    # VS Code 디버깅 설정
│   └── launch.json             # Flask와 Gradio를 동시에 실행하기 위한 설정
├── backend/                    # 백엔드 모듈
│   ├── exchangerate.py         # 네이버 환율 데이터 크롤러
│   ├── stockchart.py           # TradingView 데이터 크롤러
│   └── stock_backend.py        # Flask 서버 및 데이터 처리
│
├── data/                       # 데이터 디렉터리
│   └── available_tickers.csv   # 사용 가능한 주식 티커 목록
│
├── frontend/                   # 프론트엔드 리소스
│   ├── images/                 # UI 이미지 리소스
│   └── index.py                # Gradio 인터페이스 및 실행 파일
│
├── project_common/             # 공통 모듈
│   ├── my_yfinance.py          # 주식 데이터 수집 및 전처리
│
├── requirements.txt            # Python 패키지 목록
└── README.md                   # 프로젝트 설명 파일
```

---

