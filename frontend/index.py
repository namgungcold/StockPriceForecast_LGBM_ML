import sys
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from io import BytesIO
from PIL import Image
import random
import base64
import platform
from matplotlib import font_manager, rc

# Add common module to path
sys.path.append(os.path.abspath(os.path.join('..')))
from common.yfinance import *

import random
import base64

def set_korean_font():
    if platform.system() == 'Windows':
        # Windows의 경우
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
    elif platform.system() == 'Darwin':
        # macOS의 경우
        rc('font', family='AppleGothic')
    else:
        # 기타 OS의 경우
        rc('font', family='NanumGothic')

# 한글 폰트 설정
set_korean_font()


# 백엔드 데이터 연동
from backend.stockchart import *
from backend.exchangerate import *
from gradio_modal import Modal

daily = dailyChart()
daily[0].head(10)
daily[1].head(10)

# 사용 가능한 티커 목록 가져오기
available_tickers = pd.read_csv('../data/available_tickers.csv')['tickers'].str.upper().tolist()
random.shuffle(available_tickers)

# 이미지 파일을 base64로 인코딩
with open("images/bg_04.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

with open("images/up.png", "rb") as image_file:
    encoded_up_image = base64.b64encode(image_file.read()).decode()

with open("images/down.png", "rb") as image_file:
    encoded_down_image = base64.b64encode(image_file.read()).decode()
    
# Gradio 인터페이스 설정
css_code = f"""
.gradio-container {{
    background-image: url(data:image/png;base64,{encoded_string});
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.8;
}}

.custom-markdown h1, p {{
    color: white;
    text-shadow: -2px 0px black, 0px 2px black, 2px 0px black, 0px -2px black;
    text-align: center;
}}

div.custom-title > div > p {{
    color: white;
}}

div.small-font {{
    height: 170px !important;
    overflow-y: auto !important;
}}

div.small-font::-webkit-scrollbar {{
  display: none;
}}

div.small-font span {{
    font-size: 12px;
}}

.table-wrap {{
    max-height: 400px !important;
    overflow-y: auto !important;
}}

.table-wrap::-webkit-scrollbar {{
  display: none;
}}

.predict_wrapper {{
    display: flex;
    justify-content: space-between;
}}

.predict_result {{
    width: 50%;
    height: 300px;
    background-color: white;
    border-radius: 10px;
    margin-left: 10px;
    padding-bottom: 5px;
}}

.predict_result_up {{
    width: 50%;
    height: 300px;
    border-radius: 10px;
    background-image: url(data:image/png;base64,{encoded_up_image});
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 1;
}}

.predict_result_down {{
    width: 50%;
    height: 300px;
    border-radius: 10px;
    background-image: url(data:image/png;base64,{encoded_down_image});
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 1;
}}

.predict_ticker, .predict_price, .start_price {{
    padding: 20px 10px 20px 10px;
    font-size: 20px;
}}

span.price_down {{
    font-size: 2.5rem;
    color: blue;
    font-weight: 900;
}}

span.price_up {{
    font-size: 3rem;
    color: red;
    font-weight: 700;
}}

.max_diff {{
    font-size: 1.5rem;
    font-weight: 700;
    color: yellow;
    padding: 2px;
    rounded: 2px;
    background-color: red;
}}

.min_diff {{
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    padding: 2px;
    rounded: 2px;
    background-color: blue;
}}
"""

import requests

# 요청 URL
# url = "http://121.162.37.155:5000/predict"
url = "http://127.0.0.1:5001/predict"

# 요청 헤더 설정
# headers = {
#     "Content-Type": "application/json"
# }

# # 요청 본문 설정 (필요한 데이터로 수정)
# payload = {
#     # API에서 요구하는 데이터를 여기에 추가하세요.
#     "ticker": "AAPL"
# }

# try:
#     # POST 요청 보내기
#     response = requests.get(url, json=payload, headers=headers)
    
#     # 요청이 성공했는지 확인
#     response.raise_for_status()
    
#     # JSON 응답 받기
#     json_response = response.json()
#     print("응답 받은 JSON:", json_response)
    
# except requests.exceptions.HTTPError as http_err:
#     print(f"HTTP 에러 발생: {http_err}")
# except Exception as err:
#     print(f"기타 에러 발생: {err}")

import plotly.graph_objects as go

# 주식 데이터를 가져와 Plotly 차트로 변환하는 함수
def get_stock_chart(ticker):
    # 주식 데이터 다운로드 (최근 6개월)
    data, img = get_yfinance_stock_data(ticker)

    data = data.sort_values(by="Date")
    
    # Plotly 라인 차트 생성
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="종가"))
    fig.add_trace
    fig.update_layout(
        title=f"{ticker.upper()} 주식 종가 라인 차트",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_white"  # 밝은 테마
    )

    return fig

def info_fn():
    information="""
    [오른쪽 X를 눌러 창 닫기]<br>
    < 분석 기준 ><br>
    - open(시작가)<br>
    - close(종가)<br>
    - InterestRate(금리)<br>
    - ExchangeRate(1유로 기준 달러 환율)<br>
    - VIX(변동성 지수(=공포 지수))<br>
    - TEDSpread(은행 간 대출 금리와 단기 미국 국채 금리 간의 차이)<br>
    - EFFR(예금 기관들이 서로에게 하룻밤 동안 예비 잔고를 대출할때 적용되는 금리)<br>
    - Gold(금 가격)<br>
    - Oil(유가)<br>
    - Close_InterestRate_Corr(종가와 이자율의 상관계수)<br>
    - Close_VIX_Corr(종가-공포지수 상관계수)<br>
    - Daily_Return(일일 수익률)<br>
    - Volatility(30일 이동평균편차)<br>
    - Rolling_Mean_Close(30일 이동평균)<br><br>

    < 분석 기준 출처 ><br>
    - open,close,Gold, Oil = YFinance<br>
    - InterestRate, ExchangeRate, VIX,<br> TEDSpread, EFFR = FRED<br>
    - 달러 환율=네이버 은행고시환율(하나은행)<br><br>

    < 분석 기간 ><br>
    2010-01-01 ~ 2024-11-07<br><br>

    < 분석 모델 ><br> 
    LGBMRegressor<br><br>
    """
    gr.Info(information,duration=None,title="")

def close_modal():
    return gr.update(visible=False)

with gr.Blocks(gr.themes.Monochrome(), css=css_code) as demo:

    with Modal(visible=True,allow_user_close=False) as modal:
            gr.Image("images/start_mobile.png", show_label=False, show_download_button=False, show_fullscreen_button=False)
            agree_btn = gr.Button("START")

    agree_btn.click(close_modal, None, modal)

    exchange_rate = exchangerate()

    with gr.Row():
        trigger_info = gr.Button(value="분석 관련 정보")
        gr.Markdown("# 천하제일 단타대회", elem_classes="custom-markdown")
        gr.Markdown(f"1 USD: {exchange_rate}", elem_classes="custom-markdown")
        
    trigger_info.click(info_fn, None, None)

    daily = dailyChart()
    volumn_top10 = daily[0].head(10)
    percent_top10 = daily[1].head(10)

    with gr.Row():
        with gr.Column(scale=1):
            gr.DataFrame(percent_top10, label="실시간 상승 주식", elem_classes="custom-title small-font")
        with gr.Column(scale=1):
            gr.DataFrame(volumn_top10, label="실시간 거래량 많은 주식", elem_classes="custom-title small-font")

    previous_tickers = gr.State(set())
    gr.Markdown("# 주식 가격 예측", elem_classes="custom-markdown")
    previous_tickers_display = gr.Radio(label="이전 조회된 티커", choices=[], interactive=True, visible=False)
    ticker_input = gr.Textbox(label="주식 티커(종목 코드) 입력", placeholder=f"예: {', '.join(available_tickers[:5])} 등")
    warning_message = gr.Markdown(value="", visible=False)
    with gr.Row():
        submit_btn = gr.Button("조회")
        reset_btn = gr.Button("초기화")

    with gr.Row():
        predict_result_up = gr.HTML(
            """
            <div class='predict_wrapper'>
                <div class='predict_result_up'>

                </div>
                <div class='predict_result'>
                    {ticker}
                </div>
            </div>
            """,
            visible=False
        )
        predict_result_up = gr.HTML(
            """
            <div class='predict_wrapper'>
                <div class='predict_result_up'>

                </div>
                <div class='predict_result'>
                    {ticker}
                </div>
            </div>
            """,
            visible=False
        )

        predict_result_up = gr.HTML(
            """
            <div class='predict_wrapper'>
                <div class='predict_result_up'>

                </div>
                <div class='predict_result'>
                    {ticker}
                </div>
            </div>
            """,
            visible=False
        )

        predict_result_down = gr.HTML(
            """
            <div class='predict_wrapper'>
                <div class='predict_result_down'>

                </div>
                <div class='predict_result'>
                    {ticker}
                </div>
            </div>
            """,
            visible=False
        )

    chart_output2 = gr.Plot(label="주식 차트", visible=False)
    
    chart_output = gr.Image(label="주가 차트", elem_classes="custom-markdown", visible=False)
    data_output = gr.Dataframe(label="일별 주가 데이터", elem_classes="custom-title", visible=False)

    def update_visibility(ticker, previous_tickers):
        if not ticker:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, None, gr.update(choices=list(previous_tickers), visible=bool(previous_tickers)), gr.update(value="티커를 입력하세요.", visible=True, elem_classes="custom-markdown")

        ticker = ticker.upper()  # Convert ticker to uppercase
        # if ticker.upper() not in available_tickers:
        #     return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, None, gr.update(choices=list(previous_tickers), visible=bool(previous_tickers)), gr.update(value="입력하신 종목의 가격 예측은 추후에 제공될 예정입니다..", visible=True)

        data, img, isUp, open, close, prediction, max_diff, min_diff = predict_stock_price3(ticker)
        open = round(open, 2)
        close = round(close, 2)
        max_diff = round(max_diff, 2)
        min_diff = round(min_diff, 2)

        previous_tickers.add(ticker)
        return (
            gr.update(visible=True, label=f"일별 주가 데이터 [{ticker}]", elem_classes="custom-title"),
            gr.update(visible=False),
            gr.update(visible=isUp, value=f"""
            <div class='predict_wrapper'>
                <div class='predict_result_up'>

                </div>
                <div class='predict_result'>
                    <div class='predict_ticker'>
                        [ {ticker} ]<br>시가 : {open}
                    </div>  
                    <div class='predict_price'>
                        예측 종가: {prediction}<br><span class='price_up'>{close}(▲)</span><br>
                        지난 10거래일 수익률 <span class='max_diff'>{max_diff}%</span>~<span class='min_diff'>{min_diff}%</span>
                    </div>
                </div>
            </div>
            """),
            gr.update(visible=not isUp, value=f"""
            <div class='predict_wrapper'>
                <div class='predict_result_down'>

                </div>
                <div class='predict_result'>
                    <div class='predict_ticker'>
                        [ {ticker} ]<br>시가 : {open}
                    </div>  
                    <div class='predict_price'>
                        예측 종가: {prediction}<br><span class='price_down'>{close}(▼)</span><br>
                        지난 10거래일 수익률 <span class='max_diff'>{max_diff}%</span>~<span class='min_diff'>{min_diff}%</span>
                    </div>
                </div>
            </div>
            """),
            data,
            gr.update(visible=True),
            gr.update(choices=list(previous_tickers), visible=True),
            gr.update(visible=False)
        )

    def reset_fields():
        return "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def update_ticker_input(selected_ticker):
        fig = get_stock_chart(selected_ticker)
        return selected_ticker, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value="", visible=False), gr.update(visible=False), fig

    # def auto_submit_ticker(selected_ticker, previous_tickers):
    #     return update_visibility(selected_ticker, previous_tickers)      

    def auto_submit_ticker(selected_ticker, previous_tickers):
        return update_visibility(selected_ticker, previous_tickers)

    # Gradio에서 입력과 출력을 연결
    ticker_input.submit(get_stock_chart, inputs=ticker_input, outputs=chart_output2)
    previous_tickers_display.change(update_ticker_input, inputs=previous_tickers_display, outputs=chart_output2)

    submit_btn.click(update_visibility, inputs=[ticker_input, previous_tickers], outputs=[data_output, chart_output, predict_result_up, predict_result_down, data_output, chart_output2, previous_tickers_display, warning_message])
    ticker_input.submit(update_visibility, inputs=[ticker_input, previous_tickers], outputs=[data_output, chart_output, predict_result_up, predict_result_down, data_output, chart_output2, previous_tickers_display, warning_message])
    reset_btn.click(reset_fields, inputs=None, outputs=[ticker_input, data_output, chart_output, predict_result_up, predict_result_down, warning_message, chart_output2])
    # previous_tickers_display.change(update_ticker_input, inputs=previous_tickers_display, outputs=[ticker_input, data_output, chart_output, predict_result_up, predict_result_down, warning_message, chart_output2])
    # previous_tickers_display.change(auto_submit_ticker, inputs=[previous_tickers_display, previous_tickers], outputs=[data_output, chart_output, predict_result_up, predict_result_down, data_output, chart_output2, previous_tickers_display, warning_message])

    # 이벤트 바인딩 수정
    previous_tickers_display.change(
        update_ticker_input, 
        inputs=previous_tickers_display, 
        outputs=[ticker_input, data_output, chart_output, predict_result_up, predict_result_down, warning_message, previous_tickers_display, chart_output2]
    )

    previous_tickers_display.change(
        auto_submit_ticker, 
        inputs=[previous_tickers_display, previous_tickers], 
        outputs=[data_output, chart_output, predict_result_up, predict_result_down, data_output, chart_output2, previous_tickers_display, warning_message]
    )

# Gradio 앱 실행
demo.launch()
# demo.launch(share=True)