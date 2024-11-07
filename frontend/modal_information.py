import gradio as gr
from gradio_modal import Modal

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
    2010-01-01 ~ 2024-11-06<br><br>

    < 분석 모델 ><br> 
    RandomForestRegression<br><br>
    """
    gr.Info(information,duration=None,title="")

with gr.Blocks() as demo:
    with gr.Row():
        trigger_info = gr.Button(value="분석 관련 정보")

        trigger_info.click(info_fn, None, None)

if __name__ == "__main__":
    demo.launch()