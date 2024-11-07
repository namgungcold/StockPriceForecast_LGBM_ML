#pip install gradio_modal

import gradio as gr
from gradio_modal import Modal

def close_modal():
    return gr.update(visible=False)

with gr.Blocks() as demo:
    with Modal(visible=True,allow_user_close=False) as modal:
            gr.Image("images/start.png")
            agree_btn = gr.Button("START")

    agree_btn.click(close_modal, None, modal)

if __name__ == "__main__":
    demo.launch()