import gradio as gr
from a1111_api import txt2img

def generate_image(prompt, neg, steps, cfg, width, height, seed):
    return txt2img(
        prompt=prompt,
        negative_prompt=neg,
        steps=int(steps),
        cfg_scale=float(cfg),
        width=int(width),
        height=int(height),
        seed=int(seed),
    ).images[0]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            neg = gr.Textbox(label="Negative Prompt")
            steps = gr.Slider(1, 50, value=20,label="Steps")
            cfg = gr.Slider(1, 15, value=9,label="CFG")
            width = gr.Number(value=512,label="Width")
            height = gr.Number(value=512,label="Height")
            seed = gr.Number(value=-1,label="Seed")
        with gr.Column():
            btn = gr.Button("Generate")
            out = gr.Image()
            btn.click(
                generate_image,
                inputs=[prompt, neg, steps, cfg, width, height, seed],
                outputs=out
            )

demo.launch(server_port=7862)