import gradio as gr
import nst

def style_transfer(content_img, style_img, epochs, style_weight=1e6, content_weight=1):
  return nst.train(
      nst.img_to_matrix(content_img),
      nst.img_to_matrix(style_img),
      nst.generate_noise_image(nst.img_to_matrix(content_img)),
      content_weight,
      style_weight,
      epochs
  )

demo = gr.Interface(
    fn=style_transfer,
    inputs=[gr.Image(type="pil"), gr.Image(type="pil"), gr.Number(value=40, label="Number of Epochs")],
    outputs=gr.Image(type="pil"),
    title="Neural Style Transfer",
    description="Upload a content image and a style image to generate a stylized output.",
)

demo.launch()