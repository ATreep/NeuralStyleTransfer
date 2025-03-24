# Neural Style Transfer
> Try on HF Space: https://huggingface.co/spaces/Treep/NeuralStyleTransfer

This is a simple neural style transfer (NST) demo on VGG-19 in PyTorch.

# Architecture
For some architecture details, this algorithm chooses a shallow CONV layer (7th layer of VGG-19) for capturing more edge details in the content image on computing content loss, and choosing 5 CONV layers in each block to capture details from shapes to colors in the style image on computing style loss, which has a good performance for style transfer through my experiments.

# Demo
With 180 training epochs and resizing input images to 300px:
<img width="933" alt="image" src="https://github.com/user-attachments/assets/e4852098-c2b0-4073-b987-31b6627daf61" />

