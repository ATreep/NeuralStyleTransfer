import torch
from torch import nn, Tensor
from torchvision import  models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
torch.set_default_device(device)
vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device)

model = vgg_model.features
content_layer_num = 7
style_layer_nums = [(2, 2), (7, 3), (16, 1), (25, 1), (34, 2)]

image_size = 128

def generate_noise_image(content_image, noise_ratio = 0.6):
    noise_image = torch.rand(content_image.size()).to(device)
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image


def normalize_image(image):
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    image = (image - cnn_normalization_mean) / cnn_normalization_std

    return image

def img_to_matrix(image):
    loader = transforms.Compose([
    transforms.Resize(128),  # scale imported image
    transforms.ToTensor()])

    image = loader(image).unsqueeze(0)
    image = image.to(device, torch.float)
    return image

def imshow(img):
    img = img.cpu().clone()
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    return img

def gram(A : Tensor):
    _, c, h, w = A.size()
    A = A.view(c, h * w)
    return torch.mm(A, A.t()) / (c * h * w)

def train(content_img, style_img, input_img, alpha, beta, epochs):
    model.eval()
    model.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    G_x = input_img.clone().detach().requires_grad_(True)

    C_x = normalize_image(content_img).detach().requires_grad_(False)
    S_x = normalize_image(style_img).detach().requires_grad_(False)

    C_activation_list = []
    S_activation_list = []

    # LBFGS is better than Adam on NTS
    optimizer = torch.optim.LBFGS([G_x])  # Warning! You should not put normalized G_x into optimizer!

    for layer_num, layer in enumerate(model.children()):
            C_x = layer(C_x)
            S_x = layer(S_x)

            if content_layer_num == layer_num:
                C_activation_list.append(C_x.detach())

            for style_layer_num, _ in style_layer_nums:
                if style_layer_num == layer_num:
                    S_activation_list.append(S_x.detach())
                    break
    epoch = [0]
    while epoch[0] <= epochs:
        def closure():
            with torch.no_grad():
                G_x.clamp_(0, 1)  # If you do not clamp the input before each training, some negative or large values will be exploding and forming noise points finally.

            optimizer.zero_grad()

            G_x_forward = normalize_image(G_x)

            content_loss = 0
            style_loss = 0

            C_list_idx = 0
            S_list_idx = 0

            # Compute content and style loss
            for layer_num, layer in enumerate(model):
                G_x_forward = layer(G_x_forward)

                _, c, h, w = G_x_forward.size()

                if content_layer_num == layer_num:
                    content_loss = nn.functional.mse_loss(C_activation_list[C_list_idx], G_x_forward)
                    C_list_idx += 1

                for style_layer_num, lambd in style_layer_nums:
                    if style_layer_num == layer_num:
                        J_l = nn.functional.mse_loss(gram(S_activation_list[S_list_idx]), gram(G_x_forward))
                        S_list_idx += 1
                        style_loss += lambd * J_l
                        break

                if C_list_idx == len(C_activation_list) and S_list_idx == len(S_activation_list):
                    break

            content_loss *= alpha
            style_loss *= beta

            loss = content_loss + style_loss
            loss.backward()

            epoch[0] += 1

            if epoch[0] % 20 == 0:
                print(f"Epoch = {epoch[0]}, Content Loss = {content_loss.item()}, Style Loss = {style_loss.item()}, Total Loss = {loss.item()}")

            return content_loss + style_loss

        optimizer.step(closure)

    with torch.no_grad():
        G_x.clamp_(0, 1)

    return imshow(G_x)

