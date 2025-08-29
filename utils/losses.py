import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from math import exp

class Charbonnier_Loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(Charbonnier_Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = torch.add(x, -y)
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class VGG_Loss(nn.Module):
    def __init__(self, n_layers=5):
        super(VGG_Loss, self).__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.cuda())
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().cuda()

    def forward(self, x, y):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            x = layer(x)
            with torch.no_grad():
                y = layer(y)
            loss += weight * self.criterion(x, y)

        return loss

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class SSIM_Loss(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def rgb_to_hsv(img):
    eps = 1e-8

    # Precompute max and min across the color channels
    max_val, max_idx = img.max(1)
    min_val = img.min(1)[0]
    diff = max_val - min_val

    # Initialize hue tensor
    hue = torch.zeros_like(max_val).to(img.device)

    # Calculate hue
    mask_r = (max_idx == 0)
    mask_g = (max_idx == 1)
    mask_b = (max_idx == 2)

    hue[mask_r] = (img[:, 1][mask_r] - img[:, 2][mask_r]) / (diff[mask_r] + eps)
    hue[mask_g] = 2.0 + (img[:, 2][mask_g] - img[:, 0][mask_g]) / (diff[mask_g] + eps)
    hue[mask_b] = 4.0 + (img[:, 0][mask_b] - img[:, 1][mask_b]) / (diff[mask_b] + eps)

    hue = (hue % 6) / 6
    hue[diff == 0] = 0.0

    # Calculate saturation
    saturation = diff / (max_val + eps)
    saturation[max_val == 0] = 0.0

    # Calculate value
    value = max_val

    # Combine hue, saturation, and value
    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)

    hsv = torch.cat([hue, saturation, value], dim=1)

    return hsv

def rgb_to_lab(rgb_image):
    device = rgb_image.device
    # Convert RGB to Lab
    # Assume rgb_image has shape [batch, 3, height, width]
    
    # Normalize RGB values to [0, 1]
    rgb_image = rgb_image.float() / 255.0
    
    # Convert to linear RGB
    linear_rgb = rgb_image.clone()
    mask = (linear_rgb > 0.04045)
    linear_rgb[mask] = ((linear_rgb[mask] + 0.055) / 1.055) ** 2.4
    linear_rgb[~mask] /= 12.92

    # Convert to XYZ
    xyz_image = torch.zeros_like(rgb_image, device=device)
    xyz_image[:, 0, :, :] = 0.4124564 * linear_rgb[:, 0, :, :] + 0.3575761 * linear_rgb[:, 1, :, :] + 0.1804375 * linear_rgb[:, 2, :, :]
    xyz_image[:, 1, :, :] = 0.2126729 * linear_rgb[:, 0, :, :] + 0.7151522 * linear_rgb[:, 1, :, :] + 0.0721750 * linear_rgb[:, 2, :, :]
    xyz_image[:, 2, :, :] = 0.0193339 * linear_rgb[:, 0, :, :] + 0.1191920 * linear_rgb[:, 1, :, :] + 0.9503041 * linear_rgb[:, 2, :, :]

    # Normalize XYZ to reference white
    xyz_image /= torch.tensor([[0.950456, 1.0, 1.088754]], device=device).view(1, 3, 1, 1)

    # Convert to Lab
    epsilon = 0.008856
    kappa = 903.3
    lab_image = torch.zeros_like(xyz_image, device=device)
    mask = (xyz_image > epsilon)
    lab_image[mask] = 116.0 * (xyz_image[mask] ** (1/3)) - 16.0
    lab_image[~mask] = kappa * xyz_image[~mask]

    return lab_image

def extract_vl(img):
    # Get HSV representations
    hsv = rgb_to_hsv(img)
    lab = rgb_to_lab(img)

    # Extract V from HSV and L from Lab
    v = hsv[:, 2, :, :]
    l = lab[:, 0, :, :]

    return v, l

class VL_Loss(nn.Module):
    def __init__(self):
        super(VL_Loss, self).__init__()
        self.eps = 1e-8

    def forward(self, x, y):
        xv, xl = extract_vl(x)
        yv, yl = extract_vl(y)
        diff_v = xv - yv
        diff_l = xl - yl
        loss = 0.3*torch.mean(torch.sqrt(diff_v ** 2 +  self.eps ** 2))+torch.mean(torch.sqrt(diff_l ** 2 + self.eps ** 2))
        return loss