import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelTransfer(nn.Module):
    r"""Transfer 2D feature channel"""

    def __init__(self, dim=48, norm_layer=None):
        super().__init__()
        self.embed_dim = dim

        if norm_layer is not None:
            self.norm = norm_layer(dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x

class ChannelReturn(nn.Module):
    r"""Return 2D feature channel"""

    def __init__(self, dim=48, norm_layer=None):
        super().__init__()
        self.embed_dim = dim
        if norm_layer is not None:
            self.norm = norm_layer(dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class CA(nn.Module):
    r"""Double-pooled CM"""

    def __init__(self, dim=48, r=16):
        super(CA, self).__init__()
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.maxp = nn.AdaptiveMaxPool2d(1)
        self.clrc = nn.Sequential(
            nn.Conv2d(dim, dim // r, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(dim // r, dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        a = self.clrc(self.avgp(x))
        m = self.clrc(self.maxp(x))
        ca = self.sigmoid(a + m)
        out = (x * ca).permute(0, 2, 3, 1).contiguous()
        return out

# Use the avgpool layer to unify the values of each region in the depth map
def region_normal(d, size):
    # Set size to kernel size and stride
    avgpool = nn.AvgPool2d(size, stride=size, padding=0)
    avg_d = avgpool(d)
    # Expand the avgpool value to the size*size region
    matrix3 = torch.ones((size, size))
    uni_d = torch.kron(avg_d, matrix3.to(avg_d.device))
    # Map the extended matrix back to the original matrix
    temp_d = d.clone()
    temp_d[:,:,:uni_d.size(2),:uni_d.size(3)] = uni_d
    d = temp_d.clone()
    return d

class SGRF(nn.Module):
    r"""Structure-guided residual fusion module"""

    def __init__(self, dim=48, avg_size=8, norm_layer=None):
        super(SGRF, self).__init__()
        self.avg_size = avg_size
        self.cr = ChannelReturn(dim=dim, norm_layer=norm_layer)
        self.cs1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.SiLU()
        )
        self.bn = nn.BatchNorm2d(dim)
        self.cs2 = nn.Sequential(
                    nn.Conv2d(dim, dim, 3, 1, 1),
                    nn.SiLU()
        )
        self.cf = ChannelTransfer(dim=dim, norm_layer=norm_layer)

    def forward(self, x, c):
        c_pre = F.interpolate(c, size=[x.size(1), x.size(2)], mode='nearest')
        c = 1.0 + c_pre
        x = self.cr(x)
        x_res = self.cs1(c * x)
        x = self.bn(x_res)
        c_avg = 1.0 - region_normal(c_pre, self.avg_size)
        x = x_res + self.cs2(c_avg * x)
        x = self.cf(x)

        return x