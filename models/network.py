import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.mamba_arch import ASSModule
from models.blocks import ChannelTransfer, ChannelReturn, SGRF

##### Dewater Mamba network  #####
##### input:256*256*3|output:256*256*3
class DWMamba(nn.Module):
    def __init__(self, in_chans=3, dims=None,
                 depths=None, drop_rate=0., d_state=16,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False,
                 **kwargs):
        super(DWMamba, self).__init__()
        if dims is None:
            dims = [24, 48, 96, 48, 24]
        if depths is None:
            depths = [1, 1, 2, 1, 1]
        self.dims = dims
        self.depths = depths

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        #####encoder####
        self.conv1 = nn.Conv2d(in_chans, self.dims[0], kernel_size=1, stride=1)
        self.cf = ChannelTransfer(dim=self.dims[0], norm_layer=norm_layer if patch_norm else None)

        self.assm1 = ASSModule(dim=self.dims[0], depth=self.depths[0],
                         d_state=math.ceil(self.dims[0] / 6) if d_state is None else d_state,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[0:self.depths[0]],
                         norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint)
        self.down1 = nn.Sequential(*[Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2),
                             nn.LayerNorm(4 * self.dims[0]),
                             nn.Linear(4 * self.dims[0], 2 * self.dims[0], bias=False)])

        self.assm2 = ASSModule(dim=self.dims[1], depth=self.depths[1],
                             d_state=math.ceil(self.dims[1] / 6) if d_state is None else d_state,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[self.depths[0]:sum(self.depths[:2])],
                             norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint)
        self.down2 = nn.Sequential(*[Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2),
                      nn.LayerNorm(4 * self.dims[1]),
                      nn.Linear(4 * self.dims[1], 2 * self.dims[1], bias=False),])

        self.assm3 = ASSModule(dim=self.dims[2], depth=self.depths[2],
                             d_state=math.ceil(self.dims[2] / 6) if d_state is None else d_state,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(self.depths[:2]):sum(self.depths[:3])],
                             norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint)

        #####decoder####
        self.up1 = nn.Sequential(*[Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2),
                                   nn.LayerNorm(self.dims[3]),
                                   nn.Linear(self.dims[3], self.dims[3], bias=False)])
        self.sgrf1 = SGRF(dim=self.dims[3], avg_size=4, norm_layer=norm_layer)
        self.assm4 = ASSModule(dim=self.dims[3], depth=self.depths[3],
                             d_state=math.ceil(self.dims[3] / 6) if d_state is None else d_state,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(self.depths[:3]):sum(self.depths[:4])],
                             norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint)

        self.up2 = nn.Sequential(*[Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2),
                    nn.LayerNorm(self.dims[4]),
                    nn.Linear(self.dims[4], self.dims[4], bias=False)])
        self.sgrf2 = SGRF(dim=self.dims[4], avg_size=8, norm_layer=norm_layer)
        self.assm5 = ASSModule(dim=self.dims[4], depth=self.depths[4],
                             d_state=math.ceil(self.dims[4] / 6) if d_state is None else d_state,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(self.depths[:4]):sum(self.depths[:5])],
                             norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint)

        self.cr = ChannelReturn(dim=self.dims[4], norm_layer=norm_layer if patch_norm else None)
        self.conv2 = nn.Conv2d(self.dims[4], in_chans, kernel_size=1, stride=1)

    def forward(self, x, c):
        #####encoder####
        x = self.conv1(x)
        x = self.cf(x)

        skip1 = self.down1(self.assm1(x))
        skip2 = self.down2(self.assm2(skip1))
        x = self.assm3(skip2)

        #####decoder####
        x = self.up1(torch.cat((x, skip2), 3))
        x = self.sgrf1(x, c)
        x = self.assm4(x)
        x = self.up2(torch.cat((x, skip1), 3))
        x = self.sgrf2(x, c)
        x = self.assm5(x)

        x = self.cr(x)
        out = self.conv2(x)

        return out