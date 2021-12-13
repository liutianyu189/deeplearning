import torch
import torch.nn as nn
import math

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def create_mask( x, H, W,window_size=7):
    img_mask = torch.zeros((1, H, W, 1), device=x.device)  # [1, 56, 56, 1]
    shift_size=window_size//2
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)  #[64,7,7,1]
    mask_windows = mask_windows.view(-1, window_size *window_size)  # [64, 49]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]=[64,49,49]
    # [nW, Mh*Mw, Mh*Mw]
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=96):
        super(PatchEmbed,self).__init__()
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=4, stride=4)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self, x):
        #x.shape [B,3,224,224]
        x = self.proj(x)#x.shape#[B,96,56,56]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)#[B,3136,96]
        return x

class PatchMerging(nn.Module):
    def __init__(self,dim):
        super(PatchMerging,self).__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
    def forward(self, x):
        B, L, C = x.shape#[B,3136,96]
        W=int(math.sqrt(L))
        x = x.view(B,W, W,C)#[B,56,56,96]
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C],表示从下标为（0，0）的元素，隔一个元素，读取一个元素
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]，表示从下标为（1，0）的元素，隔一个元素，读取一个元素
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]表示从下标为（0，1）的元素，隔一个元素，读取一个元素
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]表示从下标为（1，1）的元素，隔一个元素，读取一个元素
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, 28, 28, 4*96]
        x = x.view(B, -1, 4 * C)  # [B, 28*28, 4*96]
        x = self.norm(x)
        x = self.reduction(x)  # [B, 768, 192]
        return x

class Mlp(nn.Module):
    def __init__(self, in_features):
        super(Mlp,self).__init__()
        hidden_features=in_features*4
        self.conv=nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(hidden_features,in_features),
            nn.Dropout(0.)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads,size,shift_size,window_size=7):
        super(WindowAttention,self).__init__()
        self.window_size = window_size  #7
        self.num_heads = num_heads#3
        self.size=size
        self.scale = (dim // num_heads) ** -0.5
        self.shift_size=shift_size#判断是否使用SW——MSA

        #偏执表创建
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # [2*7-1 * 2*7-1, 3]
        #给偏执表赋值
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        #创建偏执矩阵
        a = torch.arange(window_size)
        b = torch.arange(window_size)
        c = torch.meshgrid([a, b])
        c = torch.stack(c)#[2, 7, 7]
        c = torch.flatten(c, 1)# [2, 49]
        c = c[:, :, None] - c[:, None, :]#[2, 49, 49]
        c = c.permute(1, 2, 0).contiguous()#[49, 49, 2]
        c[:, :, 0] += window_size - 1
        c[:, :, 1] += window_size - 1
        c[:, :, 0] *= 2 * window_size - 1
        c = c[:, :, 0] + c[:, :, 1]# [49, 49]
        self.relative_position_index = c

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_numw, N, C = x.shape#[64*4,7*7,96]
        qkv = self.qkv(x)#[64*4,7*7,288]
        qkv=qkv.reshape( B_numw, N, 3, self.num_heads, C // self.num_heads)#[64*4,7*7,3,3,32]
        qkv=qkv.permute(2, 0, 3, 1, 4)#[3,64*4,3,7*7,32]
        q, k, v = qkv.unbind(0)#[64*4,3,7*7,32]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))#[64*4,3,49,49]

        #从偏执表中取值赋给偏执矩阵
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)#[49,49,3]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [3, 7*7, 7*7]
        #加上偏执矩阵
        attn = attn + relative_position_bias.unsqueeze(0)#[64*4,3,7*7,7*7]

        #判断W-MSA还是SW-MSA
        if self.shift_size >0:
            attn_mask = create_mask(x, self.size, self.size)  # [64,49,49]
            nW = attn_mask.shape[0]  # 64
            attn = attn.view( B_numw // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)#[4,64,3,49,49]
            attn = attn.view(-1, self.num_heads, N, N)#[64*4,3,49,49]
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)#[64*4,3,49,49]

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_numw, N, C)#[64*4,7*7,96]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x#[64*4,7*7,96]

class Block1(nn.Module):
    def __init__(self, dim, num_heads,size,window_size=7,shift_size=0):
        super(Block1,self).__init__()
        self.dim = dim#96
        self.size=size
        self.num_heads = num_heads#3
        self.window_size = window_size#7
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=7, num_heads=num_heads,shift_size=shift_size,size=size)
        self.drop_path = DropPath(0)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim)
    def forward(self, x):
        B, L, C = x.shape#[4,56*56,96]
        shortcut = x#[4,56*56,96]
        x = self.norm1(x)
        x = x.view(B, self.size, self.size, C)#[4,56,56,96]
        B, H, W, C = x.shape#[4,56,56,96]
        x = window_partition(x, self.window_size)# [64*4, 7, 7, 96]
        x = x.view(-1, self.window_size * self.window_size, C)  # [64*4, 7*7, 96]
        x = self.attn(x)
        x = x.view(-1, self.window_size, self.window_size, C)  # [64*4, 7, 7, 96]
        x = window_reverse(x, self.window_size, H, W)  # [4, 56, 56, 96]
        x = x.view(B, self.size* self.size, C)#[4,56*56,96]
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x#[4,56*56,96]

class Block2(nn.Module):
    def __init__(self, dim, num_heads,size,window_size=7,shift_size=3):
        super(Block2,self).__init__()
        self.dim = dim#96
        self.size=size
        self.num_heads = num_heads#3
        self.window_size = window_size#7
        self.shift_size = shift_size#0or3
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=7, num_heads=num_heads,shift_size=shift_size,size=size)
        self.drop_path = DropPath(0)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim)
    def forward(self, x):
        B, L, C = x.shape#[4,56*56,96]
        shortcut = x#[4,56*56,96]
        x = self.norm1(x)
        x = x.view(B, self.size, self.size, C)#[4,56,56,96]
        B, H, W, C = x.shape#[4,56,56,96]
        x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))#[4,56,56,96]
        x = window_partition(x, self.window_size)  # [64*4, 7, 7, 96]
        x = x.view(-1, self.window_size * self.window_size, C)  # [64*4, 7*7, 96]
        x = self.attn(x)  # [64*4, 7*7, 96]
        x = x.view(-1, self.window_size, self.window_size, C)  # [64*4, 7, 7, 96]
        x = window_reverse(x, self.window_size, H, W)  # [4, 56, 56, 96]
        x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, self.size* self.size, C)#[4,56*56,96]
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x#[4,56*56,96]

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1000):
        super(SwinTransformer,self).__init__()
        shift_size = 3# 3
        self.patch_embed = PatchEmbed(in_c=3, embed_dim=96)
        self.drop = nn.Dropout(0)
        self.layers = nn.ModuleList(
            [Block1(dim=96, num_heads=3, window_size=7, shift_size=0, size=56),
            Block2(dim=96, num_heads=3, window_size=7, shift_size=shift_size, size=56),
            PatchMerging(dim=96),
            Block1(dim=192, num_heads=6, window_size=7, shift_size=0, size=28),
            Block2(dim=192, num_heads=6, window_size=7, shift_size=shift_size, size=28),
            PatchMerging(dim=192),
            Block1(dim=384, num_heads=12, window_size=7, shift_size=0, size=14),
            Block2(dim=384, num_heads=12, window_size=7, shift_size=shift_size, size=14),
            Block1(dim=384, num_heads=12, window_size=7, shift_size=0, size=14),
            Block2(dim=384, num_heads=12, window_size=7, shift_size=shift_size, size=14),
            Block1(dim=384, num_heads=12, window_size=7, shift_size=0, size=14),
            Block2(dim=384, num_heads=12, window_size=7, shift_size=shift_size, size=14),
            PatchMerging(dim=384),
            Block1(dim=768, num_heads=24, window_size=7, shift_size=0, size=7),
            Block2(dim=768, num_heads=24, window_size=7, shift_size=shift_size, size=7)]
        )
        self.norm = nn.LayerNorm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(768, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x = self.patch_embed(x)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
#
# model=SwinTransformer(num_classes=3)
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.size())
# params = list(model.parameters())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i.size())))
#     for j in i.size():
#         l *= j
#     print("该层参数和：" + str(l))
#     k = k + l
# print("总参数数量和：" + str(k))
