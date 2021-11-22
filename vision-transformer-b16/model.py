import torch
import torch.nn as nn

def _init_vit_weights(m):#权重初始化
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def drop_path(x, drop_prob: float = 0., training: bool = False):
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
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_c, embed_dim):
        super(PatchEmbed,self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x.shape=[4,3,224,224]
        x = self.proj(x)  # [4,768,14,14]
        x = torch.flatten(x, 2)  # [4,768,196]
        x = torch.transpose(x, 1, 2)  # [4, 196, 768]
        return x

class Attention(nn.Module):
    def __init__(self,
                 embed_dim,   # 输入每个token的dim,768
                 num_heads,#12
                 qkv_bias,):
        super(Attention, self).__init__()
        self.num_heads = num_heads#12
        head_dim = embed_dim // num_heads#768/12=64
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.drop = nn.Dropout(0)
        self.proj = nn.Linear(embed_dim, embed_dim)#最后的b·W
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape#[4, 197, 768]
        qkv = self.qkv(x)#[4, 197, 2304]
        qkv=qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)#[4, 197, 3,12,64]
        qkv=qkv.permute(2, 0, 3, 1, 4)#[3,4,12,197,64]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [4,12,197,64]
        kT=k.transpose(-2,-1)#[4, 12, 64, 197]
        attn = (q @ kT) * self.scale#4, 12, 197, 197]
        attn = attn.softmax(dim=-1)#4, 12, 197, 197]
        attn = self.drop(attn)
        x = (attn @ v)#[4, 12, 197, 64]
        x=x.transpose(1,2)#[4, 197, 12, 64]
        x=x.reshape(B, N, C)#[4, 197, 768]
        x = self.proj(x)
        x = self.drop(x)
        return x

class Mlp_Block(nn.Module):
    def __init__(self, in_features):
        super(Mlp_Block,self).__init__()
        hidden_features = in_features*4#3072
        self.layers=nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(0)
        )
    def forward(self, x):
        x=self.layers(x)
        return x

class Encoder_Block(nn.Module):
    def __init__(self,
                 embed_dim,#768
                 num_heads,#12
                 qkv_bias,
                 DropPath_ratio
                ):
        super(Encoder_Block, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(DropPath_ratio) if DropPath_ratio > 0. else nn.Identity()
        self.mlp = Mlp_Block(in_features=embed_dim)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224,patch_size=16, in_c=3, num_classes=3,
                 embed_dim=768, repeat=12, num_heads=12,  qkv_bias=True,
                    drop_path_ratio=0.
                 ):
        super(VisionTransformer, self).__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)#[4, 196, 768]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches +1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(0)
        layers=[]
        for i in range(repeat):
            layers.append(Encoder_Block(embed_dim=embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,
                                        DropPath_ratio=drop_path_ratio))
        self.blocks = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head1 = nn.Linear(embed_dim, embed_dim)
        self.act=nn.Tanh()
        self.head2 = nn.Linear(embed_dim, num_classes)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)#[B,1,768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x=x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head1(x)
        x=self.act(x)
        x=self.head2(x)
        return x
