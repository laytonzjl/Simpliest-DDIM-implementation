import torch
import math
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract(v, t, x_shape):
    t = t.to(device)
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class AddNoise(nn.Module):
    def __init__(self, T: int = 1000, start: float = 0.0001, end: float = 0.02):
        super().__init__()
        self.T = T

        # 线性插值计算 beta 值
        betas = torch.linspace(start, end, T).to(device)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # 计算扩散过程所需的 alpha_bar 和 sqrt 值
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1 - alphas_bar))

    def forward(self, x_0, t):
        x_0 = x_0.to(device)
        noise = torch.randn_like(x_0).to(device)

        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        return x_t, noise


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T: int, d_model: int, dim: int):
        super().__init__()
        self.T = T
        self.d_model = d_model

        TimeMatrix = torch.zeros(T, d_model)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        ratio = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))

        TimeMatrix[:, 0::2] = torch.sin(position * ratio)
        TimeMatrix[:, 1::2] = torch.cos(position * ratio)
        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(TimeMatrix),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.embedding(x)


# 图像由(B,C,H,W)->(B,C,H/2,W/2)
class DownCOV(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.cov = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x, temb):
        return self.cov(x)


# 图像由(B,C,H/2,W/2)->(B,C,H,W)
class UpCOV(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.cov = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x, temb):
        _, _, h, w = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.cov(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.projection_q = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.projection_k = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.projection_v = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.projection = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        b, c, H, W = x.shape
        x_o = x
        x = self.group_norm(x)
        q = self.projection_q(x)
        k = self.projection_k(x)
        v = self.projection_v(x)
        # [b, h*w, c]*[b, c, h*w]->[b, h*w, h*w], [b, h*w, h*w]*[b, h*w, c]->[b, h*w, c]->[b, c, h, w]
        q = q.permute(0, 2, 3, 1).view(b, H*W, c)
        k = k.view(b, c, H*W)
        w = torch.bmm(q, k)/(int(c)**0.5)
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(b, H*W, c)
        x = torch.bmm(w, v)
        x = x.view(b, H, W, c).permute(0, 3, 1, 2)
        x = self.projection(x)

        return x_o + x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        self.time_emb = nn.Sequential(
            Swish(),
            nn.Linear(t_dim, out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttentionBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        # [B, out_ch]->[B, out_ch, H, W](自动匹配h,w)， 对于每一个channel都加对应的emb值
        h += self.time_emb(t_emb)[:, :, None, None]
        h = self.block2(h)
        # 残差连接
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    # t是时间， ch是初始通道（128），ch_multi是一维倍数数组[1,2,2,2]
    def __init__(self, t, ch, ch_multi, attn, num_resblock, dropout):
        super().__init__()
        # 更高的维度的t_dim可以表达更丰富的信息，ResBlock里面的time_emb模块会自动匹配维度
        t_dim = ch*4
        self.time_embedding = TimeEmbedding(t, ch, t_dim)

        self.head = nn.Conv2d(3, ch, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.downblock = nn.ModuleList()
        ch_save = []
        now_ch = ch
        ch_save.append(now_ch)
        for i, multi in enumerate(ch_multi):
            out_ch = ch * multi
            for _ in range(num_resblock):
                self.downblock.append(ResBlock(now_ch, out_ch, t_dim, dropout, (i in attn)))
                now_ch = out_ch
                ch_save.append(now_ch)
            if i != (len(ch_multi)-1):
                self.downblock.append(DownCOV(now_ch))
                ch_save.append(now_ch)

        self.middleblock = nn.Sequential(
            ResBlock(now_ch, now_ch, t_dim, dropout, True),
            ResBlock(now_ch, now_ch, t_dim, dropout, False)
        )

        self.upblock = nn.ModuleList()
        for i, multi in reversed(list(enumerate(ch_multi))):
            out_ch = ch * multi
            for _ in range(num_resblock + 1):
                self.upblock.append(ResBlock(now_ch+ch_save.pop(), out_ch, t_dim, dropout, (i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblock.append(UpCOV(now_ch))

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, x, t):
        x = x.to(device)
        t = t.to(device)
        h_for_resconnet = []
        t_emb = self.time_embedding(t)

        h = self.head(x)
        h_for_resconnet.append(h)

        for layer in self.downblock:
            h = layer(h, t_emb)
            h_for_resconnet.append(h)

        for layer in self.middleblock:
            h = layer(h, t_emb)

        for layer in self.upblock:
            # 只有ResBlock需要拼接
            if isinstance(layer, ResBlock):
                h = torch.cat([h, h_for_resconnet.pop()], dim=1)
            h = layer(h, t_emb)

        h = self.tail(h)

        return h











