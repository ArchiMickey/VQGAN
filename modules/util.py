import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from collections import namedtuple

CONFIG = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)
device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
if device_properties.major == 8 and device_properties.minor == 0:
    cuda_config = CONFIG(True, False, False)
else:
    cuda_config = CONFIG(False, True, True)

# Discriminator related functions


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


# Generator related functions


class GroupNorm(nn.Module):
    def __init__(self, channels, groups):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=groups, num_channels=channels, eps=1e-6, affine=True
        )

    def forward(self, x):
        return self.gn(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups) -> None:
        super(Block, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = GroupNorm(dim_out, groups)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups) -> None:
        super(ResnetBlock, self).__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.block = nn.Sequential(
            Block(dim, dim_out, groups), Block(dim_out, dim_out, groups)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return self.res_conv(x) + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(UpsampleBlock, self).__init__()
        dim_out = dim_out if dim_out else dim
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownsampleBlock(nn.Module):
    def __init__(self, dim, dim_out=None):
        super(DownsampleBlock, self).__init__()
        dim_out = dim_out if dim_out else dim
        self.conv = nn.Conv2d(dim * 4, dim_out, 1)
        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)

    def forward(self, x):
        x = self.rearrange(x)
        return self.conv(x)


class RMSNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )
        q = q * self.scale

        with torch.backends.cuda.sdp_kernel(**cuda_config._asdict()):
            q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=32,
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels

        init_dim = init_dim if init_dim else dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.blocks = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, resnet_block_groups),
                        ResnetBlock(dim_in, dim_in, resnet_block_groups),
                        LinearAttention(dim_in),
                        DownsampleBlock(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )
        self.final_res_block = nn.Sequential(
            ResnetBlock(dim_out, dim_out, resnet_block_groups),
            Attention(dim_out),
            ResnetBlock(dim_out, dim_out, resnet_block_groups),
        )

        out_dim = out_dim if out_dim else dim_out
        self.out_dim = out_dim
        self.final_norm = GroupNorm(dim_out, resnet_block_groups)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(dim_out, out_dim, 1)

    def forward(self, x):
        x = self.init_conv(x)

        for resnet_blocks in self.blocks:
            for block in resnet_blocks:
                if isinstance(block, (Attention, LinearAttention)):
                    x = block(x) + x
                else:
                    x = block(x)

        for block in self.final_res_block:
            if isinstance(block, Attention):
                x = block(x) + x
            else:
                x = block(x)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)


class Decoder(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        init_dim=None,
        dim_div=(1, 2, 4, 8),
        resnet_block_groups=32,
    ):
        super().__init__()

        init_dim = init_dim if init_dim else dim
        self.init_conv = nn.Conv2d(init_dim, dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim // m, dim_div)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.mid_blocks = nn.ModuleList(
            [
                ResnetBlock(dim, dim, resnet_block_groups),
                Attention(dim),
                ResnetBlock(dim, dim, resnet_block_groups),
            ]
        )

        self.blocks = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == (num_resolutions - 1)

            self.blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, resnet_block_groups),
                        ResnetBlock(dim_in, dim_in, resnet_block_groups),
                        ResnetBlock(dim_in, dim_in, resnet_block_groups),
                        LinearAttention(dim_in),
                        UpsampleBlock(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = ResnetBlock(dim_out, dim_out, resnet_block_groups)
        out_dim = out_dim if out_dim else dim_out
        self.final_norm = GroupNorm(dim_out, resnet_block_groups)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(dim_out, out_dim, 1)

    def forward(self, x):
        x = self.init_conv(x)

        block1, attn, block2 = self.mid_blocks
        x = block1(x)
        x = attn(x) + x
        x = block2(x)

        for resnet_blocks in self.blocks:
            for block in resnet_blocks:
                if isinstance(block, (Attention, LinearAttention)):
                    x = block(x) + x
                else:
                    x = block(x)

        x = self.final_res_block(x)
        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)


if __name__ == "__main__":
    from icecream import install

    install()
    encoder = Encoder(32, out_dim=512)
    decoder = Decoder(512, 3)
    x = torch.randn(1, 3, 256, 256)
    z = encoder(x)
    ic(z.shape)
    x_hat = decoder(z)
    ic(x_hat.shape)
