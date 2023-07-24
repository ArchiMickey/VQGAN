from einops import rearrange
import torch.nn as nn
import torch

from modules.util import Encoder, Decoder
from modules.vector_quantize import VectorQuantize


class VQVAE(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        codebook_size=1024,
        codebook_dim=512,
    ):
        super().__init__()

        self.encoder = Encoder(
            dim=dim,
            init_dim=init_dim,
            dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups=resnet_block_groups,
        )

        self.decoder = Decoder(
            dim=self.encoder.out_dim,
            out_dim=out_dim if out_dim else 3,
            init_dim=None,
            dim_div=dim_mults[::-1],
            resnet_block_groups=resnet_block_groups,
        )

        self.codebook = VectorQuantize(
            dim=self.encoder.out_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            beta=0.25,
            commitment_weight=1
        )

    def encode(self, x):
        z = self.encoder(x)
        return {"z": z}

    def decode(self, z_q):
        x_hat = self.decoder(z_q)
        return {"x_hat": x_hat}

    def quantize(self, z):
        b, c, h, w = z.shape
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, sample_idxs, commit_loss = self.codebook(z)
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
        return {
            "z_q": z_q,
            "sample_idxs": sample_idxs,
            "commit_loss": commit_loss,
        }
        

    def forward(self, x):
        ret = {"x": x}

        ret |= self.encode(x)

        ret |= self.quantize(ret["z"])

        ret |= self.decode(ret["z_q"])

        return ret


if __name__ == "__main__":
    from icecream import install

    install()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vqvae = VQVAE(32).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    ret = vqvae(x)
    ic(ret["x"].shape)
