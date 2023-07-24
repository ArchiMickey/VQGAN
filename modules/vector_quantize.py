import torch
from torch import nn


class VectorQuantize(nn.Module):
    def __init__(
        self, dim, codebook_size, codebook_dim=None, beta=0.25, commitment_weight=1
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim if codebook_dim else dim
        self.beta = beta
        self.commitment_weight = commitment_weight

        self.quant_proj = (
            nn.Linear(self.dim, self.codebook_dim)
            if self.dim != self.codebook_dim
            else nn.Identity()
        )
        self.post_quant_proj = (
            nn.Linear(self.codebook_dim, self.dim)
            if self.dim != self.codebook_dim
            else nn.Identity()
        )

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )

    def forward(self, z):
        z = self.quant_proj(z)

        d = (
            torch.sum(z**2, dim=2, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )

        sample_idxs = torch.argmin(d, dim=2)
        z_q = self.embedding(sample_idxs)

        commit_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )
        commit_loss = self.commitment_weight * commit_loss

        z_q = z + (z_q - z).detach()

        z_q = self.post_quant_proj(z_q)

        return z_q, sample_idxs, commit_loss
