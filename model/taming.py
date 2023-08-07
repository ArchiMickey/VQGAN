import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.mingpt import GPT
from modules.vqvae import VQVAE
from loguru import logger
from torchvision.utils import make_grid


class VQTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.sos_token = config["sos_token"]

        self.vqvae = VQVAE(**config["vqvae"])
        self.load_vqvae(config["vqvae_ckpt"])

        transformer_config = config["transformer"]
        self.transformer = GPT(**transformer_config)

        self.pkeep = config["pkeep"]

    def load_vqvae(self, ckpt_path):
        logger.info(f"Loading VQVAE from {ckpt_path}")
        vqgan_ckpt = torch.load(ckpt_path, map_location="cpu")
        self.vqvae.load_state_dict(vqgan_ckpt["vqvae"])
        self.vqvae.eval()

    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_ret = self.vqvae.quantize(self.vqvae.encode(x)["z"])
        return codebook_ret["z_q"], codebook_ret["sample_idxs"]

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        z = (
            self.vqvae.codebook.embedding(indices)
            .reshape(indices.shape[0], p1, p2, -1)
            .permute(0, 3, 1, 2)
        )
        img = self.vqvae.decode(z)["x_hat"]
        return img

    def forward(self, x):
        _, indices = self.encode_to_z(x)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(x.device)

        mask = torch.bernoulli(torch.ones(indices.shape) * self.pkeep)
        mask = mask.round().long().to(x.device)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat([sos_tokens, new_indices], dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def training_step(self, x):
        logits, targets = self(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return {"loss": loss}

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for _ in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1] :]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        z_q, indices = self.encode_to_z(x)
        p1, p2 = z_q.shape[-2], z_q.shape[-1]
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, : indices.shape[1] // 2]
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1]
        )
        half_sample = self.z_to_image(sample_indices, p1=p1, p2=p2)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices, p1=p1, p2=p2)

        x_rec = self.z_to_image(indices, p1=p1, p2=p2)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample
        
        log_img = make_grid(torch.cat([x, x_rec, half_sample, full_sample], dim=0), nrow=x.shape[0])

        return log, log_img
