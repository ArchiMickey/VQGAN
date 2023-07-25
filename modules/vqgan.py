from loguru import logger
import torch.nn as nn
import torch

from modules.util import weights_init
from modules.vqvae import VQVAE
from modules.discriminator import Discriminator
from modules.lpips import LPIPS


class VQGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vqvae = VQVAE(**config["vqvae"])
        self.discriminator = Discriminator(**config["discriminator"])
        self.discriminator.apply(weights_init)

        self.recon_loss = lambda x, y: torch.abs(x.contiguous() - y.contiguous())
        self.perceptual_loss = LPIPS().eval()

        self.discrimator_loss = nn.BCEWithLogitsLoss()

        self.use_adv = False
        self.perceptual_weight = config["perceptual_weight"]

        if config["model_ckpt"] is not None:
            logger.info(f"Loading checkpoint from {config['model_ckpt']}")
            self.load_checkpoint(config["model_ckpt"], config["loading_modules"])

    
    def enable_adv(self):
        self.use_adv = True

    def load_checkpoint(self, checkpoint_path, loading_module=None):
        available_modules = ["vqvae", "discriminator"]
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if loading_module is None:
            for module in available_modules:
                getattr(self, module).load_state_dict(ckpt[module])

        else:
            for module in loading_module:
                getattr(self, module).load_state_dict(ckpt[module])
    
    def calculate_lambda(self, nll_loss, gan_loss):
        last_layer = self.vqvae.decoder.final_conv
        last_layer_weight = last_layer.weight

        nll_loss_grad = torch.autograd.grad(
            nll_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grad = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lam = torch.norm(nll_loss_grad) / (torch.norm(gan_loss_grad) + 1e-4)
        lam = torch.clamp(lam, 0, 1e4).detach()
        return 0.8 * lam

    def encode(self, x):
        z = self.vqvae.encoder(x)
        z = self.vqvae.quant_conv(z)

        codebook_ret = self.vqvae.vector_quantize(z)

        ret = {"z": z, "codebook_ret": codebook_ret}
        return ret

    def decode(self, z):
        z = self.vqvae.post_quant_conv(z)
        x_hat = self.vqvae.decoder(z)
        return x_hat

    def generator_step(self, x):
        vqvae_ret = self.vqvae(x)
        ret = vqvae_ret

        recon_loss = self.recon_loss(x, vqvae_ret["x_hat"])
        perceptual_loss = self.perceptual_loss(x, vqvae_ret["x_hat"])
        preceptual_recon_loss = torch.mean(recon_loss + self.perceptual_weight * perceptual_loss)

        commit_loss = vqvae_ret["commit_loss"]

        adv_loss = self.discrimator_loss(
            vqvae_ret["x_hat"], torch.ones_like(vqvae_ret["x_hat"])
        )
        lam = self.calculate_lambda(preceptual_recon_loss, adv_loss)
        disc_weight = 1.0 if self.use_adv else 0.0
        adv_loss = disc_weight * lam * adv_loss

        total_loss = preceptual_recon_loss + commit_loss + adv_loss 

        losses = {
            "recon_loss": recon_loss.mean(),
            "perceptual_loss": perceptual_loss.mean(),
            "perceptual_recon_loss": preceptual_recon_loss,
            "commit_loss": commit_loss,
            "adv_loss": adv_loss,
            "total_loss": total_loss,
        }
        ret |= {"losses": losses}

        return ret

    def discriminator_step(self, generator_ret):
        disc_real = self.discriminator(generator_ret["x"])
        disc_fake = self.discriminator(generator_ret["x_hat"].detach())
        disc_loss = (
            self.discrimator_loss(disc_real, torch.ones_like(disc_real))
            + self.discrimator_loss(disc_fake, torch.zeros_like(disc_fake))
        ) / 2

        disc_weight = 1.0 if self.use_adv else 0.0
        disc_loss = disc_weight * disc_loss

        ret = {
            "disc_real": disc_real.mean(),
            "disc_fake": disc_fake.mean(),
            "disc_loss": disc_loss,
        }

        return ret

    def forward(self, x):
        return self.vqvae(x)["x_hat"]
