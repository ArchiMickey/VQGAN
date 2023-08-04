import torch.nn as nn
import torch

from modules.discriminator import hinge_d_loss
from modules.lpips import LPIPS


class VQGANLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recon_loss = lambda x, y: torch.abs(x.contiguous() - y.contiguous())
        self.perceptual_loss = LPIPS().eval()
        self.adv_loss = lambda x: -torch.mean(x)

        self.disc_loss = hinge_d_loss
        self.use_adv = False
        self.perceptual_weight = config["perceptual_weight"]
        self.adv_weight = config["adv_weight"]

    def enable_adv(self):
        self.use_adv = True

    def calculate_lambda(self, vqgan, nll_loss, gan_loss):
        last_layer = vqgan.vqvae.decoder.final_conv
        last_layer_weight = last_layer.weight

        nll_loss_grad = torch.autograd.grad(
            nll_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grad = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lam = torch.norm(nll_loss_grad) / (torch.norm(gan_loss_grad) + 1e-4)
        lam = torch.clamp(lam, 0, 1e4).detach()
        return self.adv_weight * lam

    def generator_loss(self, vqgan, vqvae_ret):
        assert vqvae_ret is not None

        recon_loss = self.recon_loss(vqvae_ret["x"], vqvae_ret["x_hat"])
        perceptual_loss = self.perceptual_loss(vqvae_ret["x"], vqvae_ret["x_hat"])
        perceptual_recon_loss = torch.mean(
            recon_loss + self.perceptual_weight * perceptual_loss
        )

        commit_loss = vqvae_ret["commit_loss"]

        adv_loss = self.adv_loss(vqgan.discriminator(vqvae_ret["x_hat"]))
        lam = self.calculate_lambda(vqgan, perceptual_recon_loss, adv_loss)
        disc_weight = 1.0 if self.use_adv else 0.0
        adv_loss = disc_weight * lam * adv_loss

        total_loss = perceptual_recon_loss + commit_loss + adv_loss

        loss_dict = {
            "recon_loss": recon_loss.mean(),
            "perceptual_loss": perceptual_loss.mean(),
            "perceptual_recon_loss": perceptual_recon_loss,
            "commit_loss": commit_loss,
            "adv_loss": adv_loss,
            "total_loss": total_loss,
        }

        return loss_dict

    def discriminator_loss(self, disc_ret):
        disc_real = disc_ret["disc_real"]
        disc_fake = disc_ret["disc_fake"]
        disc_weight = 1.0 if self.use_adv else 0.0
        disc_loss = disc_weight * self.disc_loss(disc_real, disc_fake)

        ret = {
            "disc_real": disc_real,
            "disc_fake": disc_fake,
            "disc_loss": disc_loss,
        }

        return ret

    def forward(self, mode, **kwargs):
        assert mode in ["generator", "discriminator"]
        if mode == "generator":
            return self.generator_loss(**kwargs)
        else:
            return self.discriminator_loss(**kwargs)
