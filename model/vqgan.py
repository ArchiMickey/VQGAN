from loguru import logger
import torch.nn as nn
import torch

from modules.losses import VQGANLoss
from modules.vqvae import VQVAE
from modules.discriminator import NLayerDiscriminator, weights_init


class VQGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vqvae = VQVAE(**config["vqvae"])
        self.discriminator = NLayerDiscriminator(**config["discriminator"])
        self.discriminator.apply(weights_init)

        self.loss = VQGANLoss(config["loss"])

        if config["model_ckpt"] is not None:
            logger.info(f"Loading checkpoint from {config['model_ckpt']}")
            self.load_checkpoint(config["model_ckpt"], config["loading_modules"])

    def load_checkpoint(self, checkpoint_path, loading_module=None):
        available_modules = ["vqvae", "discriminator"]
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if loading_module is None:
            for module in available_modules:
                getattr(self, module).load_state_dict(ckpt[module])

        else:
            for module in loading_module:
                getattr(self, module).load_state_dict(ckpt[module])

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

        losses = self.loss(mode="generator", vqgan=self, vqvae_ret=vqvae_ret)

        ret |= {"losses": losses}

        return ret

    def discriminator_step(self, generator_ret):
        disc_real = self.discriminator(generator_ret["x"])
        disc_fake = self.discriminator(generator_ret["x_hat"].detach())
        disc_ret = {"disc_real": disc_real, "disc_fake": disc_fake}

        ret = self.loss(mode="discriminator", disc_ret=disc_ret)

        return ret

    def forward(self, x):
        return self.vqvae(x)["x_hat"]
