from hydra.utils import get_original_cwd
import wandb
import os
from datetime import datetime
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
from loguru import logger as console_logger
from rich import print as rprint
from omegaconf import OmegaConf


class Logger:
    def __init__(self, config):
        logger_config = config["logger"]
        self.log_img_interval = logger_config["log_img_interval"]
        self.save_interval = logger_config["save_interval"]
        self.loss_list = {}
        self.accumulated_ret = {"losses": {}, "disc": {}}
        self.loss_names = logger_config["loss_names"]
        for name in self.loss_names:
            self.loss_list[name] = []
            self.accumulated_ret["losses"][name] = []
        for k in ["disc_real", "disc_fake"]:
            self.accumulated_ret["disc"][k] = []
        now = datetime.now()
        dirname = now.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_path = os.path.join(logger_config["checkpoint_path"], dirname)

        self.global_step = 0
        self.epoch = 0
        
        wandb.init(
            project=config["project"],
            name=config["name"],
            dir=f"{get_original_cwd()}",
            config=OmegaConf.to_container(config, resolve=True),
        )
        self.transform = ToPILImage()

    def normalize_0_1(self, x):
        b, c, h, w = x.shape
        x = x.view(b, -1)
        x -= x.min(dim=1, keepdim=True)[0]
        x /= x.max(dim=1, keepdim=True)[0]
        x = x.view(b, c, h, w)
        return x

    def log_img(self, training_step_ret):
        x = training_step_ret["generator_ret"]["x"]
        x_hat = training_step_ret["generator_ret"]["x_hat"]
        x = self.normalize_0_1(x)
        x_hat = x_hat.clamp(-1, 1)
        x_hat = self.normalize_0_1(x_hat)

        b, c, h, w = x.shape
        canvas = Image.new("RGB", (w * 2, h * b))
        for i in range(x.shape[0]):
            canvas.paste(self.transform(x[i]), (0, i * h))
            canvas.paste(self.transform(x_hat[i]), (w, i * h))
        return wandb.Image(canvas)

    def save_checkpoint(self, training_step_ret):
        model = training_step_ret["model"]
        checkpoint = {
            "vqvae": model.vqvae.state_dict(),
            "discriminator": model.discriminator.state_dict(),
            "optimizer_g": training_step_ret["optimizer_g"].state_dict(),
            "optimizer_d": training_step_ret["optimizer_d"].state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        console_logger.info(f"Saving checkpoint to {self.checkpoint_path}")
        torch.save(
            checkpoint,
            os.path.join(self.checkpoint_path, f"checkpoint_{self.global_step}.pt"),
        )

    def log_to_wandb(self):
        log_dict = {
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        for k in self.accumulated_ret["losses"].keys():
            log_dict[k] = sum(self.accumulated_ret["losses"][k]) / len(
                self.accumulated_ret["losses"][k]
            )
            self.accumulated_ret["losses"][k].clear()

        for k in ["disc_real", "disc_fake"]:
            log_dict[f"discriminator/{k}"] = wandb.Histogram(
                self.accumulated_ret["disc"][k]
            )
            self.accumulated_ret["disc"][k].clear()
        wandb.log(log_dict)

    def log_iter(self, training_step_ret):
        if self.global_step != training_step_ret["global_step"]:
            self.log_to_wandb()
            self.global_step = training_step_ret["global_step"]

            if self.global_step != 0 and self.global_step % self.save_interval == 0:
                self.save_checkpoint(training_step_ret)

        losses = training_step_ret["losses"]
        for k, v in losses.items():
            self.loss_list[k].append(v.item())
            self.accumulated_ret["losses"][k].append(v.item())

        for k in ["disc_real", "disc_fake"]:
            self.accumulated_ret["disc"][k].extend(
                training_step_ret["discriminator_ret"][k]
                .detach()
                .cpu()
                .flatten()
                .tolist()
            )

    def log_epoch(self, epoch, training_step_ret):
        self.epoch = epoch
        log_dict = {"global_step": self.global_step, "epoch": self.epoch}
        for k, v in self.loss_list.items():
            log_dict[f"{k}_epoch"] = sum(v) / len(v)
        log_dict |= {"log_img": self.log_img(training_step_ret)}
        wandb.log(log_dict)
        rprint(log_dict)
        for k in self.loss_list.keys():
            self.loss_list[k] = []
