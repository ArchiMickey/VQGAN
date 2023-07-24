import hydra
from tqdm import tqdm
from bitsandbytes.optim import Adam8bit
from torch.optim import Adam
import torch
import lightning as L

from icecream import install

install()

from modules.vqgan import VQGAN
from dataset import ImageDataset
from logger import Logger
from loguru import logger as console_logger


@hydra.main(config_path="config", config_name="debug")
def main(config):
    # Load config
    ic(dict(config))
    train_params = config["train_params"]
    torch.set_float32_matmul_precision("high")

    # Setup Fabric
    fabric = L.Fabric(
        accelerator="cuda", devices=1, precision=train_params["precision"]
    )
    fabric.launch()
    fabric.seed_everything(train_params["seed"])

    # Create model and dataloader
    model = VQGAN(config["model"])
    dataset = ImageDataset(**config["data"])
    dataloader = fabric.setup_dataloaders(
        dataset.get_dataloader(train_params["batch_size"], True, 8)
    )

    # Setup optimizer
    use_8bit_optim = train_params["optimizer"]["use_8bit"]
    base_optim = Adam8bit if use_8bit_optim else Adam
    if use_8bit_optim:
        console_logger.info("Using 8bit optimizer")

    lr = train_params["optimizer"]["lr"]
    lr_ratio = train_params["batch_size"] * train_params["grad_accumulation"]
    console_logger.info(f"Setting learning rate to {lr} * {lr_ratio} = {lr * lr_ratio}")
    lr = lr * lr_ratio

    optimizer_g = base_optim(
        model.vqvae.parameters(),
        lr=lr,
        betas=(0.5, 0.9),
    )

    optimizer_d = base_optim(
        model.discriminator.parameters(),
        lr=lr,
        betas=(0.5, 0.9),
    )

    model, optimizer_g, optimizer_d = fabric.setup(model, optimizer_g, optimizer_d)

    # Setup logger
    logger = Logger(config)

    # training loop
    epoch = 0
    global_step = 0
    log_ret = None
    while True:
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True) as pbar:
            pbar.set_postfix({"loss_g": 0, "loss_d": 0})
            for batch_idx, x in enumerate(pbar):
                ret = {
                    "batch_idx": batch_idx,
                    "epoch": epoch,
                }
                generator_ret = model.generator_step(x)
                ret |= {"generator_ret": generator_ret}

                discriminator_ret = model.discriminator_step(generator_ret)
                ret |= {"discriminator_ret": discriminator_ret}

                ret["losses"] = {}
                for k, v in generator_ret["losses"].items():
                    ret["losses"][f"generator/{k}"] = v
                ret["losses"]["discriminator/loss"] = discriminator_ret["disc_loss"]

                fabric.backward(generator_ret["losses"]["total_loss"])
                fabric.backward(discriminator_ret["disc_loss"])

                if batch_idx != 0 and (
                    batch_idx == len(dataloader) - 1
                    or batch_idx % train_params["grad_accumulation"] == 0
                ):
                    optimizer_g.step()
                    optimizer_d.step()
                    optimizer_g.zero_grad()
                    optimizer_d.zero_grad()
                    
                    global_step += 1
                    if global_step == train_params["disc_start"]:
                        console_logger.info(
                            f"Start using discriminator at step {global_step}"
                        )
                        model.enable_adv()

                ret |= {
                    "global_step": global_step,
                    "model": model,
                    "optimizer_g": optimizer_g,
                    "optimizer_d": optimizer_d,
                }

                logger.log_iter(ret)

                if batch_idx == len(dataloader) - 2:
                    log_ret = ret

                pbar.set_postfix(
                    {
                        "loss_g": generator_ret["losses"]["total_loss"].item(),
                        "loss_d": discriminator_ret["disc_loss"].item(),
                    }
                )

            epoch += 1
            logger.log_epoch(epoch, log_ret)


if __name__ == "__main__":
    main()
