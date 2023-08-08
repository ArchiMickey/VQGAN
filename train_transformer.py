from tqdm import tqdm
from bitsandbytes.optim import Adam8bit
from torch.optim import Adam
import torch
import lightning as L

from dataset import ImageDataset
from logger import Logger
from loguru import logger as console_logger
from rich import print as rprint

from modules.lr_scheduler import NoamLR
from model.taming import VQTransformer


def train_transformer(config):
    # Load config
    rprint("Config:")
    rprint(dict(config))
    train_params = config["train_params"]
    torch.set_float32_matmul_precision("high")

    # Setup Fabric
    fabric = L.Fabric(
        accelerator="cuda", devices=1, precision=train_params["precision"]
    )
    fabric.launch()
    fabric.seed_everything(train_params["seed"])

    # Create model and dataloader
    model = VQTransformer(config["model"])
    dataset = ImageDataset(**config["data"])
    dataloader = fabric.setup_dataloaders(
        dataset.get_dataloader(train_params["batch_size"], True, config["data"]["num_workers"])
    )

    # Setup optimizer
    use_8bit_optim = train_params["optimizer"]["use_8bit"]
    base_optim = Adam8bit if use_8bit_optim else Adam
    if use_8bit_optim:
        console_logger.info("Using 8bit optimizer")

    lr = train_params["optimizer"]["lr"]

    optimizer = base_optim(model.parameters(), lr=lr)

    model, optimizer = fabric.setup(model, optimizer)
    
    scheduler = NoamLR(optimizer, train_params["optimizer"]["warmup_steps"])

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
                    "x": x,
                }

                transformer_ret = model.training_step(x)
                ret |= {"transformer_ret": transformer_ret}
                
                ret["losses"] = {}
                ret["losses"]["loss"] = transformer_ret["loss"]

                fabric.backward(transformer_ret["loss"])

                if batch_idx != 0 and (
                    batch_idx == len(dataloader) - 1
                    or batch_idx % train_params["grad_accumulation"] == 0
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    scheduler.step()

                    global_step += 1

                ret |= {
                    "global_step": global_step,
                    "model": model,
                    "optimizer": optimizer,
                }

                logger.log_iter(ret)

                if batch_idx == len(dataloader) - 2:
                    log_ret = ret

                pbar.set_postfix(
                    {
                        "loss": ret["losses"]["loss"].item(),
                    }
                )

            epoch += 1
            logger.log_epoch(epoch, log_ret)
