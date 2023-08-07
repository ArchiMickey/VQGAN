import hydra

from train_vqgan import train_vqgan
from train_transformer import train_transformer

from icecream import install
install()


@hydra.main(version_base=None, config_path="config", config_name="debug")
def main(config):
    assert config.stage in ["vqgan", "transformer"]
    if config.stage == "vqgan":
        return train_vqgan(config)
    if config.stage == "transformer":
        return train_transformer(config)

if __name__ == "__main__":
    main()
