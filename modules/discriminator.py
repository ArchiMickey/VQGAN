import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4, 8), channels=3):
        super().__init__()
        self.channels = channels
        init_channels = channels

        dims = [init_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.blocks = nn.ModuleList([])
        for dim_in, dim_out in in_out:
            is_last = dim_out == dims[-1]

            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim_in, dim_out, 4, 2 if not is_last else 1, 1, bias=False
                    ),
                    nn.BatchNorm2d(dim_out),
                    nn.LeakyReLU(0.2, True),
                )
            )
        self.blocks.append(nn.Conv2d(dims[-1], 1, 4, 1, 1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    import torch
    from icecream import install

    install()

    disc = Discriminator(64)
    x = torch.randn(1, 3, 256, 256)
    ic(disc(x).shape)
