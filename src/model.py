import torch


class DownSample(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """a downsampling layer that adds to skip list its output if it is passed, if HxW is a power of 2 cuts it in half

        Args:
            in_dims (int): NxCxHxW
            out_dims (int): NxCxHxW
        """

        super().__init__()

        self.in_ch = in_ch

        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.LeakyReLU(),
        )

        self.pooler = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.module(x)

        pooled = self.pooler(output)

        return pooled, output


class UpSample(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """_summary_

        Args:
            in_ch (int): NxCxHxW
            out_ch (int): NxCxHxW
            skip_tens (torch.Tensor): th skip tensor to use
        """

        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.LeakyReLU(),
        )

    def forward(
        self, x: torch.Tensor, skip_tens: torch.Tensor | None = None
    ) -> torch.Tensor:

        if skip_tens != None:
            assert (
                skip_tens.shape[2] == x.shape[2] * 2
            ), f"non compatible H of x: {x.shape[2]} and skip: {skip_tens.shape[2]}"
            assert (
                skip_tens.shape[3] == x.shape[3] * 2
            ), f"non compatible H of x: {x.shape[3]} and skip: {skip_tens.shape[3]}"

        up = self.upsample(x)

        skipped = torch.concat([skip_tens, up], dim=1) if skip_tens != None else up

        return self.module(skipped)


class AttentionGate(torch.nn.Module):
    def __init__(self, in_ch_skip: int, in_ch_act: int):
        """this is an attention gat that takes in a skip conn and a lower level feature representation

        Args:
            in_ch (int): in channel count
            out_ch (int): out channel count
        """
        super().__init__()

        self.scale_down = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_ch_act, out_channels=in_ch_act, kernel_size=1, stride=1
            ),
            torch.nn.BatchNorm2d(in_ch_act),
        )

        self.increase_depth = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_ch_skip,
                out_channels=in_ch_act,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.BatchNorm2d(in_ch_act),
        )

        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch_act, out_channels=1, kernel_size=1),
            torch.nn.Sigmoid(),
        )

        self.relu = torch.nn.LeakyReLU()
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(
        self, skip_features: torch.Tensor, lower_level_features: torch.Tensor
    ) -> torch.Tensor:

        # we expect skip features to have maybe Dx512x32x32
        # we expect lower level features to have maybe Dx1024x16x16

        # upsample the lower feature

        upped = self.increase_depth(skip_features)

        downed = self.scale_down(lower_level_features)

        added = self.relu(upped + downed)

        activated = self.psi(added)

        resampled = self.upsample(activated)

        return skip_features * resampled


class UNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # in_dims to be like ( N , 3 , 512 , 512 )

        self.d1 = DownSample(in_ch=3, out_ch=64)  # (N , 64 , 256 , 256)
        self.d2 = DownSample(in_ch=64, out_ch=128)  # (N , 128 , 128 , 128)
        self.d3 = DownSample(in_ch=128, out_ch=256)  # (N , 256 , 64 , 64)
        self.d4 = DownSample(in_ch=256, out_ch=512)  # (N , 512 , 32 , 32)

        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1
            ),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(),
        )  # (N , 1024 , 16 , 16)

        self.att1 = AttentionGate(in_ch_skip=512, in_ch_act=1024)

        self.up1 = UpSample(in_ch=1024 + 512, out_ch=512)  # (N , 512 , 32 , 32)

        self.att2 = AttentionGate(in_ch_skip=256, in_ch_act=512)

        self.up2 = UpSample(in_ch=512 + 256, out_ch=256)  # (N , 256 , 64 , 64)

        self.att3 = AttentionGate(in_ch_skip=128, in_ch_act=256)

        self.up3 = UpSample(in_ch=256 + 128, out_ch=128)  # (N , 128 , 128 , 128)

        self.att4 = AttentionGate(in_ch_skip=64, in_ch_act=128)

        self.up4 = UpSample(in_ch=128 + 64, out_ch=64)  # (N , 64 , 128 , 128)

        self.finalize = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): Assumes (NxCxHxW) , C = 3 , H = 512 , W = 512
        """

        assert (
            x.shape[1] == 3
        ), f"invalid channel dimension of: {x.shape[1]} , expected 3."
        assert (
            x.shape[2] == 512 and x.shape[3] == 512
        ), f"invalid HxW dimension of: {x.shape[2]}x{x.shape[3]} , expected 512x512."

        d1 = self.d1.forward(x)
        d2 = self.d2.forward(d1[0])
        d3 = self.d3.forward(d2[0])
        d4 = self.d4.forward(d3[0])

        neck = self.neck.forward(d4[0])

        up1 = self.up1.forward(neck, self.att1(d4[1], neck))
        # up1 = self.up1.forward(neck, d4[1])

        up2 = self.up2.forward(up1, self.att2(d3[1], up1))
        # up2 = self.up2.forward(up1, d3[1])

        up3 = self.up3.forward(up2, self.att3(d2[1], up2))
        # up3 = self.up3.forward(up2, d2[1])

        up4 = self.up4.forward(up3, self.att4(d1[1], up3))
        # up4 = self.up4.forward(up3, d1[1])

        logits = self.finalize(up4)

        return logits
