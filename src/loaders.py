import torch
from torchvision.transforms import v2
import os
import rasterio
from rasterio.io import DatasetReader
from torchvision import tv_tensors


class ImageLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        img_chips_dir: str,
        ground_truth_chips_dir: str,
        use_transform: bool = True,
        scale_factor : int = 255
    ):
        """Load NAIP imagery for processing

        Args:
            img_chips_dir (str): this is expected to be a dir 3x512x512 image chips that are NAIP satelite imagery or similar
            ground_truth_chips_dir (str): this is expected to be a dir of 1x512x512 image chips that are the ground truth masks for the training image chips. They should have the same names as the image chips in the train dir so they can be mapped to eachother. Expects 1 for posiive class and 0 for background.
        """

        super().__init__()

        assert (
            os.path.exists(img_chips_dir)
            and os.path.exists(ground_truth_chips_dir)
            and os.path.isdir(img_chips_dir)
            and os.path.isdir(ground_truth_chips_dir)
        ), "data/ground truth directories not found"

        self.train_dat_dir: str = img_chips_dir
        self.img_chips_pths: list[str] = list(
            set(
                [
                    os.path.abspath(os.path.join(img_chips_dir, file))
                    for file in os.listdir(img_chips_dir)
                ]
            )
        )

        self.ground_truth_mask_dir: str = ground_truth_chips_dir
        self.ground_truth_chips_pths: list[str] = list(
            set(
                [
                    os.path.abspath(os.path.join(ground_truth_chips_dir, file))
                    for file in os.listdir(ground_truth_chips_dir)
                ]
            )
        )

        self.size: int = len(self.img_chips_pths)

        self.affine = v2.RandomAffine(degrees=10, scale=(1.02, 1.1), shear=(1.5, 1.8))
        self.flip = v2.RandomHorizontalFlip(1.0)
        self.rotate = v2.RandomRotation(degrees=10)
        self.color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2)

        self.choose_transform = v2.RandomChoice(
            [
                self.affine,
                self.flip,
                self.rotate,
                self.color_jitter,
                torch.nn.Identity(),
            ],
            p=(
                [0.1, 0.1, 0.1, 0.1, 0.6]
                if use_transform
                else [0.0, 0.0, 0.0, 0.0, 1.0]
            ),
        )

    def __len__(self):
        return self.size

    def __size__(self):
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """returns a specific image chip by index

        Args:
            idx (int): the index to retrieve

        Returns:
            tuple[torch.Tensor , torch.Tensor]: index 0 is the train chip and index 1 is the ground truth mask.
        """

        img_reader: DatasetReader = rasterio.open(self.img_chips_pths[idx], "r")

        data = img_reader.read((1, 2, 3))

        gt_reader: DatasetReader = rasterio.open(
            self.ground_truth_chips_pths[
                self.ground_truth_chips_pths.index(
                    os.path.join(
                        self.ground_truth_mask_dir,
                        os.path.basename(self.img_chips_pths[idx]),
                    )
                )
            ],
            "r",
        )

        gt = gt_reader.read(1)

        gt_reader.close()
        img_reader.close()

        transformed: tuple[tv_tensors.Image, tv_tensors.Mask] = self.choose_transform(
            (
                tv_tensors.Image(torch.tensor(data / scale_factor , dtype=torch.float32)),
                tv_tensors.Mask(torch.tensor(gt, dtype=torch.float32).unsqueeze(0)),
            )
        )

        return (
            transformed[0].as_subclass(torch.Tensor),
            transformed[1].as_subclass(torch.Tensor),
        )
