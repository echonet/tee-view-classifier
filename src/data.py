import os
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from numpy.typing import ArrayLike
import torch
from torch.utils.data import Dataset


def crop_resize(
    img: ArrayLike, res: Tuple[int], interpolation=cv2.INTER_CUBIC
) -> ArrayLike:
    """Takes an image, a resolution, and a zoom factor as input, returns the
    zoomed/cropped image."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    # Crop to correct aspect ratio
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]

    # Resize image
    img = cv2.resize(img, res, interpolation=interpolation)

    return img


def read_video(
    path: Union[str, Path],
    n_frames: int = 16,
    sample_period: int = 2,
    res: Tuple[int] = (112, 112),  # (width, height)
) -> ArrayLike:
    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)

    for frame_i in range(n_frames):
        _, frame = cap.read()
        if res is not None:
            frame = crop_resize(frame, res)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    return out, vid_size, fps


class EchoDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.filenames = os.listdir(self.data_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        video, _, _ = read_video(
            self.data_path / filename,
            n_frames=16,
            res=(112, 112),
            sample_period=2,
        )

        video = torch.from_numpy(video)
        video = torch.movedim(video / 255, -1, 0).to(torch.float32)

        return video, filename
