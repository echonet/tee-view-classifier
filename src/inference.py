import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18
from tqdm import tqdm

from .data import EchoDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LIST = [
    "ME 2-Chamber View",
    "ME 4-Chamber View",
    "ME AV SAX View",
    "ME Bicaval View",
    "ME Left Atrial Appendage View",
    "ME Long Axis View",
    "TG LV SAX View",
    "Aortic View",
    "Other",
]


def predict(data_path, weights_path, save_predictions_path, batch_size, num_workers):
    with torch.inference_mode():
        dataset = EchoDataset(
            data_path=data_path,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

        model = r2plus1d_18(num_classes=len(CLASS_LIST))

        print(model.load_state_dict(torch.load(weights_path)))

        model.eval()
        model = model.to(DEVICE)

        all_predictions = {}
        for i, (videos, filenames) in enumerate(tqdm(dataloader)):
            videos = videos.to(DEVICE)
            predictions = torch.softmax(model(videos), dim=-1)

            if i == 0:
                all_predictions["predictions"] = predictions
                all_predictions["filename"] = filenames
            else:
                all_predictions["predictions"] = torch.cat(
                    [all_predictions["predictions"], predictions]
                )
                all_predictions["filename"] += filenames

        all_predictions["predictions"] = all_predictions["predictions"].cpu().numpy()

        for i, view in enumerate(CLASS_LIST):
            view = view.replace(" ", "_").replace("-", "_").lower()
            view = f"{view}_preds"
            all_predictions[view] = all_predictions["predictions"][:, i]
        all_predictions.pop("predictions")

        df = pd.DataFrame(all_predictions)

        df.to_csv(save_predictions_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--save_predictions", type=str, default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    predict(
        data_path=Path(args.data_path),
        weights_path=Path(args.weights_path),
        save_predictions_path=Path(args.save_predictions),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
