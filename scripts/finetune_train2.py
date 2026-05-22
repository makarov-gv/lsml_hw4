import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml_utils import ImageDataset, build_resnet18, build_transforms, save_json, train_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.params, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)["finetune"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(params["tensorboard_dir"])
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    save_json(params, run_dir / "params.json")

    train_dataset = ImageDataset(
        csv_file=params["train_csv"],
        root_dir=params["train_root"],
        transform=build_transforms(params["image_size"], train=True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )

    model = build_resnet18(num_classes=params["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(params["base_model_path"], map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params["scheduler_step_size"],
        gamma=params["scheduler_gamma"],
    )

    train_rows = []
    best_f1 = 0.0
    best_model_path = Path(params["best_model_path"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(params["num_epochs"]):
        print(f"\nFinetune epoch {epoch + 1}/{params['num_epochs']}")
        metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        row = {"epoch": epoch + 1, **metrics, "learning_rate": scheduler.get_last_lr()[0]}
        train_rows.append(row)
        for metric_name, metric_value in row.items():
            if metric_name != "epoch":
                writer.add_scalar(f"finetune_train/{metric_name}", metric_value, epoch + 1)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Loss: {metrics['loss']:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}"
        )

    last_model_path = Path(params["last_model_path"])
    last_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), last_model_path)

    metrics_path = Path(params["train_metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_csv(metrics_path, index=False)
    writer.flush()
    writer.close()

    print(f"Best finetune F1: {best_f1:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved train metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
