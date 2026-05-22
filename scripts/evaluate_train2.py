import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml_utils import ImageDataset, build_resnet18, build_transforms, evaluate, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.params, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)["finetune"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = ImageDataset(
        csv_file=params["test_csv"],
        root_dir=params["test_root"],
        transform=build_transforms(params["image_size"], train=False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )

    model = build_resnet18(num_classes=params["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(params["best_model_path"], map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)
    save_json(test_metrics, params["test_metrics_path"])

    writer = SummaryWriter(log_dir=params["tensorboard_dir"])
    for metric_name, metric_value in test_metrics.items():
        writer.add_scalar(f"finetune_test/{metric_name}", metric_value, params["num_epochs"])
    writer.flush()
    writer.close()

    train_metrics = pd.read_csv(params["train_metrics_path"])
    plot_path = Path(params["metrics_plot_path"])
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_metrics["epoch"], train_metrics["loss"], marker="o", label="loss")
    axes[0].plot(train_metrics["epoch"], train_metrics["accuracy"], marker="o", label="accuracy")
    axes[0].plot(train_metrics["epoch"], train_metrics["f1"], marker="o", label="f1")
    axes[0].set_title("Train_2 finetune metrics by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    bar_metrics = ["accuracy", "f1", "precision", "recall"]
    axes[1].bar(
        bar_metrics,
        [test_metrics[name] for name in bar_metrics],
        color=["#4C78A8", "#F58518", "#54A24B", "#E45756"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test_2 quality metrics")
    axes[1].grid(True, axis="y", alpha=0.3)
    for index, metric_name in enumerate(bar_metrics):
        axes[1].text(index, test_metrics[metric_name] + 0.01, f"{test_metrics[metric_name]:.3f}", ha="center")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print("Test_2 metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
