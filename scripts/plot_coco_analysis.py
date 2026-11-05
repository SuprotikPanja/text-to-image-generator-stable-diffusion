
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "coco"
OUTPUT = ROOT / "outputs" / "coco_analysis"
FIGURES = OUTPUT / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


def save_caption_distribution():
    train_file = DATA / "annotations" / "captions_train2017.json"
    val_file = DATA / "annotations" / "captions_val2017.json"

    train = json.load(open(train_file, encoding="utf-8"))
    val = json.load(open(val_file, encoding="utf-8"))

    train_lengths = [
        len(item["caption"].split())
        for item in train["annotations"]
    ]

    val_lengths = [
        len(item["caption"].split())
        for item in val["annotations"]
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(train_lengths, bins=25, alpha=0.6, label="Training")
    plt.hist(val_lengths, bins=25, alpha=0.6, label="Validation")
    plt.xlabel("Number of words")
    plt.ylabel("Number of captions")
    plt.title("COCO Caption Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "caption_length_distribution.png", dpi=200)
    plt.close()


def save_category_distribution():
    file = OUTPUT / "category_distribution.csv"
    df = pd.read_csv(file)

    train = df[df["split"] == "train"].head(15)
    train = train.sort_values("annotation_count")

    plt.figure(figsize=(10, 7))
    plt.barh(train["category_name"], train["annotation_count"])
    plt.xlabel("Number of object annotations")
    plt.ylabel("Object category")
    plt.title("Top 15 COCO Object Categories")
    plt.tight_layout()
    plt.savefig(FIGURES / "category_distribution.png", dpi=200)
    plt.close()


def save_resolution_distribution():
    file = OUTPUT / "image_metadata.csv"
    df = pd.read_csv(file)

    val = df[df["split"] == "val"]
    resolutions = (
        val.groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    labels = [
        f"{width}x{height}"
        for width, height in resolutions.index
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, resolutions.values)
    plt.xlabel("Number of images")
    plt.ylabel("Resolution")
    plt.title("Top 10 COCO Validation Image Resolutions")
    plt.tight_layout()
    plt.savefig(FIGURES / "image_resolution_distribution.png", dpi=200)
    plt.close()


def save_split_comparison():
    report_file = OUTPUT / "dataset_report.json"

    with open(report_file, encoding="utf-8") as file:
        reports = json.load(file)

    splits = [item["split"].upper() for item in reports]
    images = [item["unique_images"] for item in reports]
    captions = [item["total_captions"] for item in reports]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(splits, images)
    axes[0].set_title("Image Count by Split")
    axes[0].set_ylabel("Number of images")

    axes[1].bar(splits, captions)
    axes[1].set_title("Caption Count by Split")
    axes[1].set_ylabel("Number of captions")

    fig.suptitle("COCO Training vs Validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "train_validation_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    save_caption_distribution()
    save_category_distribution()
    save_resolution_distribution()
    save_split_comparison()

    print("All four COCO visualisations generated successfully!")
    print("Saved to:", FIGURES)
