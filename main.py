import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms.v2 as transforms

from src.predict import predict
from src.train import evaluate_vgg, evaluate_tm
from src.train import load_weights
from src.train import create_dataloader
from src.utils.model import create_model

class_names = ['ambulance', 'autobus', 'kamyun', 'kamyunet', 'minibus', 'savari', 'taxi', 'vanet']

WEIGHTS = {
    'vgg': 'models/vgg.pth',
    'xception': 'models/ best_xception_model.pth',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_one(image_path, model, output, device):
    result = predict(
        image_path=image_path,
        model=model,
        class_names=class_names,
        output_path=output,
        device=device,
    )
    print(f"  Predicted class : {result['prediction']['class']}")
    print(f"  Confidence      : {result['prediction']['confidence_percent']}%")
    print(f"  Result saved to : {output}")


def collect_images(path: Path):
    """Return a list of image paths. Accepts a single file or a directory."""
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        return images
    raise FileNotFoundError(f"Path does not exist: {path}")


def run_predict(args, model, device):
    print("Prediction mode. Press Ctrl+C to quit.")
    print("You can enter an image path or a directory path.\n")

    pending = args.image_path
    while True:
        if pending is None:
            pending = input("\nImage / directory path: ").strip().strip('"').strip("'")

        if pending:
            try:
                path = Path(pending)
                images = collect_images(path)

                if not images:
                    print(f"No images found in: {path}")
                else:
                    print(f"Found {len(images)} image(s). Starting prediction...")
                    for img in images:
                        # unique output name per image
                        if len(images) == 1 and args.output:
                            out = args.output
                        else:
                            out = f"{args.output}/{img.stem}_prediction.json"

                        print(f"\n→ {img}")
                        predict_one(str(img), model, out, device)

            except Exception as e:
                print(f"Error: {e}")

        pending = None


def run_evaluate(args, model, device, evaluate, type='vgg'):
    valid_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    loader, _ = create_dataloader(path=args.data_dir, transformation=valid_transform, val=True)
    criterion = nn.CrossEntropyLoss()
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)
    f1_metric = torchmetrics.F1Score(
        task="multiclass",
        num_classes=8,
        average="macro",
    ).to(device)

    if type == 'vgg':
        loss, acc, f1 = evaluate(
            model=model,
            loader=loader,
            loss_fn=criterion,
            acc_metric=accuracy_metric,
            f1_metric=f1_metric,
            device=device,
        )
    else: 
        loss, acc, f1 = evaluate(
            model=model,
            data_loader=loader,
            loss_fn=criterion,
            accuracy_metric=accuracy_metric,
            f1_metric=f1_metric,
            device=device,
        )

    if loss is not None:
        print(f"Loss            : {loss:.6f}")
    print(f"Accuracy        : {acc:.2f}%")
    print(f"F1 (macro)      : {f1:.2f}%")


def main():
    device = get_device()

    parser = argparse.ArgumentParser(
        description="Predict on images (runs until Ctrl+C) or evaluate a model on a dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- predict ----------
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict on a single image or a whole directory (then interactive)",
    )
    predict_parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Optional first image **or directory**; afterwards you are prompted for more",
    )
    predict_parser.add_argument(
        "--model-name",
        dest="model_name",
        choices=sorted(WEIGHTS),
        default="vgg",
        help="Model to use (default: vgg)",
    )
    predict_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="predictions",
        help="Output JSON for a single image (ignored when predicting a directory)",
    )

    # ---------- evaluate ----------
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate model on an ImageFolder dataset"
    )
    eval_parser.add_argument(
        "data_dir",
        type=str,
        help="Path to data directory (ImageFolder structure: class subfolders)",
    )
    eval_parser.add_argument(
        "model_name", type=str, choices=sorted(WEIGHTS), help="Name of the trained model"
    )

    args = parser.parse_args()

    model = create_model(args.model_name, 8)
    model = load_weights(model, WEIGHTS[args.model_name], device)

    if args.command == "predict":
        run_predict(args, model, device)
    elif args.command == "evaluate":
        if args.model_name == "vgg":
            eval_type = 'vgg'
            evaluate = evaluate_vgg
        else:
            evaluate = evaluate_tm
            eval_type = 'xception'
        run_evaluate(args, model, device, evaluate, eval_type)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        sys.exit(0)