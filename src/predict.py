import json
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.v2 as transforms


# Use the SAME preprocessing that was used during training.
transform = transforms.Compose(
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


def predict(image_path, model, class_names, output_path="prediction.json", device='cuda'):
    """
    Load an image and a trained PyTorch model, perform prediction,
    and save an informative JSON result.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    model : 
        trained PyTorch model.

    class_names : list[str]
        List of class names. The index must correspond to the
        class index used during training.

    output_path : str
        Path where the prediction JSON will be saved.

    Returns
    -------
    dict
        Prediction results.
    """

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file does not exist: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)

    # Add batch dimension:
    # [C, H, W] -> [1, C, H, W]
    input_tensor = input_tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)

        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    predicted_class = class_names[predicted_index]

    class_probabilities = {
        class_names[i]: float(probabilities[0, i])
        for i in range(len(class_names))
    }

    result = {
        "status": "success",

        "input": {
            "image_path": str(image_path.resolve()),
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        },

        "model": {
            "model_path": str(model.__class__.__name__),
            "device": str(device)
        },

        "prediction": {
            "class": predicted_class,
            "class_index": predicted_index,
            "confidence": round(confidence, 6),
            "confidence_percent": round(confidence * 100, 2)
        },

        "class_probabilities": class_probabilities
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    return result

