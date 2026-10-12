# VehicleDetection

Image classification of vehicles into eight categories using deep learning (PyTorch). The project compares custom lightweight Xception-style CNNs with an ImageNet-pretrained **VGG16**, and ships a small command-line tool for prediction and evaluation.

**Classes:** `ambulance`, `autobus`, `kamyun`, `kamyunet`, `minibus`, `savari`, `taxi`, `vanet`

A ninth category, `neysan`, is deliberately **excluded from training** and reserved as an unseen class for open-set / out-of-distribution (OOD) evaluation.

The full write-up (data cleaning, preprocessing, architectures, training, model selection, evaluation) is in [`report.md`](report.md).

## Results

| Model | Best val. accuracy | Best val. macro-F1 | Eval. set accuracy |
| ----- | -----------------: | -----------------: | -----------------: |
| SmallXception1 | 95.89% | 95.88% | 350 / 365 (95.89%) |
| **VGG16** (final model) | **97.81%** | **96.00%** | 356 / 365 (97.53%) |

Most remaining errors are between visually similar classes (`kamyun`, `kamyunet`, `minibus`, `savari`). `vanet` is the smallest class (162 images) and has the lowest recall (93.33% for both models).

Misclassified examples:

| SmallXception1 | VGG16 |
| :---: | :---: |
| ![SmallXception1 errors](image-1.png) | ![VGG16 errors](image.png) |

## Dataset

- 3,329 cleaned images across the 8 known classes (162 for `vanet` up to 543 for `kamyunet`), plus 535 `neysan` images for OOD evaluation.
- Cleaning removed duplicates (SHA-256), non-JPEG files, unreadable images, and images that are too small (< 32 px) or have extreme aspect ratios (outside 0.33 to 3.0). 32 files were removed from the known classes.
- Split: stratified 80% train / 20% validation (seed 42), with balanced batch sampling (4 images per class per batch).
- Input: 224 x 224 crops with ImageNet normalization. Training uses random resized crop, horizontal flip, brightness/contrast jitter, Gaussian blur, and Gaussian noise.

The dataset itself is not included in this repository.

## Models

- **SmallXception1 / 2 / 3** - compact Xception-style networks built from depthwise separable convolutions and residual blocks (variant 3 adds Squeeze-and-Excitation blocks).
- **VGG16** - ImageNet-pretrained, first 24 feature layers frozen, new 8-class head, fine-tuned with a lower learning rate for the backbone (1e-4) than the head (1e-3).

The command-line tool exposes the two final models: `vgg` and `xception` (SmallXception1).

## Repository layout

```
.
├── main.py        # command-line interface (predict / evaluate)
├── main.ipynb     # notebook used for experiments
├── report.md      # full project report
├── src/           # data loading, models, training, prediction code
├── image.png      # VGG16 misclassification examples
└── image-1.png    # SmallXception1 misclassification examples
```

## Setup

Install the dependencies in a Python environment of your choice:

```bash
pip install torch torchvision torchmetrics pillow numpy matplotlib
```

The notebook and training code may need additional packages (for example `scikit-learn` for the stratified split, and `ultralytics` for YOLO-based vehicle cropping).

Trained weights are expected at these paths, relative to the project root:

```
models/vgg.pth
models/best_xception_model.pth
```

Create the `models/` folder and put your trained weight files there. Run all commands from the project root so `src/` and `models/` resolve correctly.

## Usage

### Predict

Runs in a loop: it asks for an image path, prints the prediction, and asks again until you press **Ctrl+C**.

```bash
python main.py predict                       # prompts for image paths (vgg model)
python main.py predict path/to/car.jpg       # predicts this image first, then keeps prompting
python main.py predict --model-name xception
python main.py predict car.jpg -o result.json
```

Each prediction prints the class and confidence and is saved to a JSON file (`prediction.json` by default; use `-o` to change it). Note that each image overwrites the previous result in the same file.

| Option | Description | Default |
| ------ | ----------- | ------- |
| `image_path` | Optional first image | none (prompt) |
| `--model-name` | `vgg` or `xception` | `vgg` |
| `--output`, `-o` | Where to save the prediction JSON | `prediction.json` |

### Evaluate

Evaluates a model on a folder in `ImageFolder` layout (one subfolder per class) and prints loss, accuracy, and macro-F1.

```bash
# single image (same as before)
python main.py predict path/to/image.jpg --model-name vgg

# whole directory
python main.py predict path/to/folder/ --model-name vgg

# interactive (no argument)
python main.py predict --model-name vgg
# then type either an image path or a directory path
```

```
data_dir/
├── ambulance/
├── autobus/
├── kamyun/
├── kamyunet/
├── minibus/
├── savari/
├── taxi/
└── vanet/
```

## Limitations

- The dataset is small and imbalanced, and several classes look alike, so some confusion remains between them.
- Performance depends on the vehicle-cropping step (YOLO11n) used to prepare the images.
- Open-set evaluation uses a single unseen class (`neysan`), which does not cover all possible unknown inputs. The command-line tool does not currently reject unknown classes.
- Only a limited set of architectures and hyperparameters was explored.

See [`report.md`](report.md) for details and future work.
