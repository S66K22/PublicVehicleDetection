# 1. Introduction

Vehicle image classification is an important computer vision task with applications in intelligent transportation systems, traffic monitoring, surveillance, and automated traffic analysis. The objective of this project is to develop an image classification system capable of identifying different types of vehicles from input images.

In this project, eight vehicle classes are considered: **ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, and vanet**. The main objective is not only to achieve high classification accuracy, but also to investigate different deep learning architectures and determine which model provides the best performance for this specific dataset.

The development process consists of several stages. First, the available image dataset is analyzed and cleaned to remove invalid, corrupted, duplicated, or unsuitable images. Image integrity, file format, image dimensions, and aspect ratio are examined during this process. In addition, vehicle regions can be extracted from images using a pretrained object detection model so that the classification models can focus more directly on the vehicle itself rather than irrelevant background information.

After preparing the dataset, multiple convolutional neural network architectures are trained and evaluated. Different model architectures and configurations are compared using appropriate classification metrics in order to identify the most suitable model for the task. In particular, both custom lightweight architectures and established architectures are investigated. **VGG-based transfer learning** is also evaluated to determine whether using features learned from a large-scale image dataset can improve classification performance.

The model selection process is based on the performance of the candidate models on previously unseen validation data. In addition to overall accuracy, class-level performance and other evaluation metrics are considered to obtain a more reliable assessment, particularly when the classes do not have exactly the same number of samples. The final model is selected based on its ability to provide a good balance between classification performance and model complexity.

A further objective of this project is to address a limitation of conventional closed-set image classifiers. A standard classifier assumes that every input belongs to one of the classes observed during training. Consequently, when presented with an image belonging to an unknown class, it may still assign the image to one of the known classes with high confidence. This behavior is undesirable in a practical vehicle classification system.

Therefore, the final system is designed to incorporate **open-set or out-of-distribution (OOD) recognition**. The system should be able to distinguish between images belonging to the eight known vehicle classes and images representing classes or objects that were not included in the training dataset. Thus, the desired behavior of the final system can be represented as:

$$
f(x) =
\begin{cases}
c_i, & \text{if } x \text{ belongs to a known class } c_i \\
\text{Unknown}, & \text{if } x \text{ does not belong to the known classes}
\end{cases}
$$

where the known classes are the eight vehicle categories defined in the dataset.

Overall, this project aims to develop and evaluate a complete vehicle image classification pipeline, beginning with dataset preparation and cleaning, followed by model development and comparison, selection of the best-performing architecture, and finally the integration of open-set recognition. The resulting system is intended to perform reliable classification of the predefined vehicle categories while reducing the risk of confidently assigning unknown inputs to an incorrect known class.


# 2. Dataset and Data Cleaning

## 2.1 Dataset Description

The dataset used in this project consists of images of different vehicle types. Eight vehicle categories were selected as the known classes for model training and classification: **ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, and vanet**. In addition to these classes, a separate **neysan** class is available in the dataset. The neysan images are deliberately excluded from the training classes and are reserved for evaluating the ability of the final system to identify previously unseen classes.

After the data cleaning process, the eight known classes contain a total of **3,329 valid images**. The distribution of the images among the classes is shown in Table 1.

| Class     | Number of Images After Cleaning |
| --------- | ------------------------------: |
| Ambulance |                             361 |
| Autobus   |                             469 |
| Kamyun    |                             408 |
| Kamyunet  |                             543 |
| Minibus   |                             395 |
| Savari    |                             500 |
| Taxi      |                             491 |
| Vanet     |                             162 |
| **Total** |                       **3,329** |

The dataset is not perfectly balanced. The number of images ranges from 162 images for the **vanet** class to 543 images for the **kamyunet** class. This imbalance is taken into consideration when evaluating the trained models, and class-level metrics are used in addition to overall accuracy.

The **neysan** class contains **535 images**. These images are not used as a ninth classification class during model training. Instead, they are kept separate and used as examples of an unseen class during the open-set/OOD evaluation. This allows the final system to be tested on a realistic scenario in which an input belongs to a class that was not present during training.

## 2.2 Data Cleaning

Before training the classification models, the dataset was examined and cleaned to ensure that invalid or duplicated files did not negatively affect the training process. A dedicated preprocessing module was developed to perform several data-quality checks automatically.

The cleaning process was applied to the available training, test, unclean, and previously cropped image directories. Each file was examined for duplication, file format, image integrity, and image dimensions.

### 2.2.1 Duplicate Detection

Duplicate images can cause problems during model development because the same image may appear multiple times in the dataset and can lead to an artificially optimistic evaluation of model performance. To identify duplicate files, the SHA-256 cryptographic hash of each file was calculated.

For each image, its contents were read in chunks and used to generate a SHA-256 hash:

[
h(I)=\operatorname{SHA256}(I)
]

If two files produced the same hash, they were considered duplicates. The duplicate file was removed from the valid dataset and moved to the corresponding corrupted/invalid-file directory.

This approach compares the actual binary contents of files rather than relying on their filenames. Therefore, two files with different filenames but identical contents can still be detected as duplicates.

### 2.2.2 File Format Validation

The MIME type of each file was checked before attempting to use it as an input image. The dataset was restricted to **JPEG images** for the subsequent processing pipeline.

Files whose detected MIME type was not `image/jpeg` were considered invalid and were separated from the dataset.

### 2.2.3 Image Integrity Validation

A file having an image extension does not necessarily mean that it contains a valid image. Therefore, each candidate image was opened using the Python Imaging Library (PIL) and verified before being accepted.

The image was first verified using the image verification functionality and then reopened and fully loaded. Reopening the image is necessary because the verification operation invalidates the image object.

An image was considered invalid if it could not be successfully verified or loaded.

### 2.2.4 Image Size and Aspect-Ratio Validation

Images with extremely small dimensions or unusual aspect ratios were also excluded. An image was required to satisfy the following minimum dimension constraint:

$$
H \geq 32,\qquad W \geq 32
$$

where (H) and (W) represent the image height and width, respectively.

The aspect ratio was calculated as:

$$
r=\frac{W}{H}
$$

and an image was accepted only when:

$$
0.33 \leq r \leq 3.0
$$

This filtering helps remove images that are too small to provide useful visual information or have extreme dimensions that are unlikely to represent suitable inputs for the classification task.

## 2.3 Cleaning Results

The cleaning process identified corrupted, invalid, or duplicate files in several of the vehicle classes. The numbers of removed files are summarized in Table 2.

| Class     | Removed Files | Images After Cleaning |
| --------- | ------------: | --------------------: |
| Ambulance |            14 |                   361 |
| Autobus   |             2 |                   469 |
| Kamyun    |             0 |                   408 |
| Kamyunet  |             4 |                   543 |
| Minibus   |            10 |                   395 |
| Savari    |             0 |                   500 |
| Taxi      |             2 |                   491 |
| Vanet     |             0 |                   162 |
| **Total** |        **32** |             **3,329** |

Therefore, **32 files** were identified and separated during the cleaning process for the eight known classes.

The cleaned dataset was subsequently used as the basis for the training and evaluation stages. Removing invalid and duplicated files before splitting the data also reduces the risk of data-quality problems affecting the comparison between different model architectures.

## 2.4 Dataset Used for Open-Set Evaluation

The **neysan** class is treated differently from the eight known classes. Its 535 images are deliberately excluded from the training set and are not assigned a ninth output class in the classification model.

This separation is important because the objective of the OOD stage is to determine whether the model can recognize that an input does not belong to any of its known training classes. Therefore, neysan images provide an unseen-class dataset for evaluating this capability.

The overall experimental setup can consequently be viewed as:

$$
\underbrace{
{\text{ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet}}
}_{\text{Known training classes}}
$$

and

$$
\underbrace{{\text{neysan}}}_{\text{Unseen class for OOD evaluation}}
$$

The classification model is trained only on the eight known classes, while the neysan images are introduced later during OOD evaluation. This separation ensures that successful recognition of neysan as **Unknown** is not the result of the model having learned neysan-specific class features during training.


# 3. Image Preprocessing and Data Augmentation

After cleaning the dataset, the images are transformed into a format suitable for training the deep learning models. The preprocessing pipeline is designed to provide a consistent input size for all models while applying data augmentation during training to improve generalization.

Two separate transformation pipelines are used: one for the training data and another for the validation/test data. Data augmentation is applied only to the training images, while validation images are processed deterministically so that model performance can be evaluated consistently.

## 3.1 Training Preprocessing

The training transformation pipeline consists of the following stages:

1. Image-to-tensor conversion
2. Resizing
3. Random resized cropping
4. Random horizontal flipping
5. Color augmentation
6. Conversion to floating-point values
7. Random Gaussian blur
8. Random Gaussian noise
9. ImageNet normalization

### 3.1.1 Image Conversion and Resizing

Each input image is first converted to an image representation compatible with the PyTorch transformation pipeline. The image is then resized so that its shorter dimension is scaled to **256 pixels**.

This provides a consistent scale before the subsequent random cropping operation.

### 3.1.2 Random Resized Crop

A random crop of size $224\times224$ pixels is extracted from the resized image. The crop is selected using a random scale between **0.7 and 1.0** of the original image area and an aspect-ratio range of **0.75 to 1.33**.

This augmentation allows the model to observe vehicles at slightly different scales and positions during training. It also reduces the model's dependence on a fixed spatial arrangement of the vehicle within the image.

The resulting input size for the classification models is therefore:

$$
224 \times 224 \times 3
$$

where the three channels correspond to RGB.

### 3.1.3 Random Horizontal Flip

A horizontal flip is applied with a probability of **0.5**.

$$
P(\text{horizontal flip})=0.5
$$

For vehicle images, horizontal flipping generally preserves the semantic class while changing the orientation of the vehicle. This provides additional variation without changing the class label.

### 3.1.4 Color Augmentation

A `ColorJitter` transformation is applied to introduce moderate variations in image brightness and contrast. The brightness parameter is set to **0.3**, while the contrast parameter is set to **0.05**.

These transformations help the model become less sensitive to differences in illumination and image capture conditions.

The augmentation is intentionally limited to brightness and contrast rather than applying strong color transformations, since visual characteristics such as vehicle color can potentially provide useful information for distinguishing some of the vehicle categories.

### 3.1.5 Conversion to Floating-Point Values

The original image values are represented as unsigned 8-bit integers in the range:

$$
[0,255]
$$

Before normalization, they are converted to `float32` and scaled to:

$$
[0,1]
$$

This conversion is necessary because the neural network operates on floating-point tensors and because the subsequent normalization is defined for floating-point image values.

### 3.1.6 Gaussian Blur

To simulate images with reduced sharpness, Gaussian blur is applied randomly with a probability of **0.15**.

A Gaussian kernel with a kernel size of 5 and a randomly selected standard deviation between 0.1 and 1.0 is used.

$$
P(\text{blur})=0.15
$$

This augmentation helps the model become more robust to variations in image focus and image quality.

### 3.1.7 Gaussian Noise

Random Gaussian noise is also applied with a probability of **0.15**. The noise has a mean of 0 and a standard deviation of 0.03.

$$
\epsilon \sim \mathcal{N}(0,0.03^2)
$$

and the augmented image can conceptually be represented as:

$$
I'=I+\epsilon
$$

The purpose of this transformation is to expose the model to small amounts of image noise that may occur because of image acquisition or compression.

### 3.1.8 ImageNet Normalization

Finally, the image channels are normalized using the mean and standard deviation commonly used for ImageNet-trained models:

$$
\mu=[0.485,\;0.456,\;0.406]
$$

$$
\sigma=[0.229,\;0.224,\;0.225]
$$

For each channel, normalization is performed as:

$$
x'=\frac{x-\mu}{\sigma}
$$

Using ImageNet normalization is particularly relevant because one of the experiments uses a VGG model with transfer learning from ImageNet-pretrained weights.

## 3.2 Validation and Test Preprocessing

A separate deterministic transformation pipeline is used for validation and test images. Unlike the training pipeline, it does not include random augmentation.

The validation/test pipeline consists of:

1. Image-to-tensor conversion
2. Resizing to 256 pixels
3. Center cropping to $224\times224$
4. Conversion to `float32` and scaling to $[0,1]$
5. ImageNet normalization

The resulting processing pipeline can be summarized as:

$$
\text{Image}
\rightarrow
\text{Resize}(256)
\rightarrow
\text{Center Crop}(224)
\rightarrow
\text{Float32}
\rightarrow
\text{ImageNet Normalization}
$$

Random transformations are intentionally avoided during validation and testing. This ensures that repeated evaluations of the same image produce the same input and that differences in model performance are attributable to the models rather than random augmentation.

## 3.3 Training and Validation Pipelines

The two pipelines can be summarized as follows:

| Stage            | Training                           | Validation/Test             |
| ---------------- | ---------------------------------- | --------------------------- |
| Image conversion | ToImage                            | ToImage                     |
| Resize           | 256                                | 256                         |
| Crop             | RandomResizedCrop $224\times224$ | CenterCrop $224\times224$ |
| Horizontal flip  | $p=0.5$                          | None                        |
| Brightness       | 0.3                                | None                        |
| Contrast         | 0.05                               | None                        |
| Gaussian blur    | $p=0.15$                         | None                        |
| Gaussian noise   | $p=0.15$                        | None                        |
| Data type        | Float32                            | Float32                     |
| Scaling          | $[0,255]\rightarrow[0,1]$        | $[0,255]\rightarrow[0,1]$ |
| Normalization    | ImageNet                           | ImageNet                    |

The training data loader uses the augmented training transformation, while the validation data loader uses the deterministic validation transformation. Consequently, the model sees different augmented versions of training images during different epochs, while the validation set remains unchanged throughout the experiments.

## 3.4 Purpose of the Augmentation Strategy

The primary purpose of the augmentation strategy is to reduce overfitting and improve the ability of the models to generalize to images that differ from the training samples in position, scale, illumination, sharpness, and noise level.

The augmentation strategy is deliberately moderate. Excessively strong transformations could remove or alter visual characteristics that are important for distinguishing similar vehicle classes. Therefore, transformations such as random cropping, horizontal flipping, moderate brightness and contrast changes, blur, and noise are used to introduce realistic variation while preserving the semantic identity of the vehicle.

The same preprocessing strategy is used when comparing the different model architectures, allowing the model comparison to focus on architectural differences rather than differences in the input preprocessing pipeline.

# 4. Model Development

The objective of the model development stage is to investigate different convolutional neural network architectures and determine which architecture is most suitable for the eight-class vehicle classification problem. Since the dataset is relatively limited in size, both lightweight custom architectures and a pretrained architecture are considered.

Four main families of models are implemented in this project: **Small Xception1, Small Xception2, Small Xception3, Small Xception with Squeeze-and-Excitation (SE), and VGG16 using transfer learning**. The custom architectures are designed to provide relatively lightweight models, while the VGG16 model provides an established pretrained architecture for comparison.

All classification models produce eight output logits corresponding to the eight known vehicle classes.

## 4.1 Separable Convolution

The custom models are based on depthwise separable convolutions. Unlike a conventional convolution, which simultaneously performs spatial filtering and channel mixing, a depthwise separable convolution separates these operations into two stages.

First, a **depthwise convolution** applies a spatial convolution independently to each input channel. This is followed by a **pointwise convolution** using a $1\times1$ kernel to combine information across channels.

Conceptually, the operation can be represented as:

$$
\text{Input}
\rightarrow
\text{Depthwise Convolution}
\rightarrow
\text{Pointwise Convolution}
\rightarrow
\text{Output}
$$

The implementation uses a depthwise convolution followed by a (1\times1) pointwise convolution. This approach reduces the number of parameters and computational operations compared with a standard convolution with the same input and output dimensions. The implementation of the separable convolution and the corresponding convolutional block is defined in the model module.

## 4.2 Xception-Style Residual Block

The custom architectures use an Xception-style residual block. Each block contains two (3\times3) depthwise separable convolutions with batch normalization and ReLU activation.

The output of the convolutional path is combined with a residual shortcut:

$$
y = F(x) + S(x)
$$

where (F(x)) represents the main convolutional path and (S(x)) represents the shortcut connection.

When the input and output dimensions are different, or when the block performs spatial downsampling, a (1\times1) convolution with the appropriate stride is used in the shortcut. Otherwise, an identity mapping is used.

Residual connections allow information from earlier layers to be propagated directly to deeper layers and help the network learn transformations relative to the input rather than learning the complete representation from scratch.

## 4.3 Small Xception1

The first custom architecture, **SmallXception1**, is designed as a compact Xception-style convolutional neural network.

The network begins with a (7\times7) convolution that maps the three RGB input channels to 32 channels, followed by batch normalization, ReLU activation, and max pooling. Four Xception-style blocks then progressively increase the number of feature channels:

$$
32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512
$$

The spatial resolution is reduced in the later blocks through a stride of 2. After the final feature extraction block, adaptive average pooling reduces the spatial dimensions to (1\times1). The resulting 512-dimensional feature vector is passed through dropout and a final linear classification layer.

The architecture can be summarized as:

```text
Input: 224 × 224 × 3
        ↓
7 × 7 Conv, 32 channels
        ↓
Max Pooling
        ↓
Xception Block: 32 → 64
        ↓
Xception Block: 64 → 128
        ↓
Xception Block: 128 → 256
        ↓
Xception Block: 256 → 512
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Fully Connected Layer
        ↓
8 Classes
```

The model also supports returning the feature vector before the final classification layer. This capability is useful later for the open-set/OOD recognition stage.

## 4.4 Small Xception2

**SmallXception2** is a smaller version of the first architecture. Its purpose is to investigate whether a substantially reduced model can provide competitive classification performance with fewer parameters and lower computational requirements.

The initial convolution uses 16 channels instead of 32, and the subsequent feature dimensions are:

$$
16 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256
$$

As with SmallXception1, four Xception-style residual blocks are used, followed by adaptive average pooling, dropout, and a fully connected classification layer.

The architecture is therefore approximately:

```text
Input: 224 × 224 × 3
        ↓
5 × 5 Conv, 16 channels
        ↓
Max Pooling
        ↓
Xception Block: 16 → 32
        ↓
Xception Block: 32 → 64
        ↓
Xception Block: 64 → 128
        ↓
Xception Block: 128 → 256
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Fully Connected Layer
        ↓
8 Classes
```

Comparing SmallXception1 and SmallXception2 allows the effect of model capacity on classification performance to be investigated.

## 4.5 Small Xception3 with Squeeze-and-Excitation

The third custom architecture, **SmallXception3**, retains the larger channel configuration of SmallXception1 but introduces **Squeeze-and-Excitation (SE) blocks** after each Xception block.

The channel dimensions are:

$$
32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512
$$

An SE block performs channel-wise attention. First, global average pooling summarizes each feature channel. The resulting channel descriptor is passed through a small fully connected network and a sigmoid activation to produce channel-wise weights. These weights are then multiplied with the original feature map:

$$
\tilde{x}=x\odot s(x)
$$

where (s(x)) represents the learned channel importance values.

In the implementation, an SE block is placed after each of the four Xception blocks.

The resulting architecture allows the network to adaptively emphasize informative feature channels and suppress less useful ones. The implementation also retains the ability to return the final feature representation for later use in OOD detection.

## 4.6 Small Xception with Integrated SE Block

A further custom architecture, **SmallXceptionSE**, investigates a different placement of the squeeze-and-excitation mechanism.

Instead of placing a separate SE block after every complete Xception block, the SE operation is integrated into the final Xception-style block. The first three blocks use the standard Xception-style structure, while the fourth block applies SE before the residual connection is added.

The overall channel progression remains:

$$
32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512
$$

and the final feature representation is reduced using adaptive average pooling before being passed through dropout and the classification layer.

This architecture provides another experimental configuration for determining whether channel attention is more effective when applied selectively within the deeper feature extraction stages.

## 4.7 VGG16 Transfer Learning

In addition to the custom architectures, **VGG16** is evaluated using transfer learning. The VGG16 backbone is initialized with weights pretrained on **ImageNet**.

The original VGG16 classification layer is replaced with a new linear layer whose output dimension corresponds to the eight vehicle classes.

The architecture used in this project can be summarized as:

```text
Input Image
     ↓
Pretrained VGG16 Features
     ↓
VGG16 Average Pooling
     ↓
Pretrained Classifier Layers
     ↓
New Classification Head
     ↓
8 Vehicle Classes
```

The final VGG16 linear layer is therefore adapted specifically to the vehicle classification task. The model implementation also returns the feature representation immediately before the final classification layer when requested. This representation is later useful for the distance-based OOD detection approach.

The use of transfer learning provides an important comparison with the custom models. The custom models must learn their visual representations from the vehicle dataset, whereas VGG16 begins with features learned from the much larger ImageNet dataset. Comparing these approaches can provide insight into whether pretrained visual representations are beneficial for the available vehicle dataset.

## 4.8 Model Factory and Experimental Comparison

A model factory is used to create the different architectures based on a model name. The implemented options include `small-xception1`, `small-xception2`, `small-xception3`, the SE-based architecture, and the VGG16-based model.

This common interface makes it possible to train and evaluate different architectures using the same general training and evaluation pipeline. Consequently, the comparison between architectures can focus on their classification performance rather than differences in implementation.

For each model, the total number of parameters and the number of trainable parameters can also be calculated. These values are considered when comparing the models, since the objective is not simply to maximize classification accuracy but to identify an architecture that provides a suitable balance between performance and model complexity.


### 4.9 Training Configuration

The classification models were trained using a common optimization and evaluation configuration. The **AdamW** optimizer was used for parameter optimization with an initial learning rate of $10^{-3}$:

$$
\text{Optimizer}=\text{AdamW},\qquad
\text{learning rate}=0.001
$$

AdamW was selected as the optimization algorithm for updating the trainable parameters of the neural networks.

The training objective was defined using **cross-entropy loss**, which is appropriate for the eight-class single-label classification problem:

$$
\mathcal{L}=\text{CrossEntropyLoss}(y,\hat{y})
$$

where $y$ represents the ground-truth class and $\hat{y}$ represents the model's output logits.

In addition to the training loss, **accuracy** and **macro F1-score** were used as evaluation metrics. Accuracy measures the proportion of correctly classified samples across all classes:

$$
\text{Accuracy}
=
\frac{\text{Number of Correct Predictions}}
{\text{Total Number of Predictions}}
$$

The macro F1-score calculates the F1-score independently for each of the eight classes and then takes their unweighted average:

$$
F1_{\text{macro}}
=
\frac{1}{8}
\sum_{i=1}^{8} F1_i
$$

Using macro F1 is particularly useful for this dataset because the class distribution is not perfectly balanced. It ensures that each vehicle class contributes equally to the reported F1-score rather than allowing classes with more training samples to dominate the metric.

The metrics were implemented using `torchmetrics` with a multiclass configuration and eight target classes. The same general training configuration was applied when training the different model architectures, allowing their classification performance to be compared under consistent optimization and evaluation conditions.

The training configuration can be summarized as follows:

| Component           | Configuration            |
| ------------------- | ------------------------ |
| Optimizer           | AdamW                    |
| Learning Rate       | $1\times10^{-3}$     |
| Loss Function       | Cross-Entropy Loss       |
| Classification Type | Multiclass               |
| Number of Classes   | 8                        |
| Evaluation Metrics  | Accuracy, Macro F1-score |

This common configuration complements the model architecture comparison described above and provides a consistent basis for evaluating the different custom Xception-based architectures and the VGG16 transfer-learning model.

# 5. Training Methodology

The training stage was designed to provide a consistent procedure for comparing the different vehicle classification architectures. The models were trained using supervised learning, with the eight known vehicle categories treated as the target classes. The training procedure includes stratified data splitting, balanced batch sampling, cross-entropy loss, AdamW optimization, accuracy and macro-F1 evaluation, and validation-based model selection.

## 5.1 Training and Validation Split

The available images from the eight known classes were divided into training and validation subsets using a **stratified split**. Stratification was used to preserve approximately the same class distribution in both subsets and prevent the smaller classes from being disproportionately represented in either split.

In the implemented training pipeline, 80% of the data is used for training and 20% for validation. A fixed random state of 42 is used to make the split reproducible. Separate dataset objects are created for the training and validation subsets so that training and validation transformations can be applied independently.

For additional experiments, a five-fold stratified cross-validation loader is also implemented. In this procedure, the dataset is divided into five folds while preserving the class distribution, with each fold serving as the validation set once.

## 5.2 Balanced Batch Sampling

Because the cleaned dataset contains an unequal number of images across the eight classes, a **balanced batch sampling** strategy is used during training.

The custom `BalancedBatchSampler` selects the same number of samples from each class for every training batch. In the main training configuration, four samples are selected from each class. Therefore, with eight classes, a complete training batch contains:

$$
8 \times 4 = 32
$$

images.

The sampler first creates a separate list of indices for every class, randomly shuffles the indices within each class, and then combines an equal number of samples from each class to construct each batch. This prevents the more frequent classes from dominating individual training batches.

Validation batches are not balanced artificially. Instead, the validation data is processed sequentially with a conventional batch size of 32 and without shuffling.

## 5.3 Optimization

For the custom Xception-based models, the **AdamW** optimizer is used. The initial learning rate is set to:

$$
\eta = 10^{-3}
$$

The optimizer updates the model parameters based on the gradients calculated from the training loss. AdamW is used because it combines adaptive gradient-based optimization with decoupled weight decay, providing a suitable optimization method for the relatively small custom CNN architectures.

The basic optimizer configuration is:

```text
Optimizer: AdamW
Initial learning rate: 0.001
```

The training loop performs the following operations for every batch:

1. Move the input images and labels to the selected device.
2. Perform a forward pass through the model.
3. Calculate the classification loss.
4. Backpropagate the loss.
5. Update the model parameters using AdamW.
6. Reset the gradients.
7. Update the training metrics.

These operations are repeated for all batches in each training epoch.

## 5.4 Classification Loss

The primary loss function is **categorical cross-entropy loss** (`CrossEntropyLoss`). Since each image belongs to exactly one of the eight known vehicle categories, the problem is formulated as a single-label multiclass classification task.

For an input image with true class (y), and model logits (z_1,\ldots,z_8), the loss can be expressed as:

$$
-\log
\left(
\frac{e^{z_y}}
{\sum_{j=1}^{8}e^{z_j}}
\right)
$$

The model therefore does not need to explicitly apply a softmax layer before passing its output to `CrossEntropyLoss`, since the loss internally applies the required log-softmax operation.

The training function also supports an alternative `BCEWithLogitsLoss` configuration using one-hot encoded targets. However, cross-entropy is the primary loss used for the standard multiclass training experiments.

## 5.5 Evaluation Metrics

Two main metrics are calculated during training and validation:

* **Classification accuracy**
* **Macro-averaged F1-score**

Accuracy measures the proportion of correctly classified samples:

$$
Accuracy =
\frac{N_{\mathrm{correct}}}{N_{\mathrm{total}}}
$$

However, because the dataset is not perfectly balanced, accuracy alone may not adequately describe the performance across all vehicle classes.

Therefore, **macro-F1** is also used. The F1-score is calculated separately for each class and then averaged equally across all eight classes:

$$
\frac{1}{8}
\sum_{c=1}^{8} F1_c
$$

This gives each vehicle category equal importance, regardless of the number of available images in that class.

The implementation uses `torchmetrics` with multiclass accuracy and macro-F1 configured for eight classes.

## 5.6 Training and Validation Procedure

At the beginning of each epoch, the model is placed in training mode. The training metrics are reset, and the model processes all batches in the training loader.

After the training phase, the model is evaluated on the validation set with gradient computation disabled. The validation procedure places the model in evaluation mode and calculates both validation accuracy and validation macro-F1.

For each epoch, the following quantities are recorded:

* Training loss
* Training accuracy
* Training macro-F1
* Validation accuracy
* Validation macro-F1
* Learning rate

These values provide a record of the training process and make it possible to analyze convergence and potential overfitting.

## 5.7 Model Selection

The **validation macro-F1 score** is used as the primary criterion for selecting the best model checkpoint.

After every epoch, the current validation macro-F1 is compared with the best value observed so far. If the current value is higher, the model state is saved as the new best model.

This criterion was selected because the dataset is class-imbalanced and macro-F1 gives equal importance to all eight vehicle categories. Consequently, model selection is not based solely on overall accuracy or the performance of the more frequent classes.

## 5.8 Learning-Rate Scheduling

The training implementation supports two learning-rate scheduling strategies:

1. **ReduceLROnPlateau**
2. **Cosine Annealing**

With `ReduceLROnPlateau`, the learning rate is reduced when the validation macro-F1 stops improving. With cosine annealing, the learning rate follows a cosine decay schedule over the training epochs.

For the general training function, scheduler usage is configurable, allowing experiments to be performed with or without learning-rate scheduling.

## 5.9 VGG16 Fine-Tuning Strategy

The VGG16 model uses a more specialized training procedure because it is initialized with ImageNet-pretrained weights.

The first 24 layers of the VGG feature extractor are frozen, which prevents their pretrained parameters from being updated during training. The remaining trainable layers are fine-tuned on the vehicle dataset.

A **discriminative learning-rate strategy** is used for VGG16. The learning rate for the unfrozen pretrained backbone is set lower than the learning rate of the newly initialized classification head:

$$
\eta_{\mathrm{backbone}} = 10^{-4}
$$

$$
\eta_{\mathrm{head}} = 10^{-3}
$$

This allows the pretrained representation to be modified gradually while allowing the newly initialized classification head to adapt more rapidly to the vehicle classification task. The optimizer is again AdamW, with weight decay applied to appropriate parameter groups.

The VGG16 training procedure additionally supports:

* cosine learning-rate scheduling with warm-up,
* validation-based learning-rate reduction,
* gradient clipping,
* automatic mixed precision on CUDA devices,
* label smoothing,
* optional class weighting,
* early stopping.

The best VGG16 model is selected according to validation macro-F1 and the corresponding model state is restored after training.

## 5.10 Reproducibility

A fixed random seed of **42** is used for the principal sources of randomness, including Python's random module, NumPy, and PyTorch. This helps make dataset splitting and training experiments more reproducible.

Overall, the training methodology combines balanced sampling, supervised multiclass optimization, class-aware evaluation, and validation-based model selection. The same general principles are used to compare the custom Xception-based architectures, while the VGG16 model receives a dedicated transfer-learning and fine-tuning strategy.

# 6. Model Selection

The model-selection stage was performed in two steps. First, several experiments were conducted using the custom Xception-based architectures and different optimization configurations. The purpose was to determine which architecture and training configuration produced the strongest validation performance. Second, the best-performing Xception configuration was trained again and compared with the VGG16 transfer-learning model.

The primary criterion for model selection was **validation macro-F1**, while validation accuracy was also reported as a complementary metric. Macro-F1 was given particular importance because the dataset contains an unequal number of images across the eight vehicle classes.

## 6.1 Xception Model Experiments

The first set of experiments focused on the custom Xception architectures. The experiments investigated the effects of learning-rate scheduling, weight decay, dropout, model capacity, and the use of Squeeze-and-Excitation (SE) blocks.

Table 3 summarizes the main experiments.

| Model    | Architecture        | Weight Decay | Scheduler         | Dropout | Best Val. Accuracy | Best Val. Macro-F1 |
| -------- | ------------------- | -----------: | ----------------- | ------: | -----------------: | -----------------: |
| Model 1  | SmallXception1      |            0 | None              | Default |             0.9375 |             0.9361 |
| Model 2  | SmallXception1      |            0 | ReduceLROnPlateau | Default |             0.9300 |         **0.9366** |
| Model 3  | SmallXception1      |       1×10⁻⁴ | ReduceLROnPlateau | Default |             0.9375 |             0.9360 |
| Model 4  | SmallXception1      |       1×10⁻⁴ | None              | Default |             0.9350 |             0.9334 |
| Model 5  | SmallXception1      |       1×10⁻⁴ | ReduceLROnPlateau |     0.5 |             0.9350 |             0.9332 |
| Model 6  | SmallXception1      |            0 | Cosine Annealing  | Default |         **0.9500** |         **0.9486** |
| Model 7  | SmallXception1      |       1×10⁻⁴ | Cosine Annealing  | Default |             0.9425 |             0.9417 |
| Model 8  | SmallXception1      |       5×10⁻⁴ | Cosine Annealing  | Default |             0.9375 |             0.9369 |
| Model 9  | SmallXception2      |            0 | Cosine Annealing  |     0.2 |             0.9225 |             0.9206 |
| Model 10 | SmallXception3 + SE |            0 | Cosine Annealing  |     0.2 |             0.9325 |             0.9300 |
| Model 11 | SmallXception3 + SE |       1×10⁻⁴ | Cosine Annealing  |     0.2 |             0.9350 |             0.9336 |

The results show that **SmallXception1 with cosine annealing and no weight decay** produced the strongest performance among the custom-model experiments. It achieved a maximum validation accuracy of **95.00%** and a validation macro-F1 of **94.86%**.

Several observations can be made from these experiments.

First, the learning-rate schedule had a noticeable effect on the performance of SmallXception1. The experiment without scheduling achieved a validation accuracy of 93.75%, whereas the cosine-annealing configuration increased this value to 95.00%.

Second, adding weight decay did not improve the performance of the best cosine-annealing configuration. With weight decay of $10^{-4}$, the validation macro-F1 decreased from 0.9486 to 0.9417. Increasing the weight decay to $5\times10^{-4}$ resulted in a further decrease to 0.9369.

Third, reducing the architecture size in **SmallXception2** resulted in lower validation performance. Its best validation accuracy was 92.25% and its macro-F1 was 92.06%. This suggests that, for this dataset, the additional representational capacity of SmallXception1 was beneficial.

Finally, introducing SE blocks in **SmallXception3** did not improve the classification performance in these experiments. The best SE configuration achieved a validation accuracy of 93.50% and a macro-F1 of 93.36%, both below the best SmallXception1 configuration.

Therefore, among the custom architectures and configurations tested, **SmallXception1 with AdamW, an initial learning rate of $10^{-3}$, no weight decay, and cosine-annealing learning-rate scheduling** was selected for further evaluation.

## 6.2 Final Training of the Selected Xception Model

After identifying the strongest configuration, a final SmallXception1 model was trained using the selected hyperparameters.

The final configuration was:

| Parameter              | Configuration       |
| ---------------------- | ------------------- |
| Architecture           | SmallXception1      |
| Number of classes      | 8                   |
| Optimizer              | AdamW               |
| Initial learning rate  | $10^{-3}$         |
| Weight decay           | 0                   |
| Loss                   | Cross-Entropy       |
| Scheduler              | Cosine Annealing    |
| Maximum epochs         | 500                 |
| Model-selection metric | Validation macro-F1 |
| Random seed            | 42                  |

The maximum number of training epochs was increased to 500 to allow the selected configuration sufficient opportunity to converge. The training procedure continued to monitor validation macro-F1 and saved the model whenever a new best validation macro-F1 was achieved.

The final training run achieved:

* **Best validation accuracy: 95.89%**
* **Best validation macro-F1: 95.88%**

The improvement over the earlier SmallXception1 experiment indicates that allowing a longer training process with the selected configuration produced a stronger model.

The resulting model checkpoint was saved as the selected Xception model and was subsequently used for further evaluation and open-set/OOD experiments.

## 6.3 Comparison with VGG16

The selected Xception model was subsequently compared with the VGG16 transfer-learning model developed in the previous stages.

The VGG16 model was initialized with ImageNet-pretrained weights. The first 24 feature layers were frozen, while the remaining trainable layers and the newly initialized classification head were fine-tuned using different learning rates for the backbone and classification head.

The VGG16 configuration used:

* Backbone learning rate: $10^{-4}$
* Classification-head learning rate: $10^{-3}$
* Weight decay: $10^{-2}$
* Frozen feature layers: 24
* Label smoothing: 0.1
* Scheduler: cosine annealing
* Maximum epochs: 60
* Early-stopping patience: 15

The VGG16 model achieved:

* **Best validation accuracy: 97.81%**
* **Best validation macro-F1: 96.00%**

The results are summarized below.

| Model                   | Best Val. Accuracy | Best Val. Macro-F1 |
| ----------------------- | -----------------: | -----------------: |
| Selected SmallXception1 |             95.89% |             95.88% |
| VGG16                   |         **97.81%** |         **96.00%** |

VGG16 therefore achieved the highest validation accuracy and the highest validation macro-F1 among the final models. Its validation accuracy was approximately **1.92 percentage points higher** than that of the selected SmallXception1 model, while its macro-F1 was approximately **0.12 percentage points higher**.

Based on these validation results, **VGG16 was selected as the final classification model** for the subsequent stages of the project.

The selected VGG16 model is particularly important for the open-set stage because its feature representation before the final classification layer can also be extracted. This representation can later be used to measure the distance between an input sample and the feature distributions of the known vehicle classes.


## 7. Final Model Evaluation

### 7.1 Confusion Matrix Analysis

After model selection, the final SmallXception1 and VGG16 models were evaluated on the evaluation set. The confusion matrices were used to analyze class-level performance and identify systematic classification errors. The rows correspond to the true classes, while the columns represent the predicted classes. The class order is:

**ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet.**

For the SmallXception1 model, 350 out of 365 samples were correctly classified, corresponding to an overall accuracy of approximately **95.89%** on this evaluation set. The normalized confusion matrix shows that the per-class recall ranged from **93.33% to 98.00%**.

The strongest recall was obtained for **ambulance (98.00%)**, followed by **autobus (96.00%)**, **minibus (96.00%)**, **savari (96.00%)**, and **taxi (96.00%)**. The lowest recall was observed for **vanet (93.33%)**, although this class still achieved relatively strong performance despite having substantially fewer training samples than most other classes.

The main misclassification patterns for SmallXception1 were:

* **kamyun → autobus** and **kamyun → taxi**
* **kamyunet → savari** and **kamyunet → autobus**
* **minibus → kamyun**
* **taxi → savari**
* **vanet → ambulance**

These errors indicate that some visually similar vehicle categories remain difficult to distinguish, particularly among the larger vehicle classes.

### 7.2 VGG16 Confusion Matrix

The VGG16 model produced **356 correct predictions out of 365 evaluation samples**, corresponding to an accuracy of approximately **97.53%** on this particular evaluation set.

The normalized confusion matrix shows that VGG16 achieved **100% recall** for **autobus, savari, and taxi**. Recall for **ambulance** was 98.00%, while **minibus** achieved 96.00%. The recall values for **kamyun** and **kamyunet** were both approximately 94.87% and 95.08%, respectively. As with SmallXception1, **vanet** had the lowest recall at 93.33%.

Compared with SmallXception1, VGG16 reduced the number of misclassified samples from **15 to 9** on this evaluation set. In particular, it eliminated several errors involving **autobus, savari, taxi, and ambulance**, although some confusion remained between **kamyun, kamyunet, and minibus**.

### 7.3 Analysis of Misclassified Images

The misclassified images provide additional insight into the failure cases of both models.


![alt text](image-1.png)
For SmallXception1, the misclassified examples show that errors are concentrated among visually similar vehicle types. For example, several **kamyun, kamyunet, minibus, and savari** samples are confused with one another. These classes can share similar body shapes, viewpoints, colors, and frontal appearances, making fine-grained classification more difficult.

![Image](image.png)
The VGG16 misclassification examples contain fewer errors. Nevertheless, similar patterns remain: **kamyun, kamyunet, and minibus** are still occasionally confused. For example, several samples visually resemble neighboring vehicle categories because of their viewing angle, vehicle size, or limited visible body structure.

Some errors also occur for **vanet**. This is particularly notable because vanet is the smallest class in the cleaned dataset, containing only **162 images**, whereas several other classes contain more than 400 samples. The lower number of training examples may contribute to the model having fewer opportunities to learn the visual variability of this class; however, the confusion matrix alone does not establish this as the cause.

### 7.4 Comparison of the Final Models

The confusion matrices demonstrate that VGG16 produced fewer classification errors than SmallXception1 on the evaluation set.

| Model          |   Correct | Incorrect | Evaluation Accuracy |
| -------------- | --------: | --------: | ------------------: |
| SmallXception1 | 350 / 365 |        15 |              95.89% |
| VGG16          | 356 / 365 |         9 |              97.53% |

VGG16 therefore correctly classified **6 additional samples** compared with SmallXception1 on this evaluation set. Its errors were also more concentrated in the visually similar **kamyun–kamyunet–minibus** group.

Overall, the confusion-matrix analysis supports the earlier model-selection results: **VGG16 provides the strongest classification performance among the evaluated models**. It was therefore selected as the final classification backbone for the subsequent **open-set/OOD evaluation**.

# 11. Limitations

Although the proposed system achieved strong classification performance, several limitations should be considered when interpreting the results.

First, the dataset is relatively limited in size and is not uniformly distributed across the eight vehicle classes. In particular, the **vanet** class contains substantially fewer images than the other classes. Although stratified splitting and balanced batch sampling were used to reduce the effect of class imbalance during training, a limited number of samples may still restrict the model's ability to learn the full visual variability of some vehicle categories.

Second, several target classes are visually similar. The confusion-matrix analysis showed that most remaining errors occurred among categories such as **kamyun, kamyunet, minibus, and savari**. These vehicles can have similar shapes, frontal appearances, colors, and viewing angles. Therefore, even the final VGG16 model occasionally confuses visually similar categories.

Third, the images have considerable variation in viewpoint, scale, illumination, cropping, and image quality. Although data cleaning and augmentation were applied to improve robustness, the available dataset may not fully represent the variability encountered in real-world deployment. Consequently, the reported performance should not automatically be interpreted as equivalent to performance on completely unseen real-world data.

Another limitation is the dependence on the vehicle-extraction stage. YOLO11n is used to detect and crop generic vehicle regions before classification. Errors in this preprocessing stage, such as inaccurate bounding boxes, incomplete crops, or missed detections, can affect the input provided to the classification model. Therefore, the complete system's performance depends not only on the classifier but also on the quality of vehicle-region extraction.

The open-set component also has a limited evaluation scope. The **neysan** category is deliberately excluded from training and is used as an unseen category for open-set evaluation. While this provides a useful test of whether the system can reject a class outside the known categories, a single unseen category cannot represent the full range of possible unknown objects or vehicle types that may occur in a real deployment environment.

Finally, the model selection experiments cover a limited set of architectures and hyperparameter configurations. The experiments compare several custom Xception-based variants and VGG16, but they do not constitute an exhaustive search over architectures, optimization strategies, augmentation policies, or hyperparameters. Therefore, it is possible that other configurations could achieve better performance.

# 12. Future Work

Several directions can be considered to further improve the proposed vehicle classification system.

One important direction is to **expand and rebalance the dataset**. Collecting additional images, particularly for underrepresented classes such as vanet, would provide the models with more examples of intra-class variation. Additional data from different cameras, locations, illumination conditions, viewpoints, and vehicle appearances could also improve generalization.

A second direction is to improve the classification of visually similar vehicle categories. More specialized fine-grained classification techniques could be investigated to distinguish classes such as **kamyun, kamyunet, minibus, and savari**. Attention mechanisms, stronger feature-extraction architectures, higher-resolution inputs, and metric-learning approaches could be explored for this purpose.

The vehicle-extraction stage could also be improved. Instead of relying only on generic COCO vehicle categories from YOLO11n, a detector specifically trained for the target vehicle types could potentially produce more accurate and consistent crops. The detector and classifier could also be investigated as a more integrated pipeline rather than treating them as completely separate stages.

Another important direction is to extend the **open-set recognition** component. The current setup uses neysan as an unseen category, but future experiments should include multiple unknown vehicle categories and more diverse out-of-distribution images. This would provide a more comprehensive evaluation of the system's ability to distinguish known classes from previously unseen inputs.

Additional OOD methods could also be compared with the current approach. For example, confidence-based thresholds, energy-based scoring, Mahalanobis distance, embedding-space methods, and other open-set recognition techniques could be evaluated under the same experimental conditions. Metrics such as AUROC, FPR95, and open-set classification accuracy could then be used to compare their effectiveness.

The final VGG16 model could also be optimized for practical deployment. Although VGG16 provided the strongest classification performance among the evaluated models, its computational and memory requirements are relatively high compared with the smaller custom architectures. Model compression techniques such as knowledge distillation, pruning, quantization, or a lightweight architecture could therefore be investigated to achieve a better trade-off between accuracy and computational cost.

Finally, the complete pipeline could be evaluated in a real-time setting. Such an evaluation would combine vehicle detection, cropping, classification, and unknown-class rejection into a single system and measure not only classification accuracy but also inference time, memory consumption, and robustness under realistic operating conditions.

# 13. Conclusion

This project developed a deep-learning-based system for classifying vehicle images into eight known categories: **ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, and vanet**. The development process included dataset cleaning, duplicate and invalid-image detection, vehicle-region extraction, image preprocessing and augmentation, development of custom Xception-style architectures, transfer learning with VGG16, and systematic model comparison.

A number of custom Xception-based models were developed using separable convolutions and residual connections, with additional experiments incorporating Squeeze-and-Excitation blocks. The experiments demonstrated that architectural complexity alone did not necessarily improve performance. Among the custom models, the best configuration was **SmallXception1 with AdamW optimization and cosine annealing**, which achieved a validation accuracy of approximately **95.89%** and a validation macro-F1 of approximately **95.88%** in the final retraining.

VGG16 with ImageNet-pretrained weights was subsequently fine-tuned using a discriminative learning-rate strategy. It achieved the strongest validation performance, with a validation accuracy of approximately **97.81%** and a macro-F1 score of approximately **96.00%**. The confusion-matrix evaluation further demonstrated the strong performance of VGG16, with **356 correct predictions out of 365 evaluation samples**, corresponding to approximately **97.53% evaluation accuracy**. The remaining errors were concentrated primarily among visually similar vehicle categories.

The analysis of misclassified images confirmed that the main difficulty is not general recognition of vehicles, but distinguishing between visually similar target categories. This highlights the importance of fine-grained visual features and motivates further work on attention mechanisms, higher-quality data, and specialized feature-learning approaches.

In addition to closed-set classification, the project establishes a foundation for **open-set recognition**, where vehicle categories not present during training can be identified as unknown. The deliberately excluded neysan class provides an initial unseen category for evaluating this capability.

Overall, the experiments demonstrate that transfer learning with VGG16 provides the strongest classification performance among the investigated models, while the smaller Xception-based architecture offers a useful alternative when computational efficiency is more important. The resulting system provides a foundation for a more complete vehicle recognition pipeline that combines detection, fine-grained classification, and open-set recognition.
