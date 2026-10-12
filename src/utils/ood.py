import random
import logging
import numpy as np
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Feature extraction
# ============================================================

def extract_features(loader, model, device):
    """
    Extract logits, deep features and labels.

    Returns
    -------
    features : np.ndarray
        Shape: [N, feature_dim]
    logits : np.ndarray
        Shape: [N, num_classes]
    labels : np.ndarray
        Shape: [N]
    """

    model.eval()

    features = []
    logits = []
    labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            batch_logits, batch_features = model(
                x,
                return_features=True
            )

            logits.append(batch_logits.cpu().numpy())
            features.append(batch_features.cpu().numpy())
            labels.append(y.cpu().numpy())

    return (
        np.concatenate(features, axis=0),
        np.concatenate(logits, axis=0),
        np.concatenate(labels, axis=0)
    )


# ============================================================
# PCA + Mahalanobis statistics
# ============================================================

def fit_mahalanobis(
    train_features,
    train_labels,
    num_classes,
    pca_dim=64
):
    """
    Fit:

        Standardization
             ↓
            PCA
             ↓
       class centroids
             ↓
       shared covariance
             ↓
      Ledoit-Wolf shrinkage

    PCA is particularly useful when the original feature
    dimension is large or highly correlated.
    """

    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)

    feature_dim = train_features.shape[1]

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca_dim = min(
        pca_dim,
        feature_dim,
        len(train_features) - 1
    )

    pca = PCA(
        n_components=pca_dim,
        whiten=True,
        random_state=42
    )

    reduced_features = pca.fit_transform(train_features)

    logger.info(
        "Original feature dimension: %d",
        feature_dim
    )

    logger.info(
        "PCA feature dimension: %d",
        pca_dim
    )

    logger.info(
        "Explained variance: %.4f",
        pca.explained_variance_ratio_.sum()
    )

    # --------------------------------------------------------
    # Class centroids
    # --------------------------------------------------------

    centroids = []

    for c in range(num_classes):

        class_features = reduced_features[
            train_labels == c
        ]

        if len(class_features) == 0:
            raise ValueError(
                f"No training samples found for class {c}"
            )

        centroids.append(
            class_features.mean(axis=0)
        )

    centroids = np.stack(centroids)

    # --------------------------------------------------------
    # Center samples around their class centroid
    # --------------------------------------------------------

    centered = np.zeros_like(reduced_features)

    for c in range(num_classes):

        mask = train_labels == c

        centered[mask] = (
            reduced_features[mask]
            - centroids[c]
        )

    # --------------------------------------------------------
    # Ledoit-Wolf covariance shrinkage
    # --------------------------------------------------------

    covariance_estimator = LedoitWolf()
    covariance_estimator.fit(centered)

    covariance = covariance_estimator.covariance_
    covariance_inv = covariance_estimator.precision_

    return {
        "pca": pca,
        "centroids": centroids,
        "covariance": covariance,
        "cov_inv": covariance_inv,
    }


# ============================================================
# Mahalanobis distance
# ============================================================

def mahalanobis_distances(
    features,
    centroids,
    cov_inv,
    pca
):
    """
    Calculate distance to every class centroid.

    Returns
    -------
    distances : [N, num_classes]
    """

    reduced = pca.transform(features)

    # [N, 1, D] - [1, C, D]
    diffs = (
        reduced[:, None, :]
        - centroids[None, :, :]
    )

    distances = np.einsum(
        "ncd,df,ncf->nc",
        diffs,
        cov_inv,
        diffs
    )

    return distances


def mahalanobis_min_distance(
    feat_vec,
    centroids,
    cov_inv,
    pca
):
    """
    Distance of one sample to its nearest class centroid.
    """

    distances = mahalanobis_distances(
        feat_vec[None, :],
        centroids,
        cov_inv,
        pca
    )[0]

    return (
        float(distances.min()),
        int(distances.argmin())
    )


# ============================================================
# Temperature scaling
# ============================================================

def fit_temperature(model, loader, device):

    model.eval()

    logits_list = []
    labels_list = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            logits_list.append(logits.cpu())
            labels_list.append(y.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Optimize log(T), rather than T directly.
    # This guarantees T > 0.
    log_temperature = nn.Parameter(
        torch.tensor(np.log(1.5), dtype=torch.float32)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.01,
        max_iter=50
    )

    criterion = nn.CrossEntropyLoss()

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)

        loss = criterion(
            logits / temperature,
            labels
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = torch.exp(
        log_temperature
    ).item()

    logger.info(
        "Temperature: %.4f",
        temperature
    )

    return temperature


# ============================================================
# Validation score
# ============================================================

def calculate_ood_scores(
    features,
    logits,
    mahalanobis_stats,
    temperature
):
    """
    Produce two OOD-related scores:

        1. Mahalanobis distance
        2. Maximum softmax probability

    Higher Mahalanobis = more OOD
    Lower MSP = more OOD
    """

    distances = mahalanobis_distances(
        features,
        mahalanobis_stats["centroids"],
        mahalanobis_stats["cov_inv"],
        mahalanobis_stats["pca"]
    )

    min_distances = distances.min(axis=1)

    probabilities = torch.softmax(
        torch.tensor(logits) / temperature,
        dim=1
    ).numpy()

    max_probs = probabilities.max(axis=1)

    return min_distances, max_probs


# ============================================================
# Threshold selection
# ============================================================

def find_threshold(
    id_scores,
    ood_scores
):
    """
    Find threshold using Youden's J statistic.

    ID = 0
    OOD = 1
    """

    scores = np.concatenate([
        id_scores,
        ood_scores
    ])

    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    j = tpr - fpr

    best_idx = np.argmax(j)

    threshold = thresholds[best_idx]

    return float(threshold)


# ============================================================
# Combined OOD threshold fitting
# ============================================================

def fit_ood_thresholds(
    id_features,
    id_logits,
    ood_features,
    ood_logits,
    mahalanobis_stats,
    temperature
):

    id_dist, id_conf = calculate_ood_scores(
        id_features,
        id_logits,
        mahalanobis_stats,
        temperature
    )

    ood_dist, ood_conf = calculate_ood_scores(
        ood_features,
        ood_logits,
        mahalanobis_stats,
        temperature
    )

    # Mahalanobis:
    # larger = more OOD
    dist_threshold = find_threshold(
        id_dist,
        ood_dist
    )

    # MSP:
    # smaller = more OOD
    #
    # Convert it into an OOD score:
    # 1 - confidence
    confidence_threshold = find_threshold(
        1.0 - id_conf,
        1.0 - ood_conf
    )

    return {
        "dist_threshold": dist_threshold,
        "confidence_threshold": 1.0 - confidence_threshold
    }


import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from PIL import Image

UNKNOWN_LABEL = "unknown"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


preprocess = transforms.Compose([
    transforms.ToImage(),

    transforms.ToDtype(
        torch.float32,
        scale=True
    ),

    transforms.Resize(
        (224, 224)
    ),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


def load_checkpoint(model, path, device):

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    model.eval()

    return model, ckpt


def predict(
    image_path,
    model,
    ckpt,
    device
):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    img = Image.open(
        image_path
    ).convert("RGB")

    x = preprocess(img)
    x = x.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        logits, features = model(
            x,
            return_features=True
        )

    # --------------------------------------------------------
    # Temperature-scaled probabilities
    # --------------------------------------------------------

    temperature = ckpt["temperature"]

    probs = torch.softmax(
        logits / temperature,
        dim=1
    )[0]

    pred_class = int(
        probs.argmax().item()
    )

    max_prob = float(
        probs.max().item()
    )

    # --------------------------------------------------------
    # Mahalanobis distance
    # --------------------------------------------------------

    feature_vector = (
        features[0]
        .cpu()
        .numpy()
    )

    distance, distance_class = (
        mahalanobis_min_distance(
            feature_vector,
            ckpt["centroids"],
            ckpt["cov_inv"],
            ckpt["pca"]
        )
    )

    # --------------------------------------------------------
    # OOD decision
    # --------------------------------------------------------

    distance_unknown = (
        distance >
        ckpt["dist_threshold"]
    )

    confidence_unknown = (
        max_prob <
        ckpt["conf_threshold"]
    )

    is_unknown = (
        distance_unknown
        or confidence_unknown
    )

    if is_unknown:
        label = UNKNOWN_LABEL
    else:
        label = ckpt["class_names"][pred_class]

    return {
        "label": label,

        "softmax_conf": max_prob,

        "mahalanobis_dist": distance,

        "mahalanobis_class":
            ckpt["class_names"][distance_class],

        "raw_pred_class":
            ckpt["class_names"][pred_class],

        "distance_unknown":
            distance_unknown,

        "confidence_unknown":
            confidence_unknown,
    }