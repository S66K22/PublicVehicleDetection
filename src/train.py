import logging
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
# import torchmetrics
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import math
import torchmetrics
import copy

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, samples_per_class):
        self.dataset = dataset
        self.samples_per_class = samples_per_class

        # Get labels from the dataset
        labels = [dataset[i][1] for i in range(len(dataset))]

        self.class_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = sorted(self.class_to_indices.keys())

    def __iter__(self):
        # Shuffle indices of every class
        class_indices = {
            cls: random.sample(indices, len(indices))
            for cls, indices in self.class_to_indices.items()
        }

        # Number of complete balanced batches
        num_batches = min(
            len(indices) // self.samples_per_class for indices in class_indices.values()
        )

        for batch_idx in range(num_batches):
            batch = []

            for cls in self.classes:
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class

                batch.extend(class_indices[cls][start:end])

            random.shuffle(batch)

            yield batch

    def __len__(self):
        return min(
            len(indices) // self.samples_per_class
            for indices in self.class_to_indices.values()
        )


def create_dataloader(
    path, transformation, samples_per_class=4, val=False, batch_size=32
):
    dataset = ImageFolder(
        path,
        transform=transformation,
    )

    batch_sampler = BalancedBatchSampler(
        dataset,
        samples_per_class=samples_per_class,
    )

    if val:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
        )

    return loader, dataset.classes


def create_cv_loaders(
    path,
    train_transform=None,
    batch_size=32,
    n_splits=5,
    samples_per_class=4,
    random_state=42,
):
    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.33),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.05,
                    hue=0.05,
                ),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    val_transform = transforms.Compose(
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

    # Separate datasets because they use different transforms.
    train_full = ImageFolder(
        path,
        transform=train_transform,
    )

    val_full = ImageFolder(
        path,
        transform=val_transform,
    )

    labels = train_full.targets
    indices = list(range(len(train_full)))

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (train_indices, val_indices) in enumerate(
        skf.split(indices, labels),
        start=1,
    ):
        train_dataset = Subset(
            train_full,
            train_indices,
        )

        val_dataset = Subset(
            val_full,
            val_indices,
        )

        batch_sampler = BalancedBatchSampler(
            train_dataset,
            samples_per_class=samples_per_class,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Log distribution for this fold.
        train_counts = Counter(labels[i] for i in train_indices)

        val_counts = Counter(labels[i] for i in val_indices)

        logger.info(f"========== Fold {fold}/{n_splits} ==========")

        for class_idx, class_name in enumerate(train_full.classes):
            logger.info(
                f"{class_name:10s} "
                f"train={train_counts[class_idx]:4d} "
                f"val={val_counts[class_idx]:4d}"
            )

        yield fold, train_loader, val_loader, train_full.classes


def create_train_val_loader(path, batch_size=32, test_size=0.2):
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.33),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.05,
                saturation=0.01,
                hue=0.05,
            ),
            # transforms.RandomGrayscale(p=0.05),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
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
    train_full = ImageFolder(
        path,
        transform=train_transform,
    )

    val_full = ImageFolder(
        path,
        transform=val_transform,
    )

    labels = train_full.targets
    indices = list(range(len(train_full)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    batch_sampler = BalancedBatchSampler(
        train_loader.dataset,
        samples_per_class=4,
    )

    balanced_train_loader = DataLoader(
        train_loader.dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
    )
    train_counts = Counter(train_full.targets[i] for i in train_indices)
    val_counts = Counter(train_full.targets[i] for i in val_indices)

    for class_idx, class_name in enumerate(train_full.classes):
        logger.info(
            f"{class_name:10s} "
            f"train={train_counts[class_idx]:4d} "
            f"val={val_counts[class_idx]:4d}"
        )

    return balanced_train_loader, val_loader, train_full.classes


def create_subset_from_loader(dataloader, num_imgs_per_class=3):
    dataset = dataloader.dataset

    # Get labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    indices = []

    for class_id in range(8):
        class_indices = np.where(labels == class_id)[0]
        indices.extend(class_indices[:3])  # 3 images per class

    small_dataset = Subset(dataset, indices)

    small_loader = DataLoader(
        small_dataset,
        batch_size=len(small_dataset),
        shuffle=True,
    )
    return small_loader


def evaluate_tm(
    model,
    data_loader,
    accuracy_metric,
    f1_metric,
    device,
    loss_fn=None
):
    model.eval()

    accuracy_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            accuracy_metric.update(y_pred, y_batch)
            f1_metric.update(y_pred, y_batch)

    accuracy = accuracy_metric.compute()
    macro_f1 = f1_metric.compute()

    if loss_fn is None:
        return accuracy, macro_f1
    else:
        return None, accuracy, macro_f1


def train(
    model,
    optimizer,
    loss_fn,
    accuracy_metric,
    f1_metric,
    train_loader,
    valid_loader,
    n_epochs,
    device,
    loss_type="ce",
    layers_to_eval=None,
    path_to_save="",
    use_scheduler=False,
    cosine_annealing=False,
    patience=10,
    factor=0.1,
):
    if use_scheduler:
        if not cosine_annealing:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=patience,
                factor=factor,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=n_epochs,
                eta_min=1e-6,
            )

    history = {
        "train_losses": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "valid_accuracy": [],
        "valid_macro_f1": [],
        "learning_rates": [],
    }

    logger.debug("=" * 70)

    best_val_macro_f1 = 0.0
    for epoch in range(n_epochs):
        total_loss = 0.0
        accuracy_metric.reset()
        f1_metric.reset()
        model.train()
        if layers_to_eval is not None:
            layers_to_eval.eval()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)

            if loss_type == "ce":
                loss = loss_fn(y_pred, y_batch)

            elif loss_type == "bce":
                targets_one_hot = F.one_hot(
                    y_batch,
                    num_classes=8
                ).float()

                loss = loss_fn(y_pred, targets_one_hot)

            else:
                raise ValueError(
                    f"Unknown loss type: {loss_type}"
                )

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accuracy_metric.update(y_pred, y_batch)
            f1_metric.update(y_pred, y_batch)

        train_accuracy = accuracy_metric.compute().item()
        train_macro_f1 = f1_metric.compute().item()

        history["train_losses"].append(
            total_loss / len(train_loader)
        )

        history["train_accuracy"].append(train_accuracy)
        history["train_macro_f1"].append(train_macro_f1)

        val_accuracy, val_macro_f1 = evaluate_tm(
            model,
            valid_loader,
            accuracy_metric,
            f1_metric,
            device,
        )

        val_accuracy = val_accuracy.item()
        val_macro_f1 = val_macro_f1.item()

        history["valid_accuracy"].append(val_accuracy)
        history["valid_macro_f1"].append(val_macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        
        history["learning_rates"].append(current_lr)
        # lr_string = ", ".join(f"{current_lr:.2e}")
        
        if use_scheduler:
            if not cosine_annealing:
                scheduler.step(val_macro_f1)
            else:
                scheduler.step()

        logger.debug(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {history['train_losses'][-1]:.4f}, "
            f"train acc: {train_accuracy:.4f}, "
            f"train macro F1: {train_macro_f1:.4f}, "
            f"val acc: {val_accuracy:.4f}, "
            f"val macro F1: {val_macro_f1:.4f}, "
            f"lr: {current_lr}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

            if len(path_to_save) > 0:
                torch.save(
                    model.state_dict(),
                    path_to_save
                )
    return history


def extract_features(loader, model, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, f = model(x.to(device), return_features=True)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def calc_centroids_cov_inv(train_feats, train_labels, num_classes=8):
    centroids = np.stack(
        [train_feats[train_labels == c].mean(axis=0) for c in range(num_classes)]
    )

    centered = np.concatenate(
        [train_feats[train_labels == c] - centroids[c] for c in range(num_classes)]
    )
    cov = np.cov(centered, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6  # regularize for invertibility
    cov_inv = np.linalg.inv(cov)

    return centroids, cov_inv


def mahalanobis_min_distance(feat_vec, centroids, cov_inv):
    diffs = feat_vec[None, :] - centroids
    dists = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)  # squared distance
    return dists.min(), dists.argmin()


def fit_temperature(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            logits_list.append(model(x.to(device)).cpu())
            labels_list.append(y)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_vgg_features(model, n_frozen=24):
    """Freeze model.features[:n_frozen]. 24 = everything up to and including conv4_3 block."""
    for p in model.features[:n_frozen].parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {n_train:,} / {n_total:,}")


def build_vgg_optimizer(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=1e-2):
    """
    AdamW with discriminative LRs:
      - unfrozen conv layers + FC trunk -> lr_backbone (pretrained, be gentle)
      - new head                        -> lr_head     (random init, can move faster)
    Frozen params are excluded; biases get no weight decay.
    """

    def split(params):
        decay, no_decay = [], []
        for p in params:
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return decay, no_decay

    groups = []
    backbone_params = list(model.features.parameters()) + list(model.classifier_trunk.parameters())
    for params, lr in ((backbone_params, lr_backbone), (list(model.head.parameters()), lr_head)):
        decay, no_decay = split(params)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, kind, n_epochs, warmup_epochs=2, plateau_patience=5, plateau_factor=0.5):
    if kind is None:
        return None
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=plateau_patience, factor=plateau_factor, min_lr=1e-7
        )
    if kind == "cosine":
        # Linear warmup, then cosine decay to ~0. Stepped once per epoch.
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {kind}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_vgg(model, loader, loss_fn, acc_metric, f1_metric, device, use_amp=False):
    model.eval()
    acc_metric.reset()
    f1_metric.reset()
    total_loss, n_samples = 0.0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(X)
            loss = loss_fn(logits, y)
        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)
        acc_metric.update(logits.float(), y)
        f1_metric.update(logits.float(), y)
    return total_loss / n_samples, acc_metric.compute().item(), f1_metric.compute().item()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_vgg(
    model,
    train_loader,
    valid_loader,
    device,
    num_classes,
    n_epochs=50,
    lr_backbone=1e-4,
    lr_head=1e-3,
    weight_decay=1e-2,
    n_frozen=24,
    label_smoothing=0.0,
    class_weights=None,          # tensor of shape [num_classes] or None
    scheduler_type="cosine",     # "cosine" | "plateau" | None
    warmup_epochs=2,
    grad_clip=1.0,
    use_amp=True,
    early_stop_patience=15,
    save_path=None,
    seed=42,
):
    """
    Fine-tunes a VGGOpenSet model with cross-entropy.
    Selects the best epoch by validation macro-F1, saves that state_dict to
    `save_path`, and returns (model_with_best_weights, history).
    """
    seed_everything(seed)
    device = torch.device(device)
    model = model.to(device)
    
    # 1. freeze first, THEN build optimizer (so frozen params are excluded)
    freeze_vgg_features(model, n_frozen)
    optimizer = build_vgg_optimizer(model, lr_backbone, lr_head, weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_type, n_epochs, warmup_epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    # validation loss is unweighted/unsmoothed so it is comparable across runs
    val_loss_fn = nn.CrossEntropyLoss()

    # 2. separate metric objects for train / val
    def make_metrics():
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        return acc, f1

    train_acc, train_f1 = make_metrics()
    val_acc, val_f1 = make_metrics()

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {k: [] for k in (
        "train_loss", "train_acc", "train_f1",
        "val_loss", "val_acc", "val_f1",
        "lr_backbone", "lr_head", "grad_norm",
    )}
    best_f1, best_epoch, best_state, epochs_no_improve = -1.0, -1, None, 0

    for epoch in range(1, n_epochs + 1):
        # ---------------- train ----------------
        model.train()
        train_acc.reset()
        train_f1.reset()
        running_loss, n_samples, grad_norms, skipped = 0.0, 0, [], 0

        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(X)
                loss = loss_fn(logits, y)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip if grad_clip else float("inf"),
            )
            grad_norms.append(gn.item())
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * X.size(0)
            n_samples += X.size(0)
            train_acc.update(logits.detach().float(), y)
            train_f1.update(logits.detach().float(), y)

        if n_samples == 0:
            raise RuntimeError("All batches produced non-finite loss. Lower the learning rate.")
        if skipped:
            logger.warning(f"Epoch {epoch}: skipped {skipped} batches with non-finite loss")

        tr_loss = running_loss / n_samples
        tr_acc, tr_f1 = train_acc.compute().item(), train_f1.compute().item()

        # ---------------- validate ----------------
        v_loss, v_acc, v_f1 = evaluate_vgg(
            model, valid_loader, val_loss_fn, val_acc, val_f1, device, amp_enabled
        )

        # ---------------- bookkeeping ----------------
        lrs = [g["lr"] for g in optimizer.param_groups]
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)
        history["lr_backbone"].append(lrs[0])
        history["lr_head"].append(lrs[-1])
        history["grad_norm"].append(float(np.mean(grad_norms)) if grad_norms else float("nan"))

        # step scheduler AFTER logging the lr that was actually used this epoch
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(v_f1)
            else:
                scheduler.step()

        logger.info(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
            f"val loss {v_loss:.4f} acc {v_acc:.4f} f1 {v_f1:.4f} | "
            f"lr bb {lrs[0]:.2e} head {lrs[-1]:.2e} | grad {history['grad_norm'][-1]:.2f}"
        )

        # ---------------- checkpoint / early stop ----------------
        if v_f1 > best_f1:
            best_f1, best_epoch, epochs_no_improve = v_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_state, save_path)
        else:
            epochs_no_improve += 1
            if early_stop_patience and epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}. Best val F1 {best_f1:.4f} @ epoch {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_f1
    return model, history


def load_weights(model, path, device, weights_only=True):   
    state_dict = torch.load(
            path,
            map_location=device,
            weights_only=weights_only,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model