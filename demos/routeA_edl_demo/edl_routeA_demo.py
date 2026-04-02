#!/usr/bin/env python3
"""
Route-A (EDL) training demo with method comparison.

Compares:
1) Softmax + CrossEntropy
2) Softmax + LabelSmoothing
3) EDL-like evidential classifier (MSE + variance + annealed evidence regularizer + small CE)

Dependencies:
- numpy
- autograd
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.extend import defvjp, primitive

warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version >=1\.23\.5 and <2\.5\.0 is required for this version of SciPy.*",
    category=UserWarning,
)
from scipy.special import digamma as sp_digamma
from scipy.special import gammaln as sp_gammaln
from scipy.special import polygamma as sp_polygamma


ArrayDict = Dict[str, anp.ndarray]


@primitive
def gammaln_p(x: anp.ndarray) -> anp.ndarray:
    return sp_gammaln(x)


defvjp(gammaln_p, lambda ans, x: lambda g: g * sp_digamma(x))


@primitive
def digamma_p(x: anp.ndarray) -> anp.ndarray:
    return sp_digamma(x)


defvjp(digamma_p, lambda ans, x: lambda g: g * sp_polygamma(1, x))


def stable_softplus(x: anp.ndarray) -> anp.ndarray:
    return anp.log1p(anp.exp(-anp.abs(x))) + anp.maximum(x, 0.0)


def relu(x: anp.ndarray) -> anp.ndarray:
    return anp.maximum(x, 0.0)


def softmax(logits: anp.ndarray) -> anp.ndarray:
    z = logits - anp.max(logits, axis=1, keepdims=True)
    ez = anp.exp(z)
    return ez / (anp.sum(ez, axis=1, keepdims=True) + 1e-12)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def clone_params(params: ArrayDict) -> ArrayDict:
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in params.items()}


def init_params(rng: np.random.Generator, input_dim: int, hidden1: int, hidden2: int, num_classes: int) -> ArrayDict:
    params: ArrayDict = {
        "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden1)),
        "b1": np.zeros(hidden1),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden1), size=(hidden1, hidden2)),
        "b2": np.zeros(hidden2),
        "W3": rng.normal(0.0, np.sqrt(2.0 / hidden2), size=(hidden2, num_classes)),
        "b3": np.zeros(num_classes),
    }
    return params


def forward_logits(params: ArrayDict, x: anp.ndarray) -> anp.ndarray:
    h1 = relu(anp.dot(x, params["W1"]) + params["b1"])
    h2 = relu(anp.dot(h1, params["W2"]) + params["b2"])
    logits = anp.dot(h2, params["W3"]) + params["b3"]
    return logits


def l2_penalty(params: ArrayDict) -> anp.ndarray:
    return anp.sum(params["W1"] ** 2) + anp.sum(params["W2"] ** 2) + anp.sum(params["W3"] ** 2)


def ce_loss(
    params: ArrayDict,
    x: anp.ndarray,
    y_onehot: anp.ndarray,
    l2: float = 1e-4,
    label_smoothing: float = 0.0,
) -> anp.ndarray:
    logits = forward_logits(params, x)
    probs = softmax(logits)
    k = y_onehot.shape[1]
    y_t = (1.0 - label_smoothing) * y_onehot + (label_smoothing / k)
    nll = -anp.sum(y_t * anp.log(probs + 1e-12), axis=1)
    return anp.mean(nll) + l2 * l2_penalty(params)


def edl_loss(
    params: ArrayDict,
    x: anp.ndarray,
    y_onehot: anp.ndarray,
    l2: float = 1e-4,
    reg_weight: float = 0.0,
    ce_aux_weight: float = 0.02,
) -> anp.ndarray:
    logits = forward_logits(params, x)
    evidence = stable_softplus(logits)
    alpha = evidence + 1.0
    alpha0 = anp.sum(alpha, axis=1, keepdims=True)
    p_mean = alpha / (alpha0 + 1e-12)

    mse = anp.sum((y_onehot - p_mean) ** 2, axis=1)
    var = anp.sum(alpha * (alpha0 - alpha) / (alpha0 * alpha0 * (alpha0 + 1.0) + 1e-12), axis=1)

    # Typical Route-A EDL regularizer: KL( Dir(alpha_tilde) || Dir(1) ), annealed.
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    alpha_tilde0 = anp.sum(alpha_tilde, axis=1, keepdims=True)
    num_classes = alpha.shape[1]

    kl = (
        gammaln_p(alpha_tilde0)
        - anp.sum(gammaln_p(alpha_tilde), axis=1, keepdims=True)
        - gammaln_p(anp.array([num_classes], dtype=alpha.dtype))
        + anp.sum((alpha_tilde - 1.0) * (digamma_p(alpha_tilde) - digamma_p(alpha_tilde0)), axis=1, keepdims=True)
    ).squeeze(-1)

    ce_aux = -anp.sum(y_onehot * anp.log(p_mean + 1e-12), axis=1)

    total = mse + var + reg_weight * kl + ce_aux_weight * ce_aux
    return anp.mean(total) + l2 * l2_penalty(params)


def predict_softmax(params: ArrayDict, x: np.ndarray) -> np.ndarray:
    logits = np.array(forward_logits(params, x))
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)


def predict_edl(params: ArrayDict, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logits = np.array(forward_logits(params, x))
    evidence = np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0.0)
    alpha = evidence + 1.0
    alpha0 = np.sum(alpha, axis=1, keepdims=True)
    p_mean = alpha / (alpha0 + 1e-12)
    return p_mean, alpha0.squeeze(-1)


def metric_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y_true))


def metric_nll(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(-np.mean(np.log(probs[np.arange(y_true.shape[0]), y_true] + 1e-12)))


def metric_brier(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    y_oh = one_hot(y_true, num_classes)
    return float(np.mean(np.sum((y_oh - probs) ** 2, axis=1)))


def metric_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    corr = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y_true.shape[0]

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)

        cnt = np.sum(mask)
        if cnt == 0:
            continue
        acc_bin = np.mean(corr[mask])
        conf_bin = np.mean(conf[mask])
        ece += (cnt / n) * abs(acc_bin - conf_bin)

    return float(ece)


def metric_auroc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    # y=1 as positive (OOD)
    y = y_true_binary.astype(np.int64)
    pos = y == 1
    neg = y == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    auc = (np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def split_train_val_test(
    rng: np.random.Generator,
    num_classes: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, num_classes, endpoint=False)
    means = np.stack([3.2 * np.cos(angles), 3.2 * np.sin(angles)], axis=1)
    cov = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)

    def sample_block(n_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for c in range(num_classes):
            x_c = rng.multivariate_normal(mean=means[c], cov=cov, size=n_per_class)
            y_c = np.full(n_per_class, c, dtype=np.int64)
            xs.append(x_c)
            ys.append(y_c)
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)

        idx = rng.permutation(x.shape[0])
        return x[idx], y[idx]

    x_train, y_train = sample_block(train_per_class)
    x_val, y_val = sample_block(val_per_class)
    x_test, y_test = sample_block(test_per_class)

    return x_train, y_train, x_val, y_val, x_test, y_test


def add_label_noise(rng: np.random.Generator, y: np.ndarray, num_classes: int, noise_rate: float) -> np.ndarray:
    y_noisy = y.copy()
    n = y.shape[0]
    m = int(n * noise_rate)
    idx = rng.choice(n, size=m, replace=False)
    for i in idx:
        old = y_noisy[i]
        candidates = [c for c in range(num_classes) if c != old]
        y_noisy[i] = candidates[rng.integers(0, len(candidates))]
    return y_noisy


def make_ood_set(rng: np.random.Generator, n_samples: int, center_ratio: float = 0.75) -> np.ndarray:
    n_half = int(n_samples * center_ratio)

    # Center blob (ambiguous region between classes)
    x_center = rng.normal(loc=0.0, scale=1.0, size=(n_half, 2))

    # Far ring (clear out-of-distribution region)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples - n_half,))
    r = rng.uniform(6.0, 8.0, size=(n_samples - n_half,))
    x_ring = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x_ring += rng.normal(0.0, 0.2, size=x_ring.shape)

    x_ood = np.concatenate([x_center, x_ring], axis=0)
    idx = rng.permutation(x_ood.shape[0])
    return x_ood[idx]


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 128
    lr: float = 1e-3
    l2: float = 1e-4
    label_smoothing: float = 0.1
    edl_ce_aux_weight: float = 0.02
    edl_reg_max: float = 0.02
    edl_anneal_ratio: float = 0.3


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_ood: np.ndarray
    num_classes: int


def normalize_features(bundle: DatasetBundle) -> DatasetBundle:
    mu = np.mean(bundle.x_train, axis=0, keepdims=True)
    std = np.std(bundle.x_train, axis=0, keepdims=True) + 1e-6

    return DatasetBundle(
        x_train=(bundle.x_train - mu) / std,
        y_train=bundle.y_train,
        x_val=(bundle.x_val - mu) / std,
        y_val=bundle.y_val,
        x_test=(bundle.x_test - mu) / std,
        y_test=bundle.y_test,
        x_ood=(bundle.x_ood - mu) / std,
        num_classes=bundle.num_classes,
    )


def train_one_method(
    method: str,
    bundle: DatasetBundle,
    seed: int,
    cfg: TrainConfig,
) -> Tuple[ArrayDict, List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    params = init_params(rng, input_dim=2, hidden1=64, hidden2=64, num_classes=bundle.num_classes)

    y_train_oh = one_hot(bundle.y_train, bundle.num_classes)

    if method == "ce":
        loss_fn = lambda p, xb, yb, regw: ce_loss(p, xb, yb, l2=cfg.l2, label_smoothing=0.0)
    elif method == "ce_ls":
        loss_fn = lambda p, xb, yb, regw: ce_loss(p, xb, yb, l2=cfg.l2, label_smoothing=cfg.label_smoothing)
    elif method == "edl":
        loss_fn = lambda p, xb, yb, regw: edl_loss(
            p,
            xb,
            yb,
            l2=cfg.l2,
            reg_weight=regw,
            ce_aux_weight=cfg.edl_ce_aux_weight,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    grad_fn = grad(loss_fn)

    m_state = {k: np.zeros_like(v) for k, v in params.items()}
    v_state = {k: np.zeros_like(v) for k, v in params.items()}

    best_params = clone_params(params)
    best_val_nll = float("inf")
    history: List[Dict[str, float]] = []

    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0

    n_train = bundle.x_train.shape[0]
    n_batches = int(np.ceil(n_train / cfg.batch_size))

    for epoch in range(cfg.epochs):
        idx = rng.permutation(n_train)
        x_train = bundle.x_train[idx]
        y_train = y_train_oh[idx]
        epoch_batch_losses: List[float] = []

        if method == "edl":
            anneal_epochs = max(1, int(cfg.epochs * cfg.edl_anneal_ratio))
            regw = cfg.edl_reg_max * min(1.0, epoch / anneal_epochs)
        else:
            regw = 0.0

        for b in range(n_batches):
            lo = b * cfg.batch_size
            hi = min((b + 1) * cfg.batch_size, n_train)
            xb = x_train[lo:hi]
            yb = y_train[lo:hi]
            batch_loss = loss_fn(params, xb, yb, regw)
            epoch_batch_losses.append(float(batch_loss))

            grads = grad_fn(params, xb, yb, regw)

            step += 1
            for k in params.keys():
                g = np.array(grads[k], dtype=np.float64)
                m_state[k] = b1 * m_state[k] + (1.0 - b1) * g
                v_state[k] = b2 * v_state[k] + (1.0 - b2) * (g * g)

                m_hat = m_state[k] / (1.0 - b1 ** step)
                v_hat = v_state[k] / (1.0 - b2 ** step)
                params[k] = params[k] - cfg.lr * m_hat / (np.sqrt(v_hat) + eps)

        # Validation NLL for model selection
        if method in ("ce", "ce_ls"):
            train_probs = predict_softmax(params, bundle.x_train)
            val_probs = predict_softmax(params, bundle.x_val)
            ood_probs = predict_softmax(params, bundle.x_ood)
            train_id_score = predictive_entropy(train_probs)
            id_score = predictive_entropy(val_probs)
            ood_score = predictive_entropy(ood_probs)
        else:
            train_probs, train_alpha0 = predict_edl(params, bundle.x_train)
            val_probs, val_alpha0 = predict_edl(params, bundle.x_val)
            ood_probs, ood_alpha0 = predict_edl(params, bundle.x_ood)
            k = bundle.num_classes
            train_id_score = predictive_entropy(train_probs) + (k / (train_alpha0 + 1e-12))
            id_score = predictive_entropy(val_probs) + (k / (val_alpha0 + 1e-12))
            ood_score = predictive_entropy(ood_probs) + (k / (ood_alpha0 + 1e-12))

        train_loss = float(np.mean(epoch_batch_losses))
        train_nll = metric_nll(bundle.y_train, train_probs)
        train_acc = metric_accuracy(bundle.y_train, train_probs)
        train_brier = metric_brier(bundle.y_train, train_probs, bundle.num_classes)
        train_ece = metric_ece(bundle.y_train, train_probs)
        train_y_ood = np.concatenate(
            [np.zeros_like(train_id_score, dtype=np.int64), np.ones_like(ood_score, dtype=np.int64)]
        )
        train_ood_auroc = metric_auroc(train_y_ood, np.concatenate([train_id_score, ood_score], axis=0))

        val_nll = metric_nll(bundle.y_val, val_probs)
        val_acc = metric_accuracy(bundle.y_val, val_probs)
        val_brier = metric_brier(bundle.y_val, val_probs, bundle.num_classes)
        val_ece = metric_ece(bundle.y_val, val_probs)
        y_ood = np.concatenate(
            [np.zeros_like(id_score, dtype=np.int64), np.ones_like(ood_score, dtype=np.int64)]
        )
        val_ood_auroc = metric_auroc(y_ood, np.concatenate([id_score, ood_score], axis=0))

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "train_nll": float(train_nll),
                "train_brier": float(train_brier),
                "train_ece": float(train_ece),
                "train_ood_auroc": float(train_ood_auroc),
                "val_accuracy": float(val_acc),
                "val_nll": float(val_nll),
                "val_brier": float(val_brier),
                "val_ece": float(val_ece),
                "val_ood_auroc": float(val_ood_auroc),
            }
        )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_params = clone_params(params)

    return best_params, history


def evaluate_method(method: str, params: ArrayDict, bundle: DatasetBundle) -> Dict[str, float]:
    if method in ("ce", "ce_ls"):
        id_probs = predict_softmax(params, bundle.x_test)
        ood_probs = predict_softmax(params, bundle.x_ood)
        id_score = predictive_entropy(id_probs)
        ood_score = predictive_entropy(ood_probs)
    else:
        id_probs, id_alpha0 = predict_edl(params, bundle.x_test)
        ood_probs, ood_alpha0 = predict_edl(params, bundle.x_ood)
        k = bundle.num_classes
        id_score = predictive_entropy(id_probs) + (k / (id_alpha0 + 1e-12))
        ood_score = predictive_entropy(ood_probs) + (k / (ood_alpha0 + 1e-12))

    y_ood = np.concatenate([np.zeros_like(id_score, dtype=np.int64), np.ones_like(ood_score, dtype=np.int64)])
    score = np.concatenate([id_score, ood_score], axis=0)

    metrics = {
        "accuracy": metric_accuracy(bundle.y_test, id_probs),
        "nll": metric_nll(bundle.y_test, id_probs),
        "brier": metric_brier(bundle.y_test, id_probs, bundle.num_classes),
        "ece": metric_ece(bundle.y_test, id_probs),
        "ood_auroc": metric_auroc(y_ood, score),
        "id_conf_mean": float(np.mean(np.max(id_probs, axis=1))),
        "ood_conf_mean": float(np.mean(np.max(ood_probs, axis=1))),
    }
    return metrics


def format_mean_std(values: List[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    return f"{arr.mean():.4f} +- {arr.std(ddof=0):.4f}"


def write_outputs(
    out_dir: str,
    per_seed_rows: List[Dict[str, object]],
    epoch_rows: List[Dict[str, object]],
    summary: Dict[str, Dict[str, str]],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "per_seed_metrics.csv")
    fields = [
        "seed",
        "method",
        "accuracy",
        "nll",
        "brier",
        "ece",
        "ood_auroc",
        "id_conf_mean",
        "ood_conf_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow(row)

    epoch_csv_path = os.path.join(out_dir, "per_epoch_metrics.csv")
    epoch_fields = [
        "seed",
        "method",
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_nll",
        "train_brier",
        "train_ece",
        "train_ood_auroc",
        "val_accuracy",
        "val_nll",
        "val_brier",
        "val_ece",
        "val_ood_auroc",
    ]
    with open(epoch_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_fields)
        writer.writeheader()
        for row in epoch_rows:
            writer.writerow(row)

    json_path = os.path.join(out_dir, "summary_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(out_dir, "comparison_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Route-A EDL Demo Comparison\n\n")
        f.write("指标说明：Accuracy 越高越好；NLL/Brier/ECE 越低越好；OOD-AUROC 越高越好。\n\n")
        f.write("| Method | Accuracy | NLL | Brier | ECE | OOD-AUROC | ID Mean Conf | OOD Mean Conf |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

        order = ["ce", "ce_ls", "edl"]
        names = {
            "ce": "Softmax-CE",
            "ce_ls": "Softmax-LabelSmoothing",
            "edl": "RouteA-EDL",
        }
        for m in order:
            s = summary[m]
            f.write(
                f"| {names[m]} | {s['accuracy']} | {s['nll']} | {s['brier']} | {s['ece']} | {s['ood_auroc']} | {s['id_conf_mean']} | {s['ood_conf_mean']} |\n"
            )

        f.write("\n## 结论模板\n\n")
        f.write("1. 若 EDL 在 `ECE` 与 `OOD-AUROC` 上优于 CE/LS，说明其不确定性建模更有效。\n")
        f.write("2. 若 Accuracy 略降但 ECE/OOD 提升显著，通常是可接受的工程权衡。\n")
        f.write("3. 若 EDL 指标不稳定，优先调 `edl_reg_max`、`edl_anneal_ratio` 与 `edl_ce_aux_weight`。\n")


def build_dataset(seed: int, num_classes: int = 4) -> DatasetBundle:
    rng = np.random.default_rng(seed)

    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(
        rng,
        num_classes=num_classes,
        train_per_class=180,
        val_per_class=150,
        test_per_class=350,
    )

    y_train = add_label_noise(rng, y_train, num_classes=num_classes, noise_rate=0.2)
    x_ood = make_ood_set(rng, n_samples=1800, center_ratio=0.8)

    bundle = DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        x_ood=x_ood,
        num_classes=num_classes,
    )
    return normalize_features(bundle)


def run_experiment(seeds: List[int], cfg: TrainConfig, out_dir: str) -> None:
    methods = ["ce", "ce_ls", "edl"]

    per_seed_rows: List[Dict[str, object]] = []
    epoch_rows: List[Dict[str, object]] = []
    aggregate: Dict[str, Dict[str, List[float]]] = {
        m: {
            "accuracy": [],
            "nll": [],
            "brier": [],
            "ece": [],
            "ood_auroc": [],
            "id_conf_mean": [],
            "ood_conf_mean": [],
        }
        for m in methods
    }

    for seed in seeds:
        bundle = build_dataset(seed)

        for method in methods:
            print(f"[seed={seed}] training method={method} ...", flush=True)
            params, history = train_one_method(method=method, bundle=bundle, seed=seed + 1000, cfg=cfg)
            metrics = evaluate_method(method, params, bundle)

            for h in history:
                epoch_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "epoch": int(h["epoch"]),
                        "train_loss": h["train_loss"],
                        "train_accuracy": h["train_accuracy"],
                        "train_nll": h["train_nll"],
                        "train_brier": h["train_brier"],
                        "train_ece": h["train_ece"],
                        "train_ood_auroc": h["train_ood_auroc"],
                        "val_accuracy": h["val_accuracy"],
                        "val_nll": h["val_nll"],
                        "val_brier": h["val_brier"],
                        "val_ece": h["val_ece"],
                        "val_ood_auroc": h["val_ood_auroc"],
                    }
                )

            row = {"seed": seed, "method": method}
            row.update(metrics)
            per_seed_rows.append(row)

            for k, v in metrics.items():
                aggregate[method][k].append(v)

            print(
                f"[seed={seed}] {method}: "
                f"acc={metrics['accuracy']:.4f}, nll={metrics['nll']:.4f}, "
                f"ece={metrics['ece']:.4f}, ood_auroc={metrics['ood_auroc']:.4f}",
                flush=True,
            )

    summary: Dict[str, Dict[str, str]] = {}
    for method in methods:
        summary[method] = {k: format_mean_std(vs) for k, vs in aggregate[method].items()}

    write_outputs(out_dir=out_dir, per_seed_rows=per_seed_rows, epoch_rows=epoch_rows, summary=summary)

    print("\n=== Summary (mean +- std) ===")
    for method in methods:
        s = summary[method]
        print(
            f"{method:>6} | acc {s['accuracy']} | nll {s['nll']} | "
            f"ece {s['ece']} | ood_auroc {s['ood_auroc']}"
        )

    print(f"\nSaved results to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route-A EDL demo: training + comparison")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds for repeated runs")
    p.add_argument("--epochs", type=int, default=120, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--out-dir", type=str, default="demo_results", help="Output directory")
    p.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    p.add_argument("--edl-reg-max", type=float, default=0.02, help="Max EDL regularizer weight")
    p.add_argument("--edl-anneal-ratio", type=float, default=0.3, help="EDL annealing ratio")
    p.add_argument("--edl-ce-aux-weight", type=float, default=0.02, help="Small CE auxiliary weight for EDL")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        label_smoothing=args.label_smoothing,
        edl_reg_max=args.edl_reg_max,
        edl_anneal_ratio=args.edl_anneal_ratio,
        edl_ce_aux_weight=args.edl_ce_aux_weight,
    )
    run_experiment(seeds=args.seeds, cfg=cfg, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
