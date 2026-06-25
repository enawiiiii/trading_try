from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from trading_agent.indicators import atr, rsi, sma, volume_zscore
from trading_agent.models import Candle


LABELS = ["bearish", "sideways", "bullish"]


@dataclass(frozen=True)
class Dataset:
    x: list[list[float]]
    y: list[int]
    feature_names: list[str]
    outcomes_pct: list[float] | None = None
    label_mode: str = "future_close"


@dataclass(frozen=True)
class MLModel:
    labels: list[str]
    feature_names: list[str]
    means: list[float]
    scales: list[float]
    weights: list[list[float]]
    bias: list[float]
    lookback: int
    horizon: int
    move_threshold_pct: float
    label_mode: str = "future_close"
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None


def build_dataset(
    candles: list[Candle],
    lookback: int = 80,
    horizon: int = 6,
    move_threshold_pct: float = 0.35,
    label_mode: str = "future_close",
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
) -> Dataset:
    if len(candles) < lookback + horizon + 1:
        raise ValueError("Not enough candles to build a machine-learning dataset.")

    x: list[list[float]] = []
    y: list[int] = []
    outcomes_pct: list[float] = []
    feature_names = _feature_names()
    tp_pct = take_profit_pct if take_profit_pct is not None else move_threshold_pct
    sl_pct = stop_loss_pct if stop_loss_pct is not None else move_threshold_pct

    for end in range(lookback, len(candles) - horizon):
        window = candles[end - lookback : end]
        entry = window[-1].close
        future_window = candles[end : end + horizon]
        x.append(_features(window))
        if label_mode == "tp_sl_path":
            label, outcome = _tp_sl_label(entry, future_window, tp_pct, sl_pct)
        elif label_mode == "future_close":
            future = candles[end + horizon].close
            move_pct = ((future - entry) / entry) * 100
            label = _label(move_pct, move_threshold_pct)
            outcome = move_pct
        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")
        y.append(label)
        outcomes_pct.append(outcome)

    return Dataset(x=x, y=y, feature_names=feature_names, outcomes_pct=outcomes_pct, label_mode=label_mode)


def train_model(
    dataset: Dataset,
    lookback: int,
    horizon: int,
    move_threshold_pct: float,
    label_mode: str | None = None,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    epochs: int = 160,
    learning_rate: float = 0.04,
    seed: int = 7,
) -> tuple[MLModel, dict[str, Any]]:
    if not dataset.x:
        raise ValueError("Empty dataset.")

    random.seed(seed)
    split = max(1, int(len(dataset.x) * 0.8))
    train_x, train_y = dataset.x[:split], dataset.y[:split]
    test_x, test_y = dataset.x[split:], dataset.y[split:]
    test_outcomes = dataset.outcomes_pct[split:] if dataset.outcomes_pct is not None else None

    means, scales = _fit_scaler(train_x)
    train_xs = [_scale(row, means, scales) for row in train_x]
    test_xs = [_scale(row, means, scales) for row in test_x]

    class_count = len(LABELS)
    class_weights = _class_weights(train_y, class_count=class_count)
    feature_count = len(dataset.feature_names)
    weights = [[random.uniform(-0.01, 0.01) for _ in range(feature_count)] for _ in range(class_count)]
    bias = [0.0 for _ in range(class_count)]

    for _ in range(epochs):
        for row, target in zip(train_xs, train_y):
            probs = _softmax([_dot(weights[class_id], row) + bias[class_id] for class_id in range(class_count)])
            sample_weight = class_weights[target]
            for class_id in range(class_count):
                error = (probs[class_id] - (1.0 if class_id == target else 0.0)) * sample_weight
                for feature_id, value in enumerate(row):
                    weights[class_id][feature_id] -= learning_rate * error * value
                bias[class_id] -= learning_rate * error

    model = MLModel(
        labels=LABELS.copy(),
        feature_names=dataset.feature_names,
        means=means,
        scales=scales,
        weights=weights,
        bias=bias,
        lookback=lookback,
        horizon=horizon,
        move_threshold_pct=move_threshold_pct,
        label_mode=label_mode or dataset.label_mode,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
    )

    metrics = evaluate_model(model, test_x, test_y, test_outcomes)
    metrics["train_samples"] = len(train_x)
    metrics["test_samples"] = len(test_x)
    metrics["label_distribution"] = dict(Counter(LABELS[label] for label in dataset.y))
    return model, metrics


def predict_model(model: MLModel, candles: list[Candle]) -> dict[str, Any]:
    if len(candles) < model.lookback:
        raise ValueError(f"Need at least {model.lookback} candles for ML prediction.")
    row = _features(candles[-model.lookback :])
    scaled = _scale(row, model.means, model.scales)
    probs = _softmax([_dot(model.weights[class_id], scaled) + model.bias[class_id] for class_id in range(len(model.labels))])
    predictions = {label: round(prob * 100, 2) for label, prob in zip(model.labels, probs)}
    scenario = max(predictions.items(), key=lambda item: item[1])[0]
    return {
        "scenario": scenario,
        "probabilities": predictions,
        "confidence": predictions[scenario],
        "top_features": _top_feature_contributions(model, scaled, scenario),
    }


def evaluate_model(
    model: MLModel,
    x: list[list[float]],
    y: list[int],
    outcomes_pct: list[float] | None = None,
    fee_pct: float = 0.1,
) -> dict[str, Any]:
    if not x:
        return {"accuracy": 0.0, "samples": 0, "by_label": {}, "macro_f1": 0.0, "confusion_matrix": {}}
    correct = 0
    by_label: dict[str, dict[str, int]] = {label: {"samples": 0, "correct": 0} for label in model.labels}
    confusion: dict[str, dict[str, int]] = {
        actual: {predicted: 0 for predicted in model.labels}
        for actual in model.labels
    }
    predicted_counts: Counter[str] = Counter()
    long_returns: list[float] = []
    for row, target in zip(x, y):
        scaled = _scale(row, model.means, model.scales)
        probs = _softmax([_dot(model.weights[class_id], scaled) + model.bias[class_id] for class_id in range(len(model.labels))])
        predicted = max(range(len(probs)), key=lambda idx: probs[idx])
        label = model.labels[target]
        predicted_label = model.labels[predicted]
        by_label[label]["samples"] += 1
        confusion[label][predicted_label] += 1
        predicted_counts[predicted_label] += 1
        if predicted == target:
            correct += 1
            by_label[label]["correct"] += 1
    if outcomes_pct is not None:
        for row, outcome in zip(x, outcomes_pct):
            scaled = _scale(row, model.means, model.scales)
            probs = _softmax([_dot(model.weights[class_id], scaled) + model.bias[class_id] for class_id in range(len(model.labels))])
            predicted = max(range(len(probs)), key=lambda idx: probs[idx])
            if model.labels[predicted] == "bullish":
                long_returns.append(outcome - fee_pct)
    return {
        "accuracy": round(correct / len(x) * 100, 2),
        "samples": len(x),
        "macro_f1": _macro_f1(confusion, model.labels),
        "confusion_matrix": confusion,
        "prediction_distribution": dict(predicted_counts),
        "long_expected_value_pct": round(mean(long_returns), 4) if long_returns else 0.0,
        "long_signal_samples": len(long_returns),
        "by_label": {
            label: {
                "samples": values["samples"],
                "accuracy": round(values["correct"] / values["samples"] * 100, 2) if values["samples"] else 0.0,
            }
            for label, values in by_label.items()
        },
    }


def save_model(model: MLModel, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model.__dict__, indent=2, ensure_ascii=True), encoding="utf-8")


def load_model(path: str | Path) -> MLModel:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw.setdefault("label_mode", "future_close")
    raw.setdefault("take_profit_pct", None)
    raw.setdefault("stop_loss_pct", None)
    return MLModel(**raw)


def _features(candles: list[Candle]) -> list[float]:
    closes = [c.close for c in candles]
    current = closes[-1]
    sma_10 = sma(closes, 10)
    sma_20 = sma(closes, 20)
    sma_50 = sma(closes, 50)
    atr_pct = (atr(candles, 14) / current) * 100 if current else 0.0
    returns = [_return_pct(closes, period) for period in (1, 3, 6, 12, 24)]
    recent_high = max(c.high for c in candles[-20:])
    recent_low = min(c.low for c in candles[-20:])
    range_position = ((current - recent_low) / (recent_high - recent_low)) if recent_high != recent_low else 0.5
    body = candles[-1].close - candles[-1].open
    candle_range = candles[-1].high - candles[-1].low
    body_pct = body / candle_range if candle_range else 0.0
    quote_volumes = [c.quote_volume for c in candles[-30:]]
    volume_ratio = quote_volumes[-1] / mean(quote_volumes) if quote_volumes and mean(quote_volumes) else 1.0

    return [
        _pct_distance(current, sma_10),
        _pct_distance(current, sma_20),
        _pct_distance(current, sma_50),
        rsi(closes, 14),
        atr_pct,
        volume_zscore(candles, 30),
        volume_ratio,
        range_position,
        body_pct,
        *returns,
    ]


def _feature_names() -> list[str]:
    return [
        "dist_sma_10_pct",
        "dist_sma_20_pct",
        "dist_sma_50_pct",
        "rsi_14",
        "atr_pct",
        "volume_zscore_30",
        "volume_ratio_30",
        "range_position_20",
        "last_candle_body_pct",
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "return_24",
    ]


def _label(move_pct: float, threshold: float) -> int:
    if move_pct <= -threshold:
        return 0
    if move_pct >= threshold:
        return 2
    return 1


def _tp_sl_label(entry: float, future_window: list[Candle], take_profit_pct: float, stop_loss_pct: float) -> tuple[int, float]:
    if entry <= 0:
        return 1, 0.0
    take_profit = entry * (1 + take_profit_pct / 100)
    stop_loss = entry * (1 - stop_loss_pct / 100)
    for candle in future_window:
        hit_tp = candle.high >= take_profit
        hit_sl = candle.low <= stop_loss
        if hit_tp and hit_sl:
            return 1, 0.0
        if hit_tp:
            return 2, take_profit_pct
        if hit_sl:
            return 0, -stop_loss_pct
    final_close = future_window[-1].close if future_window else entry
    return 1, ((final_close - entry) / entry) * 100


def _macro_f1(confusion: dict[str, dict[str, int]], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in labels if actual != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        scores.append(f1)
    return round(mean(scores) * 100, 2) if scores else 0.0


def _return_pct(closes: list[float], period: int) -> float:
    if len(closes) <= period or closes[-period - 1] == 0:
        return 0.0
    return ((closes[-1] - closes[-period - 1]) / closes[-period - 1]) * 100


def _pct_distance(value: float, reference: float) -> float:
    return ((value - reference) / reference) * 100 if reference else 0.0


def _fit_scaler(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    columns = list(zip(*rows))
    means = [mean(column) for column in columns]
    scales = [pstdev(column) or 1.0 for column in columns]
    return means, scales


def _class_weights(labels: list[int], class_count: int) -> list[float]:
    counts = Counter(labels)
    if not labels:
        return [1.0 for _ in range(class_count)]
    weights: list[float] = []
    for class_id in range(class_count):
        count = counts.get(class_id, 0)
        if not count:
            weights.append(1.0)
            continue
        balanced = len(labels) / (class_count * count)
        weights.append(min(3.0, max(0.5, balanced)))
    return weights


def _scale(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    return [(value - means[index]) / scales[index] for index, value in enumerate(row)]


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _dot(weights: list[float], row: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, row))


def _top_feature_contributions(model: MLModel, scaled: list[float], scenario: str, limit: int = 5) -> list[dict[str, Any]]:
    class_id = model.labels.index(scenario)
    contributions = [
        {
            "feature": name,
            "contribution": round(model.weights[class_id][index] * scaled[index], 4),
        }
        for index, name in enumerate(model.feature_names)
    ]
    return sorted(contributions, key=lambda item: abs(item["contribution"]), reverse=True)[:limit]
