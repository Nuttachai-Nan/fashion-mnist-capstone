"""Train baseline and CNN; select using validation loss and evaluate test once."""
from __future__ import annotations
import argparse
import copy
import csv
import time
from pathlib import Path
import torch
from torch import nn
from .data import make_loaders, set_seed, CLASS_NAMES
from .models import build_model
from .utils import (evaluate_model, plot_learning_curves, plot_confusion,
                    plot_failure_cases, plot_success_cases, write_history, write_json)


def train_one(name, train_loader, val_loader, test_loader, test_set, args, device):
    model = build_model(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_n = y.size(0)
            train_loss_sum += loss.item() * batch_n
            train_count += batch_n
            train_correct += (logits.argmax(1) == y).sum().item()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * y.size(0)
                val_count += y.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_accuracy": train_correct / max(train_count, 1),
            "val_loss": val_loss_sum / max(val_count, 1),
            "val_accuracy": val_correct / max(val_count, 1),
        }
        history.append(row)
        print(f"[{name}] epoch {epoch:02d}/{args.epochs} "
              f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
              f"train_acc={row['train_accuracy']:.4f} val_acc={row['val_accuracy']:.4f}")

        if row["val_loss"] < best_val_loss - args.min_delta:
            best_val_loss = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{name}] early stopping after epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": name, "state_dict": best_state, "seed": args.seed,
                "config": vars(args), "best_val_loss": best_val_loss}, checkpoint_dir / f"{name}.pt")

    test_result = evaluate_model(model, test_loader, device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_history(results_dir / f"history_{name}.csv", history)
    metric = {
        "model": name,
        "test_accuracy": test_result["accuracy"],
        "test_macro_f1": test_result["macro_f1"],
        "best_validation_loss": float(best_val_loss),
        "epochs_completed": len(history),
        "training_seconds": round(time.time() - started, 2),
        "checkpoint": str(checkpoint_dir / f"{name}.pt"),
        "seed": args.seed,
        "config": {"epochs_max": args.epochs, "batch_size": args.batch_size,
                   "learning_rate": args.lr, "weight_decay": args.weight_decay,
                   "patience": args.patience},
    }
    return metric, history, test_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "cnn", "all"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader, test_loader, test_set = make_loaders(
        args.data_dir, args.batch_size, args.seed, args.num_workers
    )
    names = ["linear", "cnn"] if args.model == "all" else [args.model]
    all_metrics, all_histories, test_results = [], {}, {}
    for name in names:
        # Reset seeds before each model for comparable repeatability.
        set_seed(args.seed)
        metric, history, result = train_one(name, train_loader, val_loader, test_loader,
                                            test_set, args, device)
        all_metrics.append(metric)
        all_histories[name] = history
        test_results[name] = result

    results_dir = Path(args.results_dir)
    write_json(results_dir / "metrics.json", {
        "dataset": "Fashion-MNIST",
        "split": {"train": 55000, "validation": 5000, "test": 10000,
                  "note": "Validation sampled from official training split with seeded random_split; official test set held out."},
        "class_names": CLASS_NAMES,
        "seed": args.seed,
        "device": str(device),
        "models": all_metrics,
    })
    plot_learning_curves(all_histories, results_dir / "learning_curves.png")
    if "cnn" in test_results:
        cnn = test_results["cnn"]
        plot_confusion(cnn["confusion_matrix"], results_dir / "confusion_matrix_cnn.png")
        np_cm = cnn["confusion_matrix"]
        with (results_dir / "confusion_matrix_cnn.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true/predicted"] + CLASS_NAMES)
            for label, row in zip(CLASS_NAMES, np_cm):
                writer.writerow([label] + row.tolist())
        plot_failure_cases(test_set, cnn, results_dir / "failure_cases.png",
                           results_dir / "failure_cases.json")
        plot_success_cases(test_set, cnn, results_dir / "success_cases.png",
                           results_dir / "success_cases.json")
    print(f"Saved metrics and plots under {results_dir.resolve()}")


if __name__ == "__main__":
    main()
