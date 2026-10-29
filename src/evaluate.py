"""Evaluate a saved checkpoint on the held-out Fashion-MNIST test set."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from .data import make_loaders, set_seed
from .models import build_model
from .utils import evaluate_model, write_json, plot_confusion, plot_failure_cases, plot_success_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "cnn"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, test_set = make_loaders(args.data_dir, args.batch_size, args.seed, args.num_workers)
    checkpoint_path = args.checkpoint or f"checkpoints/{args.model}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    result = evaluate_model(model, test_loader, device)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "test_accuracy": result["accuracy"], "test_macro_f1": result["macro_f1"],
               "checkpoint": checkpoint_path, "seed": args.seed}
    write_json(out / f"evaluation_{args.model}.json", summary)
    plot_confusion(result["confusion_matrix"], out / f"confusion_matrix_{args.model}.png")
    if args.model == "cnn":
        plot_failure_cases(test_set, result, out / "failure_cases.png", out / "failure_cases.json")
        plot_success_cases(test_set, result, out / "success_cases.png", out / "success_cases.json")
    print(summary)


if __name__ == "__main__":
    main()
