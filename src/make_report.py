"""Generate a 5-page PDF report from actual training artefacts (or an explicit draft)."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]


def architecture_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    blocks = [
        (0.15, 1.15, 1.15, 0.7, "Input\n1 x 28 x 28"),
        (1.55, 1.15, 1.3, 0.7, "Conv 1->32\nReLU + Pool"),
        (3.08, 1.15, 1.3, 0.7, "Conv 32->64\nReLU + Pool"),
        (4.62, 1.15, 1.15, 0.7, "Flatten\n3136"),
        (6.02, 1.15, 1.25, 0.7, "Dense 128\nReLU + Dropout"),
        (7.53, 1.15, 1.45, 0.7, "Dense 10\nClass logits"),
    ]
    for x, y, w, h, label in blocks:
        rect = plt.Rectangle((x, y), w, h, facecolor="#eaf2f8", edgecolor="#2c3e50", linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=8.5)
    for i in range(len(blocks)-1):
        x, y, w, h, _ = blocks[i]
        nx, ny, _, nh, _ = blocks[i+1]
        ax.annotate("", xy=(nx-0.04, ny+nh/2), xytext=(x+w+0.04, y+h/2),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#34495e"))
    ax.text(5, 0.45, "Two 3 x 3 convolution layers; 2 x 2 max-pooling after each block", ha="center", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def placeholder_curve(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    ax.axis("off")
    ax.text(0.5, 0.59, "Learning curves will appear here after training", ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.43, "Run: python -m src.train --model all --epochs 12 --batch-size 128 --seed 42", ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.27, "Do not submit this draft before replacing the placeholder with real curves.", ha="center", va="center", fontsize=9, color="#a93226")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def error_placeholder(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    axes[0].set_title("Test confusion matrix: pending")
    axes[0].set_xticks(range(10)); axes[0].set_yticks(range(10))
    axes[0].set_xlim(-0.5, 9.5); axes[0].set_ylim(9.5, -0.5)
    axes[0].set_xticklabels([]); axes[0].set_yticklabels([])
    axes[0].set_xticks([i - 0.5 for i in range(11)], minor=True)
    axes[0].set_yticks([i - 0.5 for i in range(11)], minor=True)
    axes[0].grid(which="minor", color="#909090", linewidth=0.45)
    axes[0].tick_params(which="both", length=0)
    axes[0].text(4.5, 4.5, "GENERATE AFTER\nREAL TEST RUN", ha="center", va="center", fontsize=10)
    axes[0].set_xlabel("Predicted class"); axes[0].set_ylabel("True class")
    axes[1].axis("off")
    axes[1].set_title("Failure case: pending")
    axes[1].text(0.5, 0.65, "Insert a genuine misclassified\nFashion-MNIST test image", ha="center", va="center", fontsize=10)
    axes[1].text(0.5, 0.37, "True label: -\nPredicted label: -\nConfidence: -", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

def load_metrics(results: Path):
    path = results / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt_metric(value):
    return "-" if value is None else f"{value:.4f}"


def make_report(output: Path, results_dir: Path) -> None:
    metrics = load_metrics(results_dir)
    actual = bool(metrics and metrics.get("models"))
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    arch_path = figures_dir / "architecture.png"
    architecture_figure(arch_path)
    curve_path = results_dir / "learning_curves.png"
    if not curve_path.exists():
        curve_path = figures_dir / "learning_curves_pending.png"
        placeholder_curve(curve_path)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22,
                              leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#17365D"), spaceAfter=10))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=11, leading=15,
                              alignment=TA_CENTER, textColor=colors.HexColor("#34495E"), spaceAfter=12))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=15, leading=18,
                              textColor=colors.HexColor("#17365D"), spaceBefore=3, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=14,
                              textColor=colors.HexColor("#1F618D"), spaceBefore=5, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9, leading=12,
                              alignment=TA_LEFT, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=9.5, spaceAfter=3))
    styles.add(ParagraphStyle(name="Warning", parent=styles["BodyText"], fontSize=9, leading=12,
                              textColor=colors.HexColor("#922B21"), borderColor=colors.HexColor("#C0392B"),
                              borderWidth=0.8, borderPadding=7, spaceBefore=4, spaceAfter=8))

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm,
                            title="Fashion-MNIST Classification Capstone", author="Capstone project")
    story = []
    if not actual:
        story.append(Paragraph("DRAFT - EXPERIMENTS NOT RUN", styles["CoverTitle"]))
        story.append(Paragraph("This report must be regenerated after training in an internet-enabled runtime.", styles["Warning"]))
    else:
        story.append(Paragraph("Fashion-MNIST Image Classification", styles["CoverTitle"]))
        story.append(Paragraph("Capstone report: linear baseline versus a custom convolutional neural network", styles["Subtitle"]))
    if not actual:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Fashion-MNIST Image Classification", styles["CoverTitle"]))
        story.append(Paragraph("Capstone report draft: linear baseline versus custom CNN", styles["Subtitle"]))
    story.append(Paragraph("Abstract", styles["H1x"]))
    story.append(Paragraph(
        "This project studies a small image-classification problem using Fashion-MNIST. A multinomial linear classifier on flattened pixels is compared with a convolutional neural network (CNN) trained from scratch. Both models use the same deterministic training/validation partition and are evaluated on the official held-out test set only after model selection. The project reports accuracy and macro-F1, and inspects errors to understand which classes remain difficult.", styles["Bodyx"]))
    story.append(Paragraph("Project question", styles["H2x"]))
    story.append(Paragraph("Does a compact CNN improve clothing-category recognition over a linear classifier by learning local spatial features, and what kinds of visual ambiguity still cause mistakes?", styles["Bodyx"]))
    if not actual:
        story.append(Paragraph("Current experimental status: source code and report automation are prepared, but this environment could not download the dataset because outbound network/DNS access was unavailable. No performance values are invented. The result table and curves below are placeholders until the project is run.", styles["Warning"]))
    story.append(Paragraph("1. Problem and data", styles["H1x"]))
    story.append(Paragraph("The task is supervised 10-class classification of 28 x 28 grayscale images of apparel. Fashion-MNIST contains 60,000 labelled training examples and 10,000 test examples. The labels are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.", styles["Bodyx"]))
    story.append(Paragraph("The code takes 5,000 examples from the official training split for validation using a fixed seed, leaving 55,000 for gradient updates. The separate official test set of 10,000 images is not used for early stopping or hyperparameter selection. Pixel tensors are normalised using mean 0.2860 and standard deviation 0.3530. Dataset files are downloaded at runtime and are not committed to the repository.", styles["Bodyx"]))
    story.append(Paragraph("Dataset source: Zalando Research, Fashion-MNIST repository, https://github.com/zalandoresearch/fashion-mnist. See Xiao, Rasul and Vollgraf (2017) in References.", styles["Smallx"]))
    story.append(Paragraph("2. Models and training", styles["H1x"]))
    story.append(Paragraph("Baseline. A single linear layer maps a flattened 784-dimensional pixel vector to 10 class logits. This model is inexpensive and tests how far global linear decision boundaries can go without learned spatial feature hierarchies.", styles["Bodyx"]))
    story.append(Paragraph("CNN. Two convolutional blocks use 3 x 3 kernels (1 to 32 channels, then 32 to 64), ReLU activations and 2 x 2 max-pooling. The resulting 64 x 7 x 7 representation is flattened and passed through a 128-unit ReLU layer, dropout of 0.30, and a 10-logit output layer. Weight sharing and local receptive fields support learning spatial patterns such as edges and garment contours.", styles["Bodyx"]))
    story.append(Image(str(arch_path), width=170*mm, height=60*mm))
    story.append(Paragraph("Training uses cross-entropy loss on logits and AdamW with learning rate 0.001 and weight decay 0.0001. Batch size is 128, with at most 12 epochs. Early stopping uses validation loss with patience 3; the checkpoint with the lowest validation loss is restored before the single held-out test evaluation. Seed 42 is set for Python, NumPy and PyTorch. Dropout, weight decay and early stopping provide regularisation.", styles["Bodyx"]))

    story.append(Paragraph("3. Evaluation protocol", styles["H1x"]))
    story.append(Paragraph("Primary metrics are test accuracy (fraction of correct predictions) and macro-F1 (the unweighted mean of per-class F1 scores). Macro-F1 complements accuracy by treating all 10 classes equally. The baseline and CNN are trained using the same train/validation split and comparable optimizer settings. The final comparison is made on the same official test images.", styles["Bodyx"]))
    story.append(Paragraph("To avoid test leakage, the test set is not used to select epochs, tune learning rate, choose dropout, or choose between checkpoints. The held-out metric is a final report card, not a feedback signal for development. If test results are used to make changes, a new untouched test set would be needed for an unbiased final claim.", styles["Bodyx"]))
    story.append(Paragraph("4. Results", styles["H1x"]))
    if actual:
        models = {m.get("model"): m for m in metrics.get("models", [])}
        table_data = [["Model", "Test accuracy", "Test macro-F1", "Epochs", "Best val. loss"]]
        for key, label in [("linear", "Linear baseline"), ("cnn", "Custom CNN")]:
            m = models.get(key)
            table_data.append([label, fmt_metric(m.get("test_accuracy") if m else None),
                               fmt_metric(m.get("test_macro_f1") if m else None),
                               str(m.get("epochs_completed", "-")) if m else "-",
                               fmt_metric(m.get("best_validation_loss") if m else None) if m else "-"])
        story.append(Paragraph("Table 1. Held-out test results generated by the provided training script.", styles["Smallx"]))
    else:
        table_data = [["Model", "Test accuracy", "Test macro-F1", "Epochs", "Best val. loss"],
                      ["Linear baseline", "PENDING", "PENDING", "PENDING", "PENDING"],
                      ["Custom CNN", "PENDING", "PENDING", "PENDING", "PENDING"]]
        story.append(Paragraph("Table 1. The table will be populated by results/metrics.json after a genuine run.", styles["Smallx"]))
    table = Table(table_data, colWidths=[43*mm, 29*mm, 29*mm, 19*mm, 29*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#AAB7B8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F4F6F7")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))
    story.append(Image(str(curve_path), width=168*mm, height=66*mm))
    if actual:
        model_map = {m.get("model"): m for m in metrics.get("models", [])}
        if "linear" in model_map and "cnn" in model_map:
            delta = model_map["cnn"]["test_accuracy"] - model_map["linear"]["test_accuracy"]
            delta_text = f"The CNN minus baseline test-accuracy difference is {delta:+.4f} ({delta*100:+.2f} percentage points)."
        else:
            delta_text = "Run both models with --model all before interpreting the baseline comparison."
        story.append(Paragraph(escape(delta_text) + " Interpret the learning curves together with the held-out metric; a growing train-validation gap may indicate overfitting.", styles["Bodyx"]))
    else:
        story.append(Paragraph("Result interpretation is intentionally withheld until the curves and metrics are produced by a real run. A submission-ready report must discuss whether training loss falls, whether validation loss diverges, and how much (if at all) the CNN improves over the baseline.", styles["Warning"]))
    story.append(PageBreak())

    story.append(Paragraph("5. Error analysis and failure case", styles["H1x"]))
    failure_img = results_dir / "failure_cases.png"
    if actual and failure_img.exists():
        story.append(Image(str(failure_img), width=154*mm, height=125*mm))
        failure_path = results_dir / "failure_cases.json"
        try:
            failures = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else []
        except json.JSONDecodeError:
            failures = []
        if failures:
            ex = failures[0]
            story.append(Paragraph(f"Example high-confidence error: the true class was {escape(ex['true_name'])}, while the CNN predicted {escape(ex['predicted_name'])} with confidence {ex['predicted_confidence']:.2f} (test index {ex['test_index']}). Inspect the image and explain whether the silhouette, low resolution or overlap in garment appearance plausibly contributed. Confidence is not proof of correctness.", styles["Bodyx"]))
        else:
            story.append(Paragraph("No misclassified examples were found, or the failure-case artefact was empty. Verify evaluation output before submitting; a failure analysis is required by the assignment.", styles["Warning"]))
    else:
        err_path = figures_dir / "error_analysis_pending.png"
        error_placeholder(err_path)
        story.append(Image(str(err_path), width=160*mm, height=54*mm))
        story.append(Paragraph("A concrete failure case must be inserted after running the evaluation. The script automatically selects high-confidence incorrect predictions and saves their true label, predicted label and predicted confidence to results/failure_cases.json, with a visual grid in results/failure_cases.png.", styles["Warning"]))
        story.append(Paragraph("What to analyse", styles["H2x"]))
        story.append(Paragraph("1. Name the true and predicted classes and quote the model confidence. 2. Inspect the actual pixels rather than assuming the model's reason. 3. Explain one plausible ambiguity (for example, similar shirt/pullover/coat silhouettes) and one limitation of the explanation. Do not present an invented example as a model prediction.", styles["Bodyx"]))
    story.append(PageBreak())
    story.append(Paragraph("6. Limitations and next steps", styles["H1x"]))
    story.append(Paragraph("The images are small, grayscale and centered; this benchmark is not representative of unconstrained real-world photographs. The CNN has limited depth and no extensive augmentation or hyperparameter search. One random train/validation split and one seed do not quantify run-to-run uncertainty. Future work could compare multiple seeds, add carefully chosen translations/crops, report per-class precision/recall, calibrate confidence, or test on higher-resolution images. Any modification should be selected using validation data rather than repeatedly consulting the test set.", styles["Bodyx"]))
    story.append(Paragraph("Reproducibility", styles["H2x"]))
    story.append(Paragraph("Run `python -m src.train --model all --epochs 12 --batch-size 128 --seed 42`, then `python -m src.make_report --output report.pdf`. The script records the device and hyperparameters in results/metrics.json. Runtime versions and hardware can change speed and may introduce small numerical differences even with a fixed seed.", styles["Bodyx"]))
    story.append(Paragraph("References", styles["H2x"]))
    story.append(Paragraph("Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. https://github.com/zalandoresearch/fashion-mnist", styles["Smallx"]))
    story.append(Paragraph("PyTorch contributors. Torchvision FashionMNIST dataset API. https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html", styles["Smallx"]))
    story.append(Paragraph("Oral-defence notes", styles["H2x"]))
    story.append(Paragraph("Be prepared to explain: (1) why the official test set is kept separate from validation; (2) why cross-entropy is paired with logits rather than applying softmax inside the model; (3) how validation loss chooses the checkpoint and triggers early stopping; (4) why macro-F1 can tell a different story from accuracy; and (5) why a high-confidence mistake does not necessarily indicate a meaningful semantic understanding by the model.", styles["Bodyx"]))
    story.append(Paragraph("Demo checklist", styles["H2x"]))
    story.append(Paragraph("The walkthrough should show the repository and reproduction command, a training or inference run, the measured baseline and CNN scores, the learning curves, one successful prediction, one genuine failure example, and one limitation. Keep the selected failure image and its true/predicted labels from the generated artefacts so the walkthrough agrees with the written report.", styles["Bodyx"]))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#566573"))
        canvas.drawString(18*mm, 9*mm, "Fashion-MNIST Capstone" + (" | DRAFT - run experiments before submission" if not actual else ""))
        canvas.drawRightString(A4[0] - 18*mm, 9*mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {output} ({'results present' if actual else 'DRAFT: no real results yet'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="report.pdf")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    make_report(Path(args.output), Path(args.results_dir))


if __name__ == "__main__":
    main()
