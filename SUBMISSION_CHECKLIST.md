# Final submission checklist

## Q1 - Public GitHub repository (90 points)
- [ ] Public repository exists and a `github.com` URL is ready to paste into the portal.
- [ ] `README.md` includes the task, dataset citation, exact split, commands, metrics and hyperparameters.
- [ ] Source includes model, training and standalone evaluation scripts.
- [ ] The seed and early-stopping rule are documented; test set is not used for model selection.
- [ ] Commit history shows genuine incremental work from the project owner.
- [ ] No data files, checkpoints, credentials or secrets are committed.

## Q2 - Report PDF (70 points)
- [ ] Run training in Colab; confirm both model runs finish and `results/metrics.json` is created.
- [ ] Run `python -m src.make_report --output report.pdf` after training.
- [ ] Read the metrics in the table and compare the CNN with the baseline; do not fill in estimates.
- [ ] Inspect learning curves and state whether the model appears underfit or overfit.
- [ ] Inspect a real item in `results/failure_cases.png`; write an evidence-based explanation.
- [ ] Re-render/review the PDF; verify it is 4-8 pages and no draft warnings remain.

## Q3 - Demo / walkthrough (40 points; 3-6 minutes)

Suggested narration plan:
- **0:00-0:35** State the task, the 10 classes and why Fashion-MNIST is a manageable benchmark.
- **0:35-1:15** Explain the linear baseline and CNN architecture diagram.
- **1:15-2:00** Show the training command, fixed seed, 55k/5k validation split and untouched 10k test split.
- **2:00-2:50** Show learning curves and quote the actual accuracy and macro-F1 numbers for both models.
- **2:50-3:50** Show a genuine correct prediction from `results/success_cases.png` and a genuine failure from `results/failure_cases.png`.
- **3:50-4:40** Explain one plausible reason for the failure and one limitation.
- **4:40-5:15** State the reproduction command and one next step. Keep within 3-6 minutes.

## Oral defence questions
- Why do we need validation data separately from the test set?
- Why does cross-entropy accept logits?
- How are the best checkpoint and early stopping selected?
- Why report macro-F1 as well as accuracy?
- What does the failure case show, and what does it not prove?
