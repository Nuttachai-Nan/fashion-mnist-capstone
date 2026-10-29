# Fashion-MNIST Capstone: Linear Baseline vs. CNN

A reproducible PyTorch deep-learning capstone that trains two models from scratch on Fashion-MNIST, compares their held-out test performance, plots learning curves, exports genuine prediction examples, and generates a report PDF from measured results.

## Project question

Does a compact convolutional neural network classify clothing images more accurately than a linear classifier on flattened pixels?

## Dataset and split

- Fashion-MNIST: 70,000 grayscale 28 × 28 images in 10 classes.
- Official 60,000-image training partition is split into 55,000 training and 5,000 validation examples using seed 42.
- Official 10,000-image test partition is kept out of training, checkpoint selection, and hyperparameter tuning.
- Dataset files are downloaded by `torchvision` at runtime and are not committed to Git.
- Source: Zalando Research, https://github.com/zalandoresearch/fashion-mnist; Xiao, Rasul & Vollgraf (2017), arXiv:1708.07747.

## Run in Google Colab (recommended)

1. Download `notebooks/capstone_fashion_mnist.ipynb` from this project (or use the directly downloadable notebook supplied with the project files).
2. In Colab, choose **File → Upload notebook** and select the `.ipynb` file. The notebook embeds all required project source code, so it does not require a GitHub clone or a separate ZIP upload.
3. Select **Runtime → Change runtime type → T4 GPU** when available.
4. Run all cells. The notebook installs dependencies, downloads Fashion-MNIST, trains both models, evaluates them, and generates `report.pdf` plus an artifact bundle.
5. Save the generated metrics, plots, failure cases, checkpoints and PDF. Do not report placeholder metrics as results.

Once this notebook is committed on the public `main` branch, the direct Colab URL will be:
https://colab.research.google.com/github/Nuttachai-Nan/fashion-mnist-capstone/blob/main/notebooks/capstone_fashion_mnist.ipynb

## Run locally

Python 3.10+ is recommended. A GPU helps, but CPU execution is supported.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train --model all --epochs 12 --batch-size 128 --seed 42 --patience 3
python -m src.make_report --output report.pdf
```

Use `--model linear` or `--model cnn` to train one model. Useful options include `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--patience`, and `--num-workers`.

## Models and training choices

- **Baseline:** a single linear layer mapping 784 flattened pixel values to 10 logits.
- **CNN:** two convolutional blocks (32 and 64 channels), ReLU and 2 × 2 max pooling, followed by a 128-unit fully connected layer and dropout.
- **Loss:** cross-entropy applied to logits.
- **Optimizer:** AdamW, learning rate 0.001, weight decay 0.0001.
- **Regularization:** dropout, weight decay and validation-loss early stopping (patience 3).
- **Seed:** 42 for Python, NumPy and PyTorch.
- **Metrics:** held-out test accuracy and macro-F1; also export confusion matrix, curves, success and failure examples.

## Results from the completed run

A real CUDA run has now been completed with seed 42. These figures are read from `results/metrics.json` and the held-out official test set; they are not placeholders.

| Model | Test accuracy | Test macro-F1 | Epochs run | Best checkpoint epoch |
|---|---:|---:|---:|---:|
| Linear baseline | 0.8386 (83.86%) | 0.8393 | 9 | 6 |
| Custom CNN | 0.9187 (91.87%) | 0.9182 | 12 | 9 |

The CNN improves accuracy by 8.01 percentage points and reduces the baseline test error rate by approximately 49.6%. See `EXPERIMENT_RESULTS.md`, `results/class_metrics_cnn.csv`, the learning curves, and confusion matrix for the analysis. The final report is `report.pdf`. The CNN ran for 12 epochs, but the checkpoint selected by lowest validation loss was from epoch 9.

## Output files

- `checkpoints/linear.pt`, `checkpoints/cnn.pt`: best checkpoints chosen by validation loss.
- `results/metrics.json`: measured performance, run configuration and device.
- `results/learning_curves.png`: train/validation loss and accuracy.
- `results/confusion_matrix_cnn.png`: CNN confusion matrix on the test set.
- `results/success_cases.png`, `results/failure_cases.png`: actual held-out examples.
- `report.pdf`: PDF generated from available experiment outputs. If no run exists, it is explicitly marked as a draft.

The dataset and trained weights/results are excluded from version control to keep the repository small. Commit source changes incrementally while doing the project; don't fabricate an experimental commit history after the fact.

## Limitations

Fashion-MNIST is a small grayscale benchmark and is not representative of unconstrained real-world photographs. A single seed does not estimate run-to-run uncertainty. Future work could compare multiple seeds, tune augmentation on validation data, or analyze per-class precision/recall.

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. https://arxiv.org/abs/1708.07747
- PyTorch/Torchvision FashionMNIST dataset API: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
