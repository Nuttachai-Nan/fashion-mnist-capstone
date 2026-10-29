import torch
from src.models import LinearBaseline, SmallCNN


def test_linear_output_shape():
    model = LinearBaseline()
    output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_cnn_output_shape():
    model = SmallCNN()
    output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)


def test_cnn_has_dropout():
    model = SmallCNN()
    assert any(isinstance(module, torch.nn.Dropout) for module in model.modules())
