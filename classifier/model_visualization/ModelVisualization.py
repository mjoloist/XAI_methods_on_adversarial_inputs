import torch
from torch import nn, manual_seed, softmax, inference_mode
from torch.nn import Sequential, ReLU, Conv2d, Linear, MaxPool2d, BatchNorm2d, Dropout, Flatten, Softmax, BatchNorm1d, CrossEntropyLoss, AdaptiveAvgPool2d
from torchvision import datasets
from torchvision.transforms import ToTensor

from classifier.FashionMNISTModel import FashionMNISTModelV1

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else torch.device('cpu')

model = FashionMNISTModelV1(input_shape=1, hidden_units=128, output_shape=10, dropout=0.5)

MODEL_NAME = "../../models/FashionMNIST/FashionMNISTModelStateDictPool.pth"

model.load_state_dict(torch.load(MODEL_NAME, map_location=device))
model.to("cpu")

test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

torch.manual_seed(42)
idx = torch.randint(10000, ()).item()
test_image, test_label = test_data[idx]
onnx_program = torch.onnx.export(model, test_image.unsqueeze(dim=0),dynamo=True)
onnx_program.save("../../classifier/model_visualization/classifier_visualization.onnx")
