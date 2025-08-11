import torch
from torch import nn, manual_seed, softmax, inference_mode
from torch.nn import Sequential, ReLU, Conv2d, Linear, MaxPool2d, BatchNorm2d, Dropout, Flatten, Softmax, BatchNorm1d, CrossEntropyLoss, AdaptiveAvgPool2d
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else torch.device('cpu')

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, dropout):
        super(FashionMNISTModelV1, self).__init__()
        # Block 1
        self.layer1 = Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer2 = ReLU()
        self.layer3 = BatchNorm2d(hidden_units)
        self.layer4 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer5 = ReLU()
        self.layer6 = BatchNorm2d(hidden_units)
        self.layer7 = MaxPool2d(2)
        self.layer8 = Dropout(dropout)
        # Block 2
        self.layer9 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer10 = ReLU()
        self.layer11 = BatchNorm2d(hidden_units)
        self.layer12 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer13 = ReLU()
        self.layer14 = BatchNorm2d(hidden_units)
        self.layer15 = MaxPool2d(2)
        self.layer16 = Dropout(dropout)
        # Block 3
        self.layer17 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer18 = ReLU()
        self.layer19 = BatchNorm2d(hidden_units)
        self.layer20 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer21 = ReLU()
        self.layer22 = BatchNorm2d(hidden_units)
        self.layer23 = Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1)
        self.layer24 = ReLU()
        self.layer25 = BatchNorm2d(hidden_units)
        self.layer26 = Dropout(dropout)
        # Global average pooling
        self.layer27 = AdaptiveAvgPool2d(1)
        self.layer28 = Flatten()
        self.layer29 = Linear(hidden_units, output_shape)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer25(x)
        x = self.layer26(x)
        x = self.layer27(x)
        x = self.layer28(x)
        x = self.layer29(x)
        return x

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
