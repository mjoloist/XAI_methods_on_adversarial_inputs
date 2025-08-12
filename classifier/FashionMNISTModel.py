import torch
from torch import nn, manual_seed, softmax, inference_mode
from torch.nn import Sequential, ReLU, Conv2d, Linear, MaxPool2d, BatchNorm2d, Dropout, Flatten, Softmax, BatchNorm1d, CrossEntropyLoss, AdaptiveAvgPool2d
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from timeit import default_timer as timer
from alive_progress import alive_bar
from early_stopping_pytorch import EarlyStopping

def train_model(data_loader: DataLoader, ta_model: nn.Module, train_loss_fn: nn.Module, train_optimizer: torch.optim.Optimizer, accuracy_fn, train_device):
    """
    This function performs for a given model a training epoch
    :param data_loader: a DataLoader object that contains the training data
    :param ta_model: the PyTorch model to be trained
    :param train_loss_fn: a loss function that takes the predicted logits as its
    first parameter and the true class as its second and returns a pytorch tensor with a single item
    :param train_optimizer: a pytorch optimizer
    :param accuracy_fn: an accuracy function that takes the predicted class as its
    first parameter and the true class as its second and returns a pytorch tensor with a single item
    :param train_device: device where the model shoould be trained on
    :return: training loss, training accuracy
    """
    pass
    train_loss, train_acc = 0, 0
    model.to(train_device)
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(train_device), y.to(train_device)
        y_logit = ta_model(x)
        loss = train_loss_fn(y_logit, y)
        train_loss += loss
        train_acc += accuracy_fn(softmax(y_logit, dim=1).argmax(dim=1), y)
        train_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%200 == 0:
            print(f"Trained on {batch*len(x)}/{len(data_loader.dataset)} samples")
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train Loss: {train_loss:.5f} | Train Accuracy: {(train_acc*100):.3f}")
    return train_loss.cpu().item(), train_acc.cpu().item()


def valid_model(data_loader: DataLoader, v_model: nn.Module, v_loss_fn: nn.Module, accuracy_fn, v_device):
    """
    This function calculates the validation loss and validation accuracy
    :param data_loader: a DataLoader object that contains the validation data
    :param v_model: a PyTorch model
    :param v_loss_fn: a loss function that takes the predicted logits as its
    first parameter and the true class as its second and returns a pytorch tensor with a single item
    :param accuracy_fn: an accuracy function that takes the predicted class as its
    first parameter and the true class as its second and returns a pytorch tensor with a single item
    :param v_device: device on which to run the model, eg. cuda or cpu
    :return: validation loss, validation accuracy
    """
    v_l, v_a = 0, 0
    model.to(v_device)
    model.eval()
    with inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(v_device), y.to(v_device)
            y_logit = v_model(x)
            loss = v_loss_fn(y_logit, y)
            v_l += loss
            v_a += accuracy_fn(softmax(y_logit, dim=1).argmax(dim=1), y)
        v_l /= len(data_loader)
        v_a /= len(data_loader)
    print(f"Validation Loss: {v_l:.5f} | Validation Accuracy: {(v_a*100):.3f}\n\n\n")
    return v_l.cpu().item(), v_a.cpu().item()

def test_model(data_loader: DataLoader, te_model: nn.Module, test_loss_fn: nn.Module, accuracy_fn, test_device):
    """
        This function calculates the test loss and test accuracy. This is separate from validation since it prints a different string
        and due to testing purposes
        :param data_loader: a DataLoader object that contains the validation data
        :param te_model: a PyTorch model
        :param test_loss_fn: a loss function that takes the predicted logits as its
        first parameter and the true class as its second and returns a pytorch tensor with a single item
        :param accuracy_fn: an accuracy function that takes the predicted class as its
        first parameter and the true class as its second and returns a pytorch tensor with a single item
        :param test_device: device on which to run the model, eg. cuda or cpu
        :return: test loss, test accuracy
        """
    test_l, test_a = 0, 0
    model.to(test_device)
    model.eval()
    with inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(test_device), y.to(test_device)
            y_logit = te_model(x)
            loss = test_loss_fn(y_logit, y)
            test_l += loss
            test_a += accuracy_fn(softmax(y_logit, dim=1).argmax(dim=1), y)
        test_l /= len(data_loader)
        test_a /= len(data_loader)
    print(f"Test Loss: {test_l:.5f} | Test Accuracy: {(test_a*100):.3f}\n\n\n")
    return test_l.cpu().item(), test_a.cpu().item()

def print_losses(train_l, test_l, num_epoch):
    """
    Plots the graph of all train and test losses
    :param train_l: train losses
    :param test_l: test losses
    :param num_epoch: number of epochs the model already trained
    :return: None
    """
    x = np.arange(1, (num_epoch+2), step=1)
    print(x)
    plt.plot(x, train_l, c="g", label="Train Loss")
    plt.plot(x, test_l, c="r", label="Test Loss")
    plt.title("Train-Test-Losses")
    plt.legend()
    plt.show()

def print_accuracies(train_a, test_a, num_epoch):
    """
    Plots the graph of all train and test accuracies
    :param train_a: train accuracies
    :param test_a: test accuracies
    :param num_epoch: number of epochs the model already trained
    :return: None
    """
    x = np.arange(1, (num_epoch+2), step=1)
    plt.plot(x, train_a, c="g", label="Train Accuracy")
    plt.plot(x, test_a, c="r", label="Test Accuracy")
    plt.title("Train-Test-Accuracy")
    plt.legend()
    plt.show()



device = "cuda" if torch.cuda.is_available() else "cpu"



class FashionMNISTModelV1(nn.Module):
    """
    This is the CNN Model inspired by the VGG-Architecture, it uses three convolution blocks, global average pooling and a final fully connected layer
    """
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

#  The following prepares the train, validation and test data
print("Preparing data...")
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

generator1 = torch.Generator().manual_seed(456)
train_split_size = int(len(train_data)*0.8)
val_split_size = len(train_data)-train_split_size
new_train_data, valid_data = random_split(train_data, [train_split_size, val_split_size], generator=generator1)

class_names = train_data.classes

BATCH_SIZE = 32
train_dataLoader = DataLoader(new_train_data, shuffle=True, batch_size=BATCH_SIZE)
validation_dataLoader = DataLoader(valid_data, shuffle=True, batch_size=BATCH_SIZE)
test_dataLoader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)
print("Done preparing data.")

# The following prepares all other classes like the CNN, the loss Function, the optimizer etc.
print("Preparing model, optimizer etc...")
manual_seed(42)
model = FashionMNISTModelV1(1, 128, 10, 0.5).to(device)
loss_fn = CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=0.00003)
accuracy_f = Accuracy(task="multiclass", num_classes=10).to(device)
early_stopper = EarlyStopping(patience=5, verbose=True)
print("Done preparing model\n")

# This initializes the variables for the training loop
print("Starting training...")
epochs = 50
train_time_start = timer()
train_losses = []
train_accs = []
validation_losses = []
validation_accs = []
test_losses = []
test_accs = []

# The following is the training loop, it first performs a full training step, then calculates test loss and accuracies and then validation loss and accuracies. After that it plots the graphs and checks for early stopping to avoid overfitting
with alive_bar(epochs, force_tty=True) as bar:
    for epoch in range(epochs):
        bar()
        print(f"\nEpoch: {epoch+1}\n----------------------\n")
        train_loss, train_acc = train_model(train_dataLoader, model, loss_fn, optimizer, accuracy_f, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_loss, test_acc = test_model(test_dataLoader, model, loss_fn, accuracy_f, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        validation_loss, validation_acc = valid_model(validation_dataLoader, model, loss_fn, accuracy_f, device)
        validation_losses.append(validation_loss)
        validation_accs.append(validation_acc)
        if (epoch+1)%5 == 0:
            print_losses(train_losses, test_losses, epoch)
            print_accuracies(train_accs, test_accs, epoch)
        early_stopper(validation_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            print_losses(train_losses, test_losses, epoch)
            print_accuracies(train_accs, test_accs, epoch)
            break
train_time_end = timer()

# This prints the final results and saves the models state_dict
print("Finished training")
print(f"Trained for {(train_time_end-train_time_start):.3f} seconds")
print(f"Best Test accuracy in epoch {np.argmax(test_accs)+1}")
MODEL_PATH = Path("../models/FashionMNIST")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "FashionMNISTModelStateDictPool.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)