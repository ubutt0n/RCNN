import torch
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import models
from collections import OrderedDict
import click
import numpy as np
import os


def load_img(img):
    file_path = "data/processed/train_img/" + str(img) + ".jpg"
    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    image = torch.tensor(image, dtype=torch.float32) / 255
    image = image.reshape(3, 224, 224)
    return image


def labels_to_probabilities(y):
    return np.hstack((y, 1 - y))


class Data(Dataset):
    def __init__(self, X, y=None):
        self.image_id = X
        self.labels = y

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):
        if self.labels is not None:
            image_id = self.image_id[idx][0]
            label = self.labels[idx]
            image = load_img(image_id)

            return image, label
        else:
            image_id = self.image_id[idx][0]
            image = load_img(image_id)

            return image


@click.command()
@click.argument("labels_path", type=click.Path())
@click.argument("model_save_path", type=click.Path())
@click.argument("epochs", type=click.INT)
@click.argument("learning_rate", type=click.FLOAT)
def train(labels_path: str, model_save_path: str, epochs: int, learning_rate: float):
    """Function create a training loop for neural net
    :param labels_path: String path for labels dataset
    :param model_save_path: String path for saving model
    :param epochs: Integer num of epochs
    :param learning_rate: Float learning rate
    """

    model = models.vgg16(pretrained=True)

    for i, param in enumerate(model.parameters()):
        if i == 20:
            break
        param.requires_grad = False

    classifier_name, _ = model._modules.popitem()

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("0", nn.Linear(25088, 4096, bias=True)),
                ("1", nn.ReLU(inplace=True)),
                ("2", nn.Dropout(p=0.5, inplace=False)),
                ("3", nn.Linear(4096, 4096, bias=True)),
                ("4", nn.ReLU(inplace=True)),
                ("5", nn.Dropout(p=0.5, inplace=False)),
                ("6", nn.Linear(4096, 2, bias=True)),
                ("7", nn.Softmax(1)),
            ]
        )
    )
    model.add_module(classifier_name, classifier)

    labels_df = pd.read_csv(labels_path)

    train_dataset = Data(
        labels_df.iloc[:, :1].values,
        labels_to_probabilities(labels_df.iloc[:, 1].values),
    )
    train_loader = DataLoader(train_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sftm = nn.Softmax(1)

    n_total_steps = len(train_loader)
    best_res = 10

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss_val = loss(output, labels)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if loss_val.item() < best_res:
                best_res = loss_val.item()
                torch.save(model.state_dict(), model_save_path)

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch: [{epoch+1}/{epochs}], Step: [{i+1}/{n_total_steps}], Loss: {loss_val.item():.4f}"
                )
            if (i + 1) % 1000 == 0:
                print(output)
                print(sftm(output))


if __name__ == "__main__":
    train()
