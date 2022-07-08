import torch
import torch.nn as nn
import torchvision
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import click
import pandas as pd


def iou(bb1: np.array, bb2: np.array) -> float:
    """Function get intersection over union from two bounding boxes
    :param bb1: Numpy array with coordinates of first bounding box
    :param bb2: Numpy array with coordinates of second bounding box
    :return: Value of iou
    """
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


@click.command()
@click.argument("test_img_path", type=click.Path())
@click.argument("ground_truth_path", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("threshold", type=click.FLOAT)
def predict(
    test_img_path: str,
    ground_truth_path: str,
    model_path: str,
    output_path: str,
    threshold: float,
):

    model = torchvision.models.vgg16(pretrained=False)
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
            ]
        )
    )
    model.add_module(classifier_name, classifier)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    sftm = nn.Softmax(1)

    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    img = cv2.imread(test_img_path)
    selective_search.setBaseImage(img)
    selective_search.switchToSelectiveSearchFast()
    res = selective_search.process()
    imgout = img.copy()
    ground_truth_dataset = pd.read_csv(ground_truth_path)
    ground_truth = []
    for row in ground_truth_dataset.iterrows():
        x1 = int(row[1][0].split(" ")[0])
        y1 = int(row[1][0].split(" ")[1])
        x2 = int(row[1][0].split(" ")[2])
        y2 = int(row[1][0].split(" ")[3])
        ground_truth.append([x1, y1, x2, y2])
    iou_val = 0
    true_bbox_counter = 0

    for e, result in enumerate(res):
        if e < 2000:
            x, y, w, h = result
            timage = imgout[y : y + h, x : x + w]
            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
            img = torch.tensor(resized, dtype=torch.float32).reshape(1, 3, 224, 224)
            with torch.no_grad():
                out = model(img)
            if sftm(out)[0][0].item() > threshold:
                true_bbox_counter += 1
                for gt in ground_truth:
                    iou_val += iou(gt, [x, y, x + w, y + h])
                cv2.rectangle(
                    imgout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA
                )

    plt.figure()
    plt.imshow(imgout)
    plt.savefig(output_path + "/model_test.jpg")


if __name__ == "__main__":
    predict()
