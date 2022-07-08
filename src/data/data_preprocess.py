import cv2
import numpy as np
import pandas as pd
import uuid
import os
import click

selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def iou(bb1: np.array, bb2: np.array) -> int:
    """Function get intersection over union from two bounding boxes
    :param bb1: Numpy array with coordinates of first bounding box
    :param bb2: Numpy array with coordinates of second bounding box
    :return: Integer value of iou
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
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
@click.argument("output_img_path")
def preprocess_data(input_path: str, output_path: str, output_img_path: str):
    """Function get 60 regions from image and save it for train model
    :param input_path: path of raw data
    :param output_path: path of processed data
    """
    if not os.path.isdir(output_img_path):
        os.mkdir(output_img_path)

    labels_df = pd.DataFrame(columns=["name", "label"])

    for e, i in enumerate(os.listdir(input_path + "/bbox/")):
        if i.startswith("airplane"):
            filename = i.split(".")[0] + ".jpg"
            image = cv2.imread(input_path + "/images/" + filename)
            df = pd.read_csv(input_path + "/bbox/" + i)
            ground_truth = []
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                ground_truth.append([x1, y1, x2, y2])
            selective_search.setBaseImage(image)
            selective_search.switchToSelectiveSearchFast()
            ss_res = selective_search.process()
            imageout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for j, result in enumerate(ss_res):
                if j < 2000 and flag == 0:
                    for gt in ground_truth:
                        x, y, w, h = result
                        iou_val = iou(gt, [x, y, x + w, y + h])
                        if counter < 30:
                            if iou_val > 0.75:
                                timage = imageout[y : y + h, x : x + w]
                                resized = cv2.resize(
                                    timage, (224, 224), interpolation=cv2.INTER_AREA
                                )
                                name = str(uuid.uuid1())
                                cv2.imwrite(
                                    output_img_path + "/" + name + ".jpg",
                                    resized,
                                )
                                labels_df = labels_df.append(
                                    {"name": name, "label": 1}, ignore_index=True
                                )
                                counter += 1
                        else:
                            fflag = 1
                        if falsecounter < 30:
                            if iou_val < 0.3:
                                timage = imageout[y : y + h, x : x + w]
                                resized = cv2.resize(
                                    timage, (224, 224), interpolation=cv2.INTER_AREA
                                )
                                name = str(uuid.uuid1())
                                cv2.imwrite(
                                    output_img_path + "/" + name + ".jpg",
                                    resized,
                                )
                                labels_df = labels_df.append(
                                    {"name": name, "label": 0}, ignore_index=True
                                )
                                falsecounter += 1
                        else:
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        flag = 1
    labels_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    preprocess_data()
