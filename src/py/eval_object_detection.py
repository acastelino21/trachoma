# import sys; sys.path.insert(0, '..')
import argparse
from tqdm import tqdm
import os
import pandas as pd
import numpy as np 

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import json
import itertools
from matplotlib.patches import Rectangle


from torchvision.ops import nms

from nets.segmentation import FasterRCNN
from loaders.tt_dataset import TTDataModuleBX, BBXImageTrainTransform, BBXImageEvalTransform, BBXImageTestTransform

from scipy.optimize import linear_sum_assignment


def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = .5 if normalize else np.sum(cm)/4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    return cm



def remove_labels(df, class_column, label_column, drop_labels=None, concat_labels=None):

    if drop_labels is not None:
        df = df[ ~ df[label_column].isin(drop_labels)]

    if concat_labels is not None:
        replacement_val = df.loc[ df['label'] == concat_labels[0]]['class'].unique()
        df.loc[ df['label'].isin(concat_labels), "class" ] = replacement_val[0]

    unique_classes = sorted(df[class_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}

    df[class_column] = df[class_column].map(class_mapping)    
    df.loc[ df[label_column] == 'Reject', class_column]  = 0

    print(f"Kept Classes : {df[label_column].unique()}, {class_mapping}")

    return df

def compute_bbx_iou(gt_box, pred_box):
    """
    Compute IoU for two bounding boxes.
    Each box should be [x1, y1, x2, y2]
    """
    x1 = max(gt_box[0], pred_box[0])
    y1 = max(gt_box[1], pred_box[1])
  
    x2 = min(gt_box[2], pred_box[2])
    y2 = min(gt_box[3], pred_box[3])

    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    gt_boxArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    pred_boxArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    iou = interArea / float(gt_boxArea + pred_boxArea - interArea)
    return iou


def match_boxes(gt_boxes, pred_boxes):
    """Match predicted boxes to ground truth boxes using the Hungarian algorithm."""
    
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], []

    # Compute cost matrix (distance between centers)
    cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

    for i, (x1_gt, y1_gt, x2_gt, y2_gt) in enumerate(gt_boxes):
        center_gt = np.array([(x1_gt + x2_gt) / 2, (y1_gt + y2_gt) / 2])
        
        for j, (x1_pred, y1_pred, x2_pred, y2_pred) in enumerate(pred_boxes):
            center_pred = np.array([(x1_pred + x2_pred) / 2, (y1_pred + y2_pred) / 2])
            cost_matrix[i, j] = np.linalg.norm(center_gt - center_pred)  # Euclidean distance

    # Solve assignment problem (minimize distance)
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    return gt_indices, pred_indices


def evaluate_with_fp_fn(gt_boxes, pred_boxes):
    gt_indices, pred_indices = match_boxes(gt_boxes, pred_boxes)

    distances = []
    ious = []
    num_pred =0
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        num_pred+=1
        x1_gt, y1_gt, x2_gt, y2_gt = gt_boxes[gt_idx]
        x1_pred, y1_pred, x2_pred, y2_pred = pred_boxes[pred_idx]

        iou = compute_bbx_iou(gt_boxes[gt_idx], pred_boxes[pred_idx])
        ious.append(iou)
        # Compute center distance
        center_gt = np.array([(x1_gt + x2_gt) / 2, (y1_gt + y2_gt) / 2])
        center_pred = np.array([(x1_pred + x2_pred) / 2, (y1_pred + y2_pred) / 2])
        distances.append(np.linalg.norm(center_gt - center_pred))
    
    # Count false positives (unmatched predictions)
    num_false_positives = len(pred_boxes) - len(pred_indices)

    # Count false negatives (unmatched GT boxes)
    num_false_negatives = len(gt_boxes) - len(gt_indices)
    return num_pred, num_false_positives, num_false_negatives, ious, distances, gt_indices, pred_indices

def main(args):

    mount_point = args.mount_point #"/CMF/data/lumargot/trachoma/B images one eye/"
    ckpt = args.model

    df_train = pd.read_csv(args.csv_train)
    df_val = pd.read_csv(args.csv_valid)
    df_test = pd.read_csv(args.csv_test)

    # map = {0:'Reject', 1:'Healthy', 2:'Bad'}
    map = {0:'Reject', 1:'Healthy', 2:'Entropion', 3:'overcorrected++', 4:'Short Incision'}

    df_test = remove_labels(df_test, args.class_column, args.label_column, args.drop_labels, args.concat_labels)
    df_train = remove_labels(df_train, args.class_column, args.label_column, args.drop_labels, args.concat_labels)
    df_val = remove_labels(df_val, args.class_column, args.label_column, args.drop_labels, args.concat_labels)

    ## remove Rejection class for evaluation but keep the same labels number (> 1)
    df_test = df_test.loc[~df_test[args.label_column].isin(['Reject'])].reset_index()
    df_val = df_val.loc[~df_val[args.label_column].isin(['Reject'])].reset_index()
    df_train = df_train.loc[~df_train[args.label_column].isin(['Reject'])].reset_index()

    ttdata = TTDataModuleBX(df_train, df_val, df_test, batch_size=1,img_column='filename', mount_point=mount_point, 
                            train_transform=BBXImageTrainTransform(), valid_transform=BBXImageEvalTransform(), test_transform=BBXImageTestTransform())


    ttdata.setup()
    dataload = ttdata.test_dataloader()

    model = FasterRCNN.load_from_checkpoint(ckpt)
    model.eval()

    num_preds, num_fps, num_fns = 0, 0, 0
    l_ious, l_distances = [], []
    gt, pred = [], []

    for idx, batch in enumerate(tqdm(dataload)):

        targets = batch
        imgs = targets.pop('img', None)
        outs = model(imgs, mode='test')
        out_img = outs[0]

        pred_indices = nms(out_img['boxes'], out_img['scores'], iou_threshold=.4) ## iou as args
        gt_indices = nms(targets['boxes'][0], torch.ones_like(targets['boxes'][0,:,0]), iou_threshold=.8) ## iou as args

        gt_boxes = targets['boxes'][0][gt_indices].cpu().detach().numpy()
        pred_boxes = out_img['boxes'][pred_indices].cpu().detach().numpy()

        n_p, n_fp, n_fn, i, d, gt_ix, pred_idx = evaluate_with_fp_fn(gt_boxes, pred_boxes)

        gt.append(targets['labels'][0][gt_ix].cpu().detach())
        pred.append(out_img['labels'][pred_idx].cpu().detach())

        num_preds += n_p
        num_fps += n_fp
        num_fns += n_fn
        l_ious.append(torch.tensor(i).reshape(-1))
        l_distances.append(torch.tensor(d))

    ious = torch.cat(l_ious, dim=0)
    dist = torch.cat(l_distances, dim=0)

    pred = torch.cat(pred)
    gt = torch.cat(gt)

    ## report
    out_dict = {'Matched Prediction': num_preds,
                'False Positives':num_fps,
                'False Negatives':num_fns,
                'Mean IoU': ious.mean().item(),
                'Mean distance (px)':dist.mean().item(),
                }
    print(json.dumps(out_dict, indent=2))
    json_filename = os.path.join(args.out, "box_prediction_ratios_report.json")

    with open(json_filename, "w") as file:
        json.dump(out_dict, file, indent=2)
    
    report = classification_report(gt, pred, output_dict=True)
    print(json.dumps(report, indent=2))
    df_report = pd.DataFrame(report).transpose()
    report_filename = os.path.join(args.out, "predictions_classification_report.csv")
    df_report.to_csv(report_filename)


    class_names = list(map.values())[1:] #remove reject
    cnf_matrix = confusion_matrix(gt, pred)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=(16,12))

    plot_confusion_matrix(cnf_matrix, classes=class_names, title='confusion matrix')
    confusion_filename =  os.path.join(args.out, "classification_confusion.png")
    fig.savefig(confusion_filename)


    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=(16,12))
    cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='confusion matrix - normalized')

    norm_confusion_filename =  os.path.join(args.out, "classification_norm_confusion.png")
    fig2.savefig(norm_confusion_filename)



    # save: outputs images directy for visualization


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT object detection Training')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    

    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')

    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')
    input_group.add_argument('--class_column', type=str, default="class", help='Name of segmentation column in csv')
    input_group.add_argument('--label_column', type=str, default="label", help='Name of label column in csv')
    input_group.add_argument('--drop_labels', type=str, default=None, nargs='+', help='drop labels in dataframe')
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')

    input_group.add_argument('--gt_iou_thr', type=float, default=0.9, help='IoU threshold for nms for predicted boxes')
    input_group.add_argument('--pred_iou_thr', type=float, default=0.8, help='IoU threshold for nms for ground truth')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    args = parser.parse_args()

    main(args)
