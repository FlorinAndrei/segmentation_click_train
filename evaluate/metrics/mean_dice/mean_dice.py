# Copyright 2022 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mean Dice (Sorensen-Dice) metric."""

from typing import Dict, Optional

import datasets
import numpy as np

import evaluate


_DESCRIPTION = """
Dice is 2x the number of elements common to both sets divided by the sum of the number of elements in each set.
For binary (two classes) or multi-class segmentation,
the mean Dice of the image is calculated by taking the Dice of each class and averaging them.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`List[ndarray]`):
        List of predicted segmentation maps, each of shape (height, width). Each segmentation map can be of a different size.
    references (`List[ndarray]`):
        List of ground truth segmentation maps, each of shape (height, width). Each segmentation map can be of a different size.
    num_labels (`int`):
        Number of classes (categories).
    ignore_index (`int`):
        Index that will be ignored during evaluation.
    nan_to_num (`int`, *optional*):
        If specified, NaN values will be replaced by the number defined by the user.
    label_map (`dict`, *optional*):
        If specified, dictionary mapping old label indices to new label indices.
    reduce_labels (`bool`, *optional*, defaults to `False`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
        and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

Returns:
    `Dict[str, float | ndarray]` comprising various elements:
    - *mean_dice* (`float`):
        Mean Sorensen-Dice (Dice averaged over all categories).
    - *mean_accuracy* (`float`):
        Mean accuracy (averaged over all categories).
    - *overall_accuracy* (`float`):
        Overall accuracy on all images.
    - *per_category_accuracy* (`ndarray` of shape `(num_labels,)`):
        Per category accuracy.
    - *per_category_dice* (`ndarray` of shape `(num_labels,)`):
        Per category Dice.
"""

_CITATION = """\
@software{MMSegmentation_Contributors_OpenMMLab_Semantic_Segmentation_2020,
author = {{MMSegmentation Contributors}},
license = {Apache-2.0},
month = {7},
title = {{OpenMMLab Semantic Segmentation Toolbox and Benchmark}},
url = {https://github.com/open-mmlab/mmsegmentation},
year = {2020}
}"""


def intersect_and_union(
    pred_label,
    label,
    num_labels,
    ignore_index: int,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    """Calculate intersection and Union.

    Args:
        pred_label (`ndarray`):
            Prediction segmentation map of shape (height, width).
        label (`ndarray`):
            Ground truth segmentation map of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

     Returns:
         area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id

    # turn into Numpy arrays
    pred_label = np.array(pred_label)
    label = np.array(label)

    total_pixels = pred_label.shape[0] * pred_label.shape[1]

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    true_positives = np.zeros((num_labels,), dtype=np.int_)
    for l in range(num_labels):
        true_positives[l] = np.sum((pred_label == l) & (label == l))

    area_pred_label = np.histogram(
        pred_label, bins=num_labels, range=(0, num_labels - 1)
    )[0]
    area_label = np.histogram(label, bins=num_labels, range=(0, num_labels - 1))[0]

    false_positives = area_pred_label - true_positives
    false_negatives = area_label - true_positives
    true_negatives = total_pixels - true_positives - false_positives - false_negatives

    return (
        true_positives,
        area_pred_label,
        area_label,
        false_positives,
        false_negatives,
        true_negatives,
    )


def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    """Calculate Total Intersection and Union, by calculating `intersect_and_union` for each (predicted, ground truth) pair.

    Args:
        results (`ndarray`):
            List of prediction segmentation maps, each of shape (height, width).
        gt_seg_maps (`ndarray`):
            List of ground truth segmentation maps, each of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

     Returns:
         total_area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         total_area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         total_area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         total_area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    total_true_positives = np.zeros((num_labels,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
    total_area_label = np.zeros((num_labels,), dtype=np.float64)
    total_false_positives = np.zeros((num_labels,), dtype=np.float64)
    total_false_negatives = np.zeros((num_labels,), dtype=np.float64)
    total_true_negatives = np.zeros((num_labels,), dtype=np.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        (
            true_positives,
            area_pred_label,
            area_label,
            false_positives,
            false_negatives,
            true_negatives,
        ) = intersect_and_union(
            result, gt_seg_map, num_labels, ignore_index, label_map, reduce_labels
        )
        total_true_positives += true_positives
        total_area_pred_label += area_pred_label
        total_area_label += area_label
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        total_true_negatives += true_negatives
    return (
        total_true_positives,
        total_area_pred_label,
        total_area_label,
        total_false_positives,
        total_false_negatives,
        total_true_negatives,
    )


def mean_dice(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: int,
    nan_to_num: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
):
    """Calculate Mean Dice (mDice).

    Args:
        results (`ndarray`):
            List of prediction segmentation maps, each of shape (height, width).
        gt_seg_maps (`ndarray`):
            List of ground truth segmentation maps, each of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        nan_to_num (`int`, *optional*):
            If specified, NaN values will be replaced by the number defined by the user.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.

    Returns:
        `Dict[str, float | ndarray]` comprising various elements:
        - *mean_dice* (`float`):
            Mean Dice (Dice averaged over all categories).
        - *mean_accuracy* (`float`):
            Mean accuracy (averaged over all categories).
        - *overall_accuracy* (`float`):
            Overall accuracy on all images.
        - *per_category_accuracy* (`ndarray` of shape `(num_labels,)`):
            Per category accuracy.
        - *per_category_dice* (`ndarray` of shape `(num_labels,)`):
            Per category Dice.
    """
    (
        total_true_positives,
        total_area_pred_label,
        total_area_label,
        total_false_positives,
        total_false_negatives,
        total_true_negatives,
    ) = total_intersect_and_union(
        results, gt_seg_maps, num_labels, ignore_index, label_map, reduce_labels
    )

    """
    print(f"TP: {total_true_positives}")
    print(f"FP: {total_false_positives}")
    print(f"TN: {total_true_negatives}")
    print(f"FN: {total_false_negatives}")
    """

    # compute metrics
    metrics = dict()

    all_acc = total_true_positives.sum() / total_area_label.sum()
    niou = total_true_positives / (
        total_true_positives + total_false_positives + total_false_negatives
    )
    dice = (
        2
        * total_true_positives
        / (2 * total_true_positives + total_false_positives + total_false_negatives)
    )
    acc = total_true_positives / total_area_label
    precision = total_true_positives / (total_true_positives + total_false_positives)
    recall = total_true_positives / (total_true_positives + total_false_negatives)

    for i in range(num_labels):
        if total_area_label[i] == 0 and total_area_pred_label[i] == 0:
            # null label, null prediction
            dice[i] = 1.0
            niou[i] = 1.0
            acc[i] = 1.0
            precision[i] = 1.0
            recall[i] = 1.0
        if total_area_label[i] != 0 and total_area_pred_label[i] == 0:
            # not null label, null prediction
            precision[i] = 0.0
        if total_area_label[i] == 0 and total_area_pred_label[i] != 0:
            # null label, not null prediction
            recall[i] = 0.0
            acc[i] = 0.0

    metrics["overall_accuracy"] = all_acc
    metrics["per_category_dice"] = dice
    metrics["per_category_niou"] = niou
    metrics["per_category_accuracy"] = acc
    metrics["per_category_precision"] = precision
    metrics["per_category_recall"] = recall

    metrics["mean_dice"] = np.nanmean(dice)
    metrics["mean_niou"] = np.nanmean(niou)
    metrics["mean_accuracy"] = np.nanmean(acc)

    metrics["total_area_label"] = total_area_label
    metrics["total_area_pred"] = total_area_pred_label

    if nan_to_num is not None:
        metrics = dict(
            {
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in metrics.items()
            }
        )

    return metrics


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MeanDice(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                # 1st Seq - height dim, 2nd - width dim
                {
                    "predictions": datasets.Sequence(
                        datasets.Sequence(datasets.Value("uint16"))
                    ),
                    "references": datasets.Sequence(
                        datasets.Sequence(datasets.Value("uint16"))
                    ),
                }
            ),
            reference_urls=[
                "https://github.com/open-mmlab/mmsegmentation/blob/71c201b1813267d78764f306a297ca717827c4bf/mmseg/core/evaluation/metrics.py"
            ],
        )

    def _compute(
        self,
        predictions,
        references,
        num_labels: int,
        ignore_index: int,
        nan_to_num: Optional[int] = None,
        label_map: Optional[Dict[int, int]] = None,
        reduce_labels: bool = False,
    ):
        dice_result = mean_dice(
            results=predictions,
            gt_seg_maps=references,
            num_labels=num_labels,
            ignore_index=ignore_index,
            nan_to_num=nan_to_num,
            label_map=label_map,
            reduce_labels=reduce_labels,
        )
        return dice_result
