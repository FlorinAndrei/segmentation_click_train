import numpy as np
import pandas as pd
from scipy import ndimage
import PIL
from PIL import Image
import os
import sys
import math
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import pickle
from datasets import load_from_disk


def gen_seg_df(s_type, segments, iou, i, segment_files):
    seg_dict = {
        "image_index": [],
        "iou": [],
        "segment_index": [],
        "segment_type": [],
        "segment_size": [],
    }
    seg_arr, seg_count = segments[0].reshape((512, 512, 1)), segments[1]
    if seg_count > 1:
        # generate multiple identical slices, as many as there are segments
        seg_arr = np.broadcast_to(seg_arr, (512, 512, seg_count)).copy()
    # this seg_index has a minimum of 1
    for seg_index in range(1, seg_count + 1):
        seg_slice = seg_arr[:, :, seg_index - 1]
        # zero out all values that do not correspond to this segment
        seg_slice[seg_slice != seg_index] = 0
        # change all non-zero values to 1
        seg_slice = seg_slice // seg_index
        seg_size = seg_slice.sum()

        seg_dict["image_index"].append(i)
        seg_dict["iou"].append(iou)
        seg_dict["segment_index"].append(seg_index)
        seg_dict["segment_type"].append(s_type)
        seg_dict["segment_size"].append(seg_size)

        with open(
            segment_files + "/" + str(i) + "/" + s_type + "/" + str(seg_index) + ".pkl",
            "wb",
        ) as seg_pkl:
            pickle.dump(seg_slice, seg_pkl, protocol=5)

    return seg_dict


def make_segments(args):
    (
        rows,
        train_dataset_path,
        predictions_path,
        segment_files,
    ) = args
    ds_hf = load_from_disk(train_dataset_path)

    seg_df = pd.DataFrame(
        columns=[
            "image_index",
            "iou",
            "segment_index",
            "segment_type",
            "segment_size",
        ]
    )
    structure_array = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    for i in rows.index:
        os.makedirs(segment_files + "/" + str(i) + "/false_positives", exist_ok=True)
        os.makedirs(segment_files + "/" + str(i) + "/false_negatives", exist_ok=True)
        os.makedirs(segment_files + "/" + str(i) + "/true_positives", exist_ok=True)
        iou = rows.loc[i, "iou"]
        label_arr = np.array(
            ds_hf[i]["label"].resize((512, 512), resample=PIL.Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        pred_arr = np.array(
            Image.open(predictions_path + "/" + str(i) + ".png"), dtype=np.uint8
        )

        false_positive_arr = pred_arr - pred_arr * label_arr
        false_positive_segments = ndimage.label(
            false_positive_arr, structure=structure_array, output=np.uint8
        )
        seg_df = pd.concat(
            [
                seg_df,
                pd.DataFrame(
                    gen_seg_df(
                        "false_positives",
                        false_positive_segments,
                        iou,
                        i,
                        segment_files,
                    )
                ),
            ]
        )

        false_negative_arr = label_arr - pred_arr * label_arr
        false_negative_segments = ndimage.label(
            false_negative_arr, structure=structure_array, output=np.uint8
        )
        seg_df = pd.concat(
            [
                seg_df,
                pd.DataFrame(
                    gen_seg_df(
                        "false_negatives",
                        false_negative_segments,
                        iou,
                        i,
                        segment_files,
                    )
                ),
            ]
        )

        true_positive_arr = pred_arr * label_arr
        true_positive_segments = ndimage.label(
            true_positive_arr, structure=structure_array, output=np.uint8
        )
        seg_df = pd.concat(
            [
                seg_df,
                pd.DataFrame(
                    gen_seg_df(
                        "true_positives",
                        true_positive_segments,
                        iou,
                        i,
                        segment_files,
                    )
                ),
            ]
        )

    return seg_df


if __name__ == "__main__":
    baseline_dir = sys.argv[1]
    train_dataset_path = sys.argv[2]
    predictions_path = sys.argv[3]

    segment_files = baseline_dir + "/segment_files"
    segment_stats_file = baseline_dir + "/segment_stats.csv"

    QUIT = False
    quit_message = ""
    if os.path.exists(segment_files):
        QUIT = True
        quit_message += f"Segment files directory {segment_files} exists. "
    if os.path.exists(segment_stats_file):
        QUIT = True
        quit_message += f"Segment stats file {segment_stats_file} exists. "
    if QUIT:
        quit_message += "Exit."
        print(quit_message)
        exit()

    os.makedirs(segment_files, exist_ok=True)

    image_perf_df = pd.read_csv(baseline_dir + "/image_perf_df.csv")

    n_cpus = multiprocessing.cpu_count()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    p = Pool(processes=n_cpus)
    mp_results = []

    step_multiplier = 10

    step_size = n_cpus * step_multiplier
    for i in tqdm(range(0, image_perf_df.shape[0], step_size)):
        rows = image_perf_df.loc[i : i + step_size - 1, :]
        rows_index = rows.index.to_list()
        arglist = list(
            zip(
                [
                    rows.loc[x : x + step_multiplier - 1, :].copy(deep=True)
                    for x in range(
                        rows_index[0],
                        rows_index[-1],
                        math.ceil(len(rows_index) / n_cpus),
                    )
                ],
                [train_dataset_path] * n_cpus,
                [predictions_path] * n_cpus,
                [segment_files] * n_cpus,
            )
        )
        mp_results.append(p.map(make_segments, arglist))

    p.close()
    segment_stats = pd.concat([pd.concat(x) for x in mp_results]).reset_index(drop=True)
    segment_stats.to_csv(segment_stats_file)
