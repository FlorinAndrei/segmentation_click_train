import numpy as np
import PIL
from PIL import Image
import multiprocessing
from datasets import Dataset, ClassLabel
from datasets import Image as ImageDS


def map_hf_train_ds(row):
    """
    input: a Pandas DF row with image and labels
    output: a dictionary with the image and the labels

    Convert mask pixel values as needed:
    - background = 0
    - lesion = 1

    Reshape all image frames to the standard size.
    """
    ret = {}

    ret["dataset"] = row["dataset"]
    # first RGB convert is to catch any non-RGB images and fix them
    # convert to L is to convert to black-and-white
    # second convert to RGB is to generate a black-and-white RGB image
    # final convert to RGB zeroes out the R and G channels
    #
    # the finer and more obscure points of the various conversions
    # are completely outsourced to PIL
    # which we treat as the gold standard of image processing
    ret["pixel_values"] = (
        Image.open(row["pixel_values"])
        .convert(mode="RGB")
        .convert(mode="L")
        .convert(mode="RGB")
        .convert(mode="RGB", matrix=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0))
        .resize(image_size, resample=PIL.Image.Resampling.BILINEAR)
    )
    # preserve original images for display
    ret["original_image"] = (
        Image.open(row["pixel_values"])
        .convert(mode="RGB")
        .convert(mode="L")
        .convert(mode="RGB")
        .resize(image_size, resample=PIL.Image.Resampling.BILINEAR)
    )

    # merge all mask images into a single mask
    mask_list = [
        np.asarray(Image.open(x).convert("L"), dtype=np.uint8) for x in row["mask"]
    ]

    # set clip values based on mask existence
    if row["tumor"] == "benign" or row["tumor"] == "malignant":
        clip_val = label2id["lesion"]
    else:
        clip_val = label2id["unlabeled"]
    # build the actual mask image
    mask_frame = Image.fromarray(
        np.clip(np.amax(np.stack(mask_list), axis=0), a_min=0, a_max=clip_val)
    ).resize(image_size, resample=PIL.Image.Resampling.NEAREST)
    ret["label"] = mask_frame

    ret["tumor"] = row["tumor"]
    ret["dataset_tumor"] = row["dataset_tumor"]
    ret["index"] = row["index"]
    return ret


def create_hf_train_dataset(df):
    """
    input: Pandas dataframe with the training dataset paths and labels
    output: HuggingFace dataset with training data
    """
    dataset = Dataset.from_dict(
        {
            "dataset": df["dataset"].to_list(),
            "pixel_values": df["image"].to_list(),
            "mask": df["mask"].to_list(),
            "tumor": df["tumor"].to_list(),
            "dataset_tumor": df["dataset_tumor"].to_list(),
            "index": train_df.index.to_list(),
        }
    )
    dataset = dataset.map(map_hf_train_ds, num_proc=multiprocessing.cpu_count())
    dataset = dataset.cast_column(
        "dataset_tumor", ClassLabel(names=df["dataset_tumor"].unique().tolist())
    )
    dataset = dataset.cast_column(
        "tumor", ClassLabel(names=df["tumor"].unique().tolist())
    )
    dataset = dataset.cast_column(
        "dataset", ClassLabel(names=df["dataset"].unique().tolist())
    )
    dataset = dataset.cast_column("pixel_values", ImageDS())
    return dataset.remove_columns("mask")
