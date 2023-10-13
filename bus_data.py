from fastai.vision.all import *
import pandas as pd


# BUSIS dataset


def busis_img_depad(img_name):
    img_name = str(img_name)
    img_index = img_name[4:]
    img_index = int(img_index)
    return "case" + str(img_index)


def busis_get_class(row):
    image_stem = row.stem
    tumor_type = busis_classes[busis_classes["img name"] == image_stem][
        "tumor type"
    ].values[0]
    if tumor_type == "B":
        return "benign"
    elif tumor_type == "M":
        return "malignant"
    else:
        return None


def busis_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    label_name = busis_dataset / "GT" / f"{fn.stem}_GT.bmp"
    return [label_name]


def busis_dataset_make(dv):
    global busis_dataset
    busis_dataset = Path(dv + "/BUS Project Home/Datasets/BUSIS")

    global busis_classes
    busis_classes = pd.read_csv(busis_dataset / "BUSIS562.csv")[
        ["img name", "Ground Truth Name", "tumor type"]
    ]
    busis_classes.dropna(inplace=True)
    busis_classes["img name"] = busis_classes["img name"].apply(busis_img_depad)

    busis_all_images = [f for f in (busis_dataset / "Original").glob("*")]
    busis_all_masks = [busis_get_label(f) for f in busis_all_images]
    busis_df = pd.DataFrame(
        {
            "dataset": ["BUSIS"] * len(busis_all_images),
            "image": busis_all_images,
            "mask": busis_all_masks,
        }
    )
    busis_df["tumor"] = busis_df["image"].apply(busis_get_class)
    return busis_df


# BUS Dataset B


def get_bus_dataset_b_class(row):
    image_stem = int(row.stem)
    tumor_type = bus_dataset_b_classes[bus_dataset_b_classes["Image"] == image_stem][
        "Type"
    ].values[0]
    if tumor_type == "Benign":
        return "benign"
    elif tumor_type == "Malignant":
        return "malignant"
    else:
        return None


def bus_dataset_b_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    label_name = bus_dataset_b / "GT" / f"{fn.stem:0>6}.png"
    return [label_name]


def bus_dataset_b_make(dv):
    global bus_dataset_b
    bus_dataset_b = Path(dv + "/BUS Project Home/Datasets/BUS_Dataset_B")

    bus_dataset_b_all_images = [f for f in (bus_dataset_b / "original").glob("*")]
    bus_dataset_b_all_masks = [
        bus_dataset_b_get_label(f) for f in bus_dataset_b_all_images
    ]
    bus_dataset_b_df = pd.DataFrame(
        {
            "dataset": ["BUS_Dataset_B"] * len(bus_dataset_b_all_images),
            "image": bus_dataset_b_all_images,
            "mask": bus_dataset_b_all_masks,
        }
    )

    global bus_dataset_b_classes
    bus_dataset_b_classes = pd.read_excel(bus_dataset_b / "DatasetB.xlsx")
    bus_dataset_b_df["tumor"] = bus_dataset_b_df["image"].apply(get_bus_dataset_b_class)
    return bus_dataset_b_df


# Dataset BUSI with GT


def dataset_busi_with_gt_get_label(fn, image_class):
    """
    Given an image file name and a folder path,
    returns the paths to all corresponding masks.
    """
    mask_folder = image_class
    return [
        f
        for f in (dataset_busi_with_gt / mask_folder).glob("*")
        if fn.stem in str(f) and "_mask" in str(f)
    ]


def dataset_busi_with_gt_make(dv):
    global dataset_busi_with_gt
    dataset_busi_with_gt = Path(dv + "/BUS Project Home/Datasets/Dataset_BUSI_with_GT")
    dataset_busi_with_gt_all_images = []
    dataset_busi_with_gt_all_masks = []
    tumor_labels = []
    for image_class in ["benign", "malignant", "normal"]:
        dataset_busi_with_gt_images = [
            f
            for f in (dataset_busi_with_gt / image_class).glob("*")
            if "_mask" not in str(f)
        ]
        dataset_busi_with_gt_masks = [
            dataset_busi_with_gt_get_label(f, image_class)
            for f in dataset_busi_with_gt_images
        ]
        dataset_busi_with_gt_all_images = (
            dataset_busi_with_gt_all_images + dataset_busi_with_gt_images
        )
        dataset_busi_with_gt_all_masks = (
            dataset_busi_with_gt_all_masks + dataset_busi_with_gt_masks
        )
        tumor_labels = tumor_labels + [image_class] * len(dataset_busi_with_gt_images)
    dataset_busi_with_gt_df = pd.DataFrame(
        {
            "dataset": ["Dataset_BUSI_with_GT"] * len(dataset_busi_with_gt_all_images),
            "image": dataset_busi_with_gt_all_images,
            "mask": dataset_busi_with_gt_all_masks,
            "tumor": tumor_labels,
        }
    )
    return dataset_busi_with_gt_df


# Mayo dataset


def mayo_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    return [
        f for f in mayo_dataset.glob("*") if fn.stem in str(f) and "_mask" in str(f)
    ]


def get_mayo_class(image_path):
    image_stem = image_path.stem
    image_id = image_stem.split("_")[0]
    pathology_values = mayo_annotations[mayo_annotations["external_id"] == image_id][
        "pathology"
    ].values
    if pathology_values.size == 1:
        return pathology_values[0]
    else:
        return None


def mayo_dataset_make(dv):
    global mayo_dataset
    mayo_dataset = Path(dv + "/BUS Project Home/Datasets/Mayo/mayo_dataset")

    mayo_all_images = [f for f in mayo_dataset.glob("*.png") if "_mask" not in str(f)]
    mayo_all_masks = [mayo_get_label(f) for f in mayo_all_images]

    out_df = pd.DataFrame(
        {
            "dataset": ["Mayo"] * len(mayo_all_images),
            "image": mayo_all_images,
            "mask": mayo_all_masks,
        }
    )

    global mayo_annotations
    mayo_annotations = pd.read_csv(
        mayo_dataset / "annotations_histology.csv", dtype={"external_id": "string"}
    )[["external_id", "pathology"]]
    mayo_annotations["pathology"] = mayo_annotations["pathology"].apply(
        lambda x: x.lower()
    )
    mayo_annotations["pathology"] = mayo_annotations["pathology"].apply(
        lambda x: "benign" if x == "elevated risk" else x
    )
    mayo_annotations = mayo_annotations[mayo_annotations["pathology"] != "unknown"]

    out_df["tumor"] = out_df["image"].apply(get_mayo_class)
    out_df.dropna(axis=0, subset=["tumor"], inplace=True)
    out_df.reset_index(drop=True, inplace=True)
    return out_df


# custom methods for the Mayo dataset


def mayo_mask_make(row):
    """
    Is called by DataBlock(getters=).
    Takes a list of paths to mask files from a Pandas column.
    Makes sure all masks are 8 bits per pixel.
    If there are multiple masks, merges them.
    Returns a PILMask.create() mask image.
    """
    f = ColReader("mask")
    # PILMask.create() probably forces 8 bits per pixel.
    all_images = [np.asarray(PILMask.create(x)) for x in f(row)]
    image_stack = np.stack(all_images)
    if row["tumor"] == "benign":
        image_stack = np.clip(image_stack, 0, 127)
    image_union = np.amax(image_stack, axis=0)
    return PILMask.create(image_union)


def mayo_image_make(row):
    """
    Receives a Pandas row. Gets an image path from the "image" column.
    Makes sure all images are 8 bits per color channel.
    (There may be multiple color channels.)
    Returns a PILImage.create() image.
    """
    f = ColReader("image")
    # PILImage.create() probably forces 8 bits per color channel.
    image_array = np.asarray(PILImage.create(f(row)))
    return PILImage.create(image_array)


class MayoCustomTransform(DisplayedTransform):
    """
    Chain of custom transforms for Mayo images:
    - resize to a common size (a few images are very large)
    - crop out useless borders (preserve actual image area)
    - resize to standard ResNet input size
    """

    def __init__(self, resize_init, box_shape, resize_final):
        # width, height
        self.resize_init = resize_init
        self.resize_final = resize_final
        # left, upper, right, lower
        self.box_shape = box_shape

    def encodes(self, x):
        x_res_init = x.resize(size=self.resize_init)
        x_cropped = x_res_init.crop(box=self.box_shape)
        x_res_final = x_cropped.resize(size=self.resize_final)
        return x_res_final


# BUV dataset


def buv_dataset_make(dv):
    buv_dataset = Path(dv + "/BUS Project Home/Datasets/BUV_dataset")

    buv_df = pd.DataFrame(
        {
            "dataset": [],
            "class": [],
            "video": [],
            "image": [],
        }
    )

    for image_class in ["benign", "malignant"]:
        buv_folders = [v for v in (buv_dataset / "rawframes" / image_class).glob("*")]
        buv_folders.sort()
        for v in buv_folders:
            buv_images = [
                f for f in (buv_dataset / "rawframes" / image_class / v.stem).glob("*")
            ]
            buv_images.sort()
            buv_df_short = pd.DataFrame(
                {
                    "dataset": ["BUV_dataset"] * len(buv_images),
                    "class": [image_class] * len(buv_images),
                    "video": [v.stem] * len(buv_images),
                    "image": buv_images,
                }
            )
            buv_df = pd.concat([buv_df, buv_df_short], ignore_index=True)

    buv_df.reset_index(drop=True, inplace=True)

    buv_df["str_index"] = buv_df.index.to_list()
    dataset_length = len(str(buv_df.shape[0]))
    buv_df["str_index"] = buv_df["str_index"].apply(
        lambda x: str(x).rjust(dataset_length, "0")
    )
    return buv_df


# Mapping classes to/from pixel values


def labels_ids_bus(multiclass=True):
    """
    Generate mappings between labels and IDs
    """
    if multiclass:
        id2label = {0: "unlabeled", 1: "benign", 2: "malignant"}
        label2id = {"unlabeled": 0, "benign": 1, "malignant": 2}
        num_labels = len(id2label)
    else:
        id2label = {0: "unlabeled", 1: "lesion"}
        label2id = {"unlabeled": 0, "lesion": 1}
        num_labels = len(id2label)
    return id2label, label2id, num_labels
