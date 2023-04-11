import numpy as np
import albumentations as A
import cv2
from transformers import SegformerFeatureExtractor


rotate_limit = 20
shear_limit = 20

# modeled loosely after the default FastAI augment values
training_augments = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            limit=(-rotate_limit, rotate_limit),
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            mask_value=0,
            p=0.9,
        ),
        A.Affine(
            scale=(0.833, 1.2),
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=None,
            shear={"x": (-shear_limit, shear_limit), "y": (-shear_limit, shear_limit)},
            interpolation=cv2.INTER_LINEAR,
            cval_mask=0,
            p=0.9,
        ),
    ]
)


def apply_clicks(images, indices, probability=1.0):
    """
    input: list of images (Numpy arrays) and list of image indices
    output: list of images with clicks applied

    Relies on a global variable guide_clicks, containing click coordinates
    """
    for i in range(len(images)):
        if (
            len(guide_clicks[indices[i]]["clicks_positive"]) > 0
            or len(guide_clicks[indices[i]]["clicks_negative"]) > 0
        ):
            img_arr = np.array(images[i])
            if np.random.uniform() <= probability:
                for c in guide_clicks[indices[i]]["clicks_positive"]:
                    cx = round(c[0] * img_arr.shape[0] / 512)
                    cy = round(c[1] * img_arr.shape[1] / 512)
                    img_arr[cx - 1 : cx + 2, cy - 1 : cy + 2, 1] = 255
                for c in guide_clicks[indices[i]]["clicks_negative"]:
                    cx = round(c[0] * img_arr.shape[0] / 512)
                    cy = round(c[1] * img_arr.shape[1] / 512)
                    img_arr[cx - 1 : cx + 2, cy - 1 : cy + 2, 0] = 255
            # images[i] = Image.fromarray(img_arr)
            images[i] = img_arr

    return images


def train_transforms(example_batch):
    """
    input: a batch of images and masks
    output: augmented and preprocessed images and masks

    This is for training.

    Images get all the augmentations, including pixel-value (brightness, contrast).
    Images also get feature-extracted (normalized).

    Masks only get the geometric transforms (rotation, etc).
    """
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    indices = [x for x in example_batch["index"]]

    if APPLY_CLICKS_TRAIN_DS == True:
        images = apply_clicks(images, indices, 0.9)

    batch_items = list(
        zip(
            images,
            labels,
        )
    )
    batch_items_aug = [
        training_augments(
            image=np.array(x[0]),
            mask=np.array(x[1]),
        )
        for x in batch_items
    ]
    images = [i["image"] for i in batch_items_aug]
    labels = [i["mask"] for i in batch_items_aug]
    inputs = feature_extractor(images, labels)
    return inputs


def val_transforms(example_batch):
    """
    input: a batch of images and masks
    output: normalized images, and masks

    This is for validation while training. No augmentations done here.

    Images get normalized by feature_extractor().

    Clicks are enabled / disabled via global variable APPLY_CLICKS_TEST_DS
    """
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    indices = [x for x in example_batch["index"]]

    if APPLY_CLICKS_TEST_DS == True:
        images = apply_clicks(images, indices)

    inputs = feature_extractor(images, labels)
    return inputs


def val_transforms_check(example_batch):
    """
    input: a batch of images and masks
    output: normalized images, masks, and some attributes from original dataset

    This is for performance check after training. No augmentations done here.
    Cannot use this function to feed inputs to the model in a training loop,
    because the model does not accept the extra attributes.

    Images get normalized by feature_extractor().

    Clicks are enabled / disabled via global variable APPLY_CLICKS_TEST_DS
    """
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    indices = [x for x in example_batch["index"]]
    datasets = [x for x in example_batch["dataset"]]
    tumors = [x for x in example_batch["tumor"]]

    if APPLY_CLICKS_TEST_DS == True:
        images = apply_clicks(images, indices)

    inputs = feature_extractor(images, labels)
    inputs["index"] = indices
    inputs["dataset"] = datasets
    inputs["tumor"] = tumors
    return inputs
