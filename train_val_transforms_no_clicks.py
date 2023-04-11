import numpy as np
import albumentations as A
import cv2
from transformers import SegformerFeatureExtractor


rotate_limit = 20
shear_limit = 20

# the workhorse of feature extraction
# our transforms are built around it
feature_extractor = SegformerFeatureExtractor(do_normalize=True)

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


def train_transforms(example_batch):
    """
    input: a batch of images and masks
    output: augmented and preprocessed images and masks

    This is for training.

    Images get all the augmentations, including pixel-value (brightness, contrast).
    Images also get feature-extracted (normalized).

    Masks only get the geometric transforms (rotation, etc).
    """
    batch_items = list(
        zip(
            [x for x in example_batch["pixel_values"]],
            [x for x in example_batch["label"]],
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

    Images get normalized.
    """
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]

    inputs = feature_extractor(images, labels)
    return inputs


def val_transforms_check(example_batch):
    """
    input: a batch of images and masks
    output: normalized images, masks, and some attributes from original dataset

    This is for performance check after training. No augmentations done here.

    Images get normalized.
    """
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    indices = [x for x in example_batch["index"]]
    datasets = [x for x in example_batch["dataset"]]
    tumors = [x for x in example_batch["tumor"]]

    inputs = feature_extractor(images, labels)
    inputs["index"] = indices
    inputs["dataset"] = datasets
    inputs["tumor"] = tumors
    return inputs
