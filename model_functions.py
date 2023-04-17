import shutil
import os
import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import torch
from torch import nn
from IPython.utils.io import capture_output
from transformers import logging
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback as ESC
import evaluate


metric = evaluate.load(
    "evaluate/metrics/mean_dice",
    num_process=multiprocessing.cpu_count(),
)


def compute_metrics(eval_pred):
    """
    Compute training / validation metrics.
    This is called by the trainer.
    """
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)

        # logits do not have the same size as the training images and labels
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=feature_extractor.reduce_labels,
        )

        mean_niou = metrics["mean_niou"]
        mean_dice = metrics["mean_dice"]
        metrics.update({"mean_niou": mean_niou})
        metrics.update({"mean_dice": mean_dice})

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_niou = metrics.pop("per_category_niou").tolist()
        per_category_dice = metrics.pop("per_category_dice").tolist()
        per_category_precision = metrics.pop("per_category_precision").tolist()
        per_category_recall = metrics.pop("per_category_recall").tolist()
        total_area_label = metrics.pop("total_area_label").tolist()
        total_area_pred = metrics.pop("total_area_pred").tolist()

        metrics.update(
            {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
        )
        metrics.update(
            {f"iou_{id2label[i]}": v for i, v in enumerate(per_category_niou)}
        )
        metrics.update(
            {f"dice_{id2label[i]}": v for i, v in enumerate(per_category_dice)}
        )
        metrics.update(
            {
                f"precision_{id2label[i]}": v
                for i, v in enumerate(per_category_precision)
            }
        )
        metrics.update(
            {f"recall_{id2label[i]}": v for i, v in enumerate(per_category_recall)}
        )
        metrics.update(
            {
                f"total_area_label_{id2label[i]}": v
                for i, v in enumerate(total_area_label)
            }
        )
        metrics.update(
            {f"total_area_pred_{id2label[i]}": v for i, v in enumerate(total_area_pred)}
        )

        return metrics


def objective(
    fold, train_ds, test_ds, model_dir, logging_dir, output_dir, folds_dir, epochs=100
):
    """
    input: k-fold number
    output: best model performance metric

    Calls train_model() with the appropriate hyperparameter values.
    Captures and logs all output from train_model().
    """
    learning_rate = 2e-4

    # define log file, log parameters
    fold_number_pad = str(fold).zfill(10)
    output_file_name = folds_dir + "/fold_" + fold_number_pad
    with open(output_file_name, "w") as cf:
        print(f"fold={fold}\n", file=cf)

    # main training step
    with capture_output(stdout=True, stderr=True, display=True) as captured:
        best_metrics = train_model(
            learning_rate=learning_rate,
            fold_number=fold,
            train_ds=train_ds,
            test_ds=test_ds,
            model_dir=model_dir,
            logging_dir=logging_dir,
            output_dir=output_dir,
            epochs=epochs,
        )
    with open(output_file_name, "a") as cf:
        print(captured.stdout, file=cf)
        print(captured.stderr, file=cf)

    del captured
    return best_metrics


def train_model(
    learning_rate,
    fold_number,
    train_ds,
    test_ds,
    model_dir,
    logging_dir,
    output_dir,
    epochs=50,
):
    """
    input: hyperparameter values and fold number
    output: best IoU from model validation

    Load a pretrained model.
    Train it using the indicated hyperparameters.
    Save the best model.
    Return the best model performance.
    """

    # delete/recreate folder with per-fold best and latest models
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # fixed hyperparameters
    # the model was chosen based on size
    pretrained_model_name = model_name_full
    epochs = epochs
    batch_size = model_batch_size
    monitor_metric = "eval_loss"

    # avoid printing useless warnings
    logging.set_verbosity(50)
    # load pretrained transformer from public repos
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
    )
    # restore normal verbosity
    logging.set_verbosity(40)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        lr_scheduler_type="linear",
        warmup_ratio=1e-2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        auto_find_batch_size=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        save_total_limit=5,
        eval_steps=20,
        logging_dir=logging_dir + "/fold-" + str(fold_number).zfill(10),
        logging_steps=1,
        log_level="error",
        eval_accumulation_steps=5,
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model=monitor_metric,
        greater_is_better=False,
        remove_unused_columns=False,
        optim="adamw_hf",
        push_to_hub=False,
        dataloader_num_workers=multiprocessing.cpu_count(),
        # dataloader_num_workers=0,
        seed=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[ESC(100)],
    )

    trainer.train()

    # extract best step data from trainer history
    validation_history = []
    for e in trainer.state.log_history:
        if "eval_iou_lesion" in e:
            validation_history.append(e)
    best_metrics = {}
    best_metrics["best_iou"] = 0
    best_metrics["best_dice"] = 0
    best_metrics["best_precision"] = 0
    best_metrics["best_recall"] = 0
    best_metrics["best_step"] = 0
    best_metrics["best_loss"] = 1
    for e in validation_history:
        if e["eval_loss"] < best_metrics["best_loss"]:
            best_metrics["best_loss"] = e["eval_loss"]
            best_metrics["best_iou"] = e["eval_iou_lesion"]
            best_metrics["best_dice"] = e["eval_dice_lesion"]
            best_metrics["best_precision"] = e["eval_precision_lesion"]
            best_metrics["best_recall"] = e["eval_recall_lesion"]
            best_metrics["best_step"] = e["step"]
    print("")
    print(f"best_step:      {best_metrics['best_step']}")
    print(f"best_iou:       {best_metrics['best_iou']}")
    print(f"best_dice:      {best_metrics['best_dice']}")
    print(f"best_precision: {best_metrics['best_precision']}")
    print(f"best_recall:    {best_metrics['best_recall']}")
    print(f"best_loss:      {best_metrics['best_loss']}")

    # move this fold's best model to model_dir
    shutil.move(
        output_dir + "/checkpoint-" + str(best_metrics["best_step"]),
        model_dir + "/fold-" + str(fold_number).zfill(10),
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    del model
    torch.cuda.empty_cache()
    return best_metrics


def check_validation_performance(model, test_ds, inputs_template, save_path):
    frame_performance = {}
    predictions_list = []
    labels_list = []

    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(len(test_ds))):
            label = test_ds[i]["labels"]
            image = test_ds[i]["pixel_values"]
            index = test_ds[i]["index"]

            inputs = inputs_template
            # fill in the contents of the template while keeping its structure unchanged
            inputs["pixel_values"] = torch.from_numpy(image[np.newaxis, ...])
            inputs.to("cuda:0")
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.shape[1:],  # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_seg_cpu = pred_seg.detach().cpu()

            predictions_list.append(pred_seg_cpu)
            labels_list.append(label)

            frame_metrics = metric._compute(
                predictions=[pred_seg_cpu],
                references=[label],
                num_labels=len(id2label),
                ignore_index=255,
                reduce_labels=feature_extractor.reduce_labels,
            )

            prediction_frame_arr = np.array(pred_seg_cpu, dtype=np.int8)
            prediction_frame = Image.fromarray(prediction_frame_arr, mode="L")
            prediction_frame.save(save_path + "/" + str(index) + ".png")

            frame_performance[index] = frame_metrics

        fold_metrics = metric._compute(
            predictions=predictions_list,
            references=labels_list,
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=feature_extractor.reduce_labels,
        )

    return frame_performance, fold_metrics


def generate_predictions(model, test_ds, pr_path):
    performance_metrics_fold = {}

    print(f"generate predictions, calculate performance")
    # For simplicity, generate once an inputs template with the right structure and types.
    # The contents will be filled in via torch.from_numpy() for each image.
    inputs_template = feature_extractor(
        images=test_ds[0]["pixel_values"], return_tensors="pt"
    )

    (
        frame_metrics_checked,
        fold_metrics_checked,
    ) = check_validation_performance(
        model=model,
        test_ds=test_ds,
        inputs_template=inputs_template,
        save_path=pr_path,
    )
    frame_means = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    fold_means = {}

    i = 0
    for k, v in frame_metrics_checked.items():
        i += 1
        for z, w in v.items():
            if isinstance(w, np.ndarray):
                v[z] = v[z].tolist()
        performance_metrics_fold[k] = v
        frame_means["dice"] += v["per_category_dice"][1]
        frame_means["iou"] += v["per_category_niou"][1]
        frame_means["precision"] += v["per_category_precision"][1]
        frame_means["recall"] += v["per_category_recall"][1]

    for k, v in frame_means.items():
        frame_means[k] = frame_means[k] / len(frame_metrics_checked.keys())

    fold_means["dice"] = fold_metrics_checked["per_category_dice"][1]
    fold_means["iou"] = fold_metrics_checked["per_category_niou"][1]
    fold_means["precision"] = fold_metrics_checked["per_category_precision"][1]
    fold_means["recall"] = fold_metrics_checked["per_category_recall"][1]

    print()
    print("performance frame by frame, averaged by fold, measured with trained model:")
    print(pd.DataFrame(frame_means, index=["values"]).T)
    print()
    print(
        "mean performance per fold, from total pixel counts across the entire fold, measured with trained model"
    )
    print(pd.DataFrame(fold_means, index=["values"]).T)

    return performance_metrics_fold
