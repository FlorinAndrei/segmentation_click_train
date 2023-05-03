# Train Image Segmentation Models to Accept User Feedback

This is an extension of my Capstone project: https://github.com/FlorinAndrei/datascience_capstone_project

Both the Capstone and this project are work done within the Computer Aided Diagnosis for Breast Ultrasound Imagery (CADBUSI) project at the University of Wisconsin-LaCrosse, under the supervision of Dr. Jeff Baggett. https://datascienceuwl.github.io/CADBUSI/

## Goal

Train off-the-shelf image segmentation models to respond to user feedback. The user provides feedback by clicking the image. The user may provide positive feedback ("this is a region of interest") or negative feedback ("this is not a region of interest").

## Algorithm

Since predictions are made on monochrome images (ultrasound scan images), the color channels (RGB) are redundant. All image data can be moved to one channel (B), leaving two channels (R and G) available for user feedback.

The dataset is split into 5 folds, and 5 baseline models are trained, one for each fold. The baseline models are used to make predictions for all images in the dataset. False positive (FP) and false negative (FN) areas are extracted from predictions.

Positive clicks (G channel) are placed in the false negative areas. Negative clicks (R channel) are placed in the false positive areas. A new set of models (the click-trained models) are now trained on the images with the clicks added.

The click-trained models will respond to user-generated clicks placed in the R and G channels.

Main notebook with the model training code: [train_models.ipynb](train_models.ipynb)

## Models

The SegFormer architecture, pretrained on ImageNet, was used for this project. Pretrained models are available at HuggingFace:

[https://huggingface.co/docs/transformers/model_doc/segformer](https://huggingface.co/docs/transformers/model_doc/segformer)

## Automating Click Generation

Manually generating clicks for training the click-trained models is tedious and does not scale. An automated procedure was devised, generalizing the concept of centroidal Voronoi tiling, and using simulated annealing, to arrive to an optimal placement of positive / negative clicks in the FN / FP areas.

Python code with the click generator: [uniform_clicks.py](uniform_clicks.py)

## Demo

You can see the click-trained models responding to user feedback in these videos:

[https://youtu.be/wc3eocA1VG8](https://youtu.be/wc3eocA1VG8)

[https://youtu.be/cPTx4NEWJC8](https://youtu.be/cPTx4NEWJC8)

Some performance metrics: [visualize_performance.ipynb](visualize_performance.ipynb)

Note: While the performance metrics (IoU, Dice) of the click-trained models appear very high, exceeding the current state of the art, this is because adding the clicks leaks data between train and test. When tested on 100% previously unseen data, the click-trained models perform the same as the baseline models.
