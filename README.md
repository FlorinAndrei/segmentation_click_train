# Train Image Segmentation Models to Accept User Feedback

This is an extension of my Data Science Capstone project: https://github.com/FlorinAndrei/datascience_capstone_project

Both the Capstone and this project are work done within the Computer Aided Diagnosis for Breast Ultrasound Imagery (CADBUSI) project at the University of Wisconsin-La Crosse, under the supervision of Dr. Jeff Baggett. https://datascienceuwl.github.io/CADBUSI/

## Goals and general description

Train off-the-shelf image segmentation models to respond to user feedback. The user provides feedback by clicking the image. The user may provide positive feedback ("this is a region of interest") or negative feedback ("this is not a region of interest").

For this project, I've created a technique for generating synthetic training data that closely simulates feedback from human operators. Human input is slow and costly to produce. This technique has drastically cut down the time and the cost for generating this data.

Articles on Towards Data Science describing the project in detail:

https://towardsdatascience.com/train-image-segmentation-models-to-accept-user-feedback-via-voronoi-tiling-part-1-8ab85d410d29

https://towardsdatascience.com/train-image-segmentation-models-to-accept-user-feedback-via-voronoi-tiling-part-2-1f02eebbddb9

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

The algorithm is explained in detail in the articles on TDS, linked above. In a nutshell, by choosing a specific energy functional associated with the click coordinates, minimizing the energy via simulated annealing leads to a placement of the clicks that closely simulates what a human operator would do.

Python code with the click generator: [uniform_clicks.py](uniform_clicks.py)

## Demo

You can see the click-trained models responding to user feedback in these videos:

[https://youtu.be/wc3eocA1VG8](https://youtu.be/wc3eocA1VG8)

[https://youtu.be/cPTx4NEWJC8](https://youtu.be/cPTx4NEWJC8)

Some performance metrics: [visualize_performance.ipynb](visualize_performance.ipynb)

The model performance (IoU, Dice) is on par with state of the art segmentation for ultrasound imaging.

The datasets and the trained models are not included in the repo. Only the code and a few artifacts (diagrams, summary performance data) are included here.
