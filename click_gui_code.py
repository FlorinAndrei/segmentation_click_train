import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets.widgets import (
    Box,
    VBox,
    HBox,
    Button,
    Dropdown,
    RadioButtons,
    FloatSlider,
    FloatLogSlider,
    Layout,
    Checkbox,
)
import os
import torch
from torch import nn
from datasets import load_from_disk
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor


class ClickWidget:
    def __init__(self, train_dataset_path, model_prefix, img_size=(512, 512)):
        # standard size of the images
        self.img_size = img_size
        # where is the dataset located
        self.train_dataset_path = train_dataset_path

        self.model = None
        self.id2label = {0: "unlabeled", 1: "lesion"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)
        self.feature_extractor = SegformerFeatureExtractor(do_normalize=True)

        self.train_df = pd.read_csv("train_df.csv", index_col=0)
        self.ds_hf = load_from_disk(self.train_dataset_path)
        self.image_ids = self.train_df.index.to_list()
        self.image_attr = (
            self.train_df.index.astype("str")
            + " - "
            + self.train_df["dataset"]
            + " - "
            + self.train_df["tumor"]
        ).to_list()
        self.image_select_tuples = list(zip(self.image_attr, self.image_ids))
        self.current_id = self.image_ids[0]

        self.model_prefix = model_prefix
        self.model_type_list = [
            f.path
            for f in os.scandir(".")
            if f.is_dir() and self.model_prefix in str(f)
        ]
        self.model_type_list.sort()
        self.model_type_current = self.model_type_list[0]
        self.get_model_folds()

        self.output_options = ["original image", "model input"]
        self.prediction_type = "segmentation"
        self.heat_map_gamma = 1.0
        self.what_to_show = self.output_options[0]
        self.clicks = []
        self.click_type = "positive"
        self.click_colors = {
            "positive": (0, 255, 0),
            "negative": (255, 0, 0),
        }
        self.click_layers = {
            "positive": 1,
            "negative": 0,
        }
        self.brightness = 1.0
        self.contrast = 1.0
        self.display_prediction = True

        self.initialize_widget()

    def get_model_folds(self):
        self.model_folders = sorted(
            [f for f in os.listdir(self.model_type_current + "/models")]
        )
        self.model_folder_current = self.model_folders[0]

    def load_model(self):
        """
        Load model from disk.

        output:
        - self.model
        """
        del self.model
        torch.cuda.empty_cache()
        model_dir = self.model_type_current + "/models/" + self.model_folder_current
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=self.num_labels,
        )
        self.model.to("cuda:0")

    def apply_image_trace(self):
        """
        Merge image with mask prediction and apply them to the display widget.

        input:
        - self.img_to_display
        - self.img_for_model
        - self.prediction_frame

        output:
        - self.img_trace
        """
        if self.what_to_show == "original image":
            temp_fig = np.array(self.img_to_display, dtype=np.uint8)
        elif self.what_to_show == "model input":
            temp_fig = self.img_for_model
        # temp_fig.shape is (512, 512, 3)
        image = Image.fromarray(temp_fig, mode="RGB")
        if self.display_prediction:
            if self.prediction_type == "segmentation":
                mask = ImageOps.colorize(
                    Image.fromarray(self.prediction_frame * 255, mode="L"),
                    black=[0, 0, 0],
                    white=[255, 255, 0],
                    blackpoint=0,
                    whitepoint=255,
                )
                image_mask_merged = Image.composite(
                    mask, image, Image.fromarray(self.prediction_frame * 64, mode="L")
                )
            elif self.prediction_type == "heat map":
                # replicate prediction on R and G channels
                # zero out the B channel
                # so we get a yellow image
                heat_map_array = np.stack(
                    [
                        self.prediction_frame,
                        self.prediction_frame,
                        np.zeros_like(self.prediction_frame),
                    ],
                    axis=-1,
                )
                heat_map = Image.fromarray(heat_map_array, mode="RGB")
                image_mask_merged = Image.blend(image, heat_map, 0.5)
        else:
            image_mask_merged = image
        self.img_trace = px.imshow(np.array(image_mask_merged, dtype=np.uint8)).data[0]

    def gen_img_trace(self):
        """
        Generate the image that's normally displayed.
        Generate the image that is input to the model.
        Apply clicks to both.

        inputs:
        - self.current_img
        - self.clicks

        outputs:
        - self.img_to_display
        - self.img_for_model
        """
        # self.current_img = the original image loaded from disk
        # self.img_to_display = Pillow object, the image to be displayed
        # self.img_for_model = Numpy array, the image for the segmentation model

        self.img_to_display = self.current_img

        # https://stackoverflow.com/questions/75883474/pillow-imageenhance-brightness-and-contrast-what-is-the-best-order-to-apply-the
        enhancer_contrast = ImageEnhance.Contrast(self.img_to_display)
        self.img_to_display = enhancer_contrast.enhance(self.contrast)
        enhancer_brightness = ImageEnhance.Brightness(self.img_to_display)
        self.img_to_display = enhancer_brightness.enhance(self.brightness)

        self.img_for_model = np.array(self.img_to_display, dtype=np.uint8)
        # zero out the R and G channels
        self.img_for_model[:, :, 0:2] = 0

        for click in self.clicks:
            # Regarding the order of x and y:
            # - Numpy and Plotly have one convention
            # - PIL has the opposite convention
            # - segmentation models take Numpy and convert it to PyTorch tensors
            # - we follow Numpy, and adjust for PIL where needed
            x = click["coord"][0]
            y = click["coord"][1]

            color = self.click_colors[click["type"]]
            # PIL x/y order
            ImageDraw.Draw(self.img_to_display).rectangle(
                xy=[(y - 1, x - 1), (y + 1, x + 1)],
                outline=color,
                fill=color,
                width=1,
            )

            layer = self.click_layers[click["type"]]
            # Numpy x/y order
            self.img_for_model[x - 1 : x + 2, y - 1 : y + 2, layer] = 255

        self.make_prediction()
        self.apply_image_trace()

    def load_image(self):
        """
        Load image from dataset.

        inputs:
        - self.current_id

        outputs:
        - self.current_img
        """
        # The result must be:
        # - Pillow object
        # - resized with BILINEAR resampling
        # - black-and-white but in RGB format
        # - 8 bits / channel
        self.current_img = self.ds_hf[self.current_id]["original_image"]

    def make_prediction(self):
        """
        Use the model to generate a mask prediction.

        inputs:
        - self.model
        - self.img_for_model

        outputs:
        - self.prediction_frame
        """
        # For np.array() shapes, this widget uses the shape you get
        # when converting from PIL, which is (512, 512, 3).
        # The model needs (3, 512, 512).
        # Shapes need to change before images are sent to the model.
        image = np.moveaxis(self.img_for_model, -1, 0).copy()
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs.to("cuda:0")
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Interpolate with BILINEAR even though this is a mask.
        # Masks pixels need to be 0 or 1.
        # It will be rounded to (0, 1) when converting to integer below.
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        if self.prediction_type == "segmentation":
            # segmentation = for each pixel, pick the index of the max value of all categories
            # i.e.: which category has the max value for this pixel?
            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_seg_cpu = pred_seg.detach().cpu()
            # prediction shape is (512, 512)
            self.prediction_frame = np.array(pred_seg_cpu, dtype=np.uint8)
        elif self.prediction_type == "heat map":
            # heat map = just copy the values from the lesion category
            # and scale them to uint8
            lesion_layer = upsampled_logits[0, self.label2id["lesion"], :, :]
            ll_min = lesion_layer.min()
            ll_max = lesion_layer.max()
            ll_heat_map = (
                torch.round(
                    255
                    * torch.pow(
                        (lesion_layer - ll_min) / (ll_max - ll_min), self.heat_map_gamma
                    )
                )
                .detach()
                .cpu()
            )
            self.prediction_frame = np.array(ll_heat_map, dtype=np.uint8)

    def on_click_figure(self, trace, points, state):
        """
        React to mouse click events. Add click coordinates to list.
        """
        x, y = points.point_inds[0]
        self.clicks.append({"coord": (x, y), "type": self.click_type})
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def initialize_figure(self):
        self.image_fig = go.FigureWidget()
        self.image_fig.update_xaxes(visible=False)
        self.image_fig.update_yaxes(visible=False)
        self.image_fig.update_layout(
            autosize=True,
            width=self.img_size[0],
            height=self.img_size[1],
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
        )
        self.load_image()
        self.gen_img_trace()
        self.image_fig.add_trace(self.img_trace)
        self.image_fig.data[0].on_click(self.on_click_figure)

    def callback_image_selector(self, change):
        self.current_id = change["new"]
        self.output_selector.value = self.output_options[0]
        self.click_selector.value = self.click_selector.default_value
        self.prediction_display_toggle.value = True
        # output type and gamma are not reset when the image is changed
        # self.prediction_selector.value = self.prediction_selector.default_value
        # self.heat_map_gamma = self.heat_map_gamma_slider.default_value
        self.load_image()
        self.clicks = []
        self.brightness_slider.value = 1.0
        self.contrast_slider.value = 1.0
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_image_selector(self):
        self.dropdown = Dropdown(
            description="Image File:", options=self.image_select_tuples
        )
        self.dropdown.observe(self.callback_image_selector, names="value")

    def callback_output_selector(self, change):
        self.what_to_show = change["new"]
        self.apply_image_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_output_selector(self):
        self.output_selector = Dropdown(
            description="Show:", options=self.output_options
        )
        self.output_selector.observe(self.callback_output_selector, names="value")

    def callback_prediction_type_selector(self, change):
        self.prediction_type = change["new"]
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_prediction_type_selector(self):
        self.prediction_selector = RadioButtons(
            description="Output:", options=["segmentation", "heat map"]
        )
        self.prediction_selector.default_value = "segmentation"
        self.prediction_type = self.prediction_selector.default_value
        self.prediction_selector.observe(
            self.callback_prediction_type_selector, names="value"
        )

    def callback_click_type_selector(self, change):
        self.click_type = change["new"]

    def build_click_type_selector(self):
        self.click_selector = RadioButtons(
            description="Click Type:", options=["positive", "negative"]
        )
        self.click_selector.default_value = "positive"
        self.click_selector.observe(self.callback_click_type_selector, names="value")

    def callback_clear_clicks_button(self, button):
        self.click_selector.value = self.click_selector.default_value
        self.clicks = []
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_clear_clicks_button(self):
        self.clear_clicks_button = Button(description="Clear All Clicks")
        self.clear_clicks_button.on_click(self.callback_clear_clicks_button)

    def callback_clear_last_click_button(self, button):
        self.clicks = self.clicks[:-1]
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_clear_last_click_button(self):
        self.clear_last_click_button = Button(description="Clear Last Click")
        self.clear_last_click_button.on_click(self.callback_clear_last_click_button)

    def callback_brightness_slider(self, change):
        self.brightness = change["new"]
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_brightness_slider(self):
        self.brightness_slider = FloatLogSlider(
            description="Brightness",
            value=self.brightness,
            base=10,
            min=-1,
            max=1,
            step=0.05,
            orientation="vertical",
        )
        self.brightness_slider.observe(self.callback_brightness_slider, names="value")

    def callback_contrast_slider(self, change):
        self.contrast = change["new"]
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_contrast_slider(self):
        self.contrast_slider = FloatLogSlider(
            description="Contrast",
            value=self.contrast,
            base=10,
            min=-1,
            max=1,
            step=0.05,
            orientation="vertical",
        )
        self.contrast_slider.observe(self.callback_contrast_slider, names="value")

    def callback_reset_enhancements_button(self, button):
        self.brightness_slider.value = 1.0
        self.contrast_slider.value = 1.0
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_reset_enhancements_button(self):
        self.reset_enhancements_button = Button(description="Reset Bright. / Contr.")
        self.reset_enhancements_button.on_click(self.callback_reset_enhancements_button)

    def callback_model_fold_selector(self, change):
        self.model_folder_current = change["new"]
        self.load_model()
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_model_fold_selector(self):
        self.model_fold_selector = Dropdown(
            description="Fold:", options=self.model_folders
        )
        self.model_fold_selector.observe(
            self.callback_model_fold_selector, names="value"
        )

    def callback_model_type_selector(self, change):
        self.model_type_current = change["new"]
        self.get_model_folds()
        self.model_fold_selector.value = self.model_folder_current
        self.load_model()
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_model_type_selector(self):
        model_type_labels = [
            d.split(self.model_prefix)[1] for d in self.model_type_list
        ]
        model_type_options = list(zip(model_type_labels, self.model_type_list))
        self.model_type_selector = Dropdown(
            description="Model:", options=model_type_options
        )
        self.model_type_selector.observe(
            self.callback_model_type_selector, names="value"
        )

    def callback_heat_map_gamma_slider(self, change):
        self.heat_map_gamma = change["new"]
        self.gen_img_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_heat_map_gamma_slider(self):
        self.heat_map_gamma_slider = FloatSlider(
            description="h.m. gamma",
            value=self.heat_map_gamma,
            min=1,
            max=5,
            step=0.1,
            orientation="horizontal",
        )
        self.heat_map_gamma_slider.observe(
            self.callback_heat_map_gamma_slider, names="value"
        )

    def callback_prediction_display_toggle(self, change):
        self.display_prediction = self.prediction_display_toggle.value
        self.apply_image_trace()
        self.image_fig.data[0].source = self.img_trace.source

    def build_prediction_display_toggle(self):
        self.prediction_display_toggle = Checkbox(
            description="Show Prediction", value=True
        )
        self.prediction_display_toggle.observe(self.callback_prediction_display_toggle)

    def initialize_widget(self):
        self.load_model()
        self.initialize_figure()
        self.build_image_selector()
        self.build_output_selector()
        # margin="top/right/bottom/left"
        image_controls_widget = HBox(
            [
                self.dropdown,
                self.output_selector,
            ],
            layout=Layout(margin="10px 0 0 0"),
        )
        self.build_click_type_selector()
        self.build_clear_clicks_button()
        self.build_clear_last_click_button()
        click_controls_widget = HBox(
            [
                self.click_selector,
                self.clear_clicks_button,
                self.clear_last_click_button,
            ],
            layout=Layout(margin="10px 0 0 0"),
        )
        self.build_brightness_slider()
        self.build_contrast_slider()
        sliders_widget = HBox(
            [
                self.brightness_slider,
                self.contrast_slider,
            ]
        )
        self.build_reset_enhancements_button()
        bc_widget = VBox(
            [
                sliders_widget,
                self.reset_enhancements_button,
            ],
            layout=Layout(margin="0 0 0 10px"),
        )
        figure_widget = HBox(
            [
                self.image_fig,
                bc_widget,
            ]
        )
        self.build_model_type_selector()
        self.build_model_fold_selector()
        self.build_prediction_display_toggle()
        self.build_prediction_type_selector()
        self.build_heat_map_gamma_slider()
        model_fold_selector_widget = HBox(
            [
                self.model_type_selector,
                self.model_fold_selector,
                self.prediction_display_toggle,
            ],
            layout=Layout(margin="10px 0 0 0"),
        )
        prediction_widget = HBox(
            [
                self.prediction_selector,
                self.heat_map_gamma_slider,
            ],
            layout=Layout(margin="10px 0 0 0"),
        )
        self.widget = VBox(
            [
                image_controls_widget,
                click_controls_widget,
                figure_widget,
                model_fold_selector_widget,
                prediction_widget,
            ],
            layout=Layout(margin="0 0 20px 20px"),
        )

    def display(self):
        display(self.widget)
