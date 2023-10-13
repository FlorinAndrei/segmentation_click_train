import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mean_dice")
launch_gradio_widget(module)
