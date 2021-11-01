from pathlib import Path

import cv2
import numpy as np
from imageio import v3 as iio

# For Visualization in Jupyter
import ipywidgets as widgets
from matplotlib import pyplot as plt
from IPython.display import display, Image, Video

# Get images and video into Jupyter from your webcam
from ipywebrtc import CameraStream, ImageRecorder, VideoRecorder

camera = CameraStream(constraints=
                      {"facing_mode": "user",
                       "audio": False,
                       "video": { "width": 640, "height": 480 }
                       })
camera

recorder = ImageRecorder(stream=camera)
recorder
    
