from .component import Component
from ..event_broker import EventBroker
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets


class VideoDisplay(Component):
    def __init__(self, initial_image, *, event_prefix=None, figsize=(6.4, 4.8), ignore_frame_change=False):
        if event_prefix is None:
            event_prefix = ""
        
        self.eventName = {
            "canvas_mouse_down": f"{event_prefix}canvas_mouse_down",
            "canvas_mouse_up": f"{event_prefix}canvas_mouse_up",
            "canvas_mouse_drag": f"{event_prefix}canvas_mouse_drag",
        }

        EventBroker.register_event("frame_changed")
        EventBroker.register_event(self.eventName["canvas_mouse_down"])
        EventBroker.register_event(self.eventName["canvas_mouse_up"])
        EventBroker.register_event(self.eventName["canvas_mouse_drag"])


        # frame display
        fig, ax = plt.subplots(figsize=figsize)
        self.image = ax.imshow(initial_image)

        self.fig = fig
        self.ax = ax
        self.frame_display = fig.canvas
        self.mouse_down = False


        self.text=ax.text(0,0, "", va="bottom", ha="left")

        cid = self.frame_display.mpl_connect(
            "button_press_event", self.mouse_down_callback
        )
        cid = self.frame_display.mpl_connect(
            "button_release_event", self.mouse_up_callback
        )
        cid = self.frame_display.mpl_connect(
            "motion_notify_event", self.mouse_move_callback
        )

        if not ignore_frame_change:
            EventBroker.register_callback("frame_changed", self.on_frame_changed)

    def build_layout(self) -> widgets.Widget:
        self.ax.axis('off')
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.frame_display.toolbar_visible = False
        self.frame_display.header_visible = False
        self.frame_display.footer_visible = False
        self.frame_display.resizable = False
        
        self.fig.tight_layout()
        
        return self.frame_display

    def on_frame_changed(self, value: np.ndarray):
        self.image.set_data(value)

    def mouse_down_callback(self, data):
        EventBroker.raise_event(self.eventName["canvas_mouse_down"], data)
        self.mouse_down = True

    def mouse_up_callback(self, data):
        EventBroker.raise_event(self.eventName["canvas_mouse_up"], data)
        self.mouse_down = False

    def mouse_move_callback(self, data):
        if not self.mouse_down:
            return

        EventBroker.raise_event(self.eventName["canvas_mouse_drag"], data)
