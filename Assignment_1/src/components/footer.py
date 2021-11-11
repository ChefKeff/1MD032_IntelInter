from .component import Component
import ipywidgets as widgets
from ..state import AppState
from ..event_broker import EventBroker


class Footer(Component):
    def __init__(self, app_state:AppState) -> None:
        EventBroker.register_event("condition_changed")
        EventBroker.register_event("frame_idx_changed")
        EventBroker.register_event("jump_frame")
        
        self.state = app_state

        self.timeline = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.state.current_video) - 1,
            description="Frames:",
            bar_style="info",
            layout=widgets.Layout(width="99%"),
        )
        self.timeline.observe(self.timeline_callback, names=["value"])

        EventBroker.register_callback("condition_changed", self.on_condition_change)
        EventBroker.register_callback("jump_frame", self.on_jump_frame)

    def on_condition_change(self, new_condition):
        self.timeline.max = len(self.state.current_video) - 1
        self.timeline.value = self.state.current_frame_idx

    def timeline_callback(self, value):
        self.state.current_frame_idx = self.timeline.value
        EventBroker.raise_event("frame_idx_changed", self.timeline.value)

    def on_jump_frame(self, new_idx):
        self.timeline.value = new_idx

    def build_layout(self) -> widgets.Widget:
        return self.timeline
