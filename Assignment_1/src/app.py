import ipywidgets as widgets
from .landmark_names import landmark_names
from .components.hand_tracker import HandTrackerComponent
from .components.joint_tracker import JointTrackerComponent
from .state import AppState
from .components.component import Component
from .components.footer import Footer
from .components.header import Header
from .event_broker import EventBroker
from typing import List


class App(Component):
    def __init__(self, conditions: List[str]) -> None:
        EventBroker.register_event("condition_changed")
        EventBroker.register_event("frame_idx_changed")
        EventBroker.register_event("frame_changed")

        self.state = AppState(conditions)

        self.state.add_landmark("hand_position")
        for landmark in landmark_names:
            self.state.add_landmark(landmark)

        EventBroker.register_callback("frame_idx_changed", self.on_frame_idx_changed)
        EventBroker.register_callback("condition_changed", self.on_condition_changed)

    def build_layout(self) -> widgets.Widget:
        header = Header(self.state).build_layout()

        content = widgets.Tab(
            children=[
                # HandTrackerComponent(self.state).build_layout(),
                JointTrackerComponent(self.state).build_layout()
            ]
        )
        # content.set_title(0, "Hand Tracking")
        content.set_title(0, "Joint Landmarks")

        footer = Footer(self.state).build_layout()

        return widgets.VBox(children=[header, content, footer])

    def on_frame_idx_changed(self, new_idx):
        EventBroker.raise_event("frame_changed", self.state.current_frame)

    def on_condition_changed(self, new_condition):
        EventBroker.raise_event("frame_changed", self.state.current_frame)


def AnnotationApp(conditions, out_file_name):
    app = App(conditions)
    app.state.csv_name = out_file_name
    return app.build_layout()
