from .component import Component
import ipywidgets as widgets
from ..state import AppState
from ..event_broker import EventBroker


class Header(Component):
    def __init__(self, app_state: AppState) -> None:
        EventBroker.register_event("condition_changed")

        self.state = app_state
        self.file_selector = widgets.Dropdown(
            options=self.state.conditions.keys(),
            value=self.state.current_condition,
            description="Video:",
        )
        self.file_selector.observe(self.file_selector_callback, names=["value"])

    def build_layout(self) -> widgets.Widget:
        save_button = widgets.Button(
            description="Save CSV",
            button_style='success',
            icon='save',
        )
        save_button.on_click(self.save_button_callback)

        return widgets.HBox(children=[save_button, self.file_selector])

    def file_selector_callback(self, value):
        self.state.current_condition = self.file_selector.value
        EventBroker.raise_event("condition_changed", self.file_selector.value)

    def save_button_callback(self, value):
        df = self.state.to_dataframe()
        df.to_csv(f"{self.state.csv_name}.csv", index_label="ID")
