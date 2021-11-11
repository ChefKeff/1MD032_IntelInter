from typing import Any
import ipywidgets as widgets


class Component:
    def eventCallback(self, event_name:str, value:Any) -> None:
        """Handles Events related to this component
        
            (ignores unrelated events)
            (may modify state of children)
        """
        raise NotImplementedError()

    def build_layout(self) -> widgets.Widget:
        """the GUI layout of this component"""
        raise NotImplementedError()
