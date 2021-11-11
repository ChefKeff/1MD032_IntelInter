from typing import Any, Callable, Dict, List

class EventBroker:
    listeners:Dict[str, List[Callable[[Any],None]]] = dict()

    @classmethod
    def raise_event(cls, event_name:str, value:Any) -> None:
        for callback in cls.listeners[event_name]:
            callback(value)

    @classmethod
    def register_event(cls, event_name:str) -> None:
        if event_name not in cls.listeners:
            cls.listeners[event_name] = list()

        if event_name+"_post" not in cls.listeners:
            cls.listeners[event_name+"_post"] = list()

    @classmethod
    def register_callback(cls, event_name:str, callback:Callable[[Any], None]) -> None:
        if callback in cls.listeners[event_name]:
            raise ValueError(f"Callback `{callback}` is already registered for event `{event_name}`.")

        cls.listeners[event_name].append(callback)
