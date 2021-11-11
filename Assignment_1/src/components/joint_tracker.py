from .component import Component
from ..event_broker import EventBroker
import ipywidgets as widgets
from .video_display import VideoDisplay
import numpy as np
from ..state import AppState
from imageio import v3 as iio


def tableau_string_to_float(rgb_string: str):
    return np.fromstring(rgb_string, dtype=int, sep=".") / 255


class JointTrackerComponent(Component):
    def __init__(self, app_state: AppState) -> None:
        EventBroker.register_event("joint_tracker_canvas_mouse_down")
        EventBroker.register_event("joint_tracker_canvas_mouse_up")
        EventBroker.register_event("joint_tracker_canvas_mouse_drag")
        EventBroker.register_event("condition_changed")
        EventBroker.register_event("frame_idx_changed")
        EventBroker.register_event("joint_changed")
        EventBroker.register_event("jump_frame")

        self.state = app_state
        self.step_joint = False
        condition_state = self.state.conditions[self.state.current_condition]
        self.ignore_conditions = ["hand_position"]

        self.primary_color = np.array([31, 119, 180, 255], dtype=float) / 255
        self.accent_color = np.array([214, 39, 40, 255], dtype=float) / 255
        self.gray_color = np.array([96, 99, 106, 255], dtype=float) / 255

        self.display = VideoDisplay(
            self.state.current_frame, event_prefix="joint_tracker_"
        )
        self.joint_display = VideoDisplay(
            iio.imread("assets/base.png"),
            event_prefix="joint_display_",
            figsize=(2.8, 3.5),
            ignore_frame_change=True,
        )
        self.joint_display_marker = self.joint_display.ax.scatter(
            [], [], s=300, color=self.accent_color, edgecolors=(1, 1, 1)
        )

        self.current_landmarks = self.display.ax.scatter(
            [], [], zorder=3, edgecolors=(1, 1, 1), linewidths=1
        )
        self.current_trajectory = self.display.ax.plot(
            [], [], zorder=2, color=self.gray_color, marker="o"
        )[0]

        self.joint_selector = widgets.Dropdown(
            options=[
                x
                for x in condition_state.tracks.keys()
                if x not in self.ignore_conditions
            ],
            value=condition_state.current_track,
            description="Joint:",
            layout=widgets.Layout(width="95%"),
        )
        self.joint_selector.observe(self.joint_selector_callback, names=["value"])

        self.bone_lines = {
            "root": [
                (
                    "thumb_1",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("31.119.180")
                    )[0],
                ),
                (
                    "index_1",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("255.127.14")
                    )[0],
                ),
                (
                    "middle_1",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("44.160.44")
                    )[0],
                ),
                (
                    "ring_1",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("214.39.40")
                    )[0],
                ),
                (
                    "pinky_1",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("148.103.189")
                    )[0],
                ),
            ],
            "thumb_1": [
                (
                    "thumb_2",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("31.119.180")
                    )[0],
                )
            ],
            "thumb_2": [
                (
                    "thumb_3",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("31.119.180")
                    )[0],
                )
            ],
            "thumb_3": [],
            "index_1": [
                (
                    "index_2",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("255.127.14")
                    )[0],
                )
            ],
            "index_2": [
                (
                    "index_3",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("255.127.14")
                    )[0],
                )
            ],
            "index_3": [
                (
                    "index_4",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("255.127.14")
                    )[0],
                )
            ],
            "index_4": [],
            "middle_1": [
                (
                    "middle_2",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("44.160.44")
                    )[0],
                )
            ],
            "middle_2": [
                (
                    "middle_3",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("44.160.44")
                    )[0],
                )
            ],
            "middle_3": [
                (
                    "middle_4",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("44.160.44")
                    )[0],
                )
            ],
            "middle_4": [],
            "ring_1": [
                (
                    "ring_2",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("214.39.40")
                    )[0],
                )
            ],
            "ring_2": [
                (
                    "ring_3",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("214.39.40")
                    )[0],
                )
            ],
            "ring_3": [
                (
                    "ring_4",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("214.39.40")
                    )[0],
                )
            ],
            "ring_4": [],
            "pinky_1": [
                (
                    "pinky_2",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("148.103.189")
                    )[0],
                )
            ],
            "pinky_2": [
                (
                    "pinky_3",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("148.103.189")
                    )[0],
                )
            ],
            "pinky_3": [
                (
                    "pinky_4",
                    self.display.ax.plot(
                        [], [], color=tableau_string_to_float("148.103.189")
                    )[0],
                )
            ],
            "pinky_4": [],
        }

        # dirty haxz
        self.joint_display_marker_positions = {
            "root": np.array([482, 920]),
            "thumb_1": np.array([227, 761]),
            "thumb_2": np.array([124, 605]),
            "thumb_3": np.array([46, 499]),
            "index_1": np.array([337, 445]),
            "index_2": np.array([329, 332]),
            "index_3": np.array([326, 194]),
            "index_4": np.array([315, 84]),
            "middle_1": np.array([493, 456]),
            "middle_2": np.array([493, 300]),
            "middle_3": np.array([493, 166]),
            "middle_4": np.array([493, 45]),
            "ring_1": np.array([627, 477]),
            "ring_2": np.array([652, 329]),
            "ring_3": np.array([670, 212]),
            "ring_4": np.array([684, 105]),
            "pinky_1": np.array([751, 531]),
            "pinky_2": np.array([790, 414]),
            "pinky_3": np.array([811, 329]),
            "pinky_4": np.array([843, 247]),
        }

        EventBroker.register_callback("condition_changed", self.on_condition_changed)
        EventBroker.register_callback(
            "joint_tracker_canvas_mouse_down", self.on_mouse_down
        )
        EventBroker.register_callback("joint_tracker_canvas_mouse_up", self.on_mouse_up)
        EventBroker.register_callback(
            "joint_tracker_canvas_mouse_drag", self.on_mouse_drag
        )
        EventBroker.register_callback("frame_idx_changed", self.on_frame_idx_changed)
        EventBroker.register_callback(
            "joint_display_canvas_mouse_down", self.on_joint_display_mouse_down
        )

    def build_layout(self) -> widgets.Widget:
        previous_keyframe = widgets.Button(
            description="Keyframe",
            button_style="primary",
            icon="arrow-left",
            layout=widgets.Layout(width="33%"),
        )
        previous_keyframe.on_click(self.previous_keyframe_callback)
        delete_keyframe = widgets.Button(
            description="Keyframe",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="32%"),
        )
        delete_keyframe.on_click(self.remove_keyframe_callback)
        next_keyframe = widgets.Button(
            description="Keyframe",
            button_style="primary",
            icon="arrow-right",
            layout=widgets.Layout(width="33%"),
        )
        next_keyframe.on_click(self.next_keyframe_callback)
        keyframe_controls = widgets.HBox(
            children=[previous_keyframe, delete_keyframe, next_keyframe]
        )

        reset_track = widgets.Button(
            description="Reset Joint",
            button_style="danger",
            icon="cross",
            layout=widgets.Layout(width="98%"),
        )
        reset_track.on_click(self.reset_track_callback)

        forward = widgets.Button(
            button_style="primary",
            icon="arrow-right",
        )
        forward.on_click(self.forward_callback)
        backward = widgets.Button(
            button_style="primary",
            icon="arrow-left",
        )
        backward.on_click(self.backward_callback)
        joint_controls = widgets.HBox(children=[backward, forward])

        menu_container = widgets.VBox(
            children=[
                keyframe_controls,
                reset_track,
                self.joint_selector,
                self.joint_display.build_layout(),
                joint_controls,
            ]
        )
        tracker_view = widgets.HBox(
            children=[self.display.build_layout(), menu_container]
        )
        return tracker_view

    def on_condition_changed(self, new_condition):
        condition_state = self.state.current_condition_state
        self.joint_selector.options = [
            x for x in condition_state.tracks.keys() if x not in self.ignore_conditions
        ]
        self.joint_selector.value = condition_state.current_track

    def joint_selector_callback(self, value):
        condition_state = self.state.current_condition_state
        new_joint = self.joint_selector.value
        condition_state.current_track = new_joint

        self.update_axes()
        EventBroker.raise_event("joint_changed", new_joint)

    def forward_callback(self, value):
        current_joint = self.joint_selector.value
        options = list(self.joint_selector.options)

        if current_joint is None:
            new_joint = options[0]
        else:
            current_idx = options.index(current_joint)
            new_joint = options[(current_idx + 1) % len(options)]

        self.joint_selector.value = new_joint
        self.joint_selector_callback(None)

    def backward_callback(self, value):
        current_joint = self.joint_selector.value
        options = list(self.joint_selector.options)

        if current_joint is None:
            new_joint = options[0]
        else:
            current_idx = options.index(current_joint)
            new_joint = options[(current_idx - 1) % len(options)]

        self.joint_selector.value = new_joint
        self.joint_selector_callback(None)

    def reset_track_callback(self, value):
        self.current_tracker.reset()
        self.update_axes()

    def on_frame_idx_changed(self, value):
        self.update_axes()

    def on_mouse_down(self, data):
        point = np.asarray((data.xdata, data.ydata))

        for track, marker_pos in self.current_tracks.items():
            if np.linalg.norm(marker_pos - point, axis=-1) < 12:
                self.joint_selector.value = track
                break
        else:
            if self.current_tracker is not None:
                self.step_joint = True
                self.current_tracker.add_annotation(self.state.current_frame_idx, point)
            else:
                self.display.text.set_text("No Joint Selected.")

        self.update_axes()

    def on_mouse_up(self, data):
        point = np.asarray((data.xdata, data.ydata))

        if self.step_joint:
            self.step_joint = False
            current_joint = self.joint_selector.value
            options = list(self.joint_selector.options)

            if current_joint is None:
                new_joint = options[0]
            else:
                current_idx = options.index(current_joint)
                new_joint = options[(current_idx + 1) % len(options)]

            self.joint_selector.value = new_joint

    def on_mouse_drag(self, data):
        point = np.asarray((data.xdata, data.ydata))

        if self.current_tracker is not None:
            self.current_tracker.add_annotation(self.state.current_frame_idx, point)
        else:
            self.display.text.set_text("No Joint Selected.")

        self.update_axes()
        # self.display.fig.canvas.draw()

    def on_joint_display_mouse_down(self, data):
        point = np.asarray((data.xdata, data.ydata))

        for joint, position in self.joint_display_marker_positions.items():
            if np.linalg.norm(point - position, axis=-1) < 30:
                self.joint_selector.value = joint

    def next_keyframe_callback(self, value):
        idx = self.state.current_frame_idx

        distance = float("inf")
        next_keyframe_idx = idx
        for keyframe_idx in self.current_tracker.annotations.keys():
            if keyframe_idx - idx > 0 and abs(keyframe_idx - idx) < distance:
                next_keyframe_idx = keyframe_idx
                distance = abs(keyframe_idx - idx)

        EventBroker.raise_event("jump_frame", next_keyframe_idx)

    def previous_keyframe_callback(self, value):
        idx = self.state.current_frame_idx

        distance = float("inf")
        next_keyframe_idx = idx
        for keyframe_idx in self.current_tracker.annotations.keys():
            if keyframe_idx - idx < 0 and abs(keyframe_idx - idx) < distance:
                next_keyframe_idx = keyframe_idx
                distance = abs(keyframe_idx - idx)

        EventBroker.raise_event("jump_frame", next_keyframe_idx)

    def remove_keyframe_callback(self, value):
        idx = self.state.current_frame_idx

        if idx not in self.current_tracker.annotations.keys():
            self.display.text.set_text("Currently not on a Keyframe.")
            return

        self.current_tracker.remove_annotation(idx)
        self.update_axes()

    def update_axes(self):
        # update selected joint trajectory
        if self.joint_selector.value is None:
            self.current_trajectory.set_data([], [])
        else:
            annotations = self.current_tracker.annotations
            timestamps = sorted(list(annotations.keys()))

            if len(timestamps) == 0:
                self.current_trajectory.set_data([], [])
            elif len(timestamps) == 1:
                pass
            else:
                points = np.stack([annotations[x] for x in timestamps])
                if len(points) > 1:
                    self.current_trajectory.set_data(points[:, 0], points[:, 1])

        # update frame annotations
        markers = self.current_tracks

        colors = list()
        joints = list()
        sizes = list()
        for start_joint in self.joint_selector.options:
            bone_start = markers[start_joint]

            for end_joint, line in self.bone_lines[start_joint]:
                bone_end = markers[end_joint]
                if np.allclose(bone_start, 0):
                    line.set_data([], [])
                elif np.allclose(bone_end, 0):
                    line.set_data([], [])
                else:
                    y_values = (bone_start[0], bone_end[0])
                    x_values = (bone_start[1], bone_end[1])
                    line.set_data(y_values, x_values)

            if np.allclose(bone_start, 0):
                continue

            joints.append(bone_start)
            if start_joint == self.joint_selector.value:
                colors.append(self.accent_color)
                sizes.append(100)
            else:
                colors.append(self.primary_color)
                sizes.append(50)

        if len(joints) > 0:
            colors = np.stack(colors)
            joints = np.stack(joints)
            sizes = np.stack(sizes)
        else:
            colors = []
            joints = []
            sizes = []

        self.current_landmarks.set_offsets(joints)
        self.current_landmarks.set_sizes(sizes)
        self.current_landmarks.set_facecolors(colors)

        # update the joint display (on the right)
        if self.joint_selector.value is None:
            self.joint_display_marker.set_offsets([])
        else:
            current_joint = self.joint_selector.value
            marker_position = self.joint_display_marker_positions[current_joint]
            self.joint_display_marker.set_offsets(marker_position)

    @property
    def current_tracker(self):
        condition_state = self.state.current_condition_state

        if condition_state.current_track is None:
            return None
        else:
            return condition_state.tracks[condition_state.current_track]

    @property
    def current_tracks(self):
        return {
            name: tracker.trajectory[self.state.current_frame_idx]
            for name, tracker in self.state.current_condition_state.tracks.items()
        }
