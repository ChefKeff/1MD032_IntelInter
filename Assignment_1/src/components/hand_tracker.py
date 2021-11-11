from .component import Component
from ..event_broker import EventBroker
import ipywidgets as widgets
from .video_display import VideoDisplay
import numpy as np
from ..skbot_replacement import angle_between
from matplotlib.path import Path as MplPath
from ..state import AppState, PositionTracker


class HandTrackerComponent(Component):
    def __init__(self, app_state:AppState) -> None:
        EventBroker.register_event("hand_tracker_canvas_mouse_down")
        EventBroker.register_event("hand_tracker_canvas_mouse_up")
        EventBroker.register_event("hand_tracker_canvas_mouse_drag")
        EventBroker.register_event("frame_idx_changed")
        EventBroker.register_event("condition_changed")

        self.state = app_state

        self.primary_color = np.array([[31, 119, 180, 255]], dtype=float) / 255
        self.accent_color = np.array([214, 39, 40, 255], dtype=float) / 255

        self.display = VideoDisplay(self.state.current_frame, event_prefix="hand_tracker_")
        self.roi_poly = self.display.ax.fill(
            [],
            [],
            self.primary_color,
            alpha=0.2,
            zorder=1,
            edgecolor=self.primary_color,
            linewidth=3,
        )[0]
        self.scatter_collection = self.display.ax.scatter([], [], zorder=2)
        self.selected_idx = None
        self.last_selected_idx = None
        self.selected_region = False
        self.region_ancor = None

        self.hand_position = np.zeros((len(self.state.current_video), 2), dtype=float)
        
        EventBroker.register_callback("hand_tracker_canvas_mouse_down", self.on_mouse_down)
        EventBroker.register_callback("hand_tracker_canvas_mouse_up", self.on_mouse_up)
        EventBroker.register_callback("hand_tracker_canvas_mouse_drag", self.on_mouse_drag)
        EventBroker.register_callback("frame_idx_changed", self.on_frame_idx_changed)

    def build_layout(self) -> widgets.Widget:
        delete_point = widgets.Button(
            button_style='danger',
            icon='trash',
        )
        delete_point.on_click(self.on_delete_point)

        track_backward = widgets.Button(
            description= "Track Backwards",
            icon='fast-backward',
        )
        track_backward.on_click(self.on_track_backward)

        track_forward = widgets.Button(
            description="Track Forward",
            icon='fast-forward',
        )
        track_forward.on_click(self.on_track_forward)

        tracking_controls = widgets.HBox(children=[track_backward, track_forward])
        menu_container = widgets.VBox(children=[delete_point, tracking_controls])
        tracker_view = widgets.HBox(children=[self.display.build_layout(), menu_container])
        return tracker_view

    def on_mouse_down(self, data):
        point = np.asarray((data.xdata, data.ydata))
        distances = np.linalg.norm(self.points - point, axis=-1)

        # the number is hard-coded until I figure out
        # how to compute on-screen size of a single blob in a scatter
        is_close = distances < 12

        if len(self.points) == 0:
            # insert a vertex; done.
            offsets = np.array(point, ndmin=2)
            self.scatter_collection.set_offsets(offsets)
            self.selected_idx = len(self.points) - 1
        elif np.any(is_close):
            # existing vertex re-selected
            # (allow dragging/deleting vertices)
            self.selected_idx = np.argmax(is_close)
        elif MplPath(self.points[:, [1, 0]]).contains_points(point[None, [1, 0]]):
            # face selected
            # (allow draging polygon face)
            self.selected_idx = None
            self.selected_region = True
            self.region_ancor = self.mask_centroid - point
        else:
            # area outside ROI selected
            # (add a new point to polygon)
            neighbourA = np.argmin(distances)
            insert_idx = (neighbourA + 1) % len(self.points)

            optionA = (neighbourA - 1) % len(self.points)
            optionB = (neighbourA + 1) % len(self.points)
            vec_point = point - self.points[neighbourA]
            vec_A = self.points[optionA] - self.points[neighbourA]
            vec_B = self.points[optionB] - self.points[neighbourA]
            scoreA = angle_between(vec_point, vec_A)
            scoreB = angle_between(vec_point, vec_B)
            if scoreA < scoreB:
                insert_idx = neighbourA

            offsets = np.insert(self.points, insert_idx, point, axis=0)
            self.scatter_collection.set_offsets(offsets)
            self.selected_idx = insert_idx

        self.last_selected_idx = self.selected_idx
        self.update_roi()

    def on_mouse_up(self, data):
        self.display.text.set_text("")
        self.selected_idx = None
        self.selected_region = False
        self.region_ancor = None

        self.current_tracker.annotations[self.state.current_frame_idx] = self.mask_centroid

        self.update_roi()
        
        # back_projection = self.current_tracker.get_back_projection(self.state.current_frame, self.mask)
        # self.display.text.set_text((back_projection.min(), back_projection.max()))
        # self.display.image.set_data(back_projection)
        # self.display.fig.canvas.draw()

    def on_mouse_drag(self, data):
        point = np.asarray((data.xdata, data.ydata))

        if self.selected_idx is not None:
            self.points[self.selected_idx] = point
        
        if self.selected_region:
            target = point + self.region_ancor
            offset = target - self.mask_centroid
            new_positions = self.points + offset[None, :]
            self.scatter_collection.set_offsets(new_positions)
        
        self.update_roi()
        self.display.fig.canvas.draw()

    def on_delete_point(self, data):
        if self.last_selected_idx is None:
            return

        offsets = np.delete(self.points, self.last_selected_idx, axis=0)
        self.scatter_collection.set_offsets(offsets)
        self.last_selected_idx = None
        self.update_roi()

    def on_track_forward(self, data):
        condition = self.state.current_condition
        tracker = self.state.conditions[condition].tracks["hand_position"]
        tracker.track_forward(self.state.current_video, self.mask, self.state.current_frame_idx)

    def on_track_backward(self, data):
        condition = self.state.current_condition
        tracker = self.state.conditions[condition].tracks["hand_position"]
        tracker.track_backward(self.state.current_video, self.mask, self.state.current_frame_idx)

    def on_frame_idx_changed(self, new_frame):
        current_center = self.mask_centroid
        new_center = self.current_tracker.trajectory[new_frame]
        offset = new_center - current_center

        new_positions = self.points + offset[None, :]
        self.scatter_collection.set_offsets(new_positions)

        self.update_roi()

    def update_roi(self):
        sizes = np.full(len(self.points), 100)
        colors = np.repeat(self.primary_color[None, ...], len(self.points), axis=0)

        if self.selected_idx is not None:
            sizes[self.selected_idx] = 150
            colors[self.selected_idx] = self.accent_color

        if self.last_selected_idx is not None:
            colors[self.last_selected_idx] = self.accent_color

        self.roi_poly.set_xy(self.points)
        self.scatter_collection.set_sizes(sizes)
        self.scatter_collection.set_facecolors(colors)

    @property
    def points(self):
        return self.scatter_collection.get_offsets()

    @property
    def mask(self):
        im_shape = self.display.image.get_array().shape[:2]

        flat_idx = np.arange(np.prod(im_shape))
        points = np.stack([flat_idx // im_shape[1], flat_idx % im_shape[1]], axis=-1)
   
        return MplPath(self.points[:, [1, 0]]).contains_points(points).reshape(im_shape)

    @property
    def mask_centroid(self):
        if len(self.points) == 0:
            return np.array((0, 0))

        # source: https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        points = self.points
        cyclic_points = np.concatenate((points, points[0][None, :]), axis=0)
        x, y = cyclic_points[:, 0], cyclic_points[:, 1]

        tmp = x[:-1]*y[1:] - x[1:]*y[:-1]

        area = 0.5*np.sum(tmp)
        c_x = 1/(6*area) * np.sum((x[:-1]+x[1:])*tmp)
        c_y = 1/(6*area) * np.sum((y[:-1]+y[1:])*tmp)

        return np.array((c_x, c_y))

    @property
    def current_tracker(self) -> PositionTracker:
        condition = self.state.current_condition
        tracker = self.state.conditions[condition].tracks["hand_position"]
        return tracker