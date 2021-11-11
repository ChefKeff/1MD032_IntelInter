from imageio import v3 as iio
import numpy as np
from typing import Dict, List
import cv2
import pandas as pd

from .skbot_replacement import linear_trajectory

class PositionTracker:
    def __init__(self, num_frames: int) -> None:
        self.num_frames = num_frames
        self.tracked_position = np.zeros((num_frames, 2), dtype=float)
        self.annotations: Dict[int, np.ndarray] = dict()

        self._trajectory = np.zeros((num_frames, 2), dtype=float)

    def add_annotation(self, frame:int, position:np.ndarray):
        self.annotations[frame] = position
        self._update_trajectory()

    def remove_annotation(self, frame:int):
        if frame not in self.annotations:
            return

        self.annotations.pop(frame)
        self._update_trajectory()

    def reset(self):
        self.tracked_position = np.zeros((self.num_frames, 2), dtype=float)
        self.annotations = dict()
        self._update_trajectory()

    def track_forward(self, video: np.ndarray, roi: np.ndarray, start_frame: int):
        """Forward-Track the center of an object in a video

        This function tracks an object with an initially known location in a
        video forward in time. The object is given by ``roi``, which is a
        pixel-level mask used to compute the object's histogram fingerprint.
        This fingerprint is used to compute the back-projection of each
        subsequent frame. The next object position is then computed via
        mean-shift on the back-projected image starting at the previous frame's
        poisition.

        Parameters
        ----------
        video : np.ndarray
            The video in which to track the given ROI. Format is ``(frames,
            height, width, channel)``
        roi : np.ndarray
            A pixel level mask of the object to track. This is a boolean array
            of shape ``(height, width)`` where ``True`` indicates that the pixel
            belongs to the object, and ``False`` indicates that the pixel does
            not belong to the object.
        start_frame : int
            The frame index in which the ROI is specified

        """

        self._update_tracker(video, roi, np.arange(start_frame, len(video)))

    def track_backward(self, video: np.ndarray, roi: np.ndarray, start_frame: int):
        """Backward-Track the center of an object in a video

        This function tracks an object with an initially known location in a
        video backwards in time. The object is given by ``roi``, which is a
        pixel-level mask used to compute the object's histogram fingerprint.
        This fingerprint is used to compute the back-projection of each
        previous frame. The previous object position is then computed via
        mean-shift on the back-projected image starting at the previous frame's
        old position.

        Parameters
        ----------
        video : np.ndarray
            The video in which to track the given ROI. Format is ``(frames,
            height, width, channel)``
        roi : np.ndarray
            A pixel level mask of the object to track. This is a boolean array
            of shape ``(height, width)`` where ``True`` indicates that the pixel
            belongs to the object, and ``False`` indicates that the pixel does
            not belong to the object.
        start_frame : int
            The frame index in which the ROI is specified

        """

        self._update_tracker(video, roi, np.arange(start_frame, -1, -1))

    @property
    def annotations_relative(self) -> np.ndarray:
        return {
            idx: (val - self.tracked_position[idx])
            for idx, val in self.annotations.items()
        }

    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory

    def get_back_projection(self, frame, roi):
        mask = roi.astype(np.uint8) * 255
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        fingerprint = cv2.calcHist(
            [hsv_frame[..., 0]], [0], mask, [100], [0, 255], accumulate=False
        )
        cv2.normalize(fingerprint, fingerprint, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return cv2.calcBackProject([hsv_frame[..., 0]], [0], fingerprint, [0, 255], scale=1)

    def _update_trajectory(self):
        ordered_frames = np.sort(list(self.annotations_relative.keys()))
        control_points = np.empty((len(ordered_frames), 2), dtype=float)
        for idx, key in enumerate(ordered_frames):
            control_points[idx] = self.annotations_relative[key]

        if len(control_points) == 0:
            ordered_frames = np.array([0])
            control_points = np.array(((0,0),), dtype=float)

        if 0 not in ordered_frames:
            ordered_frames = np.insert(ordered_frames, 0, 0, axis=0)
            control_points = np.insert(control_points, 0, control_points[0], axis=0)

        if self.num_frames - 1 not in ordered_frames:
            ordered_frames = np.insert(ordered_frames, -1, self.num_frames - 1, axis=0)
            control_points = np.insert(control_points, -1, control_points[-1], axis=0)

        control_trajectory = linear_trajectory(
            np.arange(self.num_frames), control_points, t_control=ordered_frames
        )

        self._trajectory = self.tracked_position + control_trajectory


    def _update_tracker(self, video:np.ndarray, roi:np.ndarray, frames_to_update:np.ndarray):
        frame_shape = video.shape[1:3]
        mask = roi.astype(np.uint8) * 255
        current_track = self.trajectory[frames_to_update]
        _, _, w, h = cv2.boundingRect(mask)

        hsv_frames = np.stack(
            [cv2.cvtColor(x, cv2.COLOR_RGB2LAB) for x in video[frames_to_update]]
        )
        initial_frame = hsv_frames[0]

        fingerprint = cv2.calcHist(
            [initial_frame[..., 0]], [0], mask, [100], [0, 255], accumulate=False
        )
        cv2.normalize(fingerprint, fingerprint, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        back_projections = np.stack(
            [
                cv2.calcBackProject([x[..., 0]], [0], fingerprint, [0, 255], scale=1)
                for x in hsv_frames
            ]
        )

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)
        half_extent = np.array([h/2, w/2])
        for idx in range(frames_to_update.size):
            frame_idx = frames_to_update[idx]
            back_projection = back_projections[idx]
            center = current_track[idx]
            initial_position = (*(center - half_extent),*(center + half_extent))
            initial_position = np.clip(initial_position, 0, [*frame_shape]*2)
            ret, position = cv2.meanShift(
                back_projection, initial_position.astype(int), term_crit
            )
            x, y, _, _ = position
            self.tracked_position[frame_idx] = (y, x)

        self._update_trajectory()


class ConditionState:
    def __init__(self, name: str, *, video_format=".mp4") -> None:
        self.name = name
        self.raw_video = iio.imread(name + video_format)
        self.tracks:Dict[str, PositionTracker] = dict()
        self.current_track = None
        self.current_frame_idx = 0

    def to_dataframe(self) -> pd.DataFrame:
        track_dfs = [pd.DataFrame.from_records({
            "joint": [name] * len(self.raw_video),
            "frame": np.arange(len(self.raw_video)),
            "y": track.trajectory[:, 0],
            "x": track.trajectory[:, 1],
        }) for name, track in self.tracks.items()]

        return pd.concat(track_dfs)


class AppState:
    def __init__(self, conditions:List[str]) -> None:
        self.conditions = {condition: ConditionState(condition) for condition in conditions}
        self.current_condition = conditions[0]
        self.csv_name = "labels"

    def add_landmark(self, landmark_name):
        for condition in self.conditions.values():
            condition.tracks[landmark_name] = PositionTracker(len(condition.raw_video))

    def to_dataframe(self) -> pd.DataFrame:
        condition_dfs = list()
        for name, condition in self.conditions.items():
            df = condition.to_dataframe()
            df.insert(0, "gesture", name)
            condition_dfs.append(df)

        return pd.concat(condition_dfs, ignore_index=True)

    @property
    def current_video(self) -> np.ndarray:
        return self.current_condition_state.raw_video

    @property
    def current_frame(self) -> np.ndarray:
        return self.current_video[self.current_frame_idx]

    @property
    def current_condition_state(self):
        return self.conditions[self.current_condition]

    @property
    def current_frame_idx(self):
        return self.current_condition_state.current_frame_idx

    @current_frame_idx.setter
    def current_frame_idx(self, new_idx):
        self.current_condition_state.current_frame_idx = new_idx
