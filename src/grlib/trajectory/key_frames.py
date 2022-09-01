from typing import List

import numpy as np

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe


def remove_outliers(landmark_sequence: List[np.ndarray]) -> List[np.ndarray]:
    """
    Idea: if dist i-1 to i+1 is less than dist i-1 to i, then i is outlier
    :param landmark_sequence:
    :return:
    """
    pass


def extract_key_frames(landmark_sequence: List[np.ndarray], target_len: int) -> List[int]:
    """
    Used only in training, as at runtime full frame list is not known due to absence of gesture
    start/stop flags.

    :param landmark_sequence: a sequence of landmarks from frames
    :param target_len: desired length of the sampled sequence
    :return: indexes of included frames
    """
    # todo: detect outliers

    displacements: List[float] = [0]
    last_pos = MediaPipe.hands_spacial_position(landmark_sequence[0])

    for i in range(1, len(landmark_sequence)):
        pos = MediaPipe.hands_spacial_position(landmark_sequence[i])
        displacements.append(np.linalg.norm(last_pos - pos))
        last_pos = pos.copy()

    total = sum(displacements)
    interval = total / (target_len - 1)
    running_sum = 0
    key_frames: List[int] = [0]

    for i in range(1, len(landmark_sequence)):
        running_sum += displacements[i]
        if running_sum >= interval:
            key_frames.append(i)
            running_sum = 0

    if len(key_frames) < target_len:
        key_frames.append(len(key_frames) - 1)

    return key_frames
