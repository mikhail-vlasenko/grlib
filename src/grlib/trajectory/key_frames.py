from typing import List

import numpy as np

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe


def extract_key_frames(landmark_sequence: List[np.ndarray], target_len: int) -> List[np.ndarray]:
    """
    Used only in training, as at runtime full frame list is not known due to absence of gesture
    start/stop flags.

    :param landmark_sequence: a sequence of landmarks from frames
    :param target_len: desired length of the sampled sequence
    :return: a sampled sequence of landmarks, length = target_len
    """
    displacements: List[float] = [0]
    last_pos = MediaPipe.hands_spacial_position(landmark_sequence[0])

    for i in range(1, len(landmark_sequence)):
        pos = MediaPipe.hands_spacial_position(landmark_sequence[i])
        displacements.append(np.linalg.norm(last_pos - pos))
        last_pos = pos.copy()

    total = sum(displacements)
    interval = total / (target_len - 1)
    running_sum = 0
    key_frames: List[np.ndarray] = [landmark_sequence[0]]

    for i in range(1, len(landmark_sequence)):
        running_sum += displacements[i]
        if running_sum >= interval:
            key_frames.append(landmark_sequence[i])
            running_sum = 0

    if len(key_frames) < target_len:
        key_frames.append(landmark_sequence[-1])

    return key_frames
