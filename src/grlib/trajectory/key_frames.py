from typing import List

import numpy as np

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe


def remove_outliers(landmark_sequence: List[np.ndarray]) -> List[np.ndarray]:
    """
    Idea: if dist i-1 to i+1 is less than dist i-1 to i or dist i to i+1, then i is outlier
    :param landmark_sequence: initial landmark sequence
    :return: reduced sequence
    """
    positions = []
    for i in range(len(landmark_sequence)):
        positions.append(MediaPipe.hands_spacial_position(landmark_sequence[i]))

    non_outliers = [landmark_sequence[0]]
    for i in range(1, len(landmark_sequence) - 1):
        if min(_distance(positions[i-1], positions[i]), _distance(positions[i], positions[i+1])) > \
                _distance(positions[i-1], positions[i+1]):
            # outlier
            continue
        non_outliers.append(landmark_sequence[i])

    non_outliers.append(landmark_sequence[-1])
    return non_outliers


def extract_key_frames(landmark_sequence: List[np.ndarray], target_len: int) -> List[int]:
    """
    Used only in training, as at runtime full frame list is not known due to absence of gesture
    start/stop flags.

    :param landmark_sequence: a sequence of landmarks from frames
    :param target_len: desired length of the sampled sequence
    :return: indexes of included frames
    """
    landmark_sequence = remove_outliers(landmark_sequence)

    displacements: List[float] = [0]
    last_pos = MediaPipe.hands_spacial_position(landmark_sequence[0])

    for i in range(1, len(landmark_sequence)):
        pos = MediaPipe.hands_spacial_position(landmark_sequence[i])
        displacements.append(_distance(last_pos, pos))
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


def _distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)
