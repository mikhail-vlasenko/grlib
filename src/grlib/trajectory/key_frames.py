from typing import List

import numpy as np

from ..feature_extraction.mediapipe_landmarks import hands_spacial_position


def remove_outliers(landmark_sequence: List[np.ndarray]) -> List[np.ndarray]:
    """
    Method: if dist i-1 to i+1 is less than dist i-1 to i and dist i to i+1, then i is outlier.
    Where dist is the distance between hand positions on the frames.
    :param landmark_sequence: initial landmark sequence
    :return: reduced sequence
    """
    positions = []
    for i in range(len(landmark_sequence)):
        positions.append(hands_spacial_position(landmark_sequence[i]))

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
    Reduces the amount of frames in the initial sequence to the target_len, based on the distance
        between hand positions on the frames.
    Ideally, the hand will travel the same distance between every 2 consecutive key frames.
    Used only in training, as at runtime full frame list is not known due to absence of gesture
    start/stop flags.

    :param landmark_sequence: a sequence of landmarks from frames
    :param target_len: desired length of the sampled sequence
    :return: indexes of included frames
    """
    landmark_sequence = remove_outliers(landmark_sequence)

    # compute displacements between neighboring frames
    displacements: List[float] = [0]
    last_pos = hands_spacial_position(landmark_sequence[0])

    for i in range(1, len(landmark_sequence)):
        pos = hands_spacial_position(landmark_sequence[i])
        displacements.append(_distance(last_pos, pos))
        last_pos = pos.copy()

    total = sum(displacements)
    # this is how often a key frame should be placed
    interval = total / (target_len - 1)
    running_sum = 0
    key_frames: List[int] = [0]

    for i in range(1, len(landmark_sequence)):
        running_sum += displacements[i]
        if running_sum >= interval:
            # the total displacement up to this point is enough to consider this frame key
            key_frames.append(i)
            running_sum = 0

    if len(key_frames) < target_len:
        # include the last frame if it wasn't
        key_frames.append(len(landmark_sequence) - 1)

    return key_frames


def _distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)
