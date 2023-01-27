from typing import List, Union

import numpy as np
from collections import deque

from src.grlib.trajectory.trajectory_candidate import TrajectoryCandidate
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier


class DynamicDetector:
    """
    Runs the dynamic gesture recognition process.
    """

    def __init__(
            self,
            start_shape_detection_model,
            start_pos_confidence: float,
            trajectory_classifier: TrajectoryClassifier,
            update_candidates_every: int = 10,
            candidate_zero_precision: float = 0.1,
            candidate_scale_zero_precision: bool = True,
            candidate_old_multiplier: float = 5,
            end_shape_detection_model=None,
            end_pos_confidence: float = 0.5,
            verbose=False,
    ):
        """
        :param start_shape_detection_model: model to predict probabilities of the start shapes.
        :param start_pos_confidence: how much certainty on the start shape detection model
        is enough to include class for trajectory analysis.
        Too high - no predictions.
        Too low - different gestures with similar trajectories will be confused.
        :param trajectory_classifier: the classifier to be used for trajectories.
        :param update_candidates_every: every x frame update() is called on candidates.
            This should be decreased if you have faster gestures,
            or the number of key frames is increased.
        :param candidate_zero_precision: displacement more than this (in image-relative coords)
            is considered movement for a candidate.
        :param candidate_scale_zero_precision: if True, the zero precision is scaled.
        :param candidate_old_multiplier: after this multiplied by update_candidates_every,
            the candidate is considered outdated and is dropped.
        :param end_shape_detection_model: model to predict probability that a completed trajectory
        represents a gesture, based on the hand's shape in the end.
        :param end_pos_confidence: how much certainty on the end shape detection model
        is enough to count the prediction complete.
        :param verbose: whether to print debug info.
        """
        self.start_detection_model = start_shape_detection_model
        self.end_detection_model = end_shape_detection_model
        self.start_pos_confidence = start_pos_confidence
        self.end_pos_confidence = end_pos_confidence
        self.trajectory_classifier: TrajectoryClassifier = trajectory_classifier

        # store the candidate as well as when it should be updated next
        self.current_candidates: deque[(TrajectoryCandidate, float)] = deque()
        self.frame_cnt = 0
        self.update_candidates_every = update_candidates_every
        self.candidate_zero_precision = candidate_zero_precision
        self.candidate_scale_zero_precision = candidate_scale_zero_precision
        self.candidate_old_multiplier = candidate_old_multiplier

        self.verbose = verbose
        # currently, only single-handed dynamic gestures are supported
        self.num_hands = 1

    def update_candidates(self, hand_position: np.ndarray) -> List[str]:
        """
        Updates the candidates and returns the prediction if there is one.
        :param hand_position: position of the hand within frame.
        :return: list of gesture classes which candidates became valid on this frame.
        """
        self.frame_cnt += 1

        valid_classes = []
        oldest_allowed_timestamp = self.frame_cnt - \
                                   self.update_candidates_every * self.candidate_old_multiplier
        # check which candidates are ready to be updated and update them.
        # because the update frequency is constant,
        # the candidates in the queue are sorted by the update timestamp
        while True:
            if len(self.current_candidates) > 0 and self.current_candidates[0][1] <= self.frame_cnt:
                candidate, _ = self.current_candidates.popleft()
                if self.verbose:
                    print(f"Updating {candidate}")
                # a candidate may also be too old to be considered
                if candidate.timestamp < oldest_allowed_timestamp:
                    continue
                # update the candidate and check if it has reached the end (is valid)
                if candidate.update(hand_position):
                    if candidate.valid:
                        valid_classes.append(candidate.pred_class)
                        if self.verbose:
                            print(f"Complete trajectory for {valid_classes[-1]}")
                    else:
                        # add back to the queue if it is not valid yet, but can be in the future
                        self.current_candidates.append(
                            (candidate, self.frame_cnt + self.update_candidates_every))
            else:
                break

        return valid_classes

    def add_candidates(self, landmarks: np.ndarray, hand_position: np.ndarray) -> List[str]:
        """
        Appends the new candidates to existing ones.
        :param landmarks: shape of the hand. Indifferent to its position within the frame.
        :param hand_position: hand position on the frame.
        :return: list of potential classes on this frame.
        """
        if landmarks.shape != (21 * 3 * self.num_hands,):
            raise ValueError(f"Expected to get hand landmarks of shape "
                             f"{(21 * 3 * self.num_hands,)}, got {landmarks.shape}")
        # proba because there is no need to be sure that it is a particular class:
        #   if there is a good chance it is, trajectory will determine it
        # taking [0] because batch size is 1
        prediction = self.start_detection_model.predict_proba(np.array([landmarks]))[0]

        possible_classes: List[str] = []
        for i in range(len(prediction)):
            if prediction[i] >= self.start_pos_confidence:
                possible_classes.append(self.start_detection_model.classes_[i])

        for possible_gesture_class in possible_classes:
            target_trajectories = self.trajectory_classifier.get_trajectories(possible_gesture_class)

            # do not add if already exists a recent candidate for the same gesture class
            exists_recent = False
            for candidate, _ in self.current_candidates:
                # trajectories don't have to be compared, as they were not updated yet
                if candidate.pred_class == possible_gesture_class and \
                        candidate.timestamp > self.frame_cnt - (self.update_candidates_every / 2):
                    exists_recent = True
                    break

            if not exists_recent:
                # add a new candidate for every representative trajectory
                for target in target_trajectories:
                    self.current_candidates.append((
                        TrajectoryCandidate(
                            target,
                            possible_gesture_class,
                            hand_position,
                            zero_precision=self.candidate_zero_precision,
                            use_scaled_zero_precision=self.candidate_scale_zero_precision,
                            start_timestamp=self.frame_cnt),
                        self.frame_cnt + self.update_candidates_every
                    ))

        return possible_classes

    def evaluate_end_shape(self, landmarks: np.ndarray, classes: List[str]) -> Union[str, None]:
        if self.end_detection_model is None:
            raise ValueError("End detection model is not set")

        prediction = None
        # if there is an end detection model, check if the end shape is correct
        end_prediction = self.end_detection_model.predict_proba(np.array([landmarks]))[0]
        for i in range(len(end_prediction)):
            if end_prediction[i] > self.end_pos_confidence:
                for gesture_class in classes:
                    if gesture_class == self.end_detection_model.classes_[i]:
                        prediction = gesture_class
                        break
        return prediction

    def get_prediction(self, landmarks: np.ndarray, hand_position: np.ndarray) -> Union[str, None]:
        """
        Updates the candidates and returns the prediction if there is one.
        Combines most of the methods of this class.
        :param landmarks: shape of the hand. Indifferent to its position within the frame.
        :param hand_position: hand position on the frame.
        :return: prediction for this frame (or None)
        """
        self.add_candidates(landmarks, hand_position)
        classes = self.update_candidates(hand_position)
        if len(classes) == 0:
            # nothing was predicted
            return None

        if self.end_detection_model is not None:
            prediction = self.evaluate_end_shape(landmarks, classes)
        else:
            if len(classes) > 1 and self.verbose:
                print(f"Warning: multiple classes predicted: {classes}. Choosing the first one.")
            prediction = classes[0]

        if prediction is not None:
            # clean the current candidates because the gesture is found
            self.current_candidates.clear()
        return prediction

    def clear_candidates(self):
        self.current_candidates.clear()
