from typing import List, Union

import cv2.cv2 as cv
import numpy as np
import pandas as pd
import sklearn.base
from sklearn.linear_model import LogisticRegression
from collections import deque

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.trajectory.trajectory_candidate import TrajectoryCandidate
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier


class DynamicDetector:
    """
    Runs the dynamic gesture recognition process.
    """

    def __init__(
            self,
            start_detection_model,
            y: np.ndarray,
            pipeline: Pipeline,
            start_pos_confidence: float,
            trajectory_classifier: TrajectoryClassifier,
            update_candidates_every=10,
            candidate_zero_precision=0.1,
            candidate_old_multiplier=5,
            verbose=False,
    ):
        """
        :param start_detection_model: model to predict probabilities of the start shapes.
        :param y: classes for these landmarks.
        :param pipeline: recognition pipeline.
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
        :param candidate_old_multiplier: after this multiplied by update_candidates_every,
            the candidate is considered outdated and is dropped.
        :param verbose: whether to print debug info.
        """
        self.start_detection_model = start_detection_model
        self.sorted_labels = sorted(y.tolist())
        self.pipeline: Pipeline = pipeline
        self.start_pos_confidence = start_pos_confidence
        self.trajectory_classifier: TrajectoryClassifier = trajectory_classifier

        # store the candidate as well as when it should be updated next
        self.current_candidates: deque[(TrajectoryCandidate, float)] = deque()
        self.frame_cnt = 0
        self.update_candidates_every = update_candidates_every
        self.candidate_zero_precision = candidate_zero_precision
        self.candidate_old_multiplier = candidate_old_multiplier
        self.last_time_pred = 0

        self.verbose = verbose

    def analyze_frame(
            self, landmarks: np.ndarray, hand_position: np.ndarray, idle_frame=False
    ) -> (List[str], str):
        """
        Runs the frame through the dynamic gesture recognition process.
        After execution, `self.last_pred` contains the most recently predicted class.
        `self.last_pred = ''` means no class is predicted.

        :param landmarks: shape of the hand. Indifferent to its position within the frame.
        :param hand_position: position of the hand within frame.
            for idle frames, it might make sense to pass last recorded position.
        :param idle_frame: if True, the landmarks and hands are not taken into account.
            however, the inner counter records a time step and the candidates are updated.
        :return: prediction for this frame ("" for none),
        list of gestures that can start on this frame.
        :raise: NoHandDetectedException
        """
        self.frame_cnt += 1

        possible_classes = []
        if not idle_frame:
            possible_classes = self.add_candidates(landmarks, hand_position)

        pred = ""
        oldest_allowed_timestamp = self.frame_cnt - \
                                   self.update_candidates_every * self.candidate_old_multiplier
        # check which candidates are ready to be updated and update them.
        # because the update frequency is constant,
        # the candidates in the queue are sorted by the update timestamp
        while True:
            if len(self.current_candidates) > 0 and self.current_candidates[0][1] <= self.frame_cnt:
                candidate, _ = self.current_candidates.popleft()
                if self.verbose:
                    print(f"updating {candidate}")
                # a candidate may also be too old to be considered
                if candidate.timestamp < oldest_allowed_timestamp:
                    continue
                # update the candidate and check if it has reached the end (is valid)
                if candidate.update(hand_position):
                    if candidate.valid:
                        pred = candidate.pred_class
                        self.last_time_pred = self.frame_cnt
                        # clean the current candidates because we found the gesture
                        self.current_candidates.clear()
                        if self.verbose:
                            print(f'Prediction: {pred}')
                        break
                    else:
                        # add back to the queue if it is not valid yet
                        self.current_candidates.append(
                            (candidate, self.frame_cnt + self.update_candidates_every))
            else:
                break

        return pred, possible_classes

    def add_candidates(self, landmarks, hand_position) -> List[str]:
        """
        Appends the new candidates to existing ones.
        :param landmarks: hand landmarks (shape).
        :param hand_position: hand position on the frame.
        :return: list of potential classes on this frame.
        """
        # proba because there is no need to be sure that it is a particular class:
        #   if there is a good chance it is, trajectory will determine it
        prediction = self.start_detection_model.predict_proba(np.array([landmarks]))[0]

        possible_classes: List[str] = []
        for i in range(len(prediction)):
            if prediction[i] >= self.start_pos_confidence:
                possible_classes.append(self.start_detection_model.classes_[i])

        for p in possible_classes:
            target_trajectories = self.trajectory_classifier.get_trajectories(p)

            # do not add if already exists a recent record
            exists_recent = False
            for candidate, _ in self.current_candidates:
                # trajectories don't have to be compared, as they were not updated yet
                if candidate.pred_class == p and candidate.timestamp > self.frame_cnt - (
                        self.update_candidates_every / 2):
                    exists_recent = True
                    break

            if not exists_recent:
                # add a new candidate for every representative trajectory
                for target in target_trajectories:
                    self.current_candidates.append((
                        TrajectoryCandidate(
                            target, p, hand_position,
                            zero_precision=self.candidate_zero_precision,
                            start_timestamp=self.frame_cnt),
                        self.frame_cnt + self.update_candidates_every
                    ))

        return possible_classes

    def extract_landmarks(
            self, frame: np.ndarray, draw_hand_position: bool = False
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """

        :param frame: the image. A circle to indicate the recognized hand position gets added.
        :param draw_hand_position: If True, puts a circle on the frame at the detected location
            of the hand.
        :return:
        """
        # this can raise NoHandDetectedException
        landmarks, handedness = self.pipeline.get_world_landmarks_from_image(frame)
        self.pipeline.optimize()

        # WARNING: this is for a single hand
        hand_position = MediaPipe.hands_spacial_position(
            # take [0] because we only need landmarks, not handedness
            self.pipeline.get_landmarks_from_image(frame)[0]
        )[0]

        if draw_hand_position:
            # draw the position of the hand, inplace
            cv.circle(
                frame,
                (round((1 - hand_position[0]) * frame.shape[1]),
                 round(hand_position[1] * frame.shape[0])),
                10,
                (0, 0, 255),
                thickness=3,
            )
        return landmarks, handedness, hand_position
