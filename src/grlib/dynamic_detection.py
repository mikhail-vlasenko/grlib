from typing import List

import cv2.cv2 as cv
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.trajectory.trajectory_candidate import TrajectoryCandidate
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier


class DynamicDetector:
    def __init__(
            self,
            start_shapes,
            y,
            pipeline,
            start_pos_confidence,
            trajectory_classifier,
            update_candidates_every=10,
            candidate_zero_precision=0.1
    ):
        """

        :param start_shapes:
        :param y:
        :param pipeline:
        :param start_pos_confidence: how much certainty is enough to include class for trajectory analysis
        """
        self.start_detection_model = LogisticRegression()
        self.start_detection_model.fit(np.array(start_shapes), y)
        self.sorted_labels = sorted(y.tolist())
        self.pipeline: Pipeline = pipeline
        self.start_pos_confidence = start_pos_confidence
        self.trajectory_classifier: TrajectoryClassifier = trajectory_classifier

        self.current_candidates: List[TrajectoryCandidate] = []
        self.frame_cnt = 0
        self.update_candidates_every = update_candidates_every
        self.candidate_zero_precision = candidate_zero_precision
        self.last_pred = ""
        self.last_time_pred = 0

    def analyze_frame(self, frame: np.ndarray):
        self.frame_cnt += 1
        if self.last_time_pred < self.frame_cnt - 30:
            self.last_pred = ""

        landmarks = self.pipeline.get_world_landmarks_from_image(frame).flatten().tolist()
        self.pipeline.optimize()

        # WARNING: this is for a single hand
        hand_position = MediaPipe.hands_spacial_position(
            self.pipeline.get_landmarks_from_image(frame)
        )[0]

        # draw the position of the hand
        cv.circle(
            frame,
            (round((1 - hand_position[0]) * frame.shape[1]),
             round(hand_position[1] * frame.shape[0])),
            10,
            (0, 0, 255),
            thickness=3,
        )

        # proba because there is no need to be sure that it is a particular class:
        #   if there is a good chance it is, trajectory will determine it
        prediction = self.start_detection_model.predict_proba(np.array([landmarks]))[0]

        possible_classes: List[str] = []
        for i in range(len(prediction)):
            if prediction[i] >= self.start_pos_confidence:
                possible_classes.append(self.start_detection_model.classes_[i])

        for p in possible_classes:
            target_traj = self.trajectory_classifier.avg_trajectories[p]

            # do not add if already exists a recent record
            exists_recent = False
            for candidate in self.current_candidates:
                if candidate.pred_class == p and candidate.timestamp > self.frame_cnt - (self.update_candidates_every / 2):
                    exists_recent = True
                    break

            if not exists_recent:
                self.current_candidates.append(TrajectoryCandidate(
                    target_traj, p, hand_position,
                    zero_precision=self.candidate_zero_precision, start_timestamp=self.frame_cnt
                ))

        if self.frame_cnt % self.update_candidates_every == 0:
            pred = self.update_candidates(hand_position)
            if pred != "":
                self.last_pred = pred
                self.last_time_pred = self.frame_cnt
                self.current_candidates = []
                print(f'Prediction: {pred}')

        return possible_classes

    def update_candidates(self, hand_position) -> str:
        i = 0
        print(f'Hand position for update: {hand_position}')
        while i < len(self.current_candidates):
            candidate = self.current_candidates[i]
            # otherwise too early to make a decision
            if candidate.timestamp <= self.frame_cnt - self.update_candidates_every:
                print(str(candidate))
                # a candidate may also be too old to be considered
                if candidate.timestamp < self.frame_cnt - self.update_candidates_every * 5 or \
                        not candidate.update(hand_position):
                    self.current_candidates.pop(i)
                    i -= 1

                if candidate.valid:
                    # clean the current candidates cause we found the gesture
                    self.current_candidates = []
                    return candidate.pred_class
            i += 1
        return ""
