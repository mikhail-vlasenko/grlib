from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.grlib.feature_extraction.mediapipe_landmarks import MediaPipe
from src.grlib.feature_extraction.pipeline import Pipeline
from src.grlib.trajectory.trajectory_candidate import TrajectoryCandidate
from src.grlib.trajectory.trajectory_classifier import TrajectoryClassifier


class DynamicDetector:
    def __init__(self, start_shapes, y, pipeline, start_pos_confidence, trajectory_classifier):
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
        self.update_candidates_every = 10
        self.last_pred = ""

    def analyze_frame(self, frame):
        landmarks = self.pipeline.get_world_landmarks_from_image(frame).flatten().tolist()
        self.pipeline.optimize()

        # WARNING: this is for a single hand
        hand_position = MediaPipe.hands_spacial_position(
            self.pipeline.get_landmarks_from_image(frame)
        )[0]
        print(hand_position)

        # proba because there is no need to be sure that it is a particular class:
        #   if there is a good chance it is, trajectory will determine it
        prediction = self.start_detection_model.predict_proba(np.array([landmarks]))[0]

        possible_classes: List[str] = []
        for i in range(len(prediction)):
            if prediction[i] >= self.start_pos_confidence:
                possible_classes.append(self.start_detection_model.classes_[i])

        for p in possible_classes:
            target_traj = self.trajectory_classifier.avg_trajectories[p]
            # todo: dont add if already exists a recent record
            self.current_candidates.append(TrajectoryCandidate(
                target_traj, p, hand_position, 0.1, self.frame_cnt
            ))

        self.frame_cnt += 1
        if self.frame_cnt % self.update_candidates_every:
            pred = self.update_candidates(hand_position)
            if pred != "":
                self.last_pred = pred

        return possible_classes

    def update_candidates(self, hand_position) -> str:
        i = 0
        while i < len(self.current_candidates):
            candidate = self.current_candidates[i]
            if candidate.timestamp < self.frame_cnt - self.update_candidates_every:
                if not candidate.update(hand_position):
                    self.current_candidates.pop(i)
                    i -= 1
                if candidate.valid:
                    # clean the current candidates cause we found the gesture
                    self.current_candidates = []
                    return candidate.pred_class
            i += 1
        return ""
