import numpy as np
from sklearn.linear_model import LogisticRegression


class DynamicDetector:
    def __init__(self, start_shapes, y, pipeline, start_pos_confidence):
        """

        :param start_shapes:
        :param y:
        :param pipeline:
        :param start_pos_confidence: how much certainty is enough to include class for trajectory analysis
        """
        self.start_detection_model = LogisticRegression()
        self.start_detection_model.fit(np.array(start_shapes), y)
        self.sorted_labels = sorted(y.tolist())
        self.pipeline = pipeline
        self.start_pos_confidence = start_pos_confidence

    def analyze_frame(self, frame):
        landmarks = self.pipeline.get_world_landmarks_from_image(frame).flatten().tolist()
        self.pipeline.optimize()

        # proba because there is no need to be sure that it is a particular class:
        #   if there is a good chance it is, trajectory will determine it
        prediction = self.start_detection_model.predict_proba(np.array([landmarks]))[0]

        possible_classes = []
        for i in range(len(prediction)):
            if prediction[i] >= self.start_pos_confidence:
                possible_classes.append(self.start_detection_model.classes_[i])

        return possible_classes

        # todo: take the trajectories of possible classes
        # dynamically detect trajectory and try to map onto a possible class
