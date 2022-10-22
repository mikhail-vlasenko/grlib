# cv2.cv2 because MediaPipe uses opencv-contrib
from collections import namedtuple
import cv2.cv2 as cv
from mediapipe import solutions as mp
import time
from typing import NamedTuple

import numpy as np


class MediaPipe:
    """
    Class to interact with MediaPipe library.
    """
    def __init__(self, num_hands: int = 2):
        self.num_hands = num_hands
        self.drawing = mp.drawing_utils
        self.drawing_styles = mp.drawing_styles
        self.hands = mp.hands.Hands(static_image_mode=True, max_num_hands=self.num_hands, min_detection_confidence=0.1)

    def process_from_path(self, img_path: str) -> NamedTuple:
        """
        Performs landmark extraction.
        :param img_path: path to image for which to recognize landmarks
        :param brightness: the additional brightness of the image
        :param rotation: the rotation of the image in degrees
        :return: recognition results
        return type: NamedTuple with fields
            multi_hand_landmarks - 21 hand landmarks where each landmark is composed of x, y and z.
                x and y are normalized to [0.0, 1.0] by the image width and height respectively.
                z represents the landmark depth with the depth at the wrist being the origin,
                and the smaller the value the closer the landmark is to the camera.
                The magnitude of z uses roughly the same scale as x.
            multi_hand_world_landmarks - x, y and z - Real-world 3D coordinates in meters
             with the origin at the hand’s approximate geometric center.
            multi_handedness - 'left' or 'right', and the certainty that the hand is there
        """
        if not (img_path.endswith('.jpg') or img_path.endswith('.jpeg') or img_path.endswith('.png')):
            print(f'Not an image: {img_path}')
            ret_tuple = namedtuple('none_tuple', 'multi_hand_landmarks multi_hand_world_landmarks multi_handedness')
            return ret_tuple(None, None, None)

        return self.process_from_image(cv.imread(img_path))

    def process_from_image(self, img: np.array) -> NamedTuple:
        """
        Identical to process_from_path except for the argument:
        :param img: the image from which to recognize landmarks
        :param brightness: the additional brightness of the image
        :param rotation: the rotation that should be applied to the image
        :return: recognition results, same as in process_from_path
        """
        # Read an image, flip it around y-axis for correct handedness output.
        image = cv.flip(img, 1)

        # Convert the BGR image to RGB before processing.
        results = self.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        return results

    def get_landmarks_from_hands(self, detected_hands) -> np.array:
        """
        Returns a list of landmarks from the given processed list of hands.
        :param detected_hands: the hands as detected by mediapipe
        :return: a list of landmarks from detected_hands
        """
        point_array = []

        for hand in detected_hands:
            for point in hand.landmark:
                point_array.append([point.x, point.y, point.z])

        # If less than num_hands hands were detected, fill in the rest of the list with zeros
        point_array.extend([[0.0, 0.0, 0.0] for _ in range(21 * self.num_hands - len(point_array))])
        return np.array(point_array)

    @staticmethod
    def hands_spacial_position(landmarks: np.ndarray) -> np.ndarray:
        """
        Encodes the hands position in the picture.
        Can be used to calculate the trajectory.
        Warning: the coordinates of the given landmarks should not be centered on the hand itself.
            Thus, "world_landmarks" are not acceptable.
        :param landmarks: array of landmarks, like the result of get_landmarks_from_hands.
        NOT world landmarks, as those are centered on the hand!
        TODO: do we want to strictly differentiate between world and other landmarks?
        TODO: make a warning if dynamic gesture appears stationary
        :return: the encoding
        """
        reshaped = landmarks.reshape((-1, 21, 3))
        return np.mean(reshaped, axis=1)

    def show_landmarks(self, img_path, results=None):
        """
        Creates debug files and pyplots of landmarks.
        saves files to /data/annotated/<timestamp>.png
        :param img_path: original image
        :param results: recognition output
        :return: None
        """
        if results is None:
            results = self.process_from_path(img_path)
        image = cv.flip(cv.imread(img_path), 1)
        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            return
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp.hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp.hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            self.drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp.hands.HAND_CONNECTIONS,
                self.drawing_styles.get_default_hand_landmarks_style(),
                self.drawing_styles.get_default_hand_connections_style())
        cv.imwrite(
            'data/annotated/' + str(int(time.time())) + '.png', cv.flip(annotated_image, 1))
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            return
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            # draw from different angles
            self.drawing.plot_landmarks(
                hand_world_landmarks, mp.hands.HAND_CONNECTIONS, azimuth=5)
            self.drawing.plot_landmarks(
                hand_world_landmarks, mp.hands.HAND_CONNECTIONS, azimuth=50, elevation=20)
            self.drawing.plot_landmarks(
                hand_world_landmarks, mp.hands.HAND_CONNECTIONS, azimuth=5, elevation=90)

    def close(self):
        self.hands.close()
