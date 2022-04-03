# cv2.cv2 because MediaPipe uses opencv-contrib
from collections import namedtuple
from itertools import permutations, combinations
import cv2.cv2 as cv
from mediapipe import solutions as mp
import time
import os
from typing import NamedTuple, List

import numpy as np

from exceptions import NoHandDetectedException


class MediaPipe:
    """
    Class to interact with MediaPipe library.
    """
    def __init__(self):
        self.drawing = mp.drawing_utils
        self.drawing_styles = mp.drawing_styles
        self.hands = mp.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    def process(self, img_path: str) -> NamedTuple:
        """
        Performs landmark extraction.
        :param img_path: path to image
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
            print('not an image')
            ret_tuple = namedtuple('none_tuple', 'multi_hand_landmarks multi_hand_world_landmarks multi_handedness')
            return ret_tuple(None, None, None)
        # Read an image, flip it around y-axis for correct handedness output.
        image = cv.flip(cv.imread(img_path), 1)
        # Convert the BGR image to RGB before processing.
        results = self.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        return results

    def get_landmarks(self, img_path: str) -> np.array:
        """
        Returns only landmarks for the first detected hand on 1 image.
        :param img_path: the path to the image from which to read the landmarks
        :return: a numpy array of landmarks
        """
        detected_hands = self.process(img_path).multi_hand_landmarks

        if detected_hands is None:
            raise NoHandDetectedException(f'No hand has been detected for {img_path}')

        return self.get_landmarks_from_hands(detected_hands)

    def get_world_landmarks(self, img_path: str) -> np.array:
        """
        Returns only world landmarks for the first detected hand on 1 image.
        :param img_path: the path to the image from which to read the landmarks
        :return: a numpy array of 63 floating point landmarks
        """
        detected_hands = self.process(img_path).multi_hand_world_landmarks

        if detected_hands is None:
            raise NoHandDetectedException(f'No hand has been detected for {img_path}')

        return self.get_landmarks_from_hands(detected_hands)

    @staticmethod
    def get_landmarks_from_hands(detected_hands) -> np.array:
        """
        Returns a list of landmarks from the given processed list of hands.
        :param detected_hands: the hands as detected by mediapipe
        :return: a list of landmarks from detected_hands
        """
        point_array = []

        for hand in detected_hands:
            for point in hand.landmark:
                point_array.append([point.x, point.y, point.z])
        # If one or zero hands were detected, fill the rest of the list with zeros
        point_array.extend([[0.0, 0.0, 0.0] for _ in range(21 * 2 - len(point_array))])
        return np.array(point_array)

    def get_landmarks_from_image(self, img: np.array) -> NamedTuple:
        image = cv.flip(img, 1)
        results = self.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        return results

    def show_landmarks(self, img_path, results=None):
        """
        Creates debug files and pyplots of landmarks.
        saves files to /data/annotated/<timestamp>.png
        :param img_path: original image
        :param results: recognition output
        :return: None
        """
        if results is None:
            results = self.process(img_path)
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
