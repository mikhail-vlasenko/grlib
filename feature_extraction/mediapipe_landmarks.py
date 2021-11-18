# cv2.cv2 because MediaPipe uses opencv-contrib
import cv2.cv2 as cv
import mediapipe as mp
import time


class MediaPipe:
    """
    Class to interact with MediaPipe library
    """
    def __init__(self):
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.hands = mp.solutions.hands

    def get_landmarks(self, img_path):
        """
        performs landmark extraction
        :param img_path: path to image
        :return: 21 3D landmarks (a complex object)
        """
        with self.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5) as hands:
            # Read an image, flip it around y-axis for correct handedness output.
            image = cv.flip(cv.imread(img_path), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            return results

    def show_landmarks(self, img_path, results):
        """
        creates debug files and pyplots of landmarks
        saves files to /data/annotated/<timestamp>.png
        :param img_path: original image
        :param results: landmarks
        :return: None
        """
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
                f'{hand_landmarks.landmark[self.hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[self.hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            self.drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.hands.HAND_CONNECTIONS,
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
                hand_world_landmarks, self.hands.HAND_CONNECTIONS, azimuth=5)
            self.drawing.plot_landmarks(
                hand_world_landmarks, self.hands.HAND_CONNECTIONS, azimuth=50, elevation=20)
            self.drawing.plot_landmarks(
                hand_world_landmarks, self.hands.HAND_CONNECTIONS, azimuth=5, elevation=90)
