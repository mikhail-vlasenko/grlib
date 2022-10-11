import cv2 as cv
import numpy as np
import os
import keyboard
import time


def save_image(image: np.array, y: str):
    if not os.path.isdir(f'out/{y}'):
        os.makedirs(f'out/{y}')

    num_files = len([name for name in os.listdir(f'out/{y}/') if os.path.isfile(f'out/{y}/{name}')])

    file_path = f'out/{y}/{num_files}.jpg'

    cv.imwrite(file_path, image)
    print(f'Saved image of class {y} to file {file_path}.')


if __name__ == '__main__':
    camera = cv.VideoCapture(0)

    cls = 'Your gesture 1'

    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        cv.imshow('Frame', frame)

        if keyboard.is_pressed('space'):
            save_image(frame, cls)
            time.sleep(0.2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()
