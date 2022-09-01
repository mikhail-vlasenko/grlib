import cv2.cv2 as cv
import numpy as np
import os
import keyboard


def save_image(image: np.array, y: str, frame_set: int, image_idx: int):
    if not os.path.isdir(f'out/{y}'):
        os.makedirs(f'out/{y}')

    file_path = f'out/{y}/{frame_set}_{image_idx}.jpg'

    cv.imwrite(file_path, image)
    print(f'Saved image of class {y} to file {file_path}.')


if __name__ == '__main__':
    camera = cv.VideoCapture(0)

    cls = 'dynamic gesture 1'

    # it is necessary to differentiate between frames taken for different instances of the
    # dynamic gesture.
    # continuous press of space makes one instance of the gesture
    last_pressed = False
    frame_set_cnt = 0
    image_idx = 0
    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        cv.imshow('Frame', frame)

        if keyboard.is_pressed('space'):
            if not last_pressed:
                frame_set_cnt += 1
                image_idx = 0
            last_pressed = True
            image_idx += 1
            save_image(frame, cls, frame_set_cnt, image_idx)
        else:
            last_pressed = False

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()