from matplotlib import pyplot as plt
import numpy as np


prefix = 'data/DHG2016/gesture_5/finger_2/subject_3/essai_1/'
frame_n = 50
file = prefix + f'depth_{frame_n}.png'


def remove_palm_center(landmarks):
    landmarks = landmarks.reshape((landmarks.shape[0], 22, 3))
    landmarks = np.concatenate((landmarks[:, 0:1, :], landmarks[:, 2:, :]), axis=1)
    return landmarks.reshape((landmarks.shape[0], 21, 3))


def recenter_landmarks(landmarks):
    landmarks = landmarks.reshape((landmarks.shape[0], 21, 3))
    palm_center = np.mean(landmarks, axis=1)
    landmarks = landmarks - palm_center[:, np.newaxis, :]
    return landmarks.reshape((landmarks.shape[0], 21, 3))


def show_image():
    depth_map = plt.imread(file)

    landmarks = np.loadtxt(prefix + 'skeleton_world.txt')
    landmarks = landmarks.reshape((landmarks.shape[0], 22, 3))

    landmarks = remove_palm_center(landmarks)
    landmarks = recenter_landmarks(landmarks)

    img_landmarks = np.loadtxt(prefix + 'skeleton_image.txt')
    img_landmarks = img_landmarks.reshape((landmarks.shape[0], 22, 2))

    plt.imshow(depth_map, cmap='gray')

    for i in range(len(landmarks[frame_n])):
        l = landmarks[frame_n][i]
        plt.text(l[0] * 800, -l[1] * 800, i, ha="center", va="center", color="r", fontsize=8)
        img_l = img_landmarks[frame_n][i]
        # plt.text(img_l[0], img_l[1], i, ha="center", va="center", color="g", fontsize=8)
    plt.show()


def show_3d():
    import matplotlib
    matplotlib.use('TkAgg')

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    landmarks = np.loadtxt(prefix + 'skeleton_world.txt')
    landmarks = landmarks.reshape((landmarks.shape[0], 22, 3))
    landmarks = remove_palm_center(landmarks)
    for i in range(len(landmarks[frame_n])):
        ax.scatter(landmarks[frame_n][i][0], landmarks[frame_n][i][1], landmarks[frame_n][i][2], c='r')
    plt.show(block=True)


if __name__ == '__main__':
    show_image()
    # show_3d()
