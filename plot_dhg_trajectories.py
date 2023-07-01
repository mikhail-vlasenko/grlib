import os

import numpy as np
import matplotlib
from natsort import natsorted


from src.grlib.exceptions import NoHandDetectedException
from src.grlib.feature_extraction.pipeline import Pipeline

matplotlib.use('TkAgg')  # use TkAgg backend which shows plots in a new window
import matplotlib.pyplot as plt


curr_path = 'data\\dynamic_dataset\\fist-up\\'
splits = [22, 23, 25]
files = [curr_path + file for file in os.listdir(curr_path)]
files = natsorted(files)

pipeline = Pipeline(num_hands=1)
pipeline.add_stage(0, 0)

landmarks = []
results = []
for file_idx, file in enumerate(files):
    print(file)
    try:
        results.append(pipeline.get_landmarks_from_path(file))
    except NoHandDetectedException:
        pass

# Remove the instances where no hand was detected (have empty list for landmarks)
results = [result for result in results if len(result[0]) > 0]

for result in results:
    landmarks.append(result[0])

data_np = np.array(landmarks)[0 : 22 + 1]
# data_np = np.array(landmarks)[22 : 22 + 23 + 1]

# DHG
# prefix = 'data/DHG2016/gesture_1/finger_2/subject_3/essai_1/'
# data_np = np.loadtxt(f'{prefix}skeleton_world.txt')

# Calculate the average position at each time point
averages = np.mean(data_np.reshape((data_np.shape[0], -1, 3)), axis=1)

# Split averages into x, y, and z for plotting
x_avg, y_avg, z_avg = averages.T

# Create a 3D plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot average positions over time
sc = ax.scatter(x_avg, y_avg, z_avg, c=np.arange(len(x_avg)), cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Average 3D position over time')

# Create an interactive colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Time')

# Show the interactive window
plt.show(block=True)
