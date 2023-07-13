import os

import numpy as np
import pandas as pd
import matplotlib
from natsort import natsorted


from src.grlib.exceptions import NoHandDetectedException
from src.grlib.feature_extraction.pipeline import Pipeline

# matplotlib.use('TkAgg')  # use TkAgg backend which shows plots in a new window
import matplotlib.pyplot as plt


curr_path = '../../../data/dynamic_dataset_online/positions.csv'
df = pd.read_csv(curr_path)
print(len(df))

# df = df[:594]  # first sequence
# df = df[594:594+1028]  # second sequence
df = df[2268:] # fourth sequence

df['frame_number'] = np.arange(len(df))  # Add a new column for frame numbers
df = df[df['0'] != -1]  # remove frames without hands

x_avg, y_avg, z_avg = np.array(df['0']), np.array(df['1']), np.array(df['2'])
frame_numbers = np.array(df['frame_number'])  # Use the frame numbers for color map


# Create a 3D plot
# plt.ion()
fig = plt.figure()
fig.set_dpi(400)
ax = fig.add_subplot(111, projection='3d')

# Plot average positions over time
sc = ax.scatter(x_avg, y_avg, z_avg, c=frame_numbers, cmap='viridis')
ax.view_init(elev=90., azim=90.)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Hand position over time')

# Create an interactive colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Frame number')

# Show the interactive window
# plt.show(block=True)
plt.savefig("../../../trajectory4.png",bbox_inches='tight')
plt.show()

