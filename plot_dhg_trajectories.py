import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # use TkAgg backend which shows plots in a new window
import matplotlib.pyplot as plt


prefix = 'data/DHG2016/gesture_1/finger_2/subject_3/essai_1/'
# Input data, each row corresponds to a time point
data = np.loadtxt(f'{prefix}skeleton_world.txt')

# Convert data into numpy array for easier manipulation
data_np = np.array(data)

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
