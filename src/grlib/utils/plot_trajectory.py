import os

import numpy as np
import pandas as pd
import matplotlib
from natsort import natsorted
from sklearn.preprocessing import MinMaxScaler

from src.grlib.exceptions import NoHandDetectedException
from src.grlib.feature_extraction.pipeline import Pipeline

# matplotlib.use('TkAgg')  # use TkAgg backend which shows plots in a new window
import matplotlib.pyplot as plt


def plot_trajectory(df):
    df['frame_number'] = np.arange(len(df))  # Add a new column for frame numbers
    df = df[df['0'] != -1]  # remove frames without hands

    x_avg, y_avg, z_avg = np.array(df['0']), np.array(df['1']), np.array(df['2'])
    frame_numbers = np.array(df['frame_number'])  # Use the frame numbers for color map

    scaler = MinMaxScaler(feature_range=(0.2, 1.0))
    scaled_z = scaler.fit_transform(z_avg.reshape(-1, 1)).flatten()

    # Invert the scaled z-values so that large z_avg means smaller marker
    scaled_z = 1.2 - scaled_z
    scale_factor = 50
    scaled_z = scaled_z * scale_factor  # Scale up the marker size

    # Create a 3D plot
    fig = plt.figure()
    fig.set_dpi(400)
    ax = fig.add_subplot(111, projection='3d')

    # Plot average positions over time, with marker size based on z-axis value
    sc = ax.scatter(x_avg, y_avg, z_avg, c=frame_numbers, s=scaled_z, cmap='viridis')
    ax.view_init(elev=90., azim=90.)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Hand position over time')

    # Create an interactive colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Frame number')

    # Create a legend for size to z-value relation
    min_z, max_z = np.min(z_avg), np.max(z_avg)
    legend_sizes = [scaler.transform(np.array([[min_z], [max_z]])).flatten()]
    legend_sizes = 1.3 - legend_sizes[0]  # Invert
    legend_labels = [f"Z = {round(min_z, 3)}", f"Z = {round(max_z, 3)}"]

    # Create proxy artists for the legend
    legend_handles = [plt.scatter([], [], s=size * scale_factor, c='gray') for size in legend_sizes]
    ax.legend(legend_handles, legend_labels, title='Marker Size to Z-Value', loc='upper right')

    # Save and show the plot
    plt.savefig("../trajectory.png", bbox_inches='tight')
    plt.show()
