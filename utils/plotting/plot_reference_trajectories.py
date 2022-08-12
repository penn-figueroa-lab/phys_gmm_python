import matplotlib.pyplot as plt
import numpy as np


def plot_reference_trajectories(Data, vel_sample, vel_size):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    M = len(Data) / 2  # store 1 Dim of Data
    if M == 2:
        # Plot the position trajectories
        plt.plot(Data[0], Data[1], 'ro', markersize=1)
        # Plot Velocities of Reference Trajectories
        vel_points = Data[:, ::vel_sample]
        U = np.zeros(len(vel_points[0]))
        V = np.zeros(len(vel_points[0]))  # （385,)
        for i in np.arange(0, len(vel_points[0])):
            dir_ = vel_points[2:, i] / np.linalg.norm(vel_points[2:, i])
            U[i] = dir_[0]
            V[i] = dir_[1]
        q = ax.quiver(vel_points[0], vel_points[1], U, V, width=0.005, scale=vel_size)

    # plt.show()

# get more function on 3D
