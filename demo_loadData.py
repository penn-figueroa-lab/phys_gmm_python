import matplotlib.pyplot as plt
import numpy as np
from fit_gmm import fig_gmm
from datasets.load_dataset import load_dataset
from utils.plotting import plot_reference_trajectories
import time
from Structs import Est_options
from gmm_stuff.my_gaussPDF import my_gaussPDF

# Step 1 (DATA LOADING): Load Datasets
plt.close("all")  # maybe need further close
pkg_dir = r'E:\ds-opt-python\ds-opt-python\phys_gmm_python'
chosen_dataset = 11
sub_sample = 2  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 7  # For real 3D data
Data = load_dataset(pkg_dir, chosen_dataset, sub_sample, nb_trajectories)


# Position/Velocity Trajectories (checked)
vel_samples = 15
vel_size = 20
plot_reference_trajectories.plot_reference_trajectories(Data, vel_samples, vel_size)

# Extract Position and Velocities (checked)
M = len(Data)
N = len(Data[0])
M = int(M / 2)
Xi_ref = Data[0:M, :]
Xi_dot_ref = Data[M:, :]

# implement 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
est_options = Est_options()

# Fit GMM to Trajectory Data
start = time.perf_counter()
fig_gmm(Xi_ref, Xi_dot_ref, est_options)
end = time.perf_counter()
print("total computation time is {}".format(start-end))


