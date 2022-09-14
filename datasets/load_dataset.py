from scipy.io import loadmat
import numpy as np


def load_dataset(pkg_dir, dataset, sub_sample, nb_trajectories):
    # add r in your dict in pkg_dir
    dataset_name = []
    if dataset == 1:
        dataset_name = r'2D_concentric.mat'
    elif dataset == 2:
        dataset_name = r'2D_opposing.mat'
    elif dataset == 3:
        dataset_name = r'2D_multiple.mat'
    elif dataset == 4:
        dataset_name = r'2D_snake.mat'
    elif dataset == 5:
        dataset_name = r'2D_messy-snake.mat'
    elif dataset == 6:
        dataset_name = r'2D_viapoint.mat'
    elif dataset == 7:
        dataset_name = r'2D_Lshape.mat'
    elif dataset == 8:
        dataset_name = r'2D_Ashape.mat'
    elif dataset == 9:
        dataset_name = r'2D_Sshape.mat'
    elif dataset == 10:
        dataset_name = r'2D_multi-behavior.mat'
    elif dataset == 11:
        dataset_name = r'3D_viapoint_2.mat'
        traj_ids = [1, 2]
    elif dataset == 12:
        dataset_name = r'3D_sink.mat'
    elif dataset == 13:
        dataset_name = r'3D_bumpy-snake.mat'

    final_dir = pkg_dir + "\\datasets\\" + dataset_name
    if dataset <= 6:
        print("we dont currently offer this function")
    elif dataset <= 10:
        data_ = loadmat(r"{}".format(final_dir))
        data = np.array(data_["data"])
        N = len(data[0])
        Data = data[0][0]
        for n in np.arange(1, N):
            data_ = np.array(data[0][n])
            Data = np.concatenate((Data, data_), axis=1)
    else:
        data_ = loadmat(r"{}".format(final_dir))
        data_ = np.array(data_["data"])
        N = len(data_)
        traj = np.random.choice(np.arange(N), nb_trajectories, replace=False)
        data = data_[traj]
        for l in np.arange(nb_trajectories):
            data[l][0] = data[l][0][:, ::sub_sample]
            if l == 0:
                Data = data[l][0]
            else:
                Data = np.concatenate((Data, data[l][0]), axis=1)

    return Data



