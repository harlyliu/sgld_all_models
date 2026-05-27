from GP_comp import GP
import random
import numpy as np
import math


def simulate_data_circle(n, r2, dim, random_seed = 2023):
    random.seed(random_seed)
    np.random.seed(random_seed)

    v_list1 = GP.generate_grids(dimensions = 2, num_grids = dim, grids_lim = np.array([-1, 1]))
    true_beta1 = (v_list1[:, 0] ** 2 + v_list1[:, 1] ** 2) < 0.25
    img1 = np.random.randn(n * dim ** 2).reshape((n, dim ** 2))

    mean_Y = 8 * np.sin(8 * img1 @ true_beta1 / np.sum(true_beta1) + 1) - 5
    true_sigma2 = np.var(mean_Y) * (1 - r2) / r2
    Y = mean_Y + np.random.normal(size = n, scale = np.sqrt(true_sigma2))
    
    return v_list1, true_beta1, img1, Y

def simulate_data_circle_square(n, r2, dim1, dim2, random_seed = 2025):
    random.seed(random_seed)
    np.random.seed(random_seed)

    v_list1 = GP.generate_grids(dimensions = 2, num_grids = dim1, grids_lim = np.array([-1, 1]))
    v_list2 = GP.generate_grids(dimensions = 2, num_grids = dim2, grids_lim = np.array([-1, 1]))

    true_beta1 = ((v_list1[:, 0] ** 2 + v_list1[:, 1] ** 2) < 0.25) + 0
    true_beta2 = (abs(v_list2[:, 0]) < 0.5) * (abs(v_list2[:, 1]) < 0.5)

    img1 = np.random.randn(n * dim1 ** 2).reshape((n, dim1 ** 2))
    img2 = np.random.randn(n * dim2 ** 2).reshape((n, dim2 ** 2))

    mean_Y = 3 * (np.sin(10 * img1 @ true_beta1 / np.sum(true_beta1) + 0.8) + np.exp(16 * img2 @ true_beta2 / np.sum(true_beta2)) - 1.7)
    true_sigma2 = np.var(mean_Y) * (1 - r2) / r2
    Y = mean_Y + np.random.normal(size = n, scale = np.sqrt(true_sigma2))
    
    return v_list1, v_list2, true_beta1, true_beta2, img1, img2, Y