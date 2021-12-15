import numpy as np
import pandas
import torch
from scipy import stats
from poincare_helper import poincare_norm

eps = 1e-5
boundary = 1 - eps

def poincare_translation(v, x):
    """
    Computes the translation of x  when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((xsq + 2 * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - vsq) * x
    b = xsq * vsq + 2 * xv + 1
    return np.dot(np.diag(1. / b), a)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--latent_file', default='final_latent.txt')
    parser.add_argument('--iroot', default=840, type=int, 
                        help='index of root cell')
    
    args = parser.parse_args()

    data = np.loadtxt(args.latent_file, delimiter=",")
    #### tranform root
    data_proj = poincare_translation(-data[args.iroot], data)
    np.savetxt("transformed_final_latent.txt", data_proj)

    #### Poincare pseodotime
    data_pt = poincare_norm(torch.tensor(data_proj, dtype=torch.float64)).data.numpy()
    data_pt = np.squeeze(data_pt)
    np.savetxt("transformed_Poincare_pseudotime.txt", data_pt)

    #### pairwise Poincare distance
    distances = hyp_distances(torch.tensor(data))[2].numpy()
    np.savetxt("pairwise_distances.txt", distances)