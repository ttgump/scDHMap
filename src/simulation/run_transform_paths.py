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

def _poinc_to_minsk(points):
    minsk_points = np.zeros((points.shape[0],3))
    minsk_points[:,0] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[0]/(1-v[0]**2-v[1]**2))
    minsk_points[:,1] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[1]/(1-v[0]**2-v[1]**2))
    minsk_points[:,2] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: (1+v[0]**2+v[1]**2)/(1-v[0]**2-v[1]**2))
    return minsk_points

def _minsk_to_poinc(points):
    poinc_points = np.zeros((points.shape[0],2))
    poinc_points[:,0] = points[:,0]/(1+points[:,2])
    poinc_points[:,1] = points[:,1]/(1+points[:,2])
    return poinc_points

def _hyperbolic_centroid(points):
    minsk_points = _poinc_to_minsk(points)
    minsk_centroid = np.mean(minsk_points,axis=0)
    normalizer = np.sqrt(np.abs(minsk_centroid[0]**2+minsk_centroid[1]**2-minsk_centroid[2]**2))
    minsk_centroid = minsk_centroid/normalizer
    return _minsk_to_poinc(minsk_centroid.reshape((1,3)))[0]

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--latent_file', default='final_latent_1.txt')
    parser.add_argument('--path_file', default='sim_1_progressions.csv')

    args = parser.parse_args()

    data = np.loadtxt(args.latent_file, delimiter=",")
    path = pandas.read_csv(args.path_file, index_col=0)

    for p in np.unique(path["group"]):
        p_path = path[path["group"]==p]
        p_path_start = p_path[p_path["percentage"]==np.min(path["percentage"])]
        p_data = data[p_path.index.values-1]
        p_data_start = data[p_path_start.index.values-1]
        p_data_start = _hyperbolic_centroid(p_data_start)
        p_data_proj = poincare_translation(-p_data_start, p_data)
        np.savetxt(str(p)+"_"+args.latent_file, p_data_proj)

        p_data_pt = poincare_norm(torch.tensor(p_data_proj, dtype=torch.float64)).data.numpy()
        p_data_pt = np.squeeze(p_data_pt)
        np.savetxt(str(p)+"_pseudotime_"+args.latent_file, p_data_pt)
        c_pt = stats.spearmanr(p_path["percentage"].values, p_data_pt)[0]

        print("Path", p, ", pt cor", c_pt)