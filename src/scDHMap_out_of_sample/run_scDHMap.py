import math, os
from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scDHMap import scDHMap
from embedding_quality_score import get_quality_metrics
import numpy as np
from single_cell_tools import geneSelection
from sklearn.decomposition import PCA
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize_training, normalize_testing, pearson_residuals

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Single-cell deep hierarchical map: hyperbolic embedding of single-cell genomics data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--data_file', default='Splatter_simulate_1.h5')
    parser.add_argument('--select_genes', default=1000, type=int)
    parser.add_argument('--n_PCA', default=50, type=int)
    parser.add_argument('--pretrain_iter', default=400, type=int)
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--minimum_iter', default=0, type=int)
    parser.add_argument('--patience', default=150, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--alpha', default=1000., type=float,
                        help='coefficient of the SNE loss')
    parser.add_argument('--beta', default=10., type=float,
                        help='coefficient of the KLD loss')
    parser.add_argument('--prob', default=0, type=float,
                        help='dropout probability')
    parser.add_argument('--perplexity', nargs="+", default=[30.], type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--ae_weights_file', default="AE_weights.pth.tar")
    parser.add_argument('--save_dir', default='ES_model/')
    parser.add_argument('--train_cells_files', default='train_index.txt')
    parser.add_argument('--test_cells_files', default='test_index.txt')
    parser.add_argument('--pretrain_latent_file', default='ae_latent.txt')
    parser.add_argument('--pretrain_test_latent_file', default='ae_test_latent.txt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--final_test_latent_file', default='final_test_latent.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    # proportion of the training set
    trainProb = 0.9

    # read data, true counts are used for calculating Q values
    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    x_true = np.array(data_mat['X_true'])
    data_mat.close()

    importantGenes = geneSelection(x, n=args.select_genes, plot=False)
    x = x[:, importantGenes]
    x_true = x_true[:, importantGenes]

    # split training and testing sets
    indx = np.arange(x.shape[0])
    np.random.shuffle(indx)
    train_cell_indx = indx[0:int(np.ceil(trainProb*x.shape[0]))]
    test_cell_indx = indx[int(np.ceil(trainProb*x.shape[0])):]

    # save cell index of training and testing sets
    np.savetxt(args.train_cells_files, train_cell_indx, fmt="%i")
    np.savetxt(args.test_cells_files, test_cell_indx, fmt="%i")

    x_train = x[train_cell_indx]
    x_test = x[test_cell_indx]
    x_true_train = x_true[train_cell_indx]
    x_true_test = x_true[test_cell_indx]

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x_train)

    adata = normalize_training(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    adata_test = sc.AnnData(x_test)

    # normalize testing set by the mean and SD of the training set
    adata_test = normalize_testing(adata_test, np.median(adata.obs.n_counts),
                      adata.var["mean"], adata.var["std"],
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    X_train_normalized = pearson_residuals(x_train, theta=100)
    X_true_train_normalized = pearson_residuals(x_true_train, theta=100)

    X_train_pca = PCA(n_components=args.n_PCA, svd_solver='full').fit_transform(X_train_normalized)
    X_true_train_pca = PCA(n_components=args.n_PCA, svd_solver='full').fit_transform(X_true_train_normalized)


    X_test_normalized = pearson_residuals(x_test, theta=100)
    X_true_test_normalized = pearson_residuals(x_true_test, theta=100)

    X_test_pca = PCA(n_components=args.n_PCA, svd_solver='full').fit_transform(X_test_normalized)
    X_true_test_pca = PCA(n_components=args.n_PCA, svd_solver='full').fit_transform(X_true_test_normalized)

    print(args)

    print(x.shape)
    print(x_true.shape)
    print(X_train_pca.shape)
    print(X_true_train_pca.shape)

    model = scDHMap(input_dim=adata.n_vars, encodeLayer=[128, 64, 32, 16], decodeLayer=[16, 32, 64, 128], 
            batch_size=args.batch_size, activation="elu", z_dim=2, alpha=args.alpha, beta=args.beta, 
            perplexity=args.perplexity, prob=args.prob, device=args.device).to(args.device)

    print(str(model))

    t0 = time()

    # pretraining
    if args.ae_weights is None:
        model.pretrain_autoencoder(adata.X.astype(np.float64), adata.raw.X.astype(np.float64), adata.obs.size_factors.astype(np.float64), 
            lr=args.lr, pretrain_iter=args.pretrain_iter, ae_save=True, ae_weights=args.ae_weights_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))
    ae_latent = model.encodeBatch(torch.tensor(adata.X).double().to(args.device))
    QM_ae = get_quality_metrics(X_true_train_pca, ae_latent, distance='P')

    np.savetxt(args.pretrain_latent_file, ae_latent, delimiter=",")

    print('Pretraining testing Q')
    ae_test_latent = model.encodeBatch(torch.tensor(adata_test.X).double().to(args.device))
    QM_ae_test = get_quality_metrics(X_true_test_pca, ae_test_latent, distance='P')

    np.savetxt(args.pretrain_test_latent_file, ae_test_latent, delimiter=",")

    # training the model with the t-SNE regularization
    model.train_model(adata.X.astype(np.float64), adata.raw.X.astype(np.float64), adata.obs.size_factors.astype(np.float64), X_train_pca.astype(np.float64), X_true_train_pca.astype(np.float64),
                    lr=args.lr, maxiter=args.maxiter, minimum_iter=args.minimum_iter,
                    patience=args.patience, save_dir=args.save_dir)
    print('Training time: %d seconds.' % int(time() - t0))

    final_latent = model.encodeBatch(torch.tensor(adata.X).double().to(args.device))
    QM_ae = get_quality_metrics(X_true_train_pca, final_latent, distance='P')

    np.savetxt(args.final_latent_file, final_latent, delimiter=",")


    print('Final testing Q')
    final_test_latent = model.encodeBatch(torch.tensor(adata_test.X).double().to(args.device))
    QM_ae_test = get_quality_metrics(X_true_test_pca, final_test_latent, distance='P')

    np.savetxt(args.final_test_latent_file, final_test_latent, delimiter=",")