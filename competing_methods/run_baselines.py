#### This is a sample script to run competing methods: PaCMap, t-SNE, UMAP, PHATE, PoincareMap


import numpy as np
import h5py
from embedding_quality_score import get_quality_metrics

from single_cell_tools import *
from sklearn.decomposition import PCA
from preprocess import read_dataset, normalize, pearson_residuals
import pacmap
from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
import umap
import scanpy as sc
import phate

#### import PoincareMap
from poincare_maps import *
from main import *
import torch


data_mat = h5py.File("Paul_Cell.h5")
x = np.array(data_mat['X'])
data_mat.close()

#### Select top 1000 genes
importantGenes = geneSelection(x, n=1000, plot=False)
x = x[:, importantGenes]

#### Analytic Pearson residuals normalization and PCA
X_normalized = pearson_residuals(x, theta=100)
X_pca = PCA(n_components=50, svd_solver='full').fit_transform(X_normalized)


#### Run PaCMap
pacmap_embedding = pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
X_pacmap_2 = pacmap_embedding.fit_transform(X_pca, init="pca")
np.savetxt("PaCMAP_2D.txt", X_pacmap_2, delimiter=",")
print("PaCMap 2D")
get_quality_metrics(X_pca, X_pacmap_2, distance='E')


#### Run t-SNE
tsne_embedding = TSNE(
                    perplexity=30,
                    initialization="pca",
                    metric="euclidean",
                    n_jobs=8,
                    random_state=42,
                )
X_tsne_2 = tsne_embedding.fit(X_pca)
np.savetxt("tsne_2D.txt", X_tsne_2, delimiter=",")
print("t-SNE 2D")
get_quality_metrics(X_pca, X_tsne_2, distance='E')


#### Run UMAP
umap_embedding = umap.UMAP()
X_umap_2 = umap_embedding.fit_transform(X_pca)
np.savetxt("umap_2D.txt", X_umap_2, delimiter=",")
print("UMAP 2D")
get_quality_metrics(X_pca, X_umap_2, distance='E')


#### Run PHATE
phate_operator = phate.PHATE()
X_phate_2 = phate_operator.fit_transform(X_pca)
np.savetxt("phate_2D.txt", X_phate_2, delimiter=",")
print("Phate 2D")
get_quality_metrics(X_pca, X_phate_2, distance='E')


#### Run PoincareMap
X_pca_tensor = torch.DoubleTensor(X_pca)
poincare_coord, _ = compute_poincare_maps(X_pca_tensor, None,
                        'pmap_res/Bif',
                        mode='features', k_neighbours=15, 
                        distlocal='minkowski', sigma=1.0, gamma=2.0,
                        color_dict=None, epochs=1000,
                        batchsize=-1, lr=0.1, earlystop=0.0001, cuda=0)


np.savetxt("PoincareMap_2D.txt", poincare_coord, delimiter=",")
print("PoincareMap")
get_quality_metrics(X_pca, poincare_coord, distance='P')