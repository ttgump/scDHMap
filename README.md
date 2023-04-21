# scDHMap

Understanding the developmental process is a critical step in single-cell analysis. This repo proposes scDHMap, a model-based deep learning approach to visualize the complex hierarchical structures of single-cell sequencing data in a low dimensional hyperbolic space. ScDHMap can be used for various dimensionality reduction tasks including revealing trajectory branches, batch correction, and denoising highly dropout counts.

## Table of contents
- [Network diagram](#diagram)
- [Requirements](#requirements)
- [Usage](#usage)
- [Reference](#reference)
- [Visualization demo](#demo)
- [Contact](#contact)

## <a name="diagram"></a>Network diagram
![alt text](https://github.com/ttgump/scDHMap/blob/main/network.png?raw=True)

## <a name="requirements"></a>Requirements
Python: 3.9.5<br/>
PyTorch: 1.9.1 (https://pytorch.org)<br/>
Scanpy: 1.7.2 (https://scanpy.readthedocs.io/en/stable)<br/>
Numpy: 1.21.2 (https://numpy.org)<br/>
sklearn: 0.24.2 (https://scikit-learn.org/stable)<br/>
Scipy: 1.6.3 (https://scipy.org)<br/>
Pandas: 1.2.5 (https://pandas.pydata.org)<br/>
h5py: 3.2.1 (https://pypi.org/project/h5py)<br/>
Optional: harmonypy (https://github.com/slowkow/harmonypy)

## <a name="usage"></a>Usage
For single-cell count data:

```sh
python run_scDHMap.py --data_file data.h5
```

For single-cell count data from multiple batches:

```sh
python run_scDHMap_batch.py --data_file data.h5
```

The real single cell datasets used in this study can be found: https://figshare.com/s/64694120e3d2b87e21c3

In the data.h5 file, cell-by-gene count matrix is stored in "X". For dataset with batches, batch IDs are one-hot encoded matrix and stored in "Y".

## <a name="parameters"></a>Parameters
--batch_size: batch size, default = 512.<br/>
--data_file: data file name.<br/>
--select_genes: number of selected genes for embedding analysis, default = 1000.<br/>
--n_PCA: number of principle components for the t-SNE part, default = 50.<br/>
--pretrain_iter: number of pretraining iterations, default = 400.<br/>
--maxiter: number of max iterations during training stage, default = 5000.<br/>
--patience: patience in training stage, default = 150.<br/>
--lr: learning rate in the Adam optimizer, default = 0.001.<br/>
--alpha: coefficient of the t-SNE regularization, default = 1000. The choice of alpha is to balance the number of genes in the ZINB reconstruction loss.<br/>
--beta: coefficient of the wrapped normal KLD loss, default = 10. If points in the embedding are all stacked near the boundary of the Poincare disk, you may choose a larger beta value.<br/>
--gamma: coefficient of the Cauchy kernel, default = 1. Larger gamma means greater repulsive force between non-neighboring points. Please note that larger gamma values will push points to the boundary of the Poincare ball. If wanting to visualize better, user can choose larger beta values for using larger gamma values. In our experience, the KLD loss value < 10 during training t-SNE loss step will result to nice visualization. See the effect of different gamma's in Supplementary Figure S23 in the manuscript.<br/>
--prob: dropout probability in encoder and decoder layers, default = 0.<br/>
--perplexity: perplexity of the t-SNE regularization, default = 30.<br/>
--final_latent_file: file name to output final latent Poincare representations, default = final_latent.txt.<br/>
--final_mean_file: file name to output denoised counts, default = denoised_mean.txt.<br/>

## <a name="reference"></a>Reference
Tian T., Cheng Z., Xiang L., Zhi W., & Hakon H. (2023). Complex hierarchical structures in single-cell genomics data unveiled by deep hyperbolic manifold learning. Genome Research. https://doi.org/10.1101/gr.277068.122

## <a name="demo"></a>Visualization demo
Visualization demo of the Paul data (Credit: Joshua Ortiga)

https://hosua.github.io/scDHMap-visual/article/2022/11/09/paul-data-visualization.html

## <a name="contact"></a>Contact
Tian Tian tt72@njit.edu
