# scDHMap

Understanding the developmental process is a critical step in single-cell analysis. This repo proposes scDHMap, a model-based deep learning approach to visualize the complex hierarchical structures of single-cell sequencing data in a low dimensional hyperbolic space. ScDHMap can be used for various dimensionality reduction tasks including revealing trajectory branches, batch correction, and denoising highly dropout counts.

![alt text](https://github.com/ttgump/scDHMap/blob/main/network.png?raw=True)

**Requirements**

Python: 3.9.5<br/>
PyTorch: 1.9.1<br/>
Scanpy: 1.7.2<br/>
Numpy: 1.21.2<br/>
sklearn: 0.24.2<br/>
Scipy: 1.6.3<br/>
Pandas: 1.2.5<br/>
h5py: 3.2.1<br/>
Optional: harmonypy (https://github.com/slowkow/harmonypy)

**Usage**

python run_scDHMap.py --data_file data.h5

python run_scDHMap_batch.py --data_file data.h5

The real single cell datasets used in this study can be found: https://figshare.com/s/64694120e3d2b87e21c3

In the data.h5 file, cell-by-gene count matrix is stored in "X". For dataset with batches, batch IDs are one-hot encoded matrix and stored in "Y".

**Parameters**

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
--prob: dropout probability in encoder and decoder layers, default = 0.<br/>
--perplexity: perplexity of the t-SNE regularization, default = 30.<br/>
--final_latent_file: file name to output final latent Poincare representations, default = final_latent.txt.<br/>
--final_mean_file: file name to output denoised counts, default = denoised_mean.txt.<br/>

Visualization demo of the Paul data (Credit: Joshua Ortiga): https://hosua.github.io/scDHMap-visual/article/2022/11/09/paul-data-visualization.html
