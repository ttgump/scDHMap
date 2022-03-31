# scDHMap

Understanding the developmental process is a critical step in single-cell analysis. This repo proposes scDHMap, a model-based deep learning approach to visualize the complex hierarchical structures of single-cell sequencing data in a low dimensional hyperbolic space. ScDHMap can be used for various dimensionality reduction tasks including revealing trajectory branches, batch correction, and denoising highly dropout counts.

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

--batch_size: batch size, default = 512<br/>
--data_file: data file name<br/>
--select_genes: number of selected genes for embedding analysis<br/>
--n_PCA: number of principle components for the t-SNE part<br/>
--pretrain_iter: number of pretraining iterations<br/>
--maxiter: number of max iterations during training stage<br/>
--patience: patience in training stage<br/>
--lr: learning rate in the Adam optimizer<br/>
--alpha: coefficient of the t-SNE regularization<br/>
--beta: coefficient of the wrapped normal KLD loss<br/>
--prob: dropout probability in encoder and decoder layers<br/>
--perplexity: perplexity of the t-SNE regularization
