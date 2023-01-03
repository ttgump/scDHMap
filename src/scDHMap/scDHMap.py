import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from layers import NBLoss, ZINBLoss, MeanAct, DispAct
from tsne_helper import compute_gaussian_perplexity
from poincare_helper import *
from lorentzian_helper import *
from embedding_quality_score import get_quality_metrics
import numpy as np
import math, os
from sklearn.metrics import pairwise_distances
from wrapped_normal import HyperbolicWrappedNorm

eps = 1e-6
weight_decay = 1e-3

def buildNetwork(layers, type, activation="relu", prob=0.):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        if prob > 0:
            net.append(nn.Dropout(p=prob))
    return nn.Sequential(*net)


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir='./'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class scDHMap(nn.Module):
    def __init__(self, input_dim, encodeLayer=[], decodeLayer=[], batch_size=512,
            activation="relu", z_dim=2, alpha=1., beta=1., gamma=1., perplexity=[30.], 
            prob=0., likelihood_type="zinb", device="cuda"):
        super(scDHMap, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.activation = activation
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma          # If gamma = 1, the Cauchy kernel will reduce to the Student's t-kernel
        self.perplexity = perplexity
        self.prob = prob
        self.likelihood_type = likelihood_type
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation, prob=prob)
        self.decoder = buildNetwork([z_dim+1]+decodeLayer, type="decode", activation=activation, prob=prob)
        self.enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self.enc_sigma = nn.Sequential(nn.Linear(encodeLayer[-1], z_dim), nn.Softplus())
        self.dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.device = device

        self.nb_loss = NBLoss().to(self.device)
        self.zinb_loss = ZINBLoss().to(self.device)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def aeForward(self, x):
        h = self.encoder(x)

        tmp = self.enc_mu(h)
        z_mu = self._polar_project(tmp)
        z_sigma_square = self.enc_sigma(h).clamp(min=1e-6, max=15)
        q_z = HyperbolicWrappedNorm(z_mu, z_sigma_square)
        z = q_z.sample()

        h = self.decoder(z)
        mean = self.dec_mean(h)
        disp = self.dec_disp(h)
        pi = self.dec_pi(h)

        return q_z, z, z_mu, mean, disp, pi

    def _polar_project(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_unit = x / torch.clamp(x_norm, min=eps)
        x_norm = torch.clamp(x_norm, 0, 32)
        z = torch.cat((torch.cosh(x_norm), torch.sinh(x_norm) * x_unit), dim=1)
        return z

    def tsne_repel(self, z, p):
        n = z.size()[0]

        ### pairwise distances
#        num = lorentz_distance_mat(z, z)**2
#        num = torch.pow(1.0 + num, -1)
        num = (lorentz_distance_mat(z, z)/self.gamma)**2
        num = 1/self.gamma/(1.0 + num)
        p = p / torch.unsqueeze(torch.sum(p, dim=1), 1)

        attraction = p * torch.log(num)
        attraction = -torch.sum(attraction)

        den = torch.sum(num, dim=1) - 1
        repellant = torch.sum(torch.log(den))

        return (repellant + attraction) / n

    def KLD(self, q_z, z):
        loc = torch.cat((torch.ones(z.shape[0], 1), torch.zeros(z.shape[0], self.z_dim)), dim=-1).to(self.device)
        p_z = HyperbolicWrappedNorm(loc, torch.ones(z.shape[0], self.z_dim).to(self.device))

        kl = q_z.log_prob(z) - p_z.log_prob(z)
        return torch.mean(kl)

    def encodeBatch(self, X):
        """
        Output latent representations and project to 2D Poincare ball for visualization
        """

        self.to(self.device)

        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/self.batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]
            inputs = Variable(xbatch)
            _, _, z, _, _, _ = self.aeForward(inputs)
            z = lorentz2poincare(z)
            encoded.append(z.data.cpu().detach())

        encoded = torch.cat(encoded, dim=0)
        self.train()
        return encoded.numpy()

    def decodeBatch(self, X):
        """
        Output denoised counts
        """

        self.to(self.device)

        decoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/self.batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]
            inputs = Variable(xbatch)
            _, _, _, mean, _, _ = self.aeForward(inputs)
            decoded.append(mean.data.cpu().detach())

        decoded = torch.cat(decoded, dim=0)
        self.train()
        return decoded.numpy()

    def pretrain_autoencoder(self, X, X_raw, size_factor, lr=0.001, pretrain_iter=400, ae_save=True, ae_weights="AE_weights.pth.tar"):
        """
        Pretrain the model with the ZINB/NB hyperbolic VAE only.

        Parameters:
        -----------
        X: array_like, shape (n_samples, n_features)
            The normalized raw counts
        X_raw: array_like, shape (n_samples, n_features)
            The raw counts, which need for the ZINB/NB loss
        size_factor: array_like, shape (n_samples)
            The size factor of each sample, which need for the ZINB/NB loss
        lr: float, defalut = 0.001
            Learning rate for the opitimizer
        pretrain_iter: int, default = 400
            Pretrain iterations
        ae_save: bool, default = True
            Whether to save the pretrained weights
        ae_weights: str
            Directory name to save the model weights
        """

        self.to(self.device)
        num = X.shape[0]
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)

        print("Pretraining stage")
        for epoch in range(pretrain_iter):
            loss_reconn_val = 0
            loss_kld_val = 0
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                q_z, z, z_mu, mean_tensor, disp_tensor, pi_tensor = self.aeForward(x_tensor)
                if self.likelihood_type == "nb":
                    loss_reconn = self.nb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, scale_factor=sf_tensor)
                elif self.likelihood_type == "zinb":
                    loss_reconn = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                else:
                    raise Exception("The likelihood type must be one of 'zinb' or 'nb'")
                loss_kld = self.KLD(q_z, z)
                loss = loss_reconn + loss_kld
                self.zero_grad()
                loss.backward()
                optimizer.step()

                loss_reconn_val += loss_reconn.item() * len(x_batch)
                loss_kld_val += loss_kld.item() * len(x_batch)
                loss_val += loss.item() * len(x_batch)
            loss_reconn_val = loss_reconn_val/num
            loss_kld_val = loss_kld_val/num
            loss_val = loss_val/num

            print('Pretraining epoch {}, Total loss:{:.8f}, reconn loss:{:.8f}, KLD loss:{:.8f}'.format(epoch+1, loss_val, loss_reconn_val, loss_kld_val))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optim_adam_state_dict': optimizer.state_dict()}, ae_weights)


    def train_model(self, X, X_raw, size_factor, X_pca, X_true_pca=None, lr=0.001, maxiter=5000, minimum_iter=0, patience=150, save_dir=""):
        """
        Train the model with the ZINB/NB hyperbolic VAE and the hyberbolic t-SNE regularization.

        Parameters:
        -----------
        X: array_like, shape (n_samples, n_features)
            The normalized raw counts
        X_raw: array_like, shape (n_samples, n_features)
            The raw counts, which need for the ZINB/NB loss
        size_factor: array_like, shape (n_samples)
            The size factor of each sample, which need for the ZINB/NB loss
        X_pca: array_like, shape (n_samples, n_PCs)
            The principal components of the analytic Pearson residual normalized raw counts
        X_true_pca: array_like, shape (n_samples, n_PCs)
            The principal components of the analytic Pearson residual normalized true counts
            This is used for evaluation of simulation experiments; for real data, it can be set to None
        lr: float, defalut = 0.001
            Learning rate for the opitimizer
        maxiter: int, default = 5000
            Maximum number of iterations
        minimum_iter: int, default = 0
            Minimum number of iterations
        patience: int, default = 150
            Patience for the early stop
        save_dir: str
            Directory name to save the model weights
        """

        self.to(self.device)
        X = torch.tensor(X)
        X_raw = torch.tensor(X_raw)
        size_factor = torch.tensor(size_factor)
        num = X.shape[0]
        sample_indices = np.arange(num)
        num_batch = int(math.ceil(1.0*num/self.batch_size))

        perplexity = np.array(self.perplexity).astype(np.double)

#        dist_X_pca = pairwise_distances(X_pca, metric="euclidean").astype(np.double)

        print("Training...")

        early_stopping = EarlyStopping(patience=patience, outdir=save_dir)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)

        for epoch in range(maxiter):
            loss_reconn_val = 0
            loss_tsne_val = 0
            loss_kld_val = 0
            loss_val = 0
            np.random.shuffle(sample_indices)
            for batch_idx in range(num_batch):
                batch_indices = sample_indices[batch_idx*self.batch_size : min((batch_idx+1)*self.batch_size, num)]

                x_batch = X[batch_indices]
                x_raw_batch = X_raw[batch_indices]
                sf_batch = size_factor[batch_indices]
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)

#                dist_X_pca_batch = dist_X_pca[batch_indices][:, batch_indices]
                dist_X_pca_batch = pairwise_distances(X_pca[batch_indices], metric="euclidean").astype(np.double)
                p_batch = compute_gaussian_perplexity(dist_X_pca_batch, perplexities=perplexity)
                p_batch = torch.tensor(p_batch)
                p_tensor = Variable(p_batch).to(self.device)

                q_z, z, z_mu, mean_tensor, disp_tensor, pi_tensor = self.aeForward(x_tensor)

                if self.likelihood_type == "nb":
                    loss_reconn = self.nb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, scale_factor=sf_tensor)
                elif self.likelihood_type == "zinb":
                    loss_reconn = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                else:
                    raise Exception("The likelihood type must be one of 'zinb' or 'nb'")
                loss_tsne = self.tsne_repel(z_mu, p_tensor)
                loss_kld = self.KLD(q_z, z)

                loss = loss_reconn + self.alpha * loss_tsne + self.beta * loss_kld 

                self.zero_grad()
                loss.backward()
                optimizer.step()

                loss_reconn_val += loss_reconn.item() * len(x_batch)
                loss_tsne_val += loss_tsne.item() * len(x_batch)
                loss_kld_val += loss_kld.item() * len(x_batch)
                loss_val += loss.item() * len(x_batch)

            loss_reconn_val = loss_reconn_val/num
            loss_tsne_val = loss_tsne_val/num
            loss_kld_val = loss_kld_val/num
            loss_val = loss_val/num

            print('Training epoch {}, Total loss:{:.8f}, reconn loss:{:.8f}, t-SNE loss:{:.8f}, KLD loss:{:.8f}'.format(epoch+1, loss_val, loss_reconn_val, loss_tsne_val, loss_kld_val))

            if X_true_pca is not None and epoch > 0 and epoch % 40 == 0:
                epoch_latent = self.encodeBatch(X.to(self.device))
                QM_ae = get_quality_metrics(X_true_pca, epoch_latent, distance='P')

            if epoch+1 >= minimum_iter:
                early_stopping(loss_tsne_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch))
                    break
