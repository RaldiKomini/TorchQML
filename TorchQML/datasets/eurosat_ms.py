from TorchQML.datasets.folder import pickdata, dataset_to_matrix
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
import torch

import tifffile
import numpy as np
import os


def ms_loader(path):
    """
    Load a EuroSAT_MS .tif as a [13, H, W] float32 tensor.
    """
    arr = tifffile.imread(path)

    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for MS image, got shape {arr.shape} for {path}")

    # [C, H, W] with C=13
    if arr.shape[0] == 13:
        # already [13, H, W]
        arr_chw = arr
    elif arr.shape[2] == 13:
        # [H, W, 13] -> [13, H, W]
        arr_chw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected MS shape {arr.shape} for {path}")

    arr_chw = arr_chw.astype("float32")


    arr_chw = arr_chw / 10000.0

    return torch.from_numpy(arr_chw)

def data_to_flat(root, used, ntr= 1000, nval = 300, nte = 300, seed = None):
    """Load EuroSAT-MS samples and flatten each image to one vector."""
    train_s, val_s, test_s = pickdata(root, used, ntr, nval, nte, seed)

    trainds = ImageFolder(root = root, loader=ms_loader)
    valds = ImageFolder(root = root, loader=ms_loader)
    testds = ImageFolder(root= root, loader=ms_loader)

    trainds.samples = train_s
    valds.samples = val_s
    testds.samples = test_s

    Xtr, ytr = dataset_to_matrix(trainds)
    Xval, yval = dataset_to_matrix(valds)
    Xte, yte = dataset_to_matrix(testds)


    train_set = TensorDataset(Xtr, ytr)
    val_set = TensorDataset(Xval, yval)
    test_set = TensorDataset(Xte, yte)

    return train_set, val_set, test_set

from sklearn.decomposition import PCA

def flat_to_pca(train_set, val_set, test_set, n_comp = 32, save_dir = None, seed = None):
    """Fit PCA on the train split and transform all splits."""
    Xtr, ytr = train_set.tensors
    Xval, yval = val_set.tensors
    Xte, yte = test_set.tensors

    Xtrnp = Xtr.cpu().numpy()
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    pca.fit(Xtrnp)
    print(f"PCA to {n_comp}D keeps {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

    def pca_tensor(pca, X):
        """Transform one tensor split with the fitted PCA."""
        Xnp = X.cpu().numpy()
        Xpca = pca.transform(Xnp)
        return torch.from_numpy(Xpca).float()

    Xtr_pca = pca_tensor(pca, Xtr)
    Xval_pca = pca_tensor(pca, Xval)
    Xte_pca = pca_tensor(pca, Xte)



    mean = Xtr_pca.mean(dim=0, keepdim=True)
    std  = Xtr_pca.std(dim=0, keepdim=True) + 1e-6
    Ztr_n  = (Xtr_pca  - mean) / std
    Zval_n = (Xval_pca - mean) / std
    Zte_n  = (Xte_pca  - mean) / std

    s = 2.5
    Ztr_n  = torch.tanh(Ztr_n  / s)
    Zval_n = torch.tanh(Zval_n / s)
    Zte_n  = torch.tanh(Zte_n  / s)


    train_pca_set = TensorDataset(Ztr_n,  ytr)
    val_pca_set   = TensorDataset(Zval_n, yval)
    test_pca_set  = TensorDataset(Zte_n,  yte)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(train_pca_set, os.path.join(save_dir, "train_ds.pt"))
        torch.save(val_pca_set,   os.path.join(save_dir, "val_ds.pt"))
        torch.save(test_pca_set,  os.path.join(save_dir, "test_ds.pt"))
        print(f"Saved TensorDatasets to {save_dir}")

    return train_pca_set, val_pca_set, test_pca_set



def data_to_pca(root, used, ntr=1000, nval=300, nte=300, ncomp=64, save_dir = None, seed = None):
    """Load EuroSAT-MS data and return PCA-compressed TensorDatasets."""
    train_set, val_set, test_set = data_to_flat(root, used, ntr=ntr, nval=nval, nte=nte, seed = seed)
    tr, va, te = flat_to_pca(train_set, val_set, test_set, n_comp=ncomp, save_dir=save_dir, seed = seed)
    return tr, va, te

def data_to_fpca(root, used, ntr=1000, nval=300, nte=300, ncomp=32):
    """Compatibility wrapper for flattened PCA data."""
    train_set, val_set, test_set = data_to_flat(root, used, ntr=ntr, nval=nval, nte=nte)
    tr, va, te = flat_to_pca(train_set, val_set, test_set, n_comp=ncomp)
    return tr, va, te


def data_to_ms(root, used, ntr=1000, nval=300, nte=300, seed=None):
    """
    Return TensorDatasets with `X` as [N, 13, H, W] float32 and `y` as labels.
    """
    train_s, val_s, test_s = pickdata(root, used, ntr, nval, nte, seed)

    trainds = ImageFolder(root=root, loader=ms_loader)
    valds   = ImageFolder(root=root, loader=ms_loader)
    testds  = ImageFolder(root=root, loader=ms_loader)

    trainds.samples = train_s
    valds.samples   = val_s
    testds.samples  = test_s

    def to_tensors(ds):
        """Load all paths in an ImageFolder split into tensors."""
        xs, ys = [], []
        for path, y in ds.samples:
            xs.append(ds.loader(path))
            ys.append(int(y))
        X = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        return X, y

    Xtr, ytr = to_tensors(trainds)
    Xval, yval = to_tensors(valds)
    Xte, yte = to_tensors(testds)

    return TensorDataset(Xtr, ytr), TensorDataset(Xval, yval), TensorDataset(Xte, yte)
