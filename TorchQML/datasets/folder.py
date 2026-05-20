from torchvision.datasets import ImageFolder
import random
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


# ---- splitting / sampling ----
def pickdata(root, used, ntr=1000, nval=300, nte=300, seed=None):
    """Pick balanced train/val/test image paths for the requested classes."""
    rng = random.Random(seed)

    base_ds = ImageFolder(root=root)
    used_idx = [base_ds.class_to_idx[u] for u in used]
    old_to_new = {old: new for new, old in enumerate(used_idx)}

    class_div = {old_lab: [] for old_lab in used_idx}
    for path, lab in base_ds.samples:
        if lab in used_idx:
            class_div[lab].append(path)

    def split(lst, ntr, nval, nte):
        """Shuffle one class split and take train/val/test slices."""
        lst = list(lst)
        rng.shuffle(lst)
        return lst[:ntr], lst[ntr:ntr+nval], lst[ntr+nval:ntr+nval+nte]

    train_samples, val_samples, test_samples = [], [], []
    for old_lab in used_idx:
        new_lab = old_to_new[old_lab]
        tr, va, te = split(class_div[old_lab], ntr, nval, nte)
        train_samples += [(p, new_lab) for p in tr]
        val_samples   += [(p, new_lab) for p in va]
        test_samples  += [(p, new_lab) for p in te]

    return train_samples, val_samples, test_samples




def dataset_to_matrix(ds, batch_size=64):
    """Flatten a dataset into one feature matrix and one label vector."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    X_list, y_list = [], []

    for xb, yb in loader:
        b = xb.size(0)
        X_list.append(xb.view(b, -1))  # flatten: [B, C*H*W]
        y_list.append(yb)

    X = torch.cat(X_list, dim=0)      # [N, D]
    y = torch.cat(y_list, dim=0)      # [N]
    return X, y




# ---- feature extraction ----
def image_to_features(img):
    """Extract a small hand-built color, grid, and edge feature vector."""

    C, H, W = img.shape
    flat = img.view(C, -1)
    rgbm = flat.mean(dim = 1)
    rgbs = flat.std(dim = 1)

    pooled = F.adaptive_avg_pool2d(img, output_size = (4, 4))
    grid = pooled.view(-1)

    gray = img.mean(dim = 0, keepdim = True)


    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

    sobel_x = sobel_x.to(img.device, img.dtype)
    sobel_y = sobel_y.to(img.device, img.dtype)

    gray_batched = gray.unsqueeze(0)

    gx = F.conv2d(gray_batched, sobel_x, padding=1)
    gy = F.conv2d(gray_batched, sobel_y, padding=1)

    grad_mag = torch.sqrt(gx**2 + gy**2)[0, 0]       # [H, W]

    edge_mean = grad_mag.mean().unsqueeze(0)         # [1]
    edge_std  = grad_mag.std().unsqueeze(0)          # [1]

    feat = torch.cat([rgbm, rgbs, grid, edge_mean, edge_std], dim=0)  # [56]

    return feat
