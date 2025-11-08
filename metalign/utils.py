__all__ = ["fix_state_dict", "set_pycortex_filestore_path", "calc_cka", "calc_class_separation"]

import configparser
import os

import torch
from cortex import options
from einops import einsum
from fastcore.script import call_parse


def fix_state_dict(state_dict: dict) -> dict:
    """
    torch compile seems to add `_orig_mod.` prefix to all module names in state_dict.
    
    this function removes that prefix if it exists.

    taken from [here](https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py#L169-L174)
    """
    unwanted_prefix = '_orig_mod.' 
    for k,_ in list(state_dict.items()): 
        if k.startswith(unwanted_prefix): 
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k) 

    return state_dict

def calc_cka(X:torch.Tensor, # batch by observations by features
                  Y:torch.Tensor # batch by observations by features
                  ):
    """
    calculate centered kernel alignment (CKA) between two batched tensors
    """

    # subtract the mean of each observation
    X -= X.mean(1, keepdim=True)
    Y -= Y.mean(1, keepdim=True)

    # batched matmul
    XTX = einsum(X, X, "b o f1, b o f2 -> b f1 f2")
    YTY = einsum(Y, Y, "b o f1, b o f2 -> b f1 f2")
    YTX = einsum(Y, X, "b o f1, b o f2 -> b f1 f2")

    # l2 (frobenius) norm of YTX squared / l2 norm of XTX * l2 norm of YTY
    top = torch.linalg.matrix_norm(YTX, ord='fro', dim=(1,2)) ** 2
    bottom = torch.linalg.matrix_norm(XTX, ord='fro', dim=(1,2)) * torch.linalg.matrix_norm(YTY, ord='fro', dim=(1,2))

    return top / bottom


def set_pycortex_filestore_path(
        new_path:str # the absolute path to the desired Pycortex database directory
        ):
    """
    programmatically sets the Pycortex filestore location in the options.cfg file.
    """
    config_file_path = options.usercfg
    
    new_path_abs = os.path.abspath(new_path)
    if not os.path.isdir(new_path_abs):
        print(f"Creating new filestore directory: {new_path_abs}")
        os.makedirs(new_path_abs, exist_ok=True)

    config = configparser.ConfigParser()
    
    if os.path.exists(config_file_path):
        config.read(config_file_path)
    else:
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        
    if 'basic' not in config:
        config['basic'] = {}
        
    config['basic']['filestore'] = new_path_abs
    
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)

def calc_class_separation(
    X:torch.Tensor, # input embeddings (observations, features)
    y:torch.Tensor  # class labels (1D long tensor or 2D binary-int tensor)
):
    """
    Calculates class separation metric $R^2$ based on the ratio of between-class variance
    to total variance. In [the original paper](https://arxiv.org/abs/2010.16402), 
    this is formulated with respect to pairwise distances, but here we use the 
    equivalent variance formulation for efficiency.
    
    Handles both single-label (1D y) and multi-label (2D binary y) cases.
    """
    
    # the metric is based on cosine distance, which is proportional to squared Euclidean distance on L2-normalized vectors
    X_norm = torch.nn.functional.normalize(X.float(), p=2, dim=1)

    # use the population variance (unbiased=False) to match the paper's formulation as an average of pairwise distances
    sigma_total_sq = X_norm.var(dim=0, unbiased=False).sum()

    if torch.isclose(sigma_total_sq, torch.tensor(0.0)): return 0.0

    total_within_var = 0.
    
    if y.dim() == 1:
        # single-label case (e.g., y = [0, 1, 0, 2, ...])
        classes = torch.unique(y)
        K = len(classes)
        for k in classes:
            X_k = X_norm[y == k]
            # .var() is 0.0 for a single-item tensor if unbiased=False
            total_within_var += X_k.var(dim=0, unbiased=False).sum()

    elif y.dim() == 2:
        # multi-label case (e.g., y = [[1, 0], [0, 1], [1, 1], ...])
        K = y.shape[1]
        for k in range(K):
            mask_k = y[:, k].bool()
            if mask_k.sum() > 0: # only add variance if class has members
                X_k = X_norm[mask_k]
                total_within_var += X_k.var(dim=0, unbiased=False).sum()
    
    else:
        raise ValueError(f"y must be 1D or 2D, but got {y.dim()}D")

    if K == 0: return 0.0

    sigma_within_sq = total_within_var / K

    r_sq = 1.0 - (sigma_within_sq / sigma_total_sq)
    
    return r_sq.item()

@call_parse
def main():
    """
    cka tests, adapted from [here](https://github.com/patrickmineault/codebook_examples/blob/main/cka/test_cka_step3.py), also some class separation tests
    """
    # some config
    bs, n_obs, n_feats = 5, 20, 2
    X, Y = torch.rand(bs, n_obs, n_feats), torch.rand(bs, n_obs, n_feats)

    # trivially cka(X,X) should be 1
    assert all(torch.isclose(calc_cka(X, X), torch.ones(bs)))

    # swapping orders of columns should not change this
    X_sw = X.clone()
    X_sw[:, :, [1,0]] = X_sw[:, :, [0,1]]
    assert all(torch.isclose(calc_cka(X, X_sw), torch.ones(bs)))

    # adding a column offset shouldnt change things
    X_coffset = X.clone()
    X_coffset[:, :, [0]] += 10
    assert all(torch.isclose(calc_cka(X, X_coffset), torch.ones(bs)))

    # isotropic scaling shouldnt change cka
    cka1 = calc_cka(X,Y)
    cka2 = calc_cka(X * 23, Y * -2)
    assert all(torch.isclose(cka1,cka2))

    # if scaling isnt isotropic, cka should be different
    cka1 = calc_cka(X,Y)
    Y_noniso = Y.clone()
    Y_noniso[:, :, [0]] *= 4
    cka2 = calc_cka(X, Y_noniso)
    assert not all(torch.isclose(cka1, cka2))

    # rotations shouldnt matter 
    rotation = torch.tensor([[1, -1], [1, 1]],dtype=X.dtype) / torch.sqrt(torch.tensor(2))
    
    # is this actually a pure rotation?
    assert torch.isclose(torch.det(rotation),torch.ones(1))
    assert torch.isclose(torch.linalg.matrix_rank(rotation), torch.tensor(rotation.shape[0]))
    
    X_rot = X @ rotation
    cka1 = calc_cka(X, Y)
    cka2 = calc_cka(X_rot, Y)
    assert all(torch.isclose(cka1,cka2))

    # Test 1: 1D Perfect Separation
    X1 = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    y1 = torch.tensor([0, 0, 1, 1])
    r1 = calc_class_separation(X1, y1)
    assert torch.isclose(torch.tensor(r1), torch.tensor(1.0)), f"Test 1 Failed: {r1}"

    # Test 2: 1D No Separation (Identical Distributions)
    X2 = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    y2 = torch.tensor([0, 0, 1, 1])
    r2 = calc_class_separation(X2, y2)
    assert torch.isclose(torch.tensor(r2), torch.tensor(0.0)), f"Test 2 Failed: {r2}"

    # Test 3: 1D Total Collapse (No variance)
    X3 = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [1., 1.]])
    y3 = torch.tensor([0, 1, 0, 1])
    r3 = calc_class_separation(X3, y3)
    assert torch.isclose(torch.tensor(r3), torch.tensor(0.0)), f"Test 3 Failed: {r3}"

    # Test 4: 1D Single point per class
    X4 = torch.tensor([[1., 0.], [0., 1.]])
    y4 = torch.tensor([0, 1])
    r4 = calc_class_separation(X4, y4)
    assert torch.isclose(torch.tensor(r4), torch.tensor(1.0)), f"Test 4 Failed: {r4}"


    # Test 5: 2D Perfect Separation (equivalent to Test 1)
    X5 = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    y5 = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    r5 = calc_class_separation(X5, y5)
    assert torch.isclose(torch.tensor(r5), torch.tensor(1.0)), f"Test 5 Failed: {r5}"

    # Test 6: 2D No Separation (equivalent to Test 2)
    X6 = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    y6 = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    r6 = calc_class_separation(X6, y6)
    assert torch.isclose(torch.tensor(r6), torch.tensor(0.0)), f"Test 6 Failed: {r6}"

    # Test 7: 2D Total Collapse (equivalent to Test 3)
    X7 = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [1., 1.]])
    y7 = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    r7 = calc_class_separation(X7, y7)
    assert torch.isclose(torch.tensor(r7), torch.tensor(0.0)), f"Test 7 Failed: {r7}"

    # Test 8: 2D Partial Overlap (Reduces separation)
    # X points are perfectly separable, but one point belongs to both classes.
    # This should *reduce* R2 from 1.0, because class 0 is no longer "pure".
    X8 = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    y8 = torch.tensor([[1, 0], [1, 0], [1, 1], [0, 1]])
    r8 = calc_class_separation(X8, y8)
    assert 0.0 < r8 < 1.0, f"Test 8 Failed: {r8}"

    # Test 9: 2D Edge Cases (Single member, No members)
    X9 = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [2., 2.]])
    # Class 0: 2 members (var=0). Class 1: 1 member (var=0). Class 2: 0 members (var=0).
    y9 = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    r9 = calc_class_separation(X9, y9)
    # Total var > 0, but total within-class var is 0.
    assert torch.isclose(torch.tensor(r9), torch.tensor(1.0)), f"Test 9 Failed: {r9}"

    # Test 10: 2D No Classes
    X10 = torch.tensor([[1., 0.], [0., 1.]])
    y10 = torch.empty(2, 0).int() # 2 obs, 0 classes
    r10 = calc_class_separation(X10, y10)
    assert torch.isclose(torch.tensor(r10), torch.tensor(0.0)), f"Test 10 Failed: {r10}"
