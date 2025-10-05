__all__ = ["fix_state_dict", "cka"]

import torch
from einops import einsum, reduce
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

def calculate_cka(X:torch.Tensor, # batch by observations by features
                  Y:torch.Tensor # batch by observations by features
                  ):
    """
    Calculate centered kernel alignment (CKA) between two batched tensors
    """

    # subtract the mean of each observation
    X -= X.mean(1, keepdim=True)
    Y -= Y.mean(1, keepdim=True)

    XTX = einsum(X, X, "b o f1, b o f2 -> b f1 f2")
    YTY = einsum(Y, Y, "b o f1, b o f2 -> b f1 f2")
    YTX = einsum(Y, X, "b o f1, b o f2 -> b f1 f2")


    top = reduce(YTX ** 2, "b f1 f2 -> b", "sum")
    bottom = torch.sqrt(reduce(XTX ** 2, "b f1 f2 -> b", "sum") * reduce(YTY ** 2, "b f1 f2 -> b", "sum"))

    return top / bottom

@call_parse
def main():
    """
    cka tests, adapted from [here](https://github.com/patrickmineault/codebook_examples/blob/main/cka/test_cka_step3.py)
    """
    # some config
    bs, n_obs, n_feats = 5, 20, 2
    X, Y = torch.rand(bs, n_obs, n_feats), torch.rand(bs, n_obs, n_feats)

    # trivially cka(X,X) should be 1
    assert all(torch.isclose(calculate_cka(X, X), torch.ones(bs)))

    # swapping orders of columns should not change this
    X_sw = X.clone()
    X_sw[:, :, [1,0]] = X_sw[:, :, [0,1]]
    assert all(torch.isclose(calculate_cka(X, X_sw), torch.ones(bs)))

    # adding a column offset shouldnt change things
    X_coffset = X.clone()
    X_coffset[:, :, [0]] += 10
    assert all(torch.isclose(calculate_cka(X, X_coffset), torch.ones(bs)))

    # isotropic scaling should be irrelevant
    cka1 = calculate_cka(X,Y)
    cka2 = calculate_cka(X * 23, Y * -2)
    assert all(torch.isclose(cka1,cka2))

    # if scaling isnt isotropic, this should change cka
    cka1 = calculate_cka(X,Y)
    Y_noniso = Y.clone()
    Y_noniso[:, :, [0]] *= 4
    cka2 = calculate_cka(X, Y_noniso)
    assert not all(torch.isclose(cka1, cka2))

    # rotations shouldnt matter 
    rotation = torch.tensor([[1, -1], [1, 1]],dtype=X.dtype) / torch.sqrt(torch.tensor(2))
    
    # is this actually a pure rotation?
    assert torch.isclose(torch.det(rotation),torch.ones(1))
    assert torch.isclose(torch.linalg.matrix_rank(rotation), torch.tensor(rotation.shape[0]))
    
    X_rot = X @ rotation
    cka1 = calculate_cka(X, Y)
    cka2 = calculate_cka(X_rot, Y)
    assert all(torch.isclose(cka1,cka2))

