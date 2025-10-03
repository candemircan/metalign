__all__ = ["fix_state_dict"]

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