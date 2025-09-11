
import torch.nn as nn


class HookPoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.fns = []
    # end def __init__

    def add_hook(self, fn):
        self.fns.append(fn)
    # end def add_hook

    def clear(self):
        self.fns = []
    # end def clear

    def forward(self, x):
        for fn in self.fns:
            rx = fn(x)
            if rx is not None:
                x = rx
            # end if
        # end for
        return x
    # end def forward

# end class HookPoint


