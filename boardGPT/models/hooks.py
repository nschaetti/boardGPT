"""
Copyright (C) 2025 boardGPT Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch.nn as nn


class HookPoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.fns = []  # end def __init__
    # end def __init__

    def add_hook(self, fn):
        self.fns.append(fn)  # end def add_hook
    # end def add_hook

    def clear(self):
        self.fns = []  # end def clear
    # end def clear

    def forward(self, x):
        for fn in self.fns:
            rx = fn(x)
            if rx is not None:
                x = rx  # end if
            # end if
        # end for
        return x  # end def forward
    # end def forward
# end class HookPoint
# end class HookPoint


