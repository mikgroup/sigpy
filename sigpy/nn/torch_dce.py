import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from sigpy.mri import dce

# %%
class DCE(nn.Module):
    def __init__(self,
                 ishape,
                 sample_time,
                 R1 = 1.,
                 M0 = 5.,
                 R1CA = 4.39,
                 FA = 15.,
                 TR = 0.006):
        super(DCE, self).__init__()

        self.ishape = list(ishape)

        self.sample_time = torch.tensor(np.squeeze(sample_time), dtype=torch.float32)

        self.R1 = torch.tensor(np.array(R1), dtype=torch.float32)
        self.M0 = torch.tensor(np.array(M0), dtype=torch.float32)
        self.R1CA = torch.tensor(np.array(R1CA), dtype=torch.float32)
        self.FA = torch.tensor(np.array(FA), dtype=torch.float32)
        self.TR = torch.tensor(np.array(TR), dtype=torch.float32)

        self.FA_radian = self.FA * np.pi / 180.
        self.M0_trans = self.M0 * torch.sin(self.FA_radian)

        E1 = torch.exp(-self.TR * self.R1)
        self.M_steady = self.M0_trans * (1 - E1) / (1 - E1 * torch.cos(self.FA_radian))

        Cp = dce.arterial_input_function(sample_time)
        self.Cp = torch.tensor(Cp, dtype=torch.float32)

    def _check_ishape(self, input):
        for i1, i2 in zip(input.shape, self.ishape):
            if i1 != i2:
                raise ValueError(
                    'input shape mismatch for {s}, got {input_shape}'.format(s=self, input_shape=input.shape))

    def _param_to_conc(self, x):
        t1_idx = torch.nonzero(self.sample_time)
        t1 = self.sample_time[t1_idx]
        dt = torch.diff(t1, dim=0)
        K_time = torch.cumsum(self.Cp, dim=0) * dt[-1]

        mult = torch.stack((K_time, self.Cp), 1)

        xr = torch.reshape(x, (self.ishape[0], np.prod(self.ishape[1:])))

        yr = torch.matmul(mult, xr)

        oshape = [len(self.sample_time)] + self.ishape[1:]
        yr = torch.reshape(yr, tuple(oshape))

        return yr

    def forward(self, x):

        if torch.is_tensor(x) is not True:
            x = torch.tensor(x, dtype=torch.float32)

        self._check_ishape(x)

        # parameters (k_trans, v_p) to concentration
        CA = self._param_to_conc(x)
        x0 = CA[0, ...]  # baseline image

        # concentration to MR signal
        E1CA = torch.exp(-self.TR * (self.R1 + self.R1CA * CA))

        CA_trans = self.M0_trans * (1 - E1CA) / (1 - E1CA * torch.cos(self.FA_radian))

        y = CA_trans + x0 - self.M_steady

        return y

# %%
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = DCE()


# for epoch in range(20):
