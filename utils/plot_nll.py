# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import ttest_ind
import torch

data_name = 'amazon'
data_path = "../data/"
current_path = "../checkpoint/model1/plot/amazon_nll_flip.txt"
# base_path = "../checkpoint/model2/plot/amazon_nll_flip.txt"
noise_path = "../checkpoint/noise_model/plot/amazon_nll_flip.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    x = torch.tensor(x).reshape(-1,1)
    y = torch.tensor(y).reshape(-1,1)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)


def load_nll(filename):
    m1s, m2s = [], []
    with open(filename) as f:
        for line in f:
            m1, m2 = line.strip().split('\t')
            if float(m1)<-5000:
                m1s.append(float(m1[3:]))
                m2s.append(float(m2[3:]))
                print(float(m1[3:]),float(m2[3:]))
            else:
                m1s.append(-float(m1))
                m2s.append(-float(m2))
    # stat = MMD(m1s, m2s,kernel='rbf')
    # print(f"MMD distance={stat:.4f}")
    # m1s = preprocessing.normalize([m1s],norm='max')
    # m2s = preprocessing.normalize([m2s],norm='max')
    return m1s, m2s

fig = plt.figure(figsize=(5,2),dpi=300)


ax6 = fig.add_subplot(121)
m1, m2 = load_nll(current_path)
ax6.hist(m1, 100, color="tab:green", alpha=0.5,range=(100,1200))
ax6.hist(m2, 100, color="tab:pink", alpha=0.5,range=(100,1200))

handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:pink"]]
labels = ["Original", "Flipped"]
ax6.legend(handles, labels, loc='upper right',fontsize=8)

ax6.set_xlabel('NLL of the Latent Variables',fontsize=8)
ax6.set_ylabel('Number of Samples',fontsize=8)
ax6.set_title(r'Flip $s$', x=0.5, fontsize=8)

# CP-VAE

ax8 = fig.add_subplot(122)
m1, m2 = load_nll(noise_path)
ax8.hist(m2, 80, color="tab:green", alpha=0.5,range=(100,1500))
ax8.hist(m1, 80, color="tab:purple", alpha=0.5,range=(100,1500))


handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:purple"]]
labels = ["Original", "Flipped"]
ax8.legend(handles, labels, loc='upper right',fontsize=8)

ax8.set_xlabel('NLL of the Latent Variables',fontsize=8)
ax8.set_title(r'Flip $\tilde{s}$', x=0.5, fontsize=8)

plt.savefig('%s_nll_comparison.png'%data_name,bbox_inches = "tight")