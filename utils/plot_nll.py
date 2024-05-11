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

data_name = 'yelp_dast'
data_path = "/mnt/Data3/hanqiyan/UDA/real_world/data/"
current_path = "/mnt/Data3/hanqiyan/style_transfer_baseline/checkpoint/cpvae_pretrain-DoCoGen_review-glove/20230129-164731/plot/"
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
            m1s.append(-float(m1))
            m2s.append(-float(m2))
    stat = MMD(m1s, m2s,kernel='rbf')
    print(f"MMD distance={stat:.4f}")
    # m1s = preprocessing.normalize([m1s],norm='max')
    # m2s = preprocessing.normalize([m2s],norm='max')
    return m1s, m2s

fig = plt.figure()

# beta-VAE one std

# ax2 = fig.add_subplot(211)
# m1, m2 = load_nll(data_path+"plot/nll.txt")
# ax2.hist(m1, 50, color="tab:green", alpha=0.5)
# ax2.hist(m2, 50, color="tab:red", alpha=0.5)

# handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:red"]]
# labels = [r"$\beta$-VAE", r"$\pm\sigma$"]
# ax2.legend(handles, labels, loc='upper right')

# ax2.set_ylabel('Number of Samples')
# ax2.set_title('(A)', x=0.0, fontsize=12)

# # beta-VAE two std

ax4 = fig.add_subplot(222)
m1, m2 = load_nll(current_path+"%s_nll_1000_flipped_shift_1.txt"%data_name)
ax4.hist(m1, 100, color="tab:green", alpha=0.5,range=(100,6000))
ax4.hist(m2, 100, color="tab:purple", alpha=0.5,range=(100,6000))

handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:purple"]]
labels = [r"CP-VAE", "$\pm1*\sigma$"]
ax4.legend(handles, labels, loc='upper right')

ax4.set_title('(B)', x=0.0, fontsize=12)

# beta-VAE extreme

ax6 = fig.add_subplot(223)
m1, m2 = load_nll(current_path+"%s_nll_1000_flipped_shift_2.txt"%data_name)
ax6.hist(m1, 100, color="tab:green", alpha=0.5,range=(100,6000))
ax6.hist(m2, 100, color="tab:pink", alpha=0.5,range=(100,6000))

handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:green", "tab:pink"]]
labels = [r"CP-VAE", "$\pm2*\sigma$"]
ax6.legend(handles, labels, loc='upper right')

ax6.set_xlabel('NLL of the Latent Codes')
ax6.set_ylabel('Number of Samples')
ax6.set_title('(C)', x=0.0, fontsize=12)

# CP-VAE

ax8 = fig.add_subplot(224)
m1, m2 = load_nll(current_path+"%s_nll_1000_flipped_shift_3.txt"%data_name)
ax8.hist(m2, 80, color="tab:orange", alpha=0.5,range=(100,6000))
ax8.hist(m1, 80, color="tab:cyan", alpha=0.5,range=(100,6000))


handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5, ec="k") for c in ["tab:cyan", "tab:orange"]]
labels = ["CP-VAE", "$\pm3*\sigma$"]
ax8.legend(handles, labels, loc='upper right')

ax8.set_xlabel('NLL of the Latent Codes')
ax8.set_title('(D)', x=0.0, fontsize=12)

plt.savefig(current_path+'%s_nll_comparison.png'%data_name)