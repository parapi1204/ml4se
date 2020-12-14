# -*- coding: utf-8 -*-
#
# 最尤推定による正規分布の推定
#
# 2015/04/23 ver1.0
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy.random import normal
from scipy.stats import norm

if __name__ == '__main__':
    fig = plt.figure()
    for c, datapoints in enumerate([2, 4, 10, 100]):  # サンプル数
        ds = normal(loc=0, scale=1, size=datapoints)
        mu = np.mean(ds)                # 平均の推定値
        sigma = np.sqrt(np.var(ds))     # 標準偏差の推定値

        ax = fig.add_subplot(2, 2, c+1)
        ax.set_title("N=%d" % datapoints)
        # 真の曲線を表示
        linex = np.arange(-10, 10.1, 0.1)
        orig = norm(loc=0, scale=1)
        ax.plot(linex, orig.pdf(linex), color='green', linestyle='--')
        # 推定した曲線を表示
        est = norm(loc=mu, scale=sigma)
        label = "Sigma=%.2f" % sigma
        ax.plot(linex, est.pdf(linex), color='red', label=label)
        ax.legend(loc=1)
        # サンプルの表示
        ax.scatter(ds, orig.pdf(ds), marker='o', color='blue')
        ax.set_xlim(-4, 4)
        ax.set_ylim(0)
    fig.show()
    plt.show()
