# -*- coding: utf-8 -*-
#
# 推定量の一致性と不偏性の確認
#
# 2015/06/01 ver1.0
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import normal


class EstimatiorBias:
    def __init__(self):
        pass

    def draw_subplot(self, ax, linex1, liney1, linex2, liney2, ylim):
        ax.set_ylim(ylim)
        ax.set_xlim(min(linex1), max(linex1)+1)
        ax.scatter(linex1, liney1)
        ax.plot(linex2, liney2, color='red', linewidth=4, label="mean")
        ax.legend(loc=0)


if __name__ == '__main__':
    mean_linex = []
    mean_mu = []
    mean_s2 = []
    mean_u2 = []
    raw_linex = []
    raw_mu = []
    raw_s2 = []
    raw_u2 = []
    for n in np.arange(2, 51):  # 観測データ数Nを変化させて実行
        for c in range(2000):  # 特定のNについて2000回の推定を繰り返す
            ds = normal(loc=0, scale=1, size=n)
            raw_mu.append(np.mean(ds))
            raw_s2.append(np.var(ds))
            raw_u2.append(np.var(ds)*n/(n-1))
            raw_linex.append(n)
        mean_mu.append(np.mean(raw_mu))  # 標本平均の平均
        mean_s2.append(np.mean(raw_s2))  # 標本分散の平均
        mean_u2.append(np.mean(raw_u2))  # 不偏分散の平均
        mean_linex.append(n)

    # プロットデータを40個に間引きする
    raw_linex = raw_linex[0:-1:50]
    raw_mu = raw_mu[0:-1:50]
    raw_s2 = raw_s2[0:-1:50]
    raw_u2 = raw_u2[0:-1:50]

    # 標本平均の結果表示
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_title('Sample mean')
    eb = EstimatiorBias()
    eb.draw_subplot(ax, raw_linex, raw_mu,
                    mean_linex, mean_mu, (-1.5, 1.5))

    plt.show()

    # 標本分散の結果表示
    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_title('Sample variance')
    eb.draw_subplot(ax, raw_linex, raw_s2,
                    mean_linex, mean_s2, (-0.5, 3.0))

    plt.show()

    # 不偏分散の結果表示
    fig3 = plt.figure()
    ax = fig3.add_subplot(1, 1, 1)
    ax.set_title('Unbiased variance')
    eb.draw_subplot(ax, raw_linex, raw_u2,
                    mean_linex, mean_u2, (-0.5, 3.0))

    plt.show()
