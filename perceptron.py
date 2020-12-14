import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal


class PrepareDataset:
    def __init__(self, variance):
        self.var = variance

    # データセット {x_n,y_n,type_n} を用意
    def prepare_dataset(self):
        cov1 = np.array([[self.var, 0], [0, self.var]])
        cov2 = np.array([[self.var, 0], [0, self.var]])

        df1 = pd.DataFrame(multivariate_normal(
            Mu1, cov1, N1), columns=['x', 'y'])
        df1['type'] = 1

        df2 = pd.DataFrame(multivariate_normal(
            Mu2, cov2, N2), columns=['x', 'y'])
        df2['type'] = -1

        df = pd.concat([df1, df2], ignore_index=True)
        df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)

        return df


class Perceptron:
    def __init__(self, trainset):
        self.train_set = trainset

    # Perceptronのアルゴリズム（確率的勾配降下法）を実行
    def run_simulation(self, data_graph, param_graph):
        train_set1 = self.train_set[self.train_set['type'] == 1]
        train_set2 = self.train_set[self.train_set['type'] == -1]
        ymin, ymax = self.train_set.y.min()-5, self.train_set.y.max()+10
        xmin, xmax = self.train_set.x.min()-5, self.train_set.x.max()+10
        data_graph.set_ylim([ymin-1, ymax+1])
        data_graph.set_xlim([xmin-1, xmax+1])
        data_graph.scatter(train_set1.x, train_set1.y, marker='o', label=None)
        data_graph.scatter(train_set2.x, train_set2.y, marker='x', label=None)

        # パラメータの初期値とbias項の設定
        w0 = w1 = w2 = 0.0
        bias = 0.5 * (self.train_set.x.abs().mean() +
                      self.train_set.y.abs().mean())

        # Iterationを30回実施
        paramhist = pd.DataFrame([[w0, w1, w2]], columns=['w0', 'w1', 'w2'])
        for i in range(30):
            for index, point in self.train_set.iterrows():
                x, y, type_ = point['x'], point['y'], point['type']
                if type_ * (w0*bias + w1*x + w2*y) <= 0:
                    w0 += type_ * bias
                    w1 += type_ * x
                    w2 += type_ * y
            paramhist = paramhist.append(
                pd.Series([w0, w1, w2], ['w0', 'w1', 'w2']),
                ignore_index=True)

        # 判定誤差の計算
        err = 0.0
        for index, point in self.train_set.iterrows():
            x, y, type_ = point.x, point.y, point.type
            if type_ * (w0*bias + w1*x + w2*y) <= 0:
                err += 1
        err_rate = err * 100 / len(self.train_set)

        # 結果の表示
        linex = np.arange(xmin-5, xmax+5)
        liney = - linex * w1 / w2 - bias * w0 / w2
        label = "ERR %.2f%%" % err_rate
        data_graph.plot(linex, liney, label=label, color='red')
        data_graph.legend(loc=1)
        paramhist.plot(ax=param_graph)
        param_graph.legend(loc=1)


# Main
N1 = 50         # クラス t=+1 のデータ数
Mu1 = [15, 10]   # クラス t=+1 の中心座標

N2 = 50         # クラス t=-1 のデータ数
Mu2 = [0, 0]     # クラス t=-1 の中心座標

Variances = [10, 30]  # 両クラス共通の分散（2種類の分散で計算を実施）

fig = plt.figure()
# 2種類の分散で実行
for c, variance in enumerate(Variances):
    prepdata = PrepareDataset(variance)
    trainset = prepdata.prepare_dataset()
    pcp = Perceptron(trainset)

    subplots1 = fig.add_subplot(2, 2, c+1)
    subplots2 = fig.add_subplot(2, 2, c+2+1)
    pcp.run_simulation(subplots1, subplots2)

plt.show()
