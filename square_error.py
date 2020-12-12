import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import normal


class CreateDataset:
    def __init__(self, N):
        self.N = N

    def create_dataset(self):
        dataset = pd.DataFrame(columns=['x', 'y'])
        for i in range(self.N):
            x = float(i)/float(self.N-1)
            y = np.sin(2*np.pi*x) + normal(scale=0.3)
            dataset = dataset.append(pd.Series([x, y], index=['x', 'y']),
                                     ignore_index=True)

        return dataset


class LeastSquares:
    def __init__(self, N, M, trainset, testset):
        self.N = N  # number of x, the position sampled
        self.M = M
        self.trainset = trainset
        self.testset = testset

    def resolve(self):
        t = self.trainset['y']
        phi = pd.DataFrame()
        for i in range(0, self.M+1):
            p = self.trainset['x']**i
            p.name = "x**%d" % i
            phi = pd.concat([phi, p], axis=1)

        tmp = np.linalg.inv(np.dot(phi.T, phi))
        ws = np.dot(np.dot(tmp, phi.T), t)

        return ws

    def f(self, x, ws):
        y = 0
        for i in range(ws.size):
            y += ws[i] * pow(x, i)

        return y

    def rms_error(self, f, ws):
        err = 0
        for index, line in self.trainset.iterrows():
            x, y = line.x, line.y
            err += 0.5 * (y - f(x, ws))**2

        return np.sqrt(2 * err/len(self.trainset))

    def rms_error_test(self, f, ws):
        err = 0
        for index, line in self.testset.iterrows():
            x, y = line.x, line.y
            err += 0.5 * (y - f(x, ws))**2

        return np.sqrt(2 * err/len(self.testset))


"""
# main
N = 20
M = [0, 1, 3, 9]
df_ws = pd.DataFrame()

cd = CreateDataset(N)
trainset = cd.create_dataset()
testset = cd.create_dataset()

fig = plt.figure()
for c, m in enumerate(M):
    ls = LeastSquares(N, m, trainset, testset)
    ws = ls.resolve()
    print(ws)
    x = ls.trainset['x']
    df_ws = df_ws.append(pd.Series(ws, name="M=%d" % m))
    ax = fig.add_subplot(2, 2, c+1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("M=%d" % m)

    # plot training set
    ax.scatter(ls.trainset['x'], ls.trainset['y'],
               marker='o', color='blue', label=None)

    # plot true curve
    linex = np.linspace(0, 1, 101)
    liney = np.sin(2*np.pi*linex)
    ax.plot(linex, liney, color='green', linestyle='--')

    # plot polynomial approximation
    linex = np.linspace(0, 1, 101)
    liney = ls.f(linex, ws)
    label = "E(RMS)=%.2f" % ls.rms_error(ls.f, ws)
    ax.plot(linex, liney, color='red', label=label)
    ax.legend(loc=1)

print(df_ws.transpose())
plt.show()

df = pd.DataFrame(columns=['Training set', 'Test set'])
for m in range(0, 10):
    ls = LeastSquares(N, m, trainset, testset)
    ws = ls.resolve()
    train_error = ls.rms_error(ls.f, ws)
    test_error = ls.rms_error_test(ls.f, ws)

    df = df.append(
        pd.Series([train_error, test_error],
                  index=['Training set', 'Test set']),
        ignore_index=True)

df.plot(title='RMS Error', style=['-', '--'], grid=True, ylim=(0, 0.9))
plt.show()
"""
