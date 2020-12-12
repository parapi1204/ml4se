import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from square_error import LeastSquares
from square_error import CreateDataset


class LogLikelihood(LeastSquares):
    def __init__(self, N, M, trainset, testset):
        super().__init__(N, M, trainset, testset)

    def log_likelihood_train(self, ws, f):
        dev = 0.0
        n = float(len(self.trainset))
        for index, line in self.trainset.iterrows():
            x, y = line.x, line.y
            dev += (y - f(x, ws))**2
        err = dev * 0.5  # E_D
        beta = n / dev
        lp = -beta*err + 0.5*n*np.log(0.5*beta/np.pi)
        return lp

    def log_likelihood_test(self, ws, f):
        dev = 0.0
        n = float(len(self.testset))
        for index, line in self.testset.iterrows():
            x, y = line.x, line.y
            dev += (y - f(x, ws))**2
        err = dev * 0.5  # E_D
        beta = n / dev
        lp = -beta*err + 0.5*n*np.log(0.5*beta/np.pi)
        return lp

    def calc_sigma(self, ws, f):
        sigma2 = 0.0
        for index, line in self.trainset.iterrows():
            x, y = line['x'], line['y']
            sigma2 += (f(x, ws) - y)**2

        sigma2 /= len(self.trainset)
        return np.sqrt(sigma2)


# Main
if __name__ == '__main__':
    N = 100
    M = [0, 1, 3, 9]
    df_ws = pd.DataFrame()

    cd = CreateDataset(N)
    trainset = cd.create_dataset()
    testset = cd.create_dataset()

    fig = plt.figure()
    for c, m in enumerate(M):
        ll = LogLikelihood(N, m, trainset, testset)
        ws = ll.resolve()
        df_ws = df_ws.append(pd.Series(ws, name="M=%d" % m))
        sigma = ll.calc_sigma(ws, ll.f)

        ax = fig.add_subplot(2, 2, c+1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("M=%d" % m)

        # plot training set
        ax.scatter(ll.trainset['x'], ll.trainset['y'],
                   marker='o', color='blue', label=None)

        # plot true curve
        linex = np.linspace(0, 1, 101)
        liney = np.sin(2*np.pi*linex)
        ax.plot(linex, liney, color='green', linestyle='--')

        # 多項式近似の曲線を表示
        linex = np.linspace(0, 1, 101)
        liney = ll.f(linex, ws)
        label = "Sigma=%.2f" % sigma
        ax.plot(linex, liney, color='red', label=label)
        ax.plot(linex, liney+sigma, color='red', linestyle='--')
        ax.plot(linex, liney-sigma, color='red', linestyle='--')
        ax.legend(loc=1)

    plt.show()

    # 多項式近似に対する最大対数尤度を計算
    df = pd.DataFrame()
    train_mlh = []
    test_mlh = []
    for m in range(0, 9):  # 多項式の次数
        ll = LogLikelihood(N, m, trainset, testset)
        ws = ll.resolve()
        sigma = ll.calc_sigma(ws, ll.f)

        train_mlh.append(ll.log_likelihood_train(ws, ll.f))
        test_mlh.append(ll.log_likelihood_test(ws, ll.f))
    df = pd.concat([df,
                    pd.DataFrame(train_mlh, columns=['Training set']),
                    pd.DataFrame(test_mlh, columns=['Test set'])],
                   axis=1)
    df.plot(title='Log likelihood for N=%d' % N, grid=True, style=['-', '--'])
    plt.show()
