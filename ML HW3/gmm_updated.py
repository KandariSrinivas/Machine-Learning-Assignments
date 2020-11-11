import numpy as np
import argparse
from csv import reader


def make_spd_matrix(n):
    generator = np.random
    A = generator.rand(n, n)
    U, _, Vt = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(U, 1.0 + np.diag(generator.rand(n))), Vt)

    return X


def pdf(x, mean, cov):
    cl = (1 / ((np.linalg.det(cov) + 0.0000001) ** 0.5) * ((np.pi * 2) ** (cov.shape[0] / 2)))
    x_mu = (x - mean)
    e = np.sqrt(np.sum(np.square(np.matmul(np.matmul(x_mu, np.linalg.pinv(cov)), x_mu.T)), axis=-1))
    ans = cl * np.exp((-1 / 2) * e)
    return ans


class GMM:
    def __init__(self, k=1):
        self.k = k
        self.w = np.ones((k)) / k
        self.means = None
        self.cov = None

    def fit(self, X):
        X = np.array(X)
        self.means = np.random.choice(X.flatten(), (self.k, X.shape[1]))

        cov = []
        for i in range(self.k):
            cov.append(np.cov(X, rowvar=False))
        cov = np.array(cov)
        assert cov.shape == (k, X.shape[1], X.shape[1])
        eps = 1e-8

        for step in range(55):
            likelihood = []
            for j in range(self.k):
                likelihood.append(pdf(x=X, mean=self.means[j], cov=cov[j])+eps)
            likelihood = np.array(likelihood)
            assert likelihood.shape == (k, len(X))

            for j in range(self.k):
                b = ((likelihood[j] * self.w[j]) / (
                        np.sum([likelihood[i] * self.w[i] for i in range(self.k)], axis=0) + eps))
                self.means[j] = np.sum(b.reshape(len(X), 1) * X, axis=0) / (np.sum(b + eps))
                cov[j] = np.dot((b.reshape(len(X), 1) * (X - self.means[j])).T, (X - self.means[j])) / (
                        np.sum(b) + eps)
                self.w[j] = np.mean(b)

        self.cov = cov

    def P(self, X):
        X = np.array(X)
        p = 0
        for j in range(self.k):
            p += self.w[j] * pdf(x=X, mean=self.means[j], cov=self.cov[j])
        return p


def load_csv(inp_file):
    dataset = []
    with open(inp_file, 'r') as file:
        csv_reader = reader(file)
        for data_row in csv_reader:
            dataset.append(data_row)
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument('--components', help='components 1|3|4', required=True)
parser.add_argument('--train', help='path to training data file', required=True)
parser.add_argument('--test', help='path to test data file', required=True)
args = vars(parser.parse_args())

k = int(args['components'])

train_data = load_csv(args['train'])

train_datas = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

for row in train_data:
    y = int(row[64])
    train_datas[y].append([float(x.strip()) for x in row[:-1]])

gmms = []
for i in range(10):
    gmm = GMM(k)
    gmm.fit(train_datas[i])
    gmms.append(gmm)

print('trained')

test_data = load_csv(args['test'])

preds = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}

for row in test_data:
    y_act = int(row[64])
    max_p = float('-inf')
    y_pred = -1
    for idx in range(len(gmms)):
        p = gmms[idx].P([float(x.strip()) for x in row[:-1]])
        if np.sum(p) > max_p:
            y_pred = idx
            max_p = np.sum(p)
    accu = 0
    if y_pred == -1:
        print('never')
    if y_act == y_pred:
        accu = 1
    preds[y_act].append(accu)

total = 0
for idx in range(len(preds)):
    sum = np.sum(np.array(preds[idx]))
    print(f'{idx}: {(sum * 100) / len(preds[idx])}')
    total += sum

print(f'total: {(total / len(test_data)) * 100}')
