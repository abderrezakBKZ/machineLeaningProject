import operator
import  numpy as np



def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in
                             self.X_train])
            dist_sorted = dist.argsort()[:self.k]
            neigh_count = {}
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(),
                                        key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0])
        return predictions