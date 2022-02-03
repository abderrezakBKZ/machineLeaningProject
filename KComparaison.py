import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from KNN import KNN


def kComparaison():
    mnist = load_digits()

    X = mnist.data
    Y = mnist.target

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

    kVals = np.arange(4, 10, 5)
    accuracies = []

    for k in kVals:
        model = KNN(k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(pred)
        acc = accuracy_score(y_test, pred)
        accuracies.append(acc)
        print("K = " + str(k) + "; Accuracy: " + str(acc))
