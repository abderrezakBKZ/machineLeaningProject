import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from KNN import KNN


def InternPrediction(number_of_tests=5):
    mnist = load_digits()

    X = mnist.data
    Y = mnist.target

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

    for i in np.random.randint(0, high=len(y_test), size=(number_of_tests,)):
        image = X_test[i]

        model = KNN(5)
        model.fit(X_train, y_train)
        prediction = model.predict([image])[0]

        imgdata = np.array(image, dtype='float')
        pixels = imgdata.reshape((8, 8))
        plt.imshow(pixels, cmap='gray')
        plt.annotate(prediction, (3, 3), bbox={'facecolor': 'white'}, fontsize=16)
        print("It's most likely a  : {}".format(prediction))
        plt.show()
        cv2.waitKey(0)
