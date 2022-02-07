import PIL
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from KNN import KNN
import cv2

def externPrediction():
    mnist = load_digits()
    X = mnist.data
    Y = mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

    # Image path here
    image = PIL.Image.open('0.png')
    image_array = np.asarray(image)[:, :, 0]


    flaten_image = image_array.flatten()

    model = KNN(5)
    model.fit(X_train, y_train)
    predict = model.predict(flaten_image)

    prediction = predict[0]

    imgdata = np.array(flaten_image, dtype='float')
    pixels = imgdata.reshape((8, 8))
    plt.imshow(pixels, cmap='gray')
    plt.annotate(prediction, (3, 3), bbox={'facecolor': 'white'}, fontsize=16)
    print("It's most likely a : {}".format(prediction))
    plt.show()
    cv2.waitKey(0)

