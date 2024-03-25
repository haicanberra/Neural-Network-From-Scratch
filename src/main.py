import numpy as np
import pandas as pd
import os.path
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from dense import Dense
from network import Network

def get_accuracy(predict, actual):
    correct = 0
    for i in range(predict.shape[0]):
        if predict[i] == actual[i]:
            correct = correct + 1
    accuracy = correct*100/predict.shape[0]
    print("Prediction: ", predict)
    print("Actual:     ", actual)
    print("Accuracy:   ", accuracy, "%")

if __name__ == '__main__':
    # NOTE: y must be one hot vector
    # X_train = np.array([[0,0], [0,1], [1,0], [1,1]])  ## 4,2
    # y_train = np.array([[1,0], [0,1], [0,1], [1,0]])  ## 4,1

    # Get Iris data
    path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(path, index_col=False)

    cols = list(df.columns)
    target = cols.pop()

    X = np.array(df[cols].copy())

    y = df[target].copy()
    y = np.array(LabelBinarizer().fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Initialize network (Delete weight files if change)
    network = Network()
    network.add(Dense(X_train[0].shape[0], 100, 'relu'))
    network.add(Dense(100, 50, 'relu'))
    network.add(Dense(50, 25, 'relu'))
    network.add(Dense(25, 3, 'sigmoid'))

    # Train (1st time use save=true load=false
    network.train(X_train=X_train, y_train=y_train, epochs=1000, lr=0.01, save_weights=True, load_weights=True)
    
    # Get accuracy
    predict = np.argmax(network.predict(X_test[:10]), axis=1)
    actual = np.argmax(y_test[:10], axis=1)
    get_accuracy(predict, actual)
    




