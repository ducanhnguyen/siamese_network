'''
Reference:
https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
https://arxiv.org/pdf/1812.09638.pdf
https://www.pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/
https://medium.com/@enoshshr/triplet-loss-and-siamese-neural-networks-5d363fdeba9b#:~:text=Another%20way%20to%20train%20a,using%20the%20triplet%20loss%20function.&text=It%20is%20a%20distance%20based,same%20class%20as%20the%20anchor
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

from config import MODEL_FOLDER
from main1 import train_Siamese1
from utils import visualize_images, gen_random_batch

if __name__ == '__main__':
    # Read dataset
    data_train = pd.read_csv('./dataset/digit-recognizer/train.csv')
    X_full = data_train.iloc[:, 1:]
    y_full = data_train.iloc[:, :1]
    x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3)

    # Normalize data
    x_train = x_train.values.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = y_train.values.astype('int')
    print('Training', x_train.shape, x_train.max())

    x_test = x_test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_test = y_test.values.astype('int')
    print('Testing', x_test.shape, x_test.max())

    # reorganize by groups
    train_groups = [x_train[np.where(y_train == i)[0]] for i in np.unique(y_train)]
    print('train groups:', [x.shape[0] for x in train_groups])

    test_groups = [x_test[np.where(y_test == i)[0]] for i in np.unique(y_train)]
    print('test groups:', [x.shape[0] for x in test_groups])

    TRAIN_ENABLED = True
    if (TRAIN_ENABLED):
        train_Siamese1(train_groups, test_groups, path=MODEL_FOLDER)
    else:
        reconstructed_model = keras.models.load_model(MODEL_FOLDER)
        reconstructed_model.summary()

        img1, img2, _ = gen_random_batch(test_groups, 5)
        pred = reconstructed_model.predict([img1, img2])
        visualize_images(img1, img2, pred)
