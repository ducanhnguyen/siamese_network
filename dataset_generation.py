import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import csv

from utils import gen_random_batch


def export(x, y, size, out):
    x = x.reshape(INPUT_SHAPES).astype('int')  # .astype('float32')  / 255.
    y = y.astype('int')

    groups = [x[np.where(y == i)[0]] for i in np.unique(y)]
    num = int(np.floor(size / 2))
    imgTrue_A, imgTrue_B, similarityTrue = gen_random_batch(groups, num, similarity=1)
    imgFalse_A, imgFalse_B, similarityFalse = gen_random_batch(groups, num, similarity=0)

    with open(out, mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for idx in range(num):
            employee_writer.writerow(imgTrue_A[idx].reshape(-1))
            employee_writer.writerow(imgTrue_B[idx].reshape(-1))
            employee_writer.writerow([similarityTrue[idx]])

            employee_writer.writerow(imgFalse_A[idx].reshape(-1))
            employee_writer.writerow(imgFalse_B[idx].reshape(-1))
            employee_writer.writerow([similarityFalse[idx]])


def read(path):
    print(f'Reading {path}')
    imgA = []
    imgB = []
    similarity = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        idx = 0
        for row in csv_reader:
            row = np.asarray(row).astype(int)
            if idx == 0:
                imgA.append(row)
            elif idx == 1:
                imgB.append(row)
            elif idx == 2:
                similarity.append(row[0])
            idx += 1
            if idx == 3:
                idx = 0

    return np.asarray(imgA)/255., np.asarray(imgB)/255., np.asarray(similarity)


if __name__ == '__main__':
    INPUT_SHAPES = (-1, 28, 28, 1)
    INPUT_SHAPE = (28, 28, 1)

    TRAINING_SET = './dataset/fashion-mnist/train.csv'
    TEST_SET = './dataset/fashion-mnist/test.csv'

    MYTRAINING = './dataset/fashion-mnist/mytraining.csv'
    TRAINING_SIZE = 100000

    MYTEST = './dataset/fashion-mnist/mytest.csv'
    TEST_SIZE = 50000

    # Create training set
    def create_trainingset():
        data_train = pd.read_csv(TRAINING_SET)
        X_full = data_train.iloc[:, 1:].to_numpy()
        y_full = data_train.iloc[:, :1].to_numpy()
        export(X_full, y_full, TRAINING_SIZE, MYTRAINING)
    create_trainingset()


    def create_testset():
        data_test = pd.read_csv(TEST_SET)
        X_full = data_test.iloc[:, 1:].to_numpy()
        y_full = data_test.iloc[:, :1].to_numpy()
        export(X_full, y_full, TEST_SIZE, MYTEST)
    create_testset()
    #
    # mode = 'read'
    # if mode == 'export':
    #
    # elif mode == 'read':
    #     imgA, imgB, similarity = read(MYTRAINING)
    #     print(imgA.shape)
    #     print(imgB.shape)
    #     print(similarity.shape)
