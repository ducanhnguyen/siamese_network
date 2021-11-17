import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Model

import dataset_generation
from utils import visualize_accuracy, visualize_images


def train_Siamese(input_shape, path):
    #  Siamese Networks
    def create_joint_model(input_shape):
        img_in = Input(shape=input_shape)
        n_layer = Conv2D(8, kernel_size=(3, 3), activation='relu')(img_in)
        n_layer = Conv2D(16, kernel_size=(3, 3), activation='relu')(n_layer)
        n_layer = MaxPool2D((2, 2))(n_layer)
        n_layer = Flatten()(n_layer)
        n_layer = Dense(32, activation='relu')(n_layer)
        joint_model = Model(inputs=[img_in], outputs=[n_layer], name='JointModel')
        return joint_model

    joint_model = create_joint_model(input_shape)

    img1_in = Input(shape=input_shape, name='ImageA_Input')
    img1_model = joint_model(img1_in)

    img2_in = Input(shape=input_shape, name='ImageB_Input')
    img2_model = joint_model(img2_in)

    merge = concatenate([img1_model, img2_model])

    merge = Dense(16, activation='relu')(merge)
    merge = Dense(4, activation='relu')(merge)
    merge = Dense(1, activation='sigmoid')(merge)
    siamese = Model(inputs=[img1_in, img2_in], outputs=[merge], name='siamese_model')
    siamese.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(siamese, to_file=MODEL_PNG, show_shapes=True, show_layer_names=True)

    # train
    train(siamese, path)

def train(siamese_model, out_path):
    img1, img2, similarity = dataset_generation.read(MY_TRAINING)

    training_size = int(np.floor(len(img1) * 0.7))
    trainingX = [img1.reshape(INPUT_SHAPES)[:training_size], img2.reshape(INPUT_SHAPES)[:training_size]]
    trainingY = similarity[:training_size]

    validationX = [img1.reshape(INPUT_SHAPES)[training_size:], img2.reshape(INPUT_SHAPES)[training_size:]]
    validationY = similarity[training_size:]

    #
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True)

    loss_history = siamese_model.fit(trainingX, trainingY,
                               batch_size=1024,
                               validation_data=(validationX, validationY),
                               epochs=40,
                               verbose=True,
                               callbacks=[model_checkpoint_callback])

    visualize_accuracy(loss_history, LOSS_PNG)

if __name__ == '__main__':
    MY_TRAINING = './dataset/fashion-mnist/mytraining.csv'
    MY_TEST = './dataset/fashion-mnist/mytest.csv'

    INPUT_SHAPES = (-1, 28, 28, 1)
    INPUT_SHAPE = (28, 28, 1)

    MODEL_PNG = './model/fashion-mnist/v1/model.png'
    LOSS_PNG = './model/fashion-mnist/v1/accuracy.png'
    MODEL_FOLDER = './model/fashion-mnist/v1/siamese_mnist'

    action = 'test'
    if action == 'train':
        train_Siamese(INPUT_SHAPE, MODEL_FOLDER)
    elif  action == 'test':
        reconstructed_model = keras.models.load_model(MODEL_FOLDER)
        reconstructed_model.summary()

        imgA, imgB, similarity = dataset_generation.read(MY_TEST)

        imgA = imgA.reshape(INPUT_SHAPES)
        imgB = imgB.reshape(INPUT_SHAPES)
        similarity = similarity.reshape(-1)

        pred = reconstructed_model.predict([imgA, imgB])
        pred = pred.reshape(-1)
        pred_label = pred > 0.5
        print(f'Training acc: {np.sum(pred_label == similarity) / len(similarity)}')

        N_VISUALIZE = 5
        visualize_images(imgA[:N_VISUALIZE], imgB[:N_VISUALIZE], pred[:N_VISUALIZE])
