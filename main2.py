import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Model

import dataset_generation
from utils import visualize_accuracy, visualize_images


def loss(margin=1):
    # https://keras.io/examples/vision/siamese_contrastive/
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


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
    siamese.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # siamese.compile(loss=loss(margin=1), optimizer="adam", metrics=["accuracy"])


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

    MODEL_PNG = './model/fashion-mnist/v2/model.png'
    LOSS_PNG = './model/fashion-mnist/v2/accuracy.png'
    MODEL_FOLDER = './model/fashion-mnist/v2/siamese_mnist'


    action = 'test'
    if action == 'train':
        train_Siamese(INPUT_SHAPE, MODEL_FOLDER)
    elif action == 'test':
        reconstructed_model = keras.models.load_model(MODEL_FOLDER, custom_objects={'contrastive_loss': loss(1)})
        reconstructed_model.summary()

        imgA, imgB, similarity = dataset_generation.read(MY_TRAINING)

        imgA = imgA.reshape(INPUT_SHAPES)
        imgB = imgB.reshape(INPUT_SHAPES)
        similarity = similarity.reshape(-1)

        pred = reconstructed_model.predict([imgA, imgB])
        pred = pred.reshape(-1)
        pred_label = pred > 0.5
        print(f'Training acc: {np.sum(pred_label == similarity) / len(similarity)}')

        N_VISUALIZE = 5
        visualize_images(imgA[:N_VISUALIZE], imgB[:N_VISUALIZE], pred[:N_VISUALIZE])
