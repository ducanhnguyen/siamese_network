import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Model
from tensorflow.keras import layers

from config import MODEL_PNG, LOSS_PNG
from utils import visualize_accuracy, gen_random_batch


def euclidean_distance(vects):
    # https://keras.io/examples/vision/siamese_contrastive/
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def create_joint_model(input_shape):
    img_in = Input(shape=input_shape)
    n_layer = Conv2D(8, kernel_size=(3, 3), activation='relu')(img_in)
    n_layer = Conv2D(16, kernel_size=(3, 3), activation='linear')(n_layer)
    n_layer = MaxPool2D((2, 2))(n_layer)
    n_layer = Flatten()(n_layer)
    n_layer = Dense(32, activation='relu')(n_layer)
    joint_model = Model(inputs=[img_in], outputs=[n_layer], name='JointModel')
    return joint_model



def train_Siamese2(train_groups, test_groups, path):
    #  Siamese Networks
    input_shape = train_groups[0][0].shape
    joint_model = create_joint_model(input_shape)

    img1_in = Input(shape=input_shape, name='ImageA_Input')
    img1_model = joint_model(img1_in)

    img2_in = Input(shape=input_shape, name='ImageB_Input')
    img2_model = joint_model(img2_in)

    # merge = concatenate([img1_model, img2_model])
    merge = layers.Lambda(euclidean_distance)([img1_model, img2_model])

    # merge = Dense(16, activation='relu')(merge)
    # merge = Dense(4, activation='relu')(merge)
    merge = Dense(1, activation='sigmoid')(merge)
    siamese = Model(inputs=[img1_in, img2_in], outputs=[merge], name='siamese_model')
    siamese.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(siamese, to_file=MODEL_PNG, show_shapes=True, show_layer_names=True)

    # train
    TRAINING_SIZE = 50000
    img1, img2, similarity = gen_random_batch(train_groups, TRAINING_SIZE)
    trainingX = [img1, img2]
    trainingY = similarity

    VALIDATION_SIZE = 10000
    img1, img2, similarity = gen_random_batch(test_groups, VALIDATION_SIZE)
    validationX = [img1, img2]
    validationY = similarity

    #
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='auto',
        save_best_only=True)

    loss_history = siamese.fit(trainingX, trainingY,
                               batch_size=1024,
                               validation_data=(validationX, validationY),
                               epochs=50,
                               verbose=True,
                               callbacks=[model_checkpoint_callback])

    visualize_accuracy(loss_history, LOSS_PNG)

