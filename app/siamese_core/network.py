import tensorflow as tf
from keras.models import Model  # pyright: ignore[reportMissingTypeStubs]
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten  # pyright: ignore[reportMissingTypeStubs]
from keras.regularizers import l2  # pyright: ignore[reportMissingTypeStubs]

# Siamese network constants
SIAMESE_INPUT_EDGE_LENGTH = 105
L2_REG = l2(1e-4)


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


class SiameseNetwork:
    def __init__(self, embedding_distance_calculator_layer):
        embedding_distance_calculator_layer._name = "distance"
        self.model = self.build_model(embedding_distance_calculator_layer)

    def build_model(self, embedding_distance_calculator_layer):
        embedding_model = self.__make_embedding()
        input_img = Input(
            name="input_img",
            shape=(SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH, 3),
        )
        validation_img = Input(
            name="validation_img",
            shape=(SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH, 3),
        )

        distances = embedding_distance_calculator_layer(
            embedding_model(input_img), embedding_model(validation_img)
        )

        classifier = Dense(units=1, activation="sigmoid", dtype='float32')(distances)

        return Model(
            inputs=[input_img, validation_img],
            outputs=classifier,
            name="SiameseNetwork",
        )

    def __make_embedding(self):
        """
        Create the embedding network
        This network will take an image as input and output a vector of size 4096
        :return:
        """
        inp = Input(
            shape=(SIAMESE_INPUT_EDGE_LENGTH, SIAMESE_INPUT_EDGE_LENGTH, 3),
            name="input_image",
        )

        # First block
        c1 = Conv2D(filters=64, kernel_size=(10, 10), activation="relu", kernel_regularizer=L2_REG)(inp)
        m1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c1)

        # Second block
        c2 = Conv2D(filters=128, kernel_size=(7, 7), activation="relu", kernel_regularizer=L2_REG)(m1)
        m2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c2)

        # Third block
        c3 = Conv2D(filters=128, kernel_size=(4, 4), activation="relu", kernel_regularizer=L2_REG)(m2)
        m3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c3)

        # Final embedding block
        c4 = Conv2D(filters=256, kernel_size=(4, 4), activation="relu", kernel_regularizer=L2_REG)(m3)
        f1 = Flatten()(c4)
        d1 = Dense(units=4096, activation="sigmoid", kernel_regularizer=L2_REG)(f1)

        return Model(inputs=[inp], outputs=[d1], name="embedding")

    def compile(self, optimizer, loss, metrics):
        pass
