from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Siamese network constants
SIAMESE_INPUT_EDGE_LENGTH = 224


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

        classifier = Dense(1, activation="sigmoid")(distances)

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
        c1 = Conv2D(64, (10, 10), activation="relu")(inp)
        m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

        # Second block
        c2 = Conv2D(128, (7, 7), activation="relu")(m1)
        m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

        # Third block
        c3 = Conv2D(128, (4, 4), activation="relu")(m2)
        m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

        # Final embedding block
        c4 = Conv2D(256, (4, 4), activation="relu")(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation="sigmoid")(f1)

        return Model(inputs=[inp], outputs=[d1], name="embedding")

    def compile(self, optimizer, loss, metrics):
        pass
