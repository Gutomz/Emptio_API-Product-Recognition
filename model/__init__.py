import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import imageio
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class SiameseModel():
    model: Model

    def __init__(self, path, input_shape):
        self.init_paths(path)
        self.init_size(input_shape)
        self.init_model()

    def validate_path(self, path, throw_error=True):
        exist = os.path.exists(path)
        if not exist and throw_error:
            raise "Error ::: can't find path {}".format(path)

        return exist

    def init_size(self, input_shape):
        self.WIDTH, self.HEIGHT, self.CHANNEL = input_shape

    def init_paths(self, root_path):
        self.validate_path(root_path)

        images_path = os.path.join(root_path, "images")
        self.validate_path(images_path)

        weights_path = os.path.join(root_path, "model_weights.h5")
        self.validate_path(weights_path)

        self.root_path = root_path
        self.images_path = images_path
        self.weights_path = weights_path

    def get_siamese_model(self, input_shape):
        """
            Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        """

        # Define the tensors for the two input images
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        # Convolutional Neural Network
        model = Sequential()
        model.add(Conv2D(32, 3, padding="same",
                  activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, 3, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, 3, padding="same", activation='relu'))
        model.add(Conv2D(64, 3, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='sigmoid'))

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(1, activation='sigmoid')(L1_distance)

        # Connect the inputs with the outputs
        siamese_net = Model(
            inputs=[left_input, right_input], outputs=prediction)

        # return the model
        return siamese_net

    def init_model(self):
        model = self.get_siamese_model((self.WIDTH, self.HEIGHT, self.CHANNEL))
        model.load_weights(self.weights_path)
        self.model = model
        return model

    def validate_images(self):
        return len(os.listdir(self.images_path)) > 0

    def generate_pairs(self, image_path):
        w, h, c = self.WIDTH, self.HEIGHT, self.CHANNEL
        image = imageio.imread(image_path).reshape(w, h, c) / 255.0

        files = os.listdir(self.images_path)
        file_count = len(files)

        pairs = [np.zeros((file_count, w, h, c)) for i in range(2)]
        classes = []

        for i in range(file_count):
            file = files[i]
            file_name, _ = os.path.splitext(file)

            classes.append(file_name)

            file_path = os.path.join(self.images_path, file)
            file_image = imageio.imread(file_path).reshape(w, h, c)

            pairs[0][i, :, :, :] = image
            pairs[1][i, :, :, :] = file_image / 255.0

        return pairs, classes

    def predict(self, image_path):
        if not self.validate_path(image_path, throw_error=False):
            return None

        if not self.validate_images():
            return None

        pairs, classes = self.generate_pairs(image_path)
        probs = self.model.predict(pairs)

        index = np.argmax(probs)
        return classes[index]
