import tensorflow as tf
import random
import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D


class DogTensor:

    def __init__(self):
        self.model = self.open_model("nela_dog.model")

    # This line is needed due to an OSX Quirk
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Categories to "learn" from
    # "Nela" does not yet have enough data to be significant.
    CATEGORIES = ["Dog", "Nela"]

    IMG_SIZE = 50

    def pickle_save(self, features, labels):
        """ pickle file creation


          Creates new "pickle" files and fills them with binary data

          Args:
            features (numpy array): array of images as numpy arrays
            labels (numpy array): array of numeric representations of the CATEGORIES

          Returns:
              nothing

        """
        pickle_out = open("features.pickle", "wb")
        pickle.dump(features, pickle_out)
        pickle_out.close()

        pickle_out = open("labels.pickle", "wb")
        pickle.dump(labels, pickle_out)
        pickle_out.close()

    def open_and_load_pickle(self, pickle_name, retries=0):
        """ opens an existing pickle files

          loads a pickle file that is located on the same level
          as this script and has provided pickleName. In case no pickle file with given name was found,
          the data extraction and creation of a pickle file is triggered

          Args:
            retries: the current retry count (default is 0)
            pickle_name: name of the pickle file to be loaded

          Returns:
              data from the pickle file that matches the provided pickle_name or None
        """
        data = None
        if retries > 1:
            return data
        try:
            pickle_in = open(pickle_name, "rb")
            data = pickle.load(pickle_in)
            pickle_in.close()
        except OSError:
            self.create_data()
            retries += 1
            data = self.open_and_load_pickle(pickle_name, retries)
        finally:
            return data

    def open_model(self, model_name, retries=0):
        """ opens an existing model

          opens and loads a model that is located on the same level
          as this script and matches the provided model_name. In case no model with given name was found,
          the creation and training of the model is triggered

          Args:
            retries: the current retry count (default is 0)
            model_name: name of the pickle file to be loaded

          Returns:
              loaded_model: that matches the provided model_name or None
        """
        loaded_model = None
        if retries > 1:
            return loaded_model
        try:
            loaded_model = tf.keras.models.load_model(model_name)
        except (ImportError, OSError):
            self.build_model(self.open_and_load_pickle("features.pickle"), self.open_and_load_pickle("labels.pickle"),
                             model_name)
            retries += 1
            loaded_model = self.open_model(model_name, retries)
        finally:
            return loaded_model

    def create_data(self):
        """ create data arrays from images and saves them as pickle


          Creates and tweaks numpy arrays that hold the data representation
          of the images provided for training. After processing the images and extracting the data,
          save the images as well as the numeric representation of the categories as pickle files

          Returns:
              nothing

        """
        training_data = []
        features = []
        labels = []
        for category in self.CATEGORIES:
            class_index = self.CATEGORIES.index(category)
            for img in os.listdir(category):
                try:
                    img_array = cv2.imread(os.path.join(category, img), cv2.IMREAD_GRAYSCALE)
                    img_array_resized = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    training_data.append([img_array_resized, class_index])
                except Exception as e:
                    pass

        random.shuffle(training_data)
        for feature, label in training_data:
            features.append(feature)
            labels.append(label)

        self.pickle_save(np.array(features).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1), np.array(labels))

    def build_model(self, features, labels, model_name):
        """builds and trains a model based on input features and labels


          Creates a and trains a model based on the input features and labels.
          A static number of layers is added to the model and a
          Tensorboard is created to review the the training cycles

          Args:
              features (numpy array): array of images as numpy arrays which is reshaped to match the image size.
              labels (numpy array): array of numeric representations of the CATEGORIES

          Returns:
              nothing

        """
        features = features / 255.0

        dense_layer = 1
        layer_size = 64
        conv_layer = 3
        name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
        print(name)

        sequential_model = Sequential()

        sequential_model.add(Conv2D(layer_size, (3, 3), input_shape=features.shape[1:]))
        sequential_model.add(Activation('relu'))
        sequential_model.add(MaxPooling2D(pool_size=(2, 2)))

        sequential_model.add(Conv2D(layer_size, (3, 3)))
        sequential_model.add(Activation('relu'))
        sequential_model.add(MaxPooling2D(pool_size=(2, 2)))

        sequential_model.add(Flatten())
        sequential_model.add(Dense(layer_size))
        sequential_model.add(Activation('relu'))

        sequential_model.add(Dense(1))
        sequential_model.add(Activation('sigmoid'))

        tensorboard = TensorBoard(log_dir="logs/{}".format(name))

        sequential_model.compile(loss='binary_crossentropy',
                                 optimizer='adam',
                                 metrics=['accuracy'],
                                 )

        sequential_model.fit(features, labels,
                             batch_size=32,
                             epochs=10,
                             validation_split=0.3,
                             callbacks=[tensorboard])
        sequential_model.save(model_name)

    def prepare(self, filepath=None, frame=None):
        """prepares an image to be predicted by the model


          Creates a and trains a model based on the input features and labels.
          A static number of layers is added to the model and a
          Tensorboard is created to review the the training cycles

          Args:
              filepath: path to the image which should be prepared for prediction


          Returns:
              new_array: containing the image data in a compatible format for being predicted

        """
        if frame is not None:
            new_array = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE)) / 255
            return new_array.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        if filepath is not None:
            img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE)) / 255
            return new_array.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

    def predict_if_dog(self, frame=None, filepath=None):
        prediction = self.model.predict([self.prepare(frame=frame, filepath=filepath)])
        return prediction


if __name__ == "__main__":
    print(DogTensor().predict_if_dog(filepath="nela.jpg"))
