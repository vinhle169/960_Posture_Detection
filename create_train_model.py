import keras
from keras.models import Sequential
from keras import layers, initializers
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import matthews_corrcoef
import openpose_helper
import numpy as np
import matplotlib.pyplot as plt
import keras
import json
import cv2
import os

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class pose_classifier():
    '''
    Class for binary classification of images of people sitting
    Labels will be either 1(good) or 0(bad)
    Functions for creating data and labels
    '''

    def __init__(self, path_to_model='openpose_model'):
        '''
        Initialize class and load in openpose model
        '''
        self.openpose = openpose_helper.load_model(path_to_model)
        self.x = None
        self.y = None

    def find_keypoints(self, image_path, visualize=False, lower_bound=5, threshold=0.5):
        '''
        Find keypoints on an image and return them
        '''
        points, frame, count = openpose_helper.find_keypoints(image_path, self.openpose, visualize=visualize, threshold=0.5)
        if visualize and count > lower_bound:
            cv2.imshow('keypoints', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return points, count

    def label_data(self, image_path, save_name='keypoint_data.json'):
        '''
        Label and save all data in a given path
        0 is bad, 1 is good
        '''
        data, labels, images = [], [], []
        for image in os.listdir(image_path):
            keypoints, count = self.find_keypoints(f'{image_path}/{image}', visualize=True)
            if count >= 5:
                images.append(image)
                data.append(keypoints)
                if 'bad' in image:
                    labels.append(0)
                elif 'good' in image:
                    labels.append(1)

        json_data = {'data': data, 'labels': labels, 'used_images': images}
        print(len(images))
        with open(save_name, 'w') as json_file:
            json.dump(json_data, json_file)

    def open_data(self, path):
        with open(path, 'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict

    def load_data(self, path, train=False, names=False):
        data_dict = self.open_data(path)
        x = data_dict['data']
        y = data_dict['labels']
        x = np.array([np.array(i).flatten() for i in x])
        y = np.array([y]).T
        if train:
            self.x = x
            self.y = y
        if names:
            z = data_dict['used_images']
            return x, y, z
        return x, y

    def build_mlp(self):
        '''
        Building our multilayer perceptron model for posture analysis based on keypoint data
        '''
        init = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        mlp = Sequential()
        mlp.add(layers.Dense(units=60, input_shape=(30,), kernel_initializer=init, bias_initializer=init, activation='relu'))
        mlp.add(layers.Dense(units=120, kernel_initializer=init, bias_initializer=init, activation='relu'))
        mlp.add(layers.Dense(units=60, kernel_initializer=init, bias_initializer=init, activation='relu'))
        mlp.add(layers.Dense(units=1, activation='sigmoid'))
        mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(mlp.summary())
        self.mlp = mlp

    def train_mlp(self, data_path='keypoint_data.json', save_path='mlp.h5', epochs=5, visualize=True):
        '''
        Train our multilayer perceptron model
        '''
        if type(self.x) != list:
            x, y = self.load_data(data_path, train=True)
        else:
            x, y = self.x, self.y
        num_epochs = epochs
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        total_acc_history, total_val_acc_history, total_test_hist = [], [], []
        if not hasattr(self, 'mlp'):
            self.build_mlp()

        for train_index, val_index in skf.split(x, y):
            x_train, x_val = np.array([x[i] for i in train_index]), np.array([x[i] for i in val_index])
            y_train, y_val = np.array([y[i] for i in train_index]), np.array([y[i] for i in val_index])
            history = self.mlp.fit(x_train, y_train,
                                   epochs=num_epochs,
                                   # callbacks=callbacks_list,
                                   validation_data=(x_val, y_val))
            if visualize:
                total_acc_history.append(history.history['accuracy'])
                total_val_acc_history.append(history.history['val_accuracy'])

        if visualize:
            total_acc_history = np.array(total_acc_history).flatten()
            total_val_acc_history = np.array(total_val_acc_history).flatten()

            plt.plot(total_acc_history)
            plt.plot(total_val_acc_history)
            plt.plot(total_test_hist)
            plt.title(f'Model Accuracy:{save_path[:-3]}')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.legend(['Training', 'Validation'], loc='upper left')
            plt.show()
        self.mlp.save(save_path)
        return total_acc_history, total_val_acc_history

    def load_mlp(self, path="mlp.h5"):
        reconstructed_model = keras.models.load_model(path)
        if hasattr(self, 'mlp'):
            del self.mlp
        self.mlp = reconstructed_model

    def test_mlp(self, test_path="test_keypoint_data.json", model_path="mlp.h5", individual=False):
        if hasattr(self, 'mlp'):
            print('yes')
            pass
        else:
            self.load_mlp(model_path)
        if not individual:
            x, y = self.load_data(test_path)
            metrics = self.mlp.evaluate(x, y)
        else:
            ypred, ytrue = [], []
            x, y, z = self.load_data(test_path, names=True)
            accuracy, names = [],[]
            for i in range(len(x)):
                pred = self.mlp.predict(np.array([x[i]]))
                ypred.append(pred)
                ytrue.append(y[i])
                if y[i]==0 and pred<.5:
                    accuracy.append(1)
                elif y[i]==1 and pred>.5:
                    accuracy.append(1)
                else:
                    accuracy.append(-1)
                names.append(z[i])
            metrics = [accuracy, names, ytrue, ypred]
        return metrics

    # def build_knn(self):
    #     '''
    #     Building our k nearest neighbors model for posture analysis based on keypoints
    #     '''
    #     knn_model = KNeighborsRegressor(n_neighbors=2)
    #     self.knn = knn_model

    # def train_knn(self, data_path='keypoint_data.json'):
    #     '''
    #     Train our multilayer perceptron model
    #     '''
    #     if type(self.x) != list:
    #         x, y = self.load_data(data_path, train=True)
    #     else:
    #         x, y = self.x, self.y
    #     skf = StratifiedKFold(n_splits=5, shuffle=True)
    #     for train_index, val_index in skf.split(x, y):
    #         x_train, x_val = np.array([x[i] for i in train_index]), np.array([x[i] for i in val_index])
    #         y_train, y_val = np.array([y[i] for i in train_index]), np.array([y[i] for i in val_index])
    #         self.knn.fit(x, y)
    #     self.knn.kneighbors_graph(x_val, 2)


if __name__ == '__main__':
    # print('begin')
    
    v = pose_classifier()
    v.load_mlp('mlp1.h5')

    w = pose_classifier()
    w.load_mlp('mlp2.h5')

    x = pose_classifier()
    x.load_mlp('mlp3.h5')

    y = pose_classifier()
    y.load_mlp('mlp4.h5')

    z = pose_classifier()
    z.load_mlp('mlp5.h5')
    models = [v,w,x,y,z]
    y_true, y_pred = [], []
    for model in models:
        metrics = model.test_mlp(individual=True)
        y_true.append(metrics[2])
        y_pred.append(metrics[3])

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_pred = np.where(y_pred < .5, y_pred, 1)
    y_pred = np.where(y_pred > .5, y_pred, 0)
    print(matthews_corrcoef(y_true, y_pred))
    # x.load_mlp('mlp2.h5')
    # metrics = x.test_mlp(individual=True)
    # print(metrics)

    # x.build_knn()
    # x.train_knn()
