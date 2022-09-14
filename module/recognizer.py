import tensorflow as tf
import numpy as np
import pickle

class Recognizer_NN:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, pca_img):
        y_pred = self.model.predict(pca_img)
        result = np.argmax(y_pred)
        confidence = max(y_pred[0])
        return result, confidence

class Recognizer_Euclidean:
    def __init__(self, pca_dict_path):
        self.pca_dict = pickle.load(open(pca_dict_path,'rb'))

    def calculate_euclidean_distance(self, a, b):
        return np.linalg.norm(a-b)

    def predict(self, pca_img):
        min_dist = float('inf')
        pred = -1
        for key, values in self.pca_dict.items():
            for val in values:
                dist = self.calculate_euclidean_distance(val, pca_img)
                if dist < min_dist:
                    min_dist = dist
                    pred = key

        return pred, min_dist
