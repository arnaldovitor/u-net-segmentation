import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns


class Visualizer():
    def __init__(self, weights, X_test, y_test, n_classes):
        self.weights = weights
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes
        self.colors = sns.color_palette(None, self.n_classes)
        self.model = None
        self.__setup()

    def __setup(self):
        self.model = tf.keras.models.load_model(self.weights)

    def __give_color(self, y):
        seg_img = np.zeros((y.shape[0], y.shape[1], 3)).astype('float')
        for c in range(self.n_classes):
            segc = (y == c)
            print(self.colors[c])
            seg_img[:, :, 0] += segc * (self.colors[c][0] * 255.0)
            seg_img[:, :, 1] += segc * (self.colors[c][1] * 255.0)
            seg_img[:, :, 2] += segc * (self.colors[c][2] * 255.0)

        return seg_img

    def run(self, sample_idx):
        X = self.X_test[sample_idx]
        y = self.__give_color(np.argmax(self.y_test[sample_idx], axis=2))
        y_pred = self.model.predict(X[np.newaxis, ...])
        y_pred = self.__give_color(np.argmax(y_pred[0, :, :, :], axis=2))

        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,8))

        ax[0].imshow(X)
        ax[0].set_title('X')
        ax[0].axis('off')

        ax[1].imshow(y.astype(np.uint8))
        ax[1].set_title('y_true')
        ax[1].axis('off')

        ax[2].imshow(y_pred.astype(np.uint8))
        ax[2].set_title('y_pred')
        ax[2].axis('off')
        plt.tight_layout()
        plt.show()
