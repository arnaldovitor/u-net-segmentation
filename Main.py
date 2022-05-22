import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

from Data import Data
from Visualizer import Visualizer
from UNet import UNet

model = UNet(input_shape=(320, 320, 1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.MeanIoU(num_classes=3)])

model.summary()

data = Data(
    train_path='/home/arnaldo/Downloads/prostate/train/',
    val_path='/home/arnaldo/Downloads/prostate/val/',
    test_path='/home/arnaldo/Downloads/prostate/test/'
)

X_train, y_train = data.get_train()
X_val, y_val = data.get_val()
X_test, y_test = data.get_test()

history = model.fit(X_train, y_train, epochs=50, batch_size=6, shuffle=True, validation_data=(X_val, y_val))
np.save('history_7.npy', history.history )

plt.plot(history.history['loss'], label='train-loss')
plt.plot(history.history['val_loss'], label='val-loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('exp_7.png')
plt.clf()

model.save('exp_7.h5')

# visualizer = Visualizer(weights='/home/arnaldo/PycharmProjects/u-net-segmentation/exp_6.h5',
#                         X_test=X_test,
#                         y_test=y_test,
#                         n_classes=3)
#
# visualizer.run(sample_idx=15)