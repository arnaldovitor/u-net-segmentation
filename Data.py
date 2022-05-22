import nibabel as nib
import numpy as np
import tensorflow as tf

import os

class Data():
  def __init__(self, train_path, val_path, test_path):
    self.train_path = train_path
    self.val_path = val_path
    self.test_path = test_path

    self.X_train = []
    self.y_train = []
    self.X_val = []
    self.y_val = []
    self.X_test = []
    self.y_test = []

    self.__setup()

  def __preprocess(self, input, mode, n_classes):
    input = np.array(input)

    if mode == 'input':
      mx = np.max(input)
      input = input / mx
      input = np.expand_dims(input, axis=-1)
      input = tf.convert_to_tensor(input, dtype=tf.float32)
      return input
    elif mode == 'target':
      input = np.expand_dims(input, axis=-1)
      input = tf.convert_to_tensor(input, dtype=tf.int32)
      stack_list = []
      for c in range(n_classes):
        mask = tf.equal(input[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))
      annotation = tf.stack(stack_list, axis=2)
      return annotation




  def __setup(self):
    train_input = os.listdir(os.path.join(self.train_path, 'input'))
    val_input = os.listdir(os.path.join(self.val_path, 'input'))
    test_input = os.listdir(os.path.join(self.test_path, 'input'))

    for img in train_input:

      X = nib.load(os.path.join(self.train_path, 'input', img)).get_fdata()
      y = nib.load(os.path.join(self.train_path, 'target', img)).get_fdata()
      if X.shape[1] == 320:
        for idx in range(X.shape[2]):
          X_idx = self.__preprocess(X[:, :, idx, 0], 'input', 3)
          y_idx = self.__preprocess(y[:, :, idx], 'target', 3)
          self.X_train.append(X_idx)
          self.y_train.append(y_idx)


    for img in val_input:
      try:
        X = nib.load(os.path.join(self.val_path, 'input', img)).get_fdata()
        y = nib.load(os.path.join(self.val_path, 'target', img)).get_fdata()
        if X.shape[1] == 320:
          for idx in range(X.shape[2]):
            X_idx = self.__preprocess(X[:, :, idx, 0], 'input', 3)
            y_idx = self.__preprocess(y[:, :, idx], 'target', 3)
            self.X_val.append(X_idx)
            self.y_val.append(y_idx)
      except:
        continue

    for img in test_input:
      try:
        X = nib.load(os.path.join(self.test_path, 'input', img)).get_fdata()
        y = nib.load(os.path.join(self.test_path, 'target', img)).get_fdata()
        if X.shape[1] == 320:
          for idx in range(X.shape[2]):
            X_idx = self.__preprocess(X[:, :, idx, 0], 'input', 3)
            y_idx = self.__preprocess(y[:, :, idx], 'target', 3)
            self.X_test.append(X_idx)
            self.y_test.append(y_idx)
      except:
        continue

    print('Finished!\nTrain size: {}\nVal size: {} \nTest size: {}\n'.format(len(self.X_train),
                                                                             len(self.X_val),
                                                                             len(self.X_test)))

  def get_train(self):
    return np.array(self.X_train), np.array(self.y_train)

  def get_val(self):
    return np.array(self.X_val), np.array(self.y_val)

  def get_test(self):
    return np.array(self.X_test), np.array(self.y_test)


if __name__ == '__main__':
  data = Data(
    train_path='/home/arnaldo/Downloads/prostate/train/',
    val_path='/home/arnaldo/Downloads/prostate/val/',
    test_path='/home/arnaldo/Downloads/prostate/test/'
  )

  X_train, y_train = data.get_train()
  X_val, y_val = data.get_val()
  X_test, y_test = data.get_test()