import tensorflow as tf

from tensorflow.keras import layers


def double_conv_block(x, n_filters, kernel_size_c1, kernel_size_c2):
    x = layers.Conv2D(n_filters, kernel_size_c1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, kernel_size_c2, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters, kernel_size_c1, kernel_size_c2):
    f = double_conv_block(x, n_filters, kernel_size_c1, kernel_size_c2)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, kernel_size_t, kernel_size_c1, kernel_size_c2):
    x = layers.Conv2DTranspose(n_filters, kernel_size_t, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters, kernel_size_c1, kernel_size_c2)
    return x

# [[1, 1], [1, 1], [2, 2], [1, 1], [1, 1], [2, 2], [4, 1], [4, 1], [2, 2], [4, 1], [4, 1], [2, 2]]
def UNet(input_shape):
    inputs = layers.Input(shape=(input_shape))

    filters_factor = 32

    f1, p1 = downsample_block(inputs, filters_factor, kernel_size_c1=5, kernel_size_c2=5)
    f2, p2 = downsample_block(p1, filters_factor*2, kernel_size_c1=5, kernel_size_c2=5)
    f3, p3 = downsample_block(p2, filters_factor*4, kernel_size_c1=3, kernel_size_c2=3)
    f4, p4 = downsample_block(p3, filters_factor*8, kernel_size_c1=3, kernel_size_c2=3)

    bottleneck = double_conv_block(p4, filters_factor*16, kernel_size_c1=3, kernel_size_c2=3)

    u5 = upsample_block(bottleneck, f4, filters_factor*8, kernel_size_t=2, kernel_size_c1=3, kernel_size_c2=3)
    u6 = upsample_block(u5, f3, filters_factor*4, kernel_size_t=2, kernel_size_c1=3, kernel_size_c2=3)
    u7 = upsample_block(u6, f2, filters_factor*2, kernel_size_t=2, kernel_size_c1=5, kernel_size_c2=5)
    u8 = upsample_block(u7, f1, filters_factor, kernel_size_t=2, kernel_size_c1=5, kernel_size_c2=5)

    outputs = layers.Conv2D(3, 1, padding="same", activation="softmax")(u8)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model
