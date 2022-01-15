from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


def conv_block_1(inputs, num_filters):

    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def conv_block_2(inputs, num_filters):

    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def conv_block_3(inputs, num_filters):

    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block_1(inputs, num_filters):
    x = conv_block_1(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def encoder_block_2(inputs, num_filters):
    x = conv_block_2(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def encoder_block_3(inputs, num_filters):
    x = conv_block_3(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block_2(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block_2(x, num_filters)
    return x

def decoder_block_1(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block_1(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block_1(inputs, 32)
    s2, p2 = encoder_block_1(p1, 64)
    s3, p3 = encoder_block_2(p2, 128)
    s4, p4 = encoder_block_2(p3, 256)

    b1 = conv_block_3(p4, 512)

    d1 = decoder_block_2(b1, s4, 256)
    d2 = decoder_block_2(d1, s3, 128)
    d3 = decoder_block_1(d2, s2, 64)
    d4 = decoder_block_1(d3, s1, 32)

    outputs = Conv2D(3, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_unet(input_shape)
    model.summary()