import keras
import os

def build_discriminator(img_shape, optimizer, load_weights):

    Dropout_rate = 0.5
    Batch_momentum = 0.8

    img = keras.layers.Input(shape=(img_shape))

    #Conv block 1 image size 28x28
    x = keras.layers.Conv2D(filters=32, kernel_size=3, strides = 2, padding = "same")(img)
    x = keras.layers.BatchNormalization(momentum = Batch_momentum)(x)
    x = keras.layers.LeakyReLU(alpha = 0.2)(x)
    x = keras.layers.Dropout(Dropout_rate)(x)

    #Conv block 1 image size 14x14
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides = 2, padding = "same")(img)
    x = keras.layers.BatchNormalization(momentum = Batch_momentum)(x)
    x = keras.layers.LeakyReLU(alpha = 0.2)(x)
    x = keras.layers.Dropout(Dropout_rate)(x)

    #Conv block 2 image size 7x7
    x = keras.layers.Conv2D(filters=128, kernel_size= 3, strides = 2, padding = "same")(x)
    x = keras.layers.BatchNormalization(momentum = Batch_momentum)(x)
    x = keras.layers.LeakyReLU(alpha = 0.2)(x)
    x = keras.layers.Dropout(Dropout_rate)(x)
    
    flatten = keras.layers.Flatten()(x)

    #Real or fake prediction
    d_pred = keras.layers.Dense(units=1, activation="sigmoid")(flatten)
    
    #Label prediction
    l_pred = keras.layers.Dense(units=1, activation="sigmoid")(flatten)

    model = keras.Model(
        img,[d_pred, l_pred]
    )

    model.summary()

    model.compile(loss = ["binary_crossentropy","binary_crossentropy"],optimizer = optimizer,metrics = ["accuracy"])
    return model


