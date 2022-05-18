import keras
import os

def build_generator(z_dimension, img_shape, num_classes):
    print("\n")
    print("Building Generator")
    print("\n")

    Batch_momentum = 0.8

    in_label = keras.Input(shape=(num_classes,))
    li = keras.layers.Embedding(num_classes, 50)(in_label)
    li = keras.layers.Dense(units = 7 * 7 * 1)(li)
    li = keras.layers.Reshape(target_shape=(7,7,num_classes))(li)

    in_z = keras.Input(shape=(z_dimension))
    zi = keras.layers.Dense(units = 7 * 7 * 512, activation = "relu")(in_z)
    zi = keras.layers.Reshape(target_shape=(7,7,512))(zi) 
    x = keras.layers.Concatenate()([li, zi])

    #Size = 16x16
    x = keras.layers.Conv2DTranspose(filters = 512, kernel_size = 5, strides=2, padding="same", use_bias = False)(x)
    x = keras.layers.BatchNormalization(momentum = Batch_momentum)(x)
    x = keras.layers.ReLU()(x)

    #Size = 32x32
    x = keras.layers.Conv2DTranspose(filters = 256, kernel_size = 5, strides=2, padding="same", use_bias = False)(x)
    x = keras.layers.BatchNormalization(momentum =Batch_momentum)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2DTranspose(filters = img_shape[-1], kernel_size = 5, strides=1, padding="same")(x)
    img = keras.layers.Activation("tanh")(x)

    model =keras.Model(
        [in_z, in_label],img
        )
    
    model.summary()
    return model


def define_gan(discriminator_model, generator_model, opimizer):
	# make weights in the discriminator not trainable
	for layer in discriminator_model.layers:
		if not isinstance(layer, keras.layers.BatchNormalization):
			layer.trainable = False

	# connect the outputs of the generator to the inputs of the discriminator
	gan_output = discriminator_model(generator_model.output)
	# define gan model as taking noise and label and outputting real/fake and label outputs
	model = keras.Model(generator_model.input, gan_output)
	# compile model
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opimizer, metrics = [])
	return model
