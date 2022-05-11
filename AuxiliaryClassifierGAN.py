import os
from sklearn.model_selection import learning_curve
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
from ACGANDiscriminator import build_discriminator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ACGANGenerator import *
from Data import Data
from data_pipeline import Data_pipeline
import winsound
import time


class CGAN:
    def __init__(self):
        # Model parameters
        #Shape of the Images
        self.img_rows = 28
        self.img_cols = 28
        self.img_depth = 1
        self.img_shape = (
            self.img_rows,
            self.img_cols,
            self.img_depth
        )

        #Number of different classes
        self.num_classes = 10
        self.z_dimension = 100
        self.g_sample = 0 
        self.OOM = 0
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-4, beta_1 = 0.5, beta_2 = 0.5)


    def build_new_models(self):
        self.discriminator_model = build_discriminator(img_shape=self.img_shape,optimizer=self.optimizer)
        self.generator_model = build_generator(z_dimension=self.z_dimension,img_shape=self.img_shape,num_classes=self.num_classes, load_weights = self.load_weights)
        self.GAN_model = define_gan(self.discriminator_model, self.generator_model, self.optimizer)

    def generate_latent_points(self, batch_size):
        self.z_input = np.random.randn(self.z_dimension*batch_size).reshape(batch_size, self.z_dimension)
        labels = np.random.randint(0, self.num_classes, batch_size)
        return[self.z_input, labels]

    def generate_fake_samples(self):
        z_input, labels_input = self.generate_latent_points(self.half_batch)
        images = self.generator_model.predict([z_input, labels_input])
        y = (np.zeros((self.half_batch, 1)))
        return [images, labels_input], y

    def generate_real_samples(self):
        #Choosing random images from all train data (x_train.shape[0]) 
        self.rand_idxs = np.random.randint(0, self.x_train.shape[0], self.half_batch)
        train_imgs = self.x_train[self.rand_idxs]
        #Transforming the images randomly
        #train_imgs = self.datagen.random_transform(train_imgs)
        train_labels = self.y_train[self.rand_idxs]
        #train_imgs,train_labels= self.pipeline.train_import(batch_size = self.batch_size)
        y = (np.ones((self.half_batch,1)))
        return [train_imgs, train_labels], y
    
    def data_import(self):
        # Load data
        data = MNIST()
        self.x_train, self.y_train = data.get_train_set()
        self.x_validation, self.y_validation = data.get_test_set()
        return

    def train(self, epochs, batch_size, sample_interval):
        self.half_batch = int(batch_size / 2)
        self.batch_size = batch_size

        for epoch in range(epochs):
            start_epoch_timer = time.time()
            #generated_imgs = self.generator_model([noise, train_labels], training=False)
            [generated_imgs, labels_fake], y_fake = self.generate_fake_samples()
            [train_imgs, labels_real], y_real = self.generate_real_samples()
            # Train the discriminator
            d_loss_r = self.discriminator_model.train_on_batch(train_imgs, [y_real, labels_real])
            d_loss_f = self.discriminator_model.train_on_batch(generated_imgs, [y_fake, labels_fake])
            # Train the generator
            y_gan = (np.ones((batch_size, 1)))
            z_input, labels_input = self.generate_latent_points(batch_size)
            g_loss= self.GAN_model.train_on_batch([z_input, labels_input], [y_gan, labels_input])
            end_epoch_timer =time.time()

            print()
            print(str(epoch) + " - Duration per epoch:  " + 
            str(round(end_epoch_timer-start_epoch_timer,3)) + "s" + "\n" + 
            f"Discriminator_loss_real:  {round(d_loss_r[1],3)}"+ "\n"
            f"Discriminator_loss_fake:  {round(d_loss_f[1],3)}"+ "\n"
            f"Generator_loss:           {round(g_loss[1],3)}" +"\n"
            f"Discriminator_acc_label:  {round(d_loss_r[4],3)}")

            if (epoch % 10) == 0:
                if self.x_validation.any:
                    validation_ones =  np.ones((self.x_validation.shape[0], 1))
                    #self.x_validation,self.y_validation= self.pipeline.validation_import(batch_size = self.batch_size)
                    validation = self.discriminator_model.evaluate(self.x_validation, [validation_ones, self.y_validation], verbose = 0, batch_size=40 )
                    print(f"validation loss :{round(validation[2],3)}"+ 
                        f"                      validation acc :{round(validation[4],3)}" )
                    from sklearn.metrics import confusion_matrix
                    prediction = self.discriminator_model.predict(self.x_validation)
                    prediction = np.round(prediction[1])
                    print(confusion_matrix( prediction, self.y_validation))

            if (epoch % sample_interval) == 0:
                print("Saving AI")
                Discriminator_weights_save_dir = PATH + os.sep + "Model" + os.sep +"discriminator_weights.h5"
                self.discriminator_model.save_weights(Discriminator_weights_save_dir)
                Generator_weights_save_dir = PATH + os.sep + "Model" + os.sep +"generator_weights.h5"
                self.generator_model.save_weights(Generator_weights_save_dir)

                if (epoch % (5*sample_interval)) == 0:
                    self.sample_images()

            if epoch > epochs:
                print("Training finished")
                time.sleep(3)
                Discriminator_save_dir = PATH + os.sep + "Model" + os.sep +"discriminator.h5"
                self.discriminator_model.save(Discriminator_save_dir)
                Generator_save_dir = PATH + os.sep + "Model" + os.sep +"generator.h5"
                self.generator_model.save(Generator_save_dir)
                break

        self.sample_images("final")
    
    def sample_images(self):
        r, c = 3, 4
        noise = np.random.normal(loc=0.0, scale=1.0, size=(r * c, self.z_dimension))
        labels = np.random.randint(0, self.num_classes,r*c)
        classification_labels = ["poor", "good"]
        gen_imgs = self.generator_model.predict([noise, labels])
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0],cmap="gray")
                axs[i, j].set_title(f"{classification_labels[labels[cnt]]}")
                axs[i, j].axis("off")
                cnt += 1

        img_name = f"{self.g_sample}.png"
        fig.savefig(os.path.join(IMAGES_PATH, img_name))
        self.g_sample = self.g_sample + 1
        plt.close()


if __name__ == "__main__":

    cgan = CGAN()
    cgan.data_import()

    if os.path.exists(os.getcwd() + os.sep + "Model"):
        feedback = None
        while True:
            feedback = input("Do you want to load a pretrained Model?  \nYES = Y No = N :")

            if feedback == "y" or feedback == "Y":
                print("Loading pretrained models...")
                cgan.build_new_models(load_weights=True)
                print("...Models sucesfully loaded")
                time.sleep(3)
                break

            elif feedback == "n"or feedback == "N":
                print("Creating new models...")
                cgan.build_new_models(load_weights=False)
                time.sleep(3)
                print("...Models succesfully created")
                time.sleep(3)
                break

            else:
                print("No correct input \nPlease type y for Yes and n for No")
    
    else:
        cgan.build_new_models(load_weights=False)


    cgan.train(
        epochs=100_000,
        batch_size=32,
        sample_interval=100
    )

