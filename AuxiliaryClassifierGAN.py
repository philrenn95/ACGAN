import os
from sklearn.model_selection import learning_curve
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from ACGANDiscriminator import build_discriminator
from ACGANGenerator import *
from mnistData import *
import cv2
import time


class ACGAN:
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
        self.discriminator_model = build_discriminator(img_shape=self.img_shape,optimizer=self.optimizer, num_classes=self.num_classes)
        self.generator_model = build_generator(z_dimension=self.z_dimension,img_shape=self.img_shape,num_classes=self.num_classes)
        self.GAN_model = define_gan(self.discriminator_model, self.generator_model, self.optimizer)

    def generate_latent_points(self, batch_size):
        self.z_input = np.random.randn(self.z_dimension*batch_size).reshape(batch_size, self.z_dimension)
        labels = np.random.randint(0, self.num_classes, batch_size)
        labels = to_categorical(labels , 10)
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
        train_labels = self.y_train[self.rand_idxs]
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

            if (epoch % sample_interval) == 0:
                if (epoch % (5*sample_interval)) == 0:
                    self.sample_images()

            if epoch > epochs:
                print("Training finished")
                time.sleep(3)
                break

        self.sample_images("final")
    
    def sample_images(self):
        r, c = 3, 4
        noise = np.random.normal(loc=0.0, scale=1.0, size=(r * c, self.z_dimension))
        labels = np.random.randint(0, self.num_classes,r*c)
        labels = to_categorical(labels , 10)
        gen_imgs = self.generator_model.predict([noise, labels])
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt , :, :, 0])
                axs[i, j].axis("off")
                cnt += 1

        img_name = f"{self.g_sample}.png"
        IMAGES_PATH = "C://Users//renne//Desktop//Master//Projekte//GAN//AuxiliaryClassifierGAN//IMAGES//"
        fig.savefig(os.path.join(IMAGES_PATH, img_name))
        self.g_sample = self.g_sample + 1
        plt.close()


if __name__ == "__main__":

    acgan = ACGAN()
    acgan.data_import()


    print("Creating new models...")
    acgan.build_new_models()
    time.sleep(3)
    print("...Models succesfully created")
    time.sleep(3)

    acgan.train(
        epochs=100_000,
        batch_size=64,
        sample_interval=100
    )

