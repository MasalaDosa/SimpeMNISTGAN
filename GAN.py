
#pip install numpy
#pip install tensorflow
#pip install keras
#pip install matplotlib

import os
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers import Dense, Reshape, BatchNormalization, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from copy import deepcopy

# The size of the noise used as input to the generator
NOISE_SIZE = 100
# The MNIST images are 28x28 grayscale.
IMAGE_W = 28
IMAGE_H = 28

# How many blocks to use in the generator, and the block's initial size.
# Eech block is a dense layer, with a leaky ReLU activation and a batch normalization.
# The first block has an input sized according to the NOISE_SIZE above and an output of BLOCK_SIZE
# Subsequent layers double the block size.
# Finally a dense layer converts the output to IMAGE_W x IMAGE_H with a tanh activation and this is reshaped for output.
GENERATOR_NUMBER_OF_BLOCKS = 4
GENERATOR_INIT_BLOCK_SIZE = 128

# How many epochs to train the gan with.
NUMBER_OF_EPOCHS = 100
# The batch size for training.
# For each batch in an epoch a coin is tossed weighted by TOSS_CHANCE_REAL_OR_FAKE and images are then taken from
# either the training data or generated
# These images are then used to train the discriminator.
# Next the same number of images are based through the whole GAN (generator + discriminator)
# in order to train the generator.
# When there are no longer enough real images left in the training data the epoch ends.
BATCH_SIZE = 128
TOSS_CHANCE_REAL_OR_FAKE = 0.5
# When training the generator occasional incorrect labelling can help.
# This determines the chance of a correct label.
TOSS_CHANCE_REAL_LABEL = 0.9
# After this many batchs/epochs a checkpoint is performed
# - some generated images are saved in the output_images dir and the generator model is saved in the output_models dir.
CHECKPOINT = 40


# Loads and scales the MNIST dataset
def load_MNIST():
    (X_train, Y_train), (_, _) = mnist.load_data()
    # Scale from -1 to 1
    X_train = (np.float32(X_train) - 127.5) / 127.5
    # expand to (60000, 28, 28, 1)
    X_train = np.expand_dims(X_train, axis=3)
    return X_train


# A weighted coin toss
def flip_coin(chance=0.5):
    return np.random.binomial(1, chance)


# Generates noise to use as input into the generator/gan
def generate_noise(instances):
    return np.random.normal(0, 1, (instances, NOISE_SIZE))


# Performs a checkpoint
def checkpoint(generator, label):
    checkpoint_generate_images(generator, str(label))
    checkpoint_save_model(generator, str(label))
    return


# Generates and saves some images to use in a checkpoint
def checkpoint_generate_images(generator, label):
    if not os.path.exists("output_images"):
        os.mkdir("output_images")
    plot_filename = os.path.join("output_images", label + ".png")
    noise = generate_noise(25)
    images = generator.predict(noise)
    plt.figure(figsize=(20, 20))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        image = images[i, :, :]
        image = np.reshape(image, [IMAGE_H, IMAGE_W])
        image = (255 * (image - np.min(image)) / np.ptp(image)).astype(int)
        plt.imshow(image, cmap='gray')

        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close("all")


# Saves the generator model at a checkpoint.
def checkpoint_save_model(generator, label):
    if not os.path.exists("output_models"):
        os.mkdir("output_models")
    model_filename = os.path.join("output_models", label + "")
    generator.save(filepath=model_filename, overwrite=True, include_optimizer=True)


# Builds the generator
def build_generator():
    optimizer = Adam(lr=0.0002, decay=8e-9)
    model = Sequential()

    block_size = GENERATOR_INIT_BLOCK_SIZE
    number_of_blocks = GENERATOR_NUMBER_OF_BLOCKS

    model.add(Dense(block_size, input_shape=(NOISE_SIZE,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    for i in range(number_of_blocks - 1):
        block_size *= 2
        model.add(Dense(block_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(IMAGE_W * IMAGE_H, activation='tanh'))
    model.add(Reshape((IMAGE_W, IMAGE_H, 1)))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# Builds the discriminator
def build_discriminator():
    capacity = IMAGE_W * IMAGE_H
    shape = (IMAGE_W, IMAGE_H, 1)

    optimizer = Adam(lr=0.0002, decay=8e-9)
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(capacity, input_shape=shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(int(capacity / 2)))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# Combine the generator and discriminator into a GAN
def build_GAN(generator, discriminator):
    optimizer = Adam(lr=0.0002, decay=8e-9)
    # Disable training of discriminator when training the GAN model
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

# Train the GAN
def train():
    X_train = load_MNIST()
    generator = build_generator()
    print(generator.summary())
    discriminator = build_discriminator()
    print(discriminator.summary())
    gan = build_GAN(generator, discriminator)
    print(gan.summary())

    for epoch in range(NUMBER_OF_EPOCHS):
        batch = 0

        X_train_temp = deepcopy(X_train)
        while len(X_train_temp) > BATCH_SIZE:
            # Firstly train the discriminator
            if flip_coin(TOSS_CHANCE_REAL_OR_FAKE):
                print("Training discriminator on real images.")
                starting_idx = np.random.randint(0, len(X_train_temp) - BATCH_SIZE)
                real_images_raw = X_train_temp[starting_idx: (starting_idx + BATCH_SIZE)]
                x_batch = real_images_raw.reshape(BATCH_SIZE, IMAGE_W, IMAGE_H, 1)
                # These are real, so label accordingly
                y_batch = np.ones([BATCH_SIZE, 1])

                X_train_temp = np.delete(X_train_temp, range(starting_idx, (starting_idx + BATCH_SIZE)), 0)
                print("Real images remaining in this epoch: " + str(len(X_train_temp)))
            else:
                print("Training discriminator on generated images")
                noise = generate_noise(BATCH_SIZE)
                x_batch = generator.predict(noise)
                # These are fake so label them accordingly.
                y_batch = np.zeros([BATCH_SIZE, 1])

            discriminator_loss = discriminator.train_on_batch(x_batch, y_batch)

            # Secondly train the generator
            # Occasionally mis-label when training the generator
            if flip_coin(TOSS_CHANCE_REAL_LABEL):
                y_generated_labels = np.ones([BATCH_SIZE, 1])
            else:
                y_generated_labels = np.zeros([BATCH_SIZE, 1])
            # Generate noise
            noise = generate_noise(BATCH_SIZE)

            # Now train the generator
            generator_loss = gan.train_on_batch(noise, y_generated_labels)

            print("Epoch: " + str(epoch) + " Batch: " + str(batch) +
                  " Discriminator Loss: " + str(discriminator_loss) +
                  " Generator Loss: " + str(generator_loss))

            if batch % CHECKPOINT == 0:
                label = str(epoch) + "_" + str(batch)
                checkpoint(generator, label)

            batch += 1
        print("Epoch: " + str(epoch) + " completed.")

        if epoch % CHECKPOINT == 0:
            checkpoint(generator, epoch)

if __name__ == "__main__":
    train()