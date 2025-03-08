#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

batch_size = 128
latent_dim = 100
epochs = 15

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)
IMG_SIZE = 32

def preprocess(example):
    image = (tf.cast(example['image'], tf.float32) / 127.5) - 1.0
    return image

ds_train = ds_train.map(preprocess).cache().batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).cache().batch(32).prefetch(tf.data.AUTOTUNE)

# def build_generator(latent_dim=100, style_dim=256):
#     # Mapping Network - Simplified
#     mapping_inputs = layers.Input(shape=(latent_dim,))
#     x = layers.Dense(style_dim)(mapping_inputs)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.Dense(style_dim)(x)
#     x = layers.LeakyReLU(0.2)(x)
#     style_codes = layers.Dense(style_dim)(x)

#     # Synthesis Network - Simplified
#     style_reshape = layers.Dense(4 * 4 * 256)(style_codes)
#     style_reshape = layers.Reshape((4, 4, 256))(style_reshape)

#     x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(style_reshape)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
#     x = layers.LeakyReLU(0.2)(x)
#     #add this layer to upscale to 32x32
#     x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
#     return tf.keras.Model(mapping_inputs, x)

def build_generator(latent_dim=100): #removed style_dim
    # Synthesis Network - Simplified
    inputs = layers.Input(shape=(latent_dim,))
    style_reshape = layers.Dense(4 * 4 * 256)(inputs)
    style_reshape = layers.Reshape((4, 4, 256))(style_reshape)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(style_reshape)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
    return tf.keras.Model(inputs, x)

def build_discriminator(input_shape=(32, 32, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs) #reduced filters
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x) #reduced filters
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs, outputs)

generator_optimizer = optimizers.Adam(learning_rate=0.0001, rho=0.8)
discriminator_optimizer = optimizers.Adam(learning_rate=0.0001, rho=0.8)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = build_generator()
discriminator = build_discriminator()

import time
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def train_step(images, generator, discriminator, latent_dim):
    noise = tf.random.normal([images.shape[0], latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, images, discriminator)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Gradient Clipping
    gradients_of_generator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_generator]
    gradients_of_discriminator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_discriminator]

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def gradient_penalty_loss(real_images, fake_images, discriminator):
    alpha = tf.random.uniform([real_images.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated_output = discriminator(interpolated_images, training=True)
    gradients = tape.gradient(interpolated_output, interpolated_images)
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_penalty = tf.reduce_mean(tf.square(tf.sqrt(gradients_sqr_sum + 1e-12) - 1.0))
    return gradient_penalty

def discriminator_loss(real_output, fake_output, real_images, discriminator):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    noise = tf.random.normal([real_images.shape[0], latent_dim]) #create noise.
    generated_images = generator(noise, training=False) #create fake images.
    gradient_penalty = gradient_penalty_loss(real_images, generated_images, discriminator)
    return fake_loss - real_loss + 10 * gradient_penalty

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def generate_and_save_images(model, epoch, test_input):

    # generate images
    predictions = model(test_input, training=False)

    # Create a figure to contain plot
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')

    # Save figure
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

wandb.init(project="cifar10-gan", config={
    "learning_rate": 0.0005,
    "batch_size": batch_size,
    "epochs": epochs,
    "latent_dim": latent_dim,
    "dataset": "cifar10"
})


def train(dataset, epochs):
    batch_size = 128
    latent_dim = 100
    epochs = 15

    # Generating sample images
    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()

        # Initialize metrics for this epoch
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        num_batches = 0

        for image_batch in dataset:
            # Train
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, latent_dim) #Corrected here

            # track losses
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss
            num_batches += 1

        # Average loss for this epoch calculation
        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches

        # Log to wanb
        wandb.log({
            'epoch': epoch,
            'generator_loss': epoch_gen_loss,
            'discriminator_loss': epoch_disc_loss,
            'time_per_epoch': time.time() - start
        })

        # generate and save images
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

            # log images to wandb
            images = generator(seed, training=False)
            images = images * 0.5 + 0.5  # scale from [-1, 1] to [0, 1]
            wandb.log({
                "generated_images": [wandb.Image(img) for img in images]
            })

        # print progress
        print(f'Epoch {epoch + 1}, Gen Loss: {epoch_gen_loss:.4f}, '
              f'Disc Loss: {epoch_disc_loss:.4f}, '
              f'Time: {time.time() - start:.2f} sec')

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    generate_and_save_images(generator, epochs, seed)

train(ds_train, epochs)
