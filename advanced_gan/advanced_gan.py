#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_datasets as tfds
from tensorflow.keras import initializers, regularizers, constraints
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

batch_size = 128
latent_dim = 100
epochs = 30

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

def build_generator(latent_dim=100, style_dim=512):
    # Mapping Network (same as before)
    mapping_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(style_dim)(mapping_inputs)
    x = layers.LeakyReLU(0.2)(x)
    for _ in range(7):
        x = layers.Dense(style_dim)(x)
        x = layers.LeakyReLU(0.2)(x)
    style_codes = layers.Dense(style_dim)(x)

    # Synthesis Network (Modified)
    inputs = layers.Input(shape=(4, 4, 512))
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x) #added layer
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
    return tf.keras.Model(mapping_inputs, x)
    
def build_discriminator(input_shape=(32, 32, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x) # batch normalization added here
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    return models.Model(inputs, outputs)

generator_optimizer = optimizers.RMSprop(learning_rate=0.0005, rho=0.8)
discriminator_optimizer = optimizers.RMSprop(learning_rate=0.0005, rho=0.8)

def train_step(images, generator, discriminator, latent_dim):
    noise = tf.random.normal([images.shape[0], latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, tf.random.normal([images.shape[0], 4, 4, 512])], training=True)

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

generator = build_generator()
discriminator = build_discriminator()

import time
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def train(dataset, generator, discriminator, latent_dim, epochs, batch_size):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, latent_dim)

            # Log losses with wandb
            wandb.log({"gen_loss": gen_loss.numpy(), "disc_loss": disc_loss.numpy()})

        # Generate sample images (e.g., every few epochs)
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, latent_dim)

        print(f"Epoch {epoch + 1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")

def generate_and_save_images(model, epoch, latent_dim):
    noise = tf.random.normal([16, latent_dim])
    generated_images = model([noise, tf.random.normal([16, 4, 4, 512])], training=False)
    generated_images = (generated_images * 0.5 + 0.5).numpy() # rescale to [0,1]

    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

    # Log to wandb
    wandb.log({"generated_images": wandb.Image(generated_images)})


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output, real_images, discriminator):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    gradient_penalty = gradient_penalty_loss(real_images, generated_images, discriminator)
    return fake_loss - real_loss + 10 * gradient_penalty

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

wandb.init(project="cifar10-gan", config={
    "learning_rate": 0.0005,
    "batch_size": batch_size,
    "epochs": epochs,
    "latent_dim": latent_dim,
    "dataset": "cifar10"
})
