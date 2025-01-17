import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
def preprocess(image):
    image = tf.image.resize(image, (128, 128)) 
    image = image / 255.0 
    return image
def create_data(dir=r"E:\Sub1\GAN\christmas_data"):
    paths = []
    for label in os.listdir(dir):
        class_path = os.path.join(dir, label)
        paths.append(class_path)
    return paths
def build_generator(z_dim):
    model = tf.keras.Sequential([
        layers.Dense(128 * 32 * 32, activation="relu", input_shape=(z_dim,)),
        layers.Reshape((32, 32, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=2, padding="same", activation="relu"),
        layers.Conv2DTranspose(64, (4, 4), strides=2, padding="same", activation="relu"),
        layers.Conv2DTranspose(3, (4, 4), strides=1, padding="same", activation="sigmoid")
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (4, 4), strides=2, padding="same", input_shape=(128, 128, 3)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (4, 4), strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1,activation='sigmoid')
    ])
    return model
z_dim = 100
generator = build_generator(z_dim)
discriminator = build_discriminator()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
gen_opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
disc_opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, z_dim])
    with tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = (real_loss + fake_loss) / 2
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    return gen_loss, disc_loss
epochs = 5000
batch_size = 16
for epoch in range(epochs):
    paths = create_data()
    batch_images = []
    gen_loss, disc_loss = 0, 0 
    for image_path in paths:
        image = load_img(image_path, target_size=(128, 128))
        image = preprocess(image) 
        batch_images.append(image)
        if len(batch_images) == batch_size:
            batch_tensor = tf.convert_to_tensor(np.array(batch_images), dtype=tf.float32)
            gen_loss, disc_loss = train_step(batch_tensor)
            batch_images = [] 
    if batch_images:
        batch_tensor = tf.convert_to_tensor(np.array(batch_images), dtype=tf.float32)
        gen_loss, disc_loss = train_step(batch_tensor)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
def generate_images(num_images=5):
    noise = tf.random.normal([num_images, z_dim])
    fake_images = generator(noise, training=False).numpy()
    fake_images = (fake_images * 255).astype(np.uint8)  
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(fake_images[i])
        ax.axis("off")
    plt.show()
generate_images()
