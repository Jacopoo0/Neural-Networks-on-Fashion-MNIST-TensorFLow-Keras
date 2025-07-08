import matplotlib.pyplot as plt           
import matplotlib.image as mpimg           
import seaborn as sns                     
import pandas as pd                        
import numpy as np                         
import os, random                   
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import TensorBoard 



(ds_train, ds_test)= tfds.load(
    'fashion_mnist',
    split=['train','test'],
    as_supervised=True,
    batch_size=None,

)

num_train = tf.data.experimental.cardinality(ds_train).numpy()



# Mostra 25 immagini random dal dataset fashion_mnist (griglia 5x5)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",   #nome delle classi da attribuire a ogni immagine nella griglia random
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.figure(figsize=(12, 12))
images = []
labels = []
for image, label in ds_train.take(25):
    images.append(image.numpy())
    labels.append(label.numpy())

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")
    plt.title(class_names[labels[i]])  # Mostra il nome della classe
plt.show()


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)  # Applica normalizzazione
ds_train = ds_train.cache()  # Tiene i dati in memoria per accelerare
ds_train = ds_train.shuffle(num_train)  # Mescola l’intero dataset
ds_train = ds_train.batch(128)  # Suddivide in batch da 128 elementi
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)  # Ottimizza caricamento dati in parallelo


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)  # Normalizza immagini
ds_test = ds_test.batch(128)  # Raggruppa in batch da 128
ds_test = ds_test.cache()  # Tiene in cache per velocizzare
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)  # Caricamento ottimizzato


# Salva i log per visualizzare l'andamento del training su TensorBoard
tensorboard_callback = TensorBoard(log_dir="./classificatore_fashion/logs")


##build model

from tensorflow.keras.callbacks import EarlyStopping

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    ds_train,
    epochs=25,
    validation_data=ds_test,
    shuffle=False,
    callbacks=[tensorboard_callback, early_stop]
)

print ("Training done, evaluating the model...")
# Valuta il modello
model.evaluate(ds_test, verbose=2)



model.save("esame/class_fash2.keras")
# Estrai le perdite di training e validazione
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Estrai l'accuratezza di training e validazione
training_accuracy = history.history['sparse_categorical_accuracy']
validation_accuracy = history.history['val_sparse_categorical_accuracy']

# Grafico delle perdite
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Grafico dell'accuratezza
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
