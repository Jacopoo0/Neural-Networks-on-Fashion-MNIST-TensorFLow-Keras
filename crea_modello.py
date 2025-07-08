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

# Carica il dataset Fashion MNIST da TensorFlow Datasets
(ds_train, ds_test)= tfds.load(
    'fashion_mnist',
    split=['train','test'],
    as_supervised=True,   # Restituisce tuple (immagine, etichetta)
    batch_size=None,      # Nessun batching automatico
)

# Conta il numero di esempi nel training set
num_train = tf.data.experimental.cardinality(ds_train).numpy()

# Mostra 25 immagini random dal dataset fashion_mnist (griglia 5x5)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",   # Nomi delle classi
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.figure(figsize=(12, 12))  # Crea una figura grande per la griglia
images = []
labels = []
for image, label in ds_train.take(25):    # Prende 25 immagini dal training set
    images.append(image.numpy())
    labels.append(label.numpy())

for i in range(25):
    plt.subplot(5, 5, i + 1)              # Posiziona il subplot nella griglia 5x5
    plt.imshow(images[i], cmap="gray")    # Mostra l'immagine in scala di grigi
    plt.axis("off")                       # Nasconde gli assi
    plt.title(class_names[labels[i]])     # Mostra il nome della classe
plt.show()                                # Visualizza la griglia

# Funzione per normalizzare le immagini (da uint8 a float32 in [0,1])
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# Applica la normalizzazione e prepara la pipeline del dataset di training
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)  # Applica normalizzazione
ds_train = ds_train.cache()  # Tiene i dati in memoria per accelerare
ds_train = ds_train.shuffle(num_train)  # Mescola l’intero dataset
ds_train = ds_train.batch(128)  # Suddivide in batch da 128 elementi
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)  # Ottimizza caricamento dati in parallelo

# Applica la normalizzazione e prepara la pipeline del dataset di test
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)  # Normalizza immagini
ds_test = ds_test.batch(128)  # Raggruppa in batch da 128
ds_test = ds_test.cache()  # Tiene in cache per velocizzare
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)  # Caricamento ottimizzato

# Salva i log per visualizzare l'andamento del training su TensorBoard
tensorboard_callback = TensorBoard(log_dir="./classificatore_fashion/logs")

## Costruzione del modello
model = tf.keras.models.Sequential([  # Modello a strati sequenziali
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Appiattisce l'immagine 2D in un vettore
  tf.keras.layers.Dense(64, activation='relu'),   # Layer completamente connesso con 64 neuroni
  tf.keras.layers.Dropout(0.5),                   # Dropout del 50% 
  tf.keras.layers.Dense(10)                       # Layer di output: 10 classi (da 0 a 9)
])
model.summary()  # Mostra l’architettura del modello a terminale

# Funzione di perdita (usa indici invece che one-hot)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compila il modello specificando ottimizzatore, funzione di perdita e metrica
model.compile(
    optimizer="adam",  # Ottimizzatore Adam
    loss=loss_fn,      # Funzione di perdita
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],  # Accuratezza come metrica
)

# Addestra il modello
history=model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    shuffle=False,  # Non serve se già shufflato prima
    callbacks=[tensorboard_callback]
)

print ("Training done, evaluating the model...")
# Valuta il modello sul test set
model.evaluate(ds_test, verbose=2)

# Salva il modello addestrato in formato Keras nella cartella 'esame'
model.save("esame/class_fash.keras")

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