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
(ds_train, ds_test) = tfds.load(
    'fashion_mnist',
    split=['train','test'],
    as_supervised=True,     # Restituisce tuple (immagine, etichetta)
    batch_size=None,        # Nessun batching automatico
)

# Carica il modello salvato precedentemente
model_path = 'esame/class_fash2.keras'
model = tf.keras.models.load_model(model_path)

# Lista dei nomi delle classi Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("\n=== INFERENZA SUL TEST SET ===")

# Prendi un batch di 25 immagini dal test set
for images, labels in ds_test.batch(25).take(1):
    # Esegui la predizione sul batch di immagini
    predictions = model.predict(images)
    # Ottieni la classe predetta per ciascuna immagine
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 12))  # Crea una figura grande per la griglia
    for i in range(min(26, len(images))):
        ax = plt.subplot(5, 5, i + 1)  # Posiziona il subplot nella griglia 5x5
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")  # Mostra l'immagine in scala di grigi
        true_label = class_names[labels[i].numpy()]           # Nome della classe reale
        predicted_label = class_names[predicted_labels[i]]    # Nome della classe predetta
        color = "green" if predicted_label == true_label else "red"  # Verde se corretto, rosso se sbagliato
        # Titolo con predizione e vero valore, colorato in base alla correttezza
        plt.title(f"Pred: {predicted_label}\n(True: {true_label})", color=color, fontsize=9)
        plt.axis("off")  # Nasconde gli assi
    plt.tight_layout()   # Migliora la disposizione dei subplot
    plt.show()           # Mostra la griglia di immagini
    break                # Esegui solo per il primo batch