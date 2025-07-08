# Fashion-MNIST Classifier with TensorFlow/Keras  
#Esame DeepLearning

![Fashion-MNIST Samples](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)  

## 📌 Overview  
This repository contains **two deep learning models** for classifying the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset:  
1. **A simple Multi-Layer Perceptron (MLP)**  
2. **A more advanced Convolutional Neural Network (CNN)**  

The project includes:  
- 🖼️ **Data visualization** of the Fashion-MNIST dataset  
- 🏗️ **Model building** with TensorFlow/Keras  
- 📊 **Training & evaluation** with accuracy/loss tracking  
- 🔍 **Model testing** with prediction visualization  

## 📂 Repository Structure  
```
.
├── crea_modello.py          # Simple MLP model (Flatten + Dense layers)  
├── crea_modello_2.py        # CNN model (Conv2D + BatchNorm + Dropout)  
├── test_modello.py          # Model testing & prediction visualization  
├── esame/                   # Saved models  
│   ├── class_fash.keras     # MLP model  
│   └── class_fash2.keras    # CNN model  
└── logs/                    # TensorBoard logs (if generated)  
```

## 🛠️ Installation & Usage  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/fashion-mnist-classifier.git
   cd fashion-mnist-classifier
   ```

2. **Install dependencies**  
   ```bash
   pip install tensorflow matplotlib numpy seaborn tensorflow-datasets
   ```

3. **Run the models**  
   - Train the **MLP**:  
     ```bash
     python crea_modello.py
     ```
   - Train the **CNN**:  
     ```bash
     python crea_modello_2.py
     ```
   - Test a saved model:  
     ```bash
     python test_modello.py
     ```

## 📈 Model Architectures  

### MLP Model (crea_modello.py)  
```python
model = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10)
])
```

### CNN Model (crea_modello_2.py)  
```python
model = tf.keras.Sequential([
    Reshape((28, 28, 1), 
    Conv2D(16, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

## 📊 Performance Comparison  

| Model       | Val Accuracy | Training Time (10 epochs) |  
|-------------|-------------|--------------------------|  
| **MLP**     | ~85%        | ~1 min (CPU)            |  
| **CNN**     | ~91%        | ~3 min (CPU)            |  


🔗 **Dataset Reference**: [Fashion-MNIST on GitHub](https://github.com/zalandoresearch/fashion-mnist)  
