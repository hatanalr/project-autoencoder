 
# # final code   image


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# 1. Load and preprocess MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# 2. Autoencoder Model
input_dim = 784
encoding_dim = 32  # Bottleneck dimension

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Full model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# 3. Train the model
history = autoencoder.fit(x_train, x_train,
                         epochs=50,
                         batch_size=256,
                         shuffle=True,
                         validation_data=(x_test, x_test))

# 4. Reconstruct test images
reconstructed = autoencoder.predict(x_test)

# 5. Compare with PCA
pca = PCA(n_components=encoding_dim)
pca.fit(x_train)
x_pca = pca.transform(x_test)
x_pca_inverse = pca.inverse_transform(x_pca)

# 6. Calculate and compare MSE
def calculate_mse(original, reconstructed, label):
    mse = np.mean(np.square(original - reconstructed))
    print(f"{label} MSE: {mse:.5f}")

calculate_mse(x_test, reconstructed, "Autoencoder")
calculate_mse(x_test, x_pca_inverse, "PCA")

# 7. Visualization
def plot_results(original, reconstructed, pca_reconstructed, n=10):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.title("Original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Autoencoder
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28))
        plt.title("Autoencoder")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # PCA
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(pca_reconstructed[i].reshape(28, 28))
        plt.title("PCA")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

plot_results(x_test, reconstructed, x_pca_inverse)

# 8. Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()



