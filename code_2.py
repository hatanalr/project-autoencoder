
#  final code number dataset



import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore

# 1. Load and preprocess diabetes data
df = pd.read_csv('diabetes.csv')
x_data = df.values

# Scale data to [0, 1] range
scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)

# Split into train/test (80/20)
split_idx = int(0.8 * len(x_data))
x_train = x_data[:split_idx]
x_test = x_data[split_idx:]

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# 2. Autoencoder Model
input_dim = x_train.shape[1]
encoding_dim = 4  # Smaller bottleneck for tabular data

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck

# Decoder
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Full model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# 3. Train the model
history = autoencoder.fit(x_train, x_train,
                         epochs=100,
                         batch_size=32,
                         shuffle=True,
                         validation_data=(x_test, x_test))

# 4. Reconstruct test data
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

# 7. Visualization - Plot first 5 features
plt.figure(figsize=(15, 10))
for feat in range(5):  # Plot first 5 features
    plt.subplot(5, 1, feat+1)
    plt.plot(x_test[:50, feat], label='Original', marker='o')
    plt.plot(reconstructed[:50, feat], label='Autoencoder', marker='x')
    plt.plot(x_pca_inverse[:50, feat], label='PCA', marker='^')
    plt.title(f'Feature {feat+1} Comparison')
    plt.legend()
plt.tight_layout()
plt.show()

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