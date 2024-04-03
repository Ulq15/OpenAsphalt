from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
import numpy as np
from  sklearn.model_selection import train_test_split

image_height = 128
image_width = 128
num_classes = 2

# Define the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten()
])

# Define the RNN model
rnn_model = Sequential([
    LSTM(128, return_sequences=True),
    Dense(num_classes, activation='softmax')
])

# Combine CNN and RNN
combined_model = Sequential([
    cnn_model,
    rnn_model
])


combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# dataset split into numpy arrays
data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
labels = np.array([0, 1])
X_train, y_train, X_test, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

combined_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
