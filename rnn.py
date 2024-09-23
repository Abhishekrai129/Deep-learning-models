import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the IMDB dataset
max_features = 10000  # Number of words to consider as features
maxlen = 200  # Maximum length of each review
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Preprocess the data: pad sequences
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Build the RNN model
model = models.Sequential()
model.add(layers.Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
model.add(layers.SimpleRNN(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Binary output

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Example input for prediction
sample_reviews = [
    "This movie was fantastic! I really enjoyed it.",
    "I did not like this movie. It was boring."
]

# Preprocess the sample reviews
sample_reviews_encoded = [[imdb.get_word_index().get(word.lower(), 0) for word in review.split()] for review in sample_reviews]
sample_reviews_padded = sequence.pad_sequences(sample_reviews_encoded, maxlen=maxlen)

# Make predictions
predictions = model.predict(sample_reviews_padded)
for review, prediction in zip(sample_reviews, predictions):
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    print(f"Review: {review}\nPredicted Sentiment: {sentiment} (Score: {prediction[0]:.4f})\n")
