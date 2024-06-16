import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Printing the shapes
print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)

# Displaying first 9 images of dataset
fig = plt.figure(figsize=(10,10))

nrows = 3
ncols = 3
for i in range(9):
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(train_images[i])
    plt.title("Digit: {}".format(train_labels[i]))
    plt.axis(False)
plt.show()

# Converting image pixel values to 0 - 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print("First Label before conversion:")
print(train_labels[0])

# Converting labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

print("First Label after conversion:")
print(train_labels[0])

# Defining Model
# Using Sequential() to build layers one after another
model = tf.keras.Sequential([

    # Flatten Layer that converts images to 1D array
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden Layer with 512 units and relu activation
    tf.keras.layers.Dense(units=512, activation='relu'),

    # Output Layer with 10 units for 10 classes and softmax activation
    tf.keras.layers.Dense(units=10, activation="sigmoid")
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    x=train_images,
    y=train_labels,
    epochs=10
)

# Showing plot for loss and accuracy
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(history.history['loss'], label='Loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:orange')
ax2.plot(history.history['accuracy'], label='Accuracy', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()


# Call evaluate to find the accuracy on test images
test_loss, test_accuracy = model.evaluate(
    x=test_images,
    y=test_labels
)

print("Test Loss: %.4f" % test_loss)
print("Test Accuracy: %.4f" % test_accuracy)

# Making Predictions
predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

# Example of digit recognition
index = 11

# Showing image
plt.imshow(test_images[index], cmap='gray')
plt.title(f"Predicted Digit: {predicted_classes[index]}")
plt.axis(False)
plt.show()

# Printing Probabilities
print("Probabilities predicted for image at index", index)
print(predicted_probabilities[index])

print()

# Printing Predicted Class
print("Predicted class for image at index", index)
print(predicted_classes[index])
