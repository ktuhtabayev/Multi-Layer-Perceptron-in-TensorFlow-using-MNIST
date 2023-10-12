

# Step 1: Import the necessary libraries.

# importing modules
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt



# Step 2: Download the dataset.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



# Step 3: Now we will convert the pixels into floating-point values.

# Cast the records into float values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize image pixel values by dividing by 255
gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale



# Step 4: Understand the structure of the dataset

print("Feature matrix:", x_train.shape)
print("Target matrix:", x_test.shape)
print("Feature matrix:", y_train.shape)
print("Target matrix:", y_test.shape)



# Step 5: Visualize the data.

fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28),
                        aspect='auto')
        k += 1
plt.show()



# Step 6: Form the Input, hidden, and output layers.

model = keras.Sequential([

    # reshape 28 row * 28 column data to 28*28 rows
    keras.layers.Flatten(input_shape=(28, 28)),

    # dense layer 1
    keras.layers.Dense(256, activation='sigmoid'),

    # dense layer 2
    keras.layers.Dense(128, activation='sigmoid'),

    # output layer
    keras.layers.Dense(10, activation='sigmoid'),
])



# Step 7: Compile the model.

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



# Step 8: Fit the model.

model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=2000,
    validation_split=0.2
)



# Step 9: Find Accuracy of the model.

results = model.evaluate(x_test,  y_test, verbose=0)
print('test loss, test acc:', results)
