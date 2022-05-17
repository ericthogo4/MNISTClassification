import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(training_dataset, training_d_labels), (testing_dataset, testing_d_labels) = keras.datasets.mnist.load_data()

len(training_dataset)

len(testing_dataset)


training_dataset = training_dataset / 255
testing_dataset = testing_dataset / 255


flattened_training_d = training_dataset.reshape(len(training_dataset), 28*28)
flattened_testing_d = testing_dataset.reshape(len(testing_dataset), 28*28)


classification_model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])


classification_model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# The model is trained in 15 epochs
classification_model.fit(flattened_training_d, training_d_labels, epochs=15)

classification_model.evaluate(flattened_testing_d, testing_d_labels)

# Using the MNIST test set, the model makes efforts of labeling images
classification_results = classification_model.predict(flattened_testing_d)

simplified_c_results = [np.argmax(i) for i in classification_results]

print('Labels of first nine MNIST test images: ',testing_d_labels[:9])
print('Classification results of first nine MNIST test images: ',simplified_c_results[:9])

print('Images of the first nine MNIST test images are about to be shown in a new window')
plt.subplot(330 + 1 + 0)
plt.imshow(testing_dataset[0])
plt.subplot(330 + 1 + 1)
plt.imshow(testing_dataset[1])
plt.subplot(330 + 1 + 2)
plt.imshow(testing_dataset[2])
plt.subplot(330 + 1 + 3)
plt.imshow(testing_dataset[3])
plt.subplot(330 + 1 + 4)
plt.imshow(testing_dataset[4])
plt.subplot(330 + 1 + 5)
plt.imshow(testing_dataset[5])
plt.subplot(330 + 1 + 6)
plt.imshow(testing_dataset[6])
plt.subplot(330 + 1 + 7)
plt.imshow(testing_dataset[7])
plt.subplot(330 + 1 + 8)
plt.imshow(testing_dataset[8])
plt.show()

for d in range(0,10):
    no_of_digit_instances = 0
    for j in testing_d_labels:
        if j==d:
            no_of_digit_instances = no_of_digit_instances + 1

    print('number of',d,'s in mnist testing dataset:',no_of_digit_instances)

    no_of_digit_instances = 0

    for k in simplified_c_results:
        if k==d:
            no_of_digit_instances = no_of_digit_instances + 1

    print('number of',d,'s in classification result:',no_of_digit_instances)
    print('')




