import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('train',mnist.train.num_examples,
      ',validation',mnist.validation.num_examples,
      ',test',mnist.test.num_examples)
def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='plasma')
    plt.show()
print('train images:', mnist.train.images.shape,
      'labels:', mnist.train.labels.shape)
print(len(mnist.train.images[3]))
print(mnist.train.labels[3])
plot_image(mnist.train.images[3])
