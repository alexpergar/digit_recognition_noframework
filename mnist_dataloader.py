"""
MNIST DataLoader and helper functions.
"""


import struct

import numpy as np
import matplotlib.pyplot as plt


class MnistDataloader(object):
    """
    MNIST DataLoader for loading and preprocessing the MNIST dataset.
    The dataset consists of 60,000 training images and 10,000 test images of
    handwritten digits (0-9). Each image is 28x28 pixels, and the labels are
    integers from 0 to 9. The images and labels are stored in IDX file format,
    which is a simple binary format for vectors and multidimensional matrices
    of various numerical types. The class provides methods to read the images
    and labels from the IDX files and return them as NumPy arrays.
    """
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Read images and labels from IDX files.
        Args:
            images_filepath (str): Path to the IDX file containing images.
            labels_filepath (str): Path to the IDX file containing labels.
        Returns:
            images (np.ndarray): NumPy array of shape (num_images, 28, 28)
                containing the images.
            labels (np.ndarray): NumPy array of shape (num_images,) containing
                the labels.
        Raises:
            ValueError: If the magic number in the IDX file does not match the expected value.
        """
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.frombuffer(file.read(), dtype=np.uint8)
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.frombuffer(file.read(), dtype=np.uint8)
        images = image_data.reshape(size, rows, cols)
        
        return images, labels
            
    def load_data(self):
        """
        Load and return the MNIST dataset.
        Returns:
            (x_train, y_train), (x_test, y_test): Tuple containing training and
                test data and labels.
        """
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    

def show_images(images, title_texts):
    """
    Show a list of images with their relating titles.
    Args:
        images (list of np.ndarray): List of images to be displayed.
        title_texts (list of str): List of titles corresponding to each image.
    """
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
