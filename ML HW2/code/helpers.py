import numpy as np
import struct
def load_mnist_images(path, mode):
    #"MNIST data\\t10k-images.idx3-ubyte"
    path = (path + "MNIST data\\t10k-images.idx3-ubyte") if mode == "test" else (path + "MNIST data\\train-images.idx3-ubyte")
    with open(path, 'rb') as f:
        magic, count = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        images = np.fromfile(f, dtype= np.dtype(np.uint8).newbyteorder('>'))
        images = images.reshape((count, rows,cols))
    return images

def load_mnist_labels(path, mode):
    path = (path + "MNIST data\\t10k-labels.idx1-ubyte") if path == "test" else (path + "MNIST data\\train-labels.idx1-ubyte")
    with open(path, 'rb') as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype= np.dtype(np.uint8).newbyteorder('>'))
    return labels