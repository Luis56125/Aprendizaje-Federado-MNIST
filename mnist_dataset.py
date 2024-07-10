"""
Source: https://github.com/google/jax/blob/main/examples/datasets.py
Code for the MNIST dataset
"""


# stdlib
import array
import gzip
import os
from os import path
import struct
import urllib.request

# third party
import numpy as np

_DATA = "/tmp/jax_example_data/"

# Esta función se encarga de descargar los archivos del conjunto de datos MNIST desde una URL dada y guardarlos en un directorio temporal 
# en el sistema de archivos local. Si el directorio no existe, lo crea. Luego, si el archivo no existe ya, lo descarga desde la URL especificada. 
def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")

# Esta función aplana todas las dimensiones de un ndarray x excepto la primera. Por ejemplo, si x es un ndarray que representa un conjunto de imágenes
# con dimensiones (número de imágenes, altura, anchura), esta función cambiará las dimensiones para que sean (número de imágenes, altura * anchura), 
# convirtiendo cada imagen en un vector mientras mantiene separadas las imágenes individuales. 

def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))

# Convierte un vector de etiquetas x en una codificación one-hot. Cada etiqueta se convierte en un vector de longitud k (el número de clases), 
# donde el índice correspondiente a la etiqueta es 1, y todos los demás índices son 0. 
def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)

# Esta función descarga y analiza los datos brutos de MNIST. Usa la función _download para obtener los archivos necesarios y luego dos funciones internas,
# parse_labels y parse_images, para analizar estos archivos. parse_labels lee las etiquetas de los datos, y parse_images lee las imágenes y las reformatea
# en un ndarray con las dimensiones apropiadas (número de imágenes, altura, anchura). 
def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels

# La función mnist() utiliza mnist_raw() para obtener los datos brutos de MNIST. Luego, procesa estos datos de la siguiente manera: 
# Utiliza _partial_flatten para transformar las matrices de imágenes 2D (altura, anchura) en vectores 1D, facilitando su uso en entradas de dimensiones fijas. 

def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()
# Divide los valores de los píxeles de las imágenes por 255.0 para escalarlos al rango [0.0, 1.0],  para mejorar el rendimiento del modelo. 
    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
# Convierte las etiquetas en representaciones one-hot utilizando _one_hot, lo que es útil para la clasificación multiclase
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)
# Si permute_train está establecido en True, permuta aleatoriamente las imágenes y las etiquetas de entrenamiento. 
# Esto puede ser útil para eliminar cualquier sesgo debido al orden en el que los datos se presentan al modelo durante el entrenamiento. 
    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels
# Finalmente, mnist() devuelve los conjuntos de imágenes y etiquetas de entrenamiento y prueba, todos preprocesados y listos para ser utilizados 
# en los cuadernos  00-data-owner-upload-data, 01-data-scientist-submit-code, 02-data-owner-review-approve-code y 03-data-scientist-download-results 