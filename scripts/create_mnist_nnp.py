import os
import numpy as np
import struct
import zlib
from PIL import Image

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.logger import logger
from nnabla.utils.save import save
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


def load_mnist(train=True):
    if train:
        image_uri = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        label_uri = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    else:
        image_uri = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        label_uri = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    logger.info('Getting label data from {}.'.format(label_uri))

    r = download(label_uri)
    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
    _, size = struct.unpack('>II', data[0:8])
    labels = np.frombuffer(data[8:], np.uint8).reshape(-1, 1)
    r.close()
    logger.info('Getting label data done.')

    logger.info('Getting image data from {}.'.format(image_uri))
    r = download(image_uri)
    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
    _, size, height, width = struct.unpack('>IIII', data[0:16])
    images = np.frombuffer(data[16:], np.uint8).reshape(
        size, 1, height, width)
    r.close()
    logger.info('Getting image data done.')

    return images, labels


class MnistDataSource(DataSource):
    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None):
        super(MnistDataSource, self).__init__(shuffle=shuffle)
        self._train = train

        self._images, self._labels = load_mnist(train)

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(MnistDataSource, self).reset()

    @property
    def images(self):
        return self._images.copy()

    @property
    def labels(self):
        return self._labels.copy()


def data_iterator_mnist(batch_size,
                        train=True,
                        rng=None,
                        shuffle=True,
                        with_memory_cache=False,
                        with_file_cache=False):
    return data_iterator(MnistDataSource(train=train, shuffle=shuffle, rng=rng),
                         batch_size,
                         rng,
                         with_memory_cache,
                         with_file_cache)


def mlp(x):
    x /= 255.0
    h = PF.affine(F.reshape(x, [-1, 28 * 28]), 64, name="affine1")
    h = F.relu(h)
    h = PF.affine(h, 64, name="affine2")
    h = F.relu(h)
    return PF.affine(h, 10, name="affine3")


def cross_entropy(y, t):
    return F.mean(F.softmax_cross_entropy(y, t))


def main():
    batch_size = 256

    # input variable
    x = nn.Variable((batch_size, 1, 28, 28))
    t = nn.Variable((batch_size, 1))

    # train graph
    y = mlp(x)
    loss = cross_entropy(y, t)

    solver = S.Adam(1e-3)
    solver.set_parameters(nn.get_parameters())

    data = data_iterator_mnist(batch_size, True)
    test_data = data_iterator_mnist(batch_size, False)

    for iters in range(1000):
        # train
        x.d, t.d = data.next()
        loss.forward()
        solver.zero_grad()
        loss.backward(clear_buffer=True)
        solver.update()

        # test
        if iters % 100 == 0:
            corrects = 0
            for _ in range(10):
                x.d, t.d = test_data.next()
                y.forward(clear_buffer=True)
                corrects += np.sum(t.d.flatten() == y.d.argmax(axis=1))
            accuracy = corrects / (batch_size * 10) * 100
            print(f"Iter={iters} Accuracy: {accuracy}%")


    # nnp contents
    contents = {
        "networks": [
            {"name": "net",
             "batch_size": batch_size,
             "outputs": {"y0": y},
             "names": {"x0": x}}],
        "executors": [
            {"name": "runtime",
             "network": "net",
             "data": ["x0"],
             "output": ["y0"]}]
    }

    # save as nnp
    save("mnist.nnp", contents)

    # save example MNIST images
    os.makedirs("mnist_images", exist_ok=True)
    x.d, t.d = data.next()
    for i in range(10):
        image = np.reshape(x.d[i], [28, 28])
        im = Image.fromarray(np.array(image, dtype=np.uint8))
        im.save(os.path.join("mnist_images", f"sample_{i}.png"))


if __name__ == "__main__":
    main()
