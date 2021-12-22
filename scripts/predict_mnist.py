import argparse

import numpy as np
import nnabla as nn
from nnabla.utils import nnp_graph
from nnabla.ext_utils import get_extension_context
from nnabla.utils.image_utils import imread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nnp', type=str)
    parser.add_argument('image', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        ctx = get_extension_context('cudnn', device_id=args.gpu)
        nn.set_default_context(ctx)
    else:
        ctx = None

    # load image file
    image = np.reshape(imread(args.image, grayscale=True), [1, 1, 28, 28])

    # build network
    nnp = nnp_graph.NnpLoader(args.nnp)
    network = nnp.get_network("net", batch_size=1)

    x = network.inputs["x0"]
    y = network.outputs["y0"]

    def callback(func):
        print(func)
        print(np.array(func.outputs[0].d))

    # predict class
    x.d = image
    y.forward(function_post_hook=callback)
    print(y.d.argmax())


if __name__ == "__main__":
    main()
