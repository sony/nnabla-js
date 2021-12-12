import argparse
import time

import nnabla as nn
from nnabla.utils.load import load
from nnabla.ext_utils import get_extension_context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nnp', type=str)
    parser.add_argument('runtime', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        ctx = get_extension_context('cudnn', device_id=args.gpu)
        nn.set_default_context(ctx)

    nnp = load(args.nnp, batch_size=1)

    executor = nnp.executors[args.runtime]

    total_time = 0.0
    for i in range(10):
        start_time = time.time()
        executor.forward_target.forward()
        end_time = time.time()
        if i > 0:
            total_time += end_time - start_time

    print(f"Average time: {total_time / 9.0 * 1000.0}ms")


if __name__ == "__main__":
    main()
