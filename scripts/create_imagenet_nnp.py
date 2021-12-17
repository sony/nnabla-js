import argparse
import nnabla as nn
from nnabla.models.imagenet import ResNet18, MobileNetV2
from nnabla.utils.save import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default="mobilenetv2")
    args = parser.parse_args()

    if args.arch == "resnet":
        model = ResNet18()
    elif args.arch == "mobilenetv2":
        model = MobileNetV2()
    else:
        raise ValueError(f"invalid architecture type: {args.arch}")

    x = nn.Variable((1,) + model.input_shape)
    y = model(x, training=False)

    # nnp contents
    contents = {
        "networks": [
            {"name": "net",
             "batch_size": 1,
             "outputs": {"y0": y},
             "names": {"x0": x}}],
        "executors": [
            {"name": "runtime",
             "network": "net",
             "data": ["x0"],
             "output": ["y0"]}]
    }

    # save as nnp
    save("imagenet.nnp", contents)


if __name__ == "__main__":
    main()
