import nnabla as nn
from nnabla.models.imagenet import ResNet18
from nnabla.utils.save import save


def main():
    model = ResNet18()
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
    save("resnet.nnp", contents)


if __name__ == "__main__":
    main()
