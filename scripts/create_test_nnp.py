import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.save import save


def main():
    batch_size = 1

    # input variable
    x0 = nn.Variable((batch_size, 100))

    # model
    h = x0
    for depth in range(34):
        h = PF.affine(h, 100, name=f"affine{depth}")
        h = F.relu(h)
    y0 = PF.affine(h, 10, name=f"affine{depth+1}")

    # nnp contents
    contents = {
        "networks": [
            {"name": "net",
             "batch_size": batch_size,
             "outputs": {"y0": y0},
             "names": {"x0": x0}}],
        "executors": [
            {"name": "runtime",
             "network": "net",
             "data": ["x0"],
             "output": ["y0"]}]
    }

    # save as nnp
    save("test.nnp", contents)


if __name__ == "__main__":
    main()
