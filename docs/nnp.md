# How to Execute Your NNP

## Save NNP
Saving your nnabla model as NNP is a very simple procedure.
Let's say you have a neural network model as follows:
```py
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def mlp(x):
    x /= 255.0
    h = PF.affine(F.reshape(x, [-1, 28 * 28]), 64, name="affine1")
    h = F.relu(h)
    h = PF.affine(h, 64, name="affine2")
    h = F.relu(h)
    return PF.affine(h, 10, name="affine3")

# input variable
x = nn.Variable((batch_size, 1, 28, 28))

# output variable
y = mlp(x)
```

In this case, you can save this model as NNP as follows:
```py
from nnabla.utils.save import save

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
save("model.nnp", contents)
```

Please note that NNP usually includes a list of `network`s and a list of `executor`s.
The `network` is defined with a pair of input variables and ouput variables.
The `executor` specifies which network to execute and what variables it takes.

## Load and execute NNP
Once you get your own NNP file, you can easily load and execute it with JavaScript though `nnabla-js`.
In this example, let's use Node.js to focus on JavaScript codes, however, you can do this on web browsers too.

Here is the example to execute inference of `model.nnp` saved above.
```js
const fs = require('fs')
const nnabla = require('nnabla-js')

// load .nnp file from your disk
fs.readFile('model.nnp', (_, data) => {
  nnabla.NNP.fromNNPData(data).then((nnp) => {
    // prepare input data
    const x = [...Array(1 * 28 * 28)].map(() => Math.random() * 2.0 - 1.0)

    // execute inference by specifying executor name
    nnp.forwardAsync('runtime', { x0: x }).then((output) => {

      // retrieve inference results
      console.log(output.y0)

    })
  })
});
```

## Release
You might want to release the resources when you no longer need `nnp` instance.
You can call `release` function to release all resources allocated to `nnp`.
```js
nnp.release()

// you can't execute inference with this instance any more
nnp.forward('runtime', { x0: x })  // raises Error!
```
