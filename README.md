# nnabla-js
[![test](https://github.com/s-takuseno/nnabla-js/actions/workflows/test.yaml/badge.svg)](https://github.com/s-takuseno/nnabla-js/actions/workflows/test.yaml)

A JavaScript runtime for Neural Network Libraries.

:warning: Currently, only few layers are supported.

## key features
- Run on web browser
- Load and execute NNP
- Support GPU (WebGL)

## example
```js
const fs = require('fs');
const nnabla = require('dist/index.js');

const x = [...Array(28 * 28)].map(() => Math.random() * 2.0 - 1.0);

// Load your NNP file
fs.readFile('mnist.nnp', (_, data) => {
  // Build computation graph from NNP
  nnabla.NNP.fromNNPData(data).then((nnp) => {
    // Forward propagation with the specified executor
    const output = nnp.forward('runtime', { 'x0': x });
  });
});
```

## build
```
$ npm install
$ npm run build
```

## test
```
$ npm run lint:fix  # code style check
$ npm test  # unit tests
```
