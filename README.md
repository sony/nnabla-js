# nnabla-js
[![test](https://github.com/s-takuseno/nnabla-js/actions/workflows/test.yaml/badge.svg)](https://github.com/s-takuseno/nnabla-js/actions/workflows/test.yaml)

A JavaScript runtime for Neural Network Libraries.

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
`protoc` command must be installed prior to build.

```
$ npm install
$ ./scripts/build_protobuf_file.sh
$ npm run build
```

## test
```
$ pip install nnabla & python scripts/create_test_nnp.py  # for the first time
$ npm run lint:fix  # code style check
$ npm test  # unit tests
```
