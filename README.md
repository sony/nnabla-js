# nnabla-js
[![test](https://github.com/nnabla/nnabla-js/actions/workflows/test.yaml/badge.svg)](https://github.com/nnabla/nnabla-js/actions/workflows/test.yaml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/nnabla/nnabla-js/blob/master/LICENSE)

A JavaScript runtime for Neural Network Libraries.

![demo](https://user-images.githubusercontent.com/72241882/163357484-4efd5533-1e80-45f0-9e9c-cd3bd3f4fd0b.gif)

## key features
- Run on web browsers
- Load and execute `.nnp` files
- Support GPU (WebGL)
- Preprocess utilities (e.g. image resizing)

## installation
```
$ npm install nnabla-js
```

## example
Check out more [examples](examples)!

```js
const fs = require('fs');
const nnabla = require('nnabla-js');

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
$ npm run build:dev
```

## test
```
$ pip install nnabla & python scripts/create_test_nnp.py  # for the first time
$ npm run lint:fix  # code style check
$ npm test  # unit tests
```
