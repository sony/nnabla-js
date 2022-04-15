# Preprocess Utilities

## Array to ImageData conversion
If you work with edge computer vision models, you might often need to convert `Array` to `ImageData` and vice versa.
`nnabla-js` provides utilities for this purpose.
Please note that `nnabla-js` always takes 1d arrays as inputs.
Therefore, you need to specify image size `(channel, width, height)` when converting `Array` back to `ImageData`.

```js
const nnabla = require('nnabla-js')

// convert ImageData to Array
const rgb = nnabla.ImageUtils.convertImageToArray(imageData, 3) // convert to RGB array
const gray = nnabla.ImageUtils.convertImageToArray(imageData, 1) // convert to gray-scaled array
// optionally you can multiply arbitrary number
const scaledRGB = nnabla.ImageUtils.convertImageToArray(imageData, 3, 1 / 255) // divide by 255

// convert Array to ImageData (width, height) = (28, 56)
const rgbImageData = nnabla.ImageUtils.convertArrayToImage(rgb, 3, 56, 28)
const grayImageData = nnabla.ImageUtils.convertArrayToImage(gray, 1, 56, 28)
// optionally you can multiply arbitrary number
const scaledImageData = nnabla.ImageUtils.convertArrayToImage(scaledRGB, 3, 56, 28, 255) // multiply by 255
```

## Image resizing
Resizing images is the necessary feature when you handle images users upload
because the neural network usually requires the specific shape of inputs.
`nnabla-js` also provides image resizing utility that leverages GPU accelaration (WebGL).
Please note that the processed output is `Texture` data defined in `gpu.js`.

```js
const nnabla = require('nnabla-js')
const { GPU } = require('gpu.js')
const gpu = new GPU()

// allow dynamic input size by specifying true at the last argument.
// resize inputs to (3, 28, 28).
const resizeKernel = nnabla.ImageUtils.createAsyncResizeKernel([3, 28, 28], gpu, true)

// image data uploaded by a user with a shape of (3, 28, 56)
const x = [...Array(3 * 56 * 56)].map(() => Math.random() * 2.0 - 1.0)

resizeKernel(x, 56, 28).then((texture) => {
  // need to convert to Array if you want to interact with the contents
  const resizedArray = texture.toArray()

  // you can directly feed the texture to NNP
  nnp.forwardAsync('runtime', { 'x0': texture }).then((output) => {
    ...
  })
})
```
