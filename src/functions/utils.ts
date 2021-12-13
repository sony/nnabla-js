import { GPU } from 'gpu.js';

export function createMatmulKernel(
  gpu: GPU,
  xShape: number[],
  yShape: number[],
  transposeX: boolean,
  transposeY: boolean,
): [(x: number[], y: number[]) => number[], number[]] {
  if (xShape.length !== 2) {
    throw Error(`invalid x shape: ${xShape}`);
  }
  if (yShape.length !== 2) {
    throw Error(`invalid y shape: ${yShape}`);
  }

  const outputShape = [];
  if (transposeX) {
    outputShape.push(xShape[1]);
  } else {
    outputShape.push(xShape[0]);
  }
  if (transposeY) {
    outputShape.push(yShape[0]);
  } else {
    outputShape.push(yShape[1]);
  }

  const kernel = gpu
    .createKernel(function (
      x: number[],
      y: number[],
      _xShape: number[],
      _yShape: number[],
      _transposeX: boolean,
      _transposeY: boolean,
    ): number {
      const [xRowSize, xColSize] = _xShape;
      const [yRowSize, yColSize] = _yShape;
      const xBase = Math.floor(this.thread.x / (_transposeY ? yRowSize : yColSize));
      const yBase = this.thread.x % (_transposeY ? yRowSize : yColSize);
      const dim = _transposeX ? xRowSize : xColSize;
      let output = 0.0;
      for (let i = 0; i < dim; i += 1) {
        let xValue: number = 0.0;
        if (_transposeX) {
          xValue = x[i * xColSize + xBase];
        } else {
          xValue = x[xBase * xColSize + i];
        }

        let yValue: number = 0.0;
        if (_transposeY) {
          yValue = y[yBase * yColSize + i];
        } else {
          yValue = y[i * yColSize + yBase];
        }

        output += xValue * yValue;
      }
      return output;
    })
    .setOutput([outputShape[0] * outputShape[1]]);

  function partialKernel(x: number[], y: number[]): number[] {
    return kernel(x, y, xShape, yShape, transposeX, transposeY) as number[];
  }

  return [partialKernel, outputShape];
}

export function createPadKernel(
  gpu: GPU,
  shape: number[],
  pad: number[],
): [(x: number[]) => number[], number[]] {
  let dataSize = 1;
  for (let i = 0; i < shape.length - 2; i += 1) {
    dataSize *= shape[i + 2];
  }

  const paddedShape = [shape[0], shape[1]];
  let paddedSize = shape[0] * shape[1];
  let paddedDataSize = 1;
  for (let i = 0; i < shape.length - 2; i += 1) {
    const dim = shape[i + 2] + 2 * pad[i];
    paddedShape.push(dim);
    paddedSize *= dim;
    paddedDataSize *= dim;
  }

  const padIndexMapping: number[] = [];
  for (let i = 0; i < paddedDataSize; i += 1) {
    // Identify index at the padded pixels
    const index = [];
    let tmp = paddedDataSize;
    let cursor = i;
    for (let j = 0; j < shape.length - 2; j += 1) {
      tmp /= paddedShape[j + 2];
      index.push(Math.floor(cursor / tmp));
      cursor -= Math.floor(cursor / tmp) * tmp;
    }

    // Check if padded pixel
    let flag = false;
    for (let j = 0; j < shape.length - 2; j += 1) {
      if (index[j] < pad[j] || index[j] >= shape[j + 2] + pad[j]) {
        flag = true;
        break;
      }
    }

    if (flag) {
      padIndexMapping.push(-1);
    } else {
      let pixIndex = 0;
      tmp = dataSize;
      for (let j = 0; j < shape.length - 2; j += 1) {
        tmp /= shape[j + 2];
        pixIndex += tmp * (index[j] - pad[j]);
      }
      padIndexMapping.push(pixIndex);
    }
  }

  const kernel = gpu
    .createKernel(function (
      x: number[],
      _shape: number[],
      _padIndexMapping: number[],
      _dataSize: number,
      _paddedDataSize: number,
    ): number {
      const index = _padIndexMapping[this.thread.x % _paddedDataSize];
      const bcIndex = Math.floor(this.thread.x / _paddedDataSize);
      if (index === -1) {
        return 0.0;
      }
      return x[bcIndex * _dataSize + index];
    })
    .setOutput([paddedSize]);

  function partialKernel(x: number[]): number[] {
    return kernel(x, shape, padIndexMapping, dataSize, paddedDataSize) as number[];
  }

  return [partialKernel, paddedShape];
}

export function createIm2ColKernel(
  gpu: GPU,
  shape: number[],
  kernelShape: number[],
  stride: number[],
): [(x: number[]) => number[], number[]] {
  // (B, C, H, W) -> (B, C, L, K) -> (B, L, C, K)
  // L is the output HxW
  // K is the kernel HxW

  const convolvedShape = [];

  // Calculate L
  let L = 1;
  for (let i = 0; i < shape.length - 2; i += 1) {
    const size = (shape[i + 2] - kernelShape[i]) / stride[i] + 1;
    L *= size;
    convolvedShape.push(size);
  }

  // Calculate K
  let K = 1;
  for (let i = 0; i < shape.length - 2; i += 1) {
    K *= kernelShape[i];
  }

  // Calculate W*H
  let dataSize = 1;
  for (let i = 0; i < shape.length - 2; i += 1) {
    dataSize *= shape[i + 2];
  }

  let kernelSize = 1;
  for (let i = 0; i < shape.length - 2; i += 1) {
    kernelSize *= kernelShape[i];
  }

  // Calculate mapping L index to pixel index (upper-left pixel in kernel)
  const l2pixMapping: number[] = [];
  for (let i = 0; i < L; i += 1) {
    // Identify index at the pixel data
    const pixelIndex = [];
    let tmp = L;
    let cursor = i;
    for (let j = 0; j < shape.length - 2; j += 1) {
      tmp /= convolvedShape[j];
      pixelIndex.push(Math.floor(cursor / tmp) * stride[j]);
      cursor -= Math.floor(cursor / tmp) * tmp;
    }

    // Convert multi-dimensional index to scalar index
    tmp = dataSize;
    let index = 0;
    for (let j = 0; j < shape.length - 2; j += 1) {
      tmp /= shape[j + 2];
      index += pixelIndex[j] * tmp;
    }
    l2pixMapping.push(index);
  }

  // Calculate mapping K index to pixel offset
  const k2pixMapping: number[] = [];
  for (let i = 0; i < K; i += 1) {
    // Identify index at the pixel data
    let tmp = kernelSize;
    const pixelIndex = [];
    let cursor = i;
    for (let j = 0; j < shape.length - 2; j += 1) {
      tmp /= kernelShape[j];
      pixelIndex.push(Math.floor(cursor / tmp));
      cursor -= Math.floor(cursor / tmp) * tmp;
    }

    // Convert multi-dimensional index to scalar index
    tmp = dataSize;
    let index = 0;
    for (let j = 0; j < shape.length - 2; j += 1) {
      tmp /= shape[j + 2];
      index += pixelIndex[j] * tmp;
    }
    k2pixMapping.push(index);
  }

  const outputShape = [shape[0], L, shape[1], K];
  const outputSize = shape[0] * shape[1] * L * K;
  const outputDataSize = L * K;

  // (B, C, H, W) -> (B, L, C, K)
  const kernel = gpu
    .createKernel(function (
      x: number[],
      _outputShape: number[],
      _outputDataSize: number,
      _l2pixMapping: number[],
      _k2pixMapping: number[],
      _dataSize: number,
    ): number {
      const [, _L, _C, _K] = _outputShape;
      let index = this.thread.x;
      const bIndex = Math.floor(index / (_L * _C * _K));
      index -= bIndex * (_L * _C * _K);
      const lIndex = Math.floor(index / (_C * _K));
      index -= lIndex * (_C * _K);
      const cIndex = Math.floor(index / _K);
      const kIndex = index % _K;

      return x[_dataSize * (_C * bIndex + cIndex) + _l2pixMapping[lIndex] + _k2pixMapping[kIndex]];
    })
    .setOutput([outputSize]);

  function partialKernel(x: number[]): number[] {
    return kernel(x, outputShape, outputDataSize, l2pixMapping, k2pixMapping, dataSize) as number[];
  }

  return [partialKernel, outputShape];
}
