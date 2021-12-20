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

export function createBatchMatmulKernel(
  gpu: GPU,
  xShape: number[],
  yShape: number[],
): [(x: number[], y: number[]) => number[], number[]] {
  if (xShape.length !== 3) {
    throw Error(`invalid x shape: ${xShape}`);
  }
  if (yShape.length !== 3) {
    throw Error(`invalid y shape: ${yShape}`);
  }
  if (xShape[0] !== 1 && yShape[0] !== 1 && xShape[0] !== yShape[0]) {
    throw Error(`invalid batch size: x=${xShape[0]}, y=${yShape[0]}`);
  }

  const outputShape = [];
  outputShape.push(Math.max(xShape[0], yShape[0]));
  outputShape.push(xShape[1]);
  outputShape.push(yShape[2]);

  const kernel = gpu
    .createKernel(function (
      x: number[],
      y: number[],
      _xShape: number[],
      _yShape: number[],
    ): number {
      const [xBatchSize, xRowSize, xColSize] = _xShape;
      const [yBatchSize, yRowSize, yColSize] = _yShape;
      const batchIndex = Math.floor(this.thread.x / (xRowSize * yColSize));
      const xBatch = xBatchSize === 1 ? 0 : batchIndex;
      const yBatch = yBatchSize === 1 ? 0 : batchIndex;
      const xOffset = xBatch * xRowSize * xColSize;
      const yOffset = yBatch * yRowSize * yColSize;
      const xBase = Math.floor((this.thread.x % (xRowSize * yColSize)) / yColSize);
      const yBase = this.thread.x % yColSize;
      const dim = xColSize;
      let output = 0.0;
      for (let i = 0; i < dim; i += 1) {
        let xValue: number = 0.0;
        xValue = x[xOffset + xBase * xColSize + i];

        let yValue: number = 0.0;
        yValue = y[yOffset + i * yColSize + yBase];

        output += xValue * yValue;
      }
      return output;
    })
    .setOutput([outputShape[0] * outputShape[1] * outputShape[2]]);

  function partialKernel(x: number[], y: number[]): number[] {
    return kernel(x, y, xShape, yShape) as number[];
  }

  return [partialKernel, outputShape];
}

export function createIm2Col2dKernel(
  gpu: GPU,
  shape: number[],
  kernelShape: number[],
  stride: number[],
  pad: number[],
): [(x: number[]) => number[], number[]] {
  // (B, C, H, W) -> (B, C, K, L)
  // L is the output HxW
  // K is the kernel HxW

  const [B, C, H, W] = shape;
  const [kH, kW] = kernelShape;
  const [sH, sW] = stride;
  const [pH, pW] = pad;

  // Calculate convoluted shape
  const oB = B;
  const oH = Math.floor((H + 2 * pH - kH) / sH) + 1;
  const oW = Math.floor((W + 2 * pW - kW) / sW) + 1;

  // Calculate im2col shape
  const K = kH * kW;
  const L = oH * oW;
  const outputShape = [oB, C, K, L];
  const outputSize = oB * C * K * L;

  const kernel = gpu
    .createKernel(function (
      x: number[],
      _H: number,
      _W: number,
      _kH: number,
      _kW: number,
      _sH: number,
      _sW: number,
      _pH: number,
      _pW: number,
      _oH: number,
      _oW: number,
    ): number {
      // (B, C, H, W) -> (B, C, K, L)
      const bcIndex = Math.floor(this.thread.x / (_kH * _kW * _oH * _oW));
      const hIdx = Math.floor(this.thread.x / _oW);
      const hIndex = hIdx % _oH;
      const wIndex = this.thread.x % _oW;
      const cIm = Math.floor(hIdx / _oH);
      const yK = Math.floor(cIm / _kW) % _kH;
      const xK = cIm % _kW;
      const yI = hIndex * _sH - _pH + yK;
      const xI = wIndex * _sW - _pW + xK;
      if (yI >= 0 && xI >= 0 && yI < _H && xI < _W) {
        return x[bcIndex * _H * _W + yI * _W + xI];
      }
      return 0.0;
    })
    .setOutput([outputSize]);

  function im2col(x: number[]): number[] {
    return kernel(x, H, W, kH, kW, sH, sW, pH, pW, oH, oW) as number[];
  }

  return [im2col, outputShape];
}

export function createIm2ColKernel(
  gpu: GPU,
  shape: number[],
  kernelShape: number[],
  stride: number[],
  pad: number[],
): [(x: number[]) => number[], number[]] {
  if (shape.length === 4) {
    return createIm2Col2dKernel(gpu, shape, kernelShape, stride, pad);
  }
  throw Error('im2col only supports (B, C, H, W) shape.');
}
