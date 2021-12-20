import { GPU, IKernelRunShortcut } from 'gpu.js';

function matmulImpl(
  idx: number,
  x: number[],
  y: number[],
  xShape: number[],
  yShape: number[],
  transposeX: boolean,
  transposeY: boolean,
): number {
  const [xRowSize, xColSize] = xShape;
  const [yRowSize, yColSize] = yShape;
  const xBase = Math.floor(idx / (transposeY ? yRowSize : yColSize));
  const yBase = idx % (transposeY ? yRowSize : yColSize);
  const dim = transposeX ? xRowSize : xColSize;
  let output = 0.0;
  for (let i = 0; i < dim; i += 1) {
    let xValue: number = 0.0;
    if (transposeX) {
      xValue = x[i * xColSize + xBase];
    } else {
      xValue = x[xBase * xColSize + i];
    }

    let yValue: number = 0.0;
    if (transposeY) {
      yValue = y[yBase * yColSize + i];
    } else {
      yValue = y[i * yColSize + yBase];
    }

    output += xValue * yValue;
  }
  return output;
}

export function createMatmulKernel(
  gpu: GPU,
  xShape: number[],
  yShape: number[],
  transposeX: boolean,
  transposeY: boolean,
  cacheX: number[] | null,
  cacheY: number[] | null,
): [IKernelRunShortcut, number[]] {
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

  const constants: { [key: string]: any } = {
    xShape,
    yShape,
    transposeX,
    transposeY,
  };

  gpu.addFunction(matmulImpl);

  let kernel: IKernelRunShortcut | null = null;
  if (cacheX !== null && cacheY !== null) {
    kernel = gpu.createKernel(function (): number {
      return matmulImpl(
        this.thread.x,
        this.constants.x as number[],
        this.constants.y as number[],
        this.constants.xShape as number[],
        this.constants.yShape as number[],
        this.constants.transposeX as boolean,
        this.constants.transposeY as boolean,
      );
    });
    constants.x = cacheX;
    constants.y = cacheY;
  } else if (cacheX !== null && cacheY === null) {
    kernel = gpu.createKernel(function (y: number[]): number {
      return matmulImpl(
        this.thread.x,
        this.constants.x as number[],
        y,
        this.constants.xShape as number[],
        this.constants.yShape as number[],
        this.constants.transposeX as boolean,
        this.constants.transposeY as boolean,
      );
    });
    constants.x = cacheX;
  } else if (cacheX === null && cacheY !== null) {
    kernel = gpu.createKernel(function (x: number[]): number {
      return matmulImpl(
        this.thread.x,
        x,
        this.constants.y as number[],
        this.constants.xShape as number[],
        this.constants.yShape as number[],
        this.constants.transposeX as boolean,
        this.constants.transposeY as boolean,
      );
    });
    constants.y = cacheY;
  } else {
    kernel = gpu.createKernel(function (x: number[], y: number[]): number {
      return matmulImpl(
        this.thread.x,
        x,
        y,
        this.constants.xShape as number[],
        this.constants.yShape as number[],
        this.constants.transposeX as boolean,
        this.constants.transposeY as boolean,
      );
    });
  }

  kernel.setConstants(constants).setOutput([outputShape[0] * outputShape[1]]);

  return [kernel, outputShape];
}

function batchMatmulImpl(
  idx: number,
  x: number[],
  y: number[],
  xShape: number[],
  yShape: number[],
): number {
  const [xBatchSize, xRowSize, xColSize] = xShape;
  const [yBatchSize, yRowSize, yColSize] = yShape;
  const batchIndex = Math.floor(idx / (xRowSize * yColSize));
  const xBatch = xBatchSize === 1 ? 0 : batchIndex;
  const yBatch = yBatchSize === 1 ? 0 : batchIndex;
  const xOffset = xBatch * xRowSize * xColSize;
  const yOffset = yBatch * yRowSize * yColSize;
  const xBase = Math.floor((idx % (xRowSize * yColSize)) / yColSize);
  const yBase = idx % yColSize;
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
}

export function createBatchMatmulKernel(
  gpu: GPU,
  xShape: number[],
  yShape: number[],
  cacheX: number[] | null,
  cacheY: number[] | null,
): [IKernelRunShortcut, number[]] {
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

  const constants: { [key: string]: any } = { xShape, yShape };

  gpu.addFunction(batchMatmulImpl);

  let kernel: IKernelRunShortcut | null = null;
  if (cacheX !== null && cacheY !== null) {
    kernel = gpu.createKernel(function (): number {
      return batchMatmulImpl(
        this.thread.x,
        this.constants.x as number[],
        this.constants.y as number[],
        this.constants.xShape as number[],
        this.constants.yShape as number[],
      );
    });
    constants.x = cacheX;
    constants.y = cacheY;
  } else if (cacheX !== null && cacheY === null) {
    kernel = gpu.createKernel(function (y: number[]): number {
      return batchMatmulImpl(
        this.thread.x,
        this.constants.x as number[],
        y,
        this.constants.xShape as number[],
        this.constants.yShape as number[],
      );
    });
    constants.x = cacheX;
  } else if (cacheX === null && cacheY !== null) {
    kernel = gpu.createKernel(function (x: number[]): number {
      return batchMatmulImpl(
        this.thread.x,
        x,
        this.constants.y as number[],
        this.constants.xShape as number[],
        this.constants.yShape as number[],
      );
    });
    constants.y = cacheY;
  } else {
    kernel = gpu.createKernel(function (x: number[], y: number[]): number {
      return batchMatmulImpl(
        this.thread.x,
        x,
        y,
        this.constants.xShape as number[],
        this.constants.yShape as number[],
      );
    });
  }

  kernel.setConstants(constants).setOutput([outputShape[0] * outputShape[1] * outputShape[2]]);

  return [kernel, outputShape];
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
