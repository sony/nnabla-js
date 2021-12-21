import { GPU, IKernelRunShortcut } from 'gpu.js';

export function createMatmulKernel(
  gpu: GPU,
  xShape: number[],
  yShape: number[],
  transposeX: boolean,
  transposeY: boolean,
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

  const [xRowSize, xColSize] = xShape;
  const [yRowSize, yColSize] = yShape;

  const kernel = gpu.createKernel(function (x: number[], y: number[]): number {
    const tXRowSize = this.constants.xRowSize as number;
    const tXColSize = this.constants.xColSize as number;
    const tYRowSize = this.constants.yRowSize as number;
    const tYColSize = this.constants.yColSize as number;
    const xBase = Math.floor(this.thread.x / ((this.constants.transposeY as boolean) ? tYRowSize : tYColSize));
    const yBase = this.thread.x % ((this.constants.transposeY as boolean) ? tYRowSize : tYColSize);
    const dim = (this.constants.transposeX as boolean) ? tXRowSize : tXColSize;
    let output = 0.0;
    for (let i = 0; i < dim; i += 1) {
      let xValue: number = 0.0;
      if (this.constants.transposeX) {
        xValue = x[i * tXColSize + xBase];
      } else {
        xValue = x[xBase * tXColSize + i];
      }

      let yValue: number = 0.0;
      if (this.constants.transposeY) {
        yValue = y[yBase * tYColSize + i];
      } else {
        yValue = y[i * tYColSize + yBase];
      }

      output += xValue * yValue;
    }
    return output;
  })
  .setConstants({
    xRowSize,
    xColSize,
    yRowSize,
    yColSize,
    transposeX,
    transposeY,
  })
  .setOutput([outputShape[0] * outputShape[1]]);

  return [kernel, outputShape];
}

export function createBatchMatmulKernel(
  gpu: GPU,
  xShape: number[],
  yShape: number[],
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

  const [xBatchSize, xRowSize, xColSize] = xShape;
  const [yBatchSize, yRowSize, yColSize] = yShape;

  const kernel = gpu.createKernel(function (x: number[], y: number[]): number {
    const tXBatchSize = this.constants.xBatchSize as number;
    const tXRowSize = this.constants.xRowSize as number;
    const tXColSize = this.constants.xColSize as number;
    const tYBatchSize = this.constants.yBatchSize as number;
    const tYRowSize = this.constants.yRowSize as number;
    const tYColSize = this.constants.yColSize as number;
    const batchIndex = Math.floor(this.thread.x / (tXRowSize * tYColSize));
    const xBatch = tXBatchSize === 1 ? 0 : batchIndex;
    const yBatch = tYBatchSize === 1 ? 0 : batchIndex;
    const xOffset = xBatch * tXRowSize * tXColSize;
    const yOffset = yBatch * tYRowSize * tYColSize;
    const xBase = Math.floor((this.thread.x % (tXRowSize * tYColSize)) / tYColSize);
    const yBase = this.thread.x % tYColSize;
    const dim = tXColSize;
    let output = 0.0;
    for (let i = 0; i < dim; i += 1) {
      let xValue: number = 0.0;
      xValue = x[xOffset + xBase * tXColSize + i];

      let yValue: number = 0.0;
      yValue = y[yOffset + i * tYColSize + yBase];

      output += xValue * yValue;
    }
    return output;
  })
  .setConstants({
    xBatchSize,
    xRowSize,
    xColSize,
    yBatchSize,
    yRowSize,
    yColSize,
  })
  .setOutput([outputShape[0] * outputShape[1] * outputShape[2]]);

  return [kernel, outputShape];
}

export function createIm2Col2dKernel(
  gpu: GPU,
  shape: number[],
  kernelShape: number[],
  stride: number[],
  pad: number[],
): [IKernelRunShortcut, number[]] {
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
    .createKernel(function (x: number[]): number {
      // (B, C, H, W) -> (B, C, K, L)
      const tH = this.constants.H as number;
      const tW = this.constants.W as number;
      const tKH = this.constants.kH as number;
      const tKW = this.constants.kW as number;
      const tSH = this.constants.sH as number;
      const tSW = this.constants.sW as number;
      const tPH = this.constants.pH as number;
      const tPW = this.constants.pW as number;
      const tOH = this.constants.oH as number;
      const tOW = this.constants.oW as number;
      const bcIndex = Math.floor(this.thread.x / (tKH * tKW * tOH * tOW));
      const hIdx = Math.floor(this.thread.x / tOW);
      const hIndex = hIdx % tOH;
      const wIndex = this.thread.x % tOW;
      const cIm = Math.floor(hIdx / tOH);
      const yK = Math.floor(cIm / tKW) % tKH;
      const xK = cIm % tKW;
      const yI = hIndex * tSH - tPH + yK;
      const xI = wIndex * tSW - tPW + xK;
      if (yI >= 0 && xI >= 0 && yI < tH && xI < tW) {
        return x[bcIndex * tH * tW + yI * tW + xI];
      }
      return 0.0;
    })
    .setConstants({
      H,
      W,
      kH,
      kW,
      sH,
      sW,
      pH,
      pW,
      oH,
      oW,
    })
    .setOutput([outputSize]);

  return [kernel, outputShape];
}

export function createIm2ColKernel(
  gpu: GPU,
  shape: number[],
  kernelShape: number[],
  stride: number[],
  pad: number[],
): [IKernelRunShortcut, number[]] {
  if (shape.length === 4) {
    return createIm2Col2dKernel(gpu, shape, kernelShape, stride, pad);
  }
  throw Error('im2col only supports (B, C, H, W) shape.');
}
