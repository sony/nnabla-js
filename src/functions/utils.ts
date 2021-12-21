import { GPU, IKernelRunShortcut } from 'gpu.js';

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

  let kernel: IKernelRunShortcut | null = null;
  if (cacheX !== null && cacheY !== null) {
    kernel = gpu.createKernel(function (): number {
      const xRowSize = (this.constants.xShape as number[])[0];
      const xColSize = (this.constants.xShape as number[])[1];
      const yRowSize = (this.constants.yShape as number[])[0];
      const yColSize = (this.constants.yShape as number[])[1];
      const xBase = Math.floor(
        this.thread.x / ((this.constants.transposeY as boolean) ? yRowSize : yColSize),
      );
      const yBase = this.thread.x % ((this.constants.transposeY as boolean) ? yRowSize : yColSize);
      const dim = (this.constants.transposeX as boolean) ? xRowSize : xColSize;
      let output = 0.0;
      for (let i = 0; i < dim; i += 1) {
        let xValue: number = 0.0;
        if (this.constants.transposeX) {
          xValue = (this.constants.x as number[])[i * xColSize + xBase];
        } else {
          xValue = (this.constants.x as number[])[xBase * xColSize + i];
        }

        let yValue: number = 0.0;
        if (this.constants.transposeY) {
          yValue = (this.constants.y as number[])[yBase * yColSize + i];
        } else {
          yValue = (this.constants.y as number[])[i * yColSize + yBase];
        }

        output += xValue * yValue;
      }
      return output;
    });
    constants.x = cacheX;
    constants.y = cacheY;
  } else if (cacheX !== null && cacheY === null) {
    kernel = gpu.createKernel(function (y: number[]): number {
      const xRowSize = (this.constants.xShape as number[])[0];
      const xColSize = (this.constants.xShape as number[])[1];
      const yRowSize = (this.constants.yShape as number[])[0];
      const yColSize = (this.constants.yShape as number[])[1];
      const xBase = Math.floor(
        this.thread.x / ((this.constants.transposeY as boolean) ? yRowSize : yColSize),
      );
      const yBase = this.thread.x % ((this.constants.transposeY as boolean) ? yRowSize : yColSize);
      const dim = (this.constants.transposeX as boolean) ? xRowSize : xColSize;
      let output = 0.0;
      for (let i = 0; i < dim; i += 1) {
        let xValue: number = 0.0;
        if (this.constants.transposeX) {
          xValue = (this.constants.x as number[])[i * xColSize + xBase];
        } else {
          xValue = (this.constants.x as number[])[xBase * xColSize + i];
        }

        let yValue: number = 0.0;
        if (this.constants.transposeY) {
          yValue = y[yBase * yColSize + i];
        } else {
          yValue = y[i * yColSize + yBase];
        }

        output += xValue * yValue;
      }
      return output;
    });
    constants.x = cacheX;
  } else if (cacheX === null && cacheY !== null) {
    kernel = gpu.createKernel(function (x: number[]): number {
      const xRowSize = (this.constants.xShape as number[])[0];
      const xColSize = (this.constants.xShape as number[])[1];
      const yRowSize = (this.constants.yShape as number[])[0];
      const yColSize = (this.constants.yShape as number[])[1];
      const xBase = Math.floor(
        this.thread.x / ((this.constants.transposeY as boolean) ? yRowSize : yColSize),
      );
      const yBase = this.thread.x % ((this.constants.transposeY as boolean) ? yRowSize : yColSize);
      const dim = (this.constants.transposeX as boolean) ? xRowSize : xColSize;
      let output = 0.0;
      for (let i = 0; i < dim; i += 1) {
        let xValue: number = 0.0;
        if (this.constants.transposeX) {
          xValue = x[i * xColSize + xBase];
        } else {
          xValue = x[xBase * xColSize + i];
        }

        let yValue: number = 0.0;
        if (this.constants.transposeY) {
          yValue = (this.constants.y as number[])[yBase * yColSize + i];
        } else {
          yValue = (this.constants.y as number[])[i * yColSize + yBase];
        }

        output += xValue * yValue;
      }
      return output;
    });
    constants.y = cacheY;
  } else {
    kernel = gpu.createKernel(function (x: number[], y: number[]): number {
      const xRowSize = (this.constants.xShape as number[])[0];
      const xColSize = (this.constants.xShape as number[])[1];
      const yRowSize = (this.constants.yShape as number[])[0];
      const yColSize = (this.constants.yShape as number[])[1];
      const xBase = Math.floor(
        this.thread.x / ((this.constants.transposeY as boolean) ? yRowSize : yColSize),
      );
      const yBase = this.thread.x % ((this.constants.transposeY as boolean) ? yRowSize : yColSize);
      const dim = (this.constants.transposeX as boolean) ? xRowSize : xColSize;
      let output = 0.0;
      for (let i = 0; i < dim; i += 1) {
        let xValue: number = 0.0;
        if (this.constants.transposeX) {
          xValue = x[i * xColSize + xBase];
        } else {
          xValue = x[xBase * xColSize + i];
        }

        let yValue: number = 0.0;
        if (this.constants.transposeY) {
          yValue = y[yBase * yColSize + i];
        } else {
          yValue = y[i * yColSize + yBase];
        }

        output += xValue * yValue;
      }
      return output;
    });
  }

  kernel.setConstants(constants).setOutput([outputShape[0] * outputShape[1]]);

  return [kernel, outputShape];
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

  let kernel: IKernelRunShortcut | null = null;
  if (cacheX !== null && cacheY !== null) {
    kernel = gpu.createKernel(function (): number {
      const xBatchSize = (this.constants.xShape as number[])[0];
      const xRowSize = (this.constants.xShape as number[])[1];
      const xColSize = (this.constants.xShape as number[])[2];
      const yBatchSize = (this.constants.yShape as number[])[0];
      const yRowSize = (this.constants.yShape as number[])[1];
      const yColSize = (this.constants.yShape as number[])[2];
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
        xValue = (this.constants.x as number[])[xOffset + xBase * xColSize + i];

        let yValue: number = 0.0;
        yValue = (this.constants.y as number[])[yOffset + i * yColSize + yBase];

        output += xValue * yValue;
      }
      return output;
    });
    constants.x = cacheX;
    constants.y = cacheY;
  } else if (cacheX !== null && cacheY === null) {
    kernel = gpu.createKernel(function (y: number[]): number {
      const xBatchSize = (this.constants.xShape as number[])[0];
      const xRowSize = (this.constants.xShape as number[])[1];
      const xColSize = (this.constants.xShape as number[])[2];
      const yBatchSize = (this.constants.yShape as number[])[0];
      const yRowSize = (this.constants.yShape as number[])[1];
      const yColSize = (this.constants.yShape as number[])[2];
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
        xValue = (this.constants.x as number[])[xOffset + xBase * xColSize + i];

        let yValue: number = 0.0;
        yValue = y[yOffset + i * yColSize + yBase];

        output += xValue * yValue;
      }
      return output;
    });
    constants.x = cacheX;
  } else if (cacheX === null && cacheY !== null) {
    kernel = gpu.createKernel(function (x: number[]): number {
      const xBatchSize = (this.constants.xShape as number[])[0];
      const xRowSize = (this.constants.xShape as number[])[1];
      const xColSize = (this.constants.xShape as number[])[2];
      const yBatchSize = (this.constants.yShape as number[])[0];
      const yRowSize = (this.constants.yShape as number[])[1];
      const yColSize = (this.constants.yShape as number[])[2];
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
        yValue = (this.constants.y as number[])[yOffset + i * yColSize + yBase];

        output += xValue * yValue;
      }
      return output;
    });
    constants.y = cacheY;
  } else {
    kernel = gpu.createKernel(function (x: number[], y: number[]): number {
      const xBatchSize = (this.constants.xShape as number[])[0];
      const xRowSize = (this.constants.xShape as number[])[1];
      const xColSize = (this.constants.xShape as number[])[2];
      const yBatchSize = (this.constants.yShape as number[])[0];
      const yRowSize = (this.constants.yShape as number[])[1];
      const yColSize = (this.constants.yShape as number[])[2];
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
