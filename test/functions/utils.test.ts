import { GPU } from 'gpu.js';
import Variable from '../../src/variable';
import {
  createBatchMatmulKernel,
  createMatmulKernel,
  createIm2ColKernel,
} from '../../src/functions/utils';
import { expectAllClose } from '../testUtils';

function transpose(x: number[], shape: number[]): number[] {
  const y: number[] = [];
  for (let i = 0; i < shape[1]; i += 1) {
    for (let j = 0; j < shape[0]; j += 1) {
      y.push(x[j * shape[1] + i]);
    }
  }
  return y;
}

function refMatmul(x: number[], y: number[], xShape: number[], yShape: number[]): number[] {
  const [xRowSize, xColSize] = xShape;
  const [, yColSize] = yShape;
  const output = [...Array(xRowSize * yColSize)].map(() => 0.0);
  for (let i = 0; i < xRowSize; i += 1) {
    for (let j = 0; j < yColSize; j += 1) {
      for (let k = 0; k < xColSize; k += 1) {
        output[i * yColSize + j] += x[i * xColSize + k] * y[k * yColSize + j];
      }
    }
  }
  return output;
}

test.each([
  [false, false],
  [true, false],
  [false, true],
  [true, true],
])('test-matmul', (transposeX: boolean, transposeY: boolean) => {
  let xShape = transposeX ? [256, 32] : [32, 256];
  const x = Variable.rand('x', xShape);
  let yShape = transposeY ? [64, 256] : [256, 64];
  const y = Variable.rand('y', yShape);
  const gpu = new GPU();

  const [matmulKernel, outputShape] = createMatmulKernel(
    gpu,
    xShape,
    yShape,
    transposeX,
    transposeY,
  );
  const z = matmulKernel(x.toArray(), y.toArray()) as number[];

  const xData = transposeX ? transpose(x.toArray(), xShape) : x.toArray();
  xShape = transposeX ? [xShape[1], xShape[0]] : xShape;
  const yData = transposeY ? transpose(y.toArray(), yShape) : y.toArray();
  yShape = transposeY ? [yShape[1], yShape[0]] : yShape;

  const refOutputShape = [32, 64];
  const refZ = refMatmul(xData, yData, xShape, yShape);
  expect(outputShape).toEqual(refOutputShape);
  expectAllClose(z, refZ, 0.0001);
});

function refBatchMatmul(x: number[], y: number[], xShape: number[], yShape: number[]): number[] {
  const [xBatchSize, xRowSize, xColSize] = xShape;
  const [yBatchSize, yRowSize, yColSize] = yShape;
  const batchSize = Math.max(xBatchSize, yBatchSize);
  const output = [...Array(batchSize * xRowSize * yColSize)].map(() => 0.0);
  for (let i = 0; i < batchSize; i += 1) {
    const oOffset = i * xRowSize * yColSize;
    const xOffset = (xBatchSize === 1 ? 0 : i) * xRowSize * xColSize;
    const yOffset = (yBatchSize === 1 ? 0 : i) * yRowSize * yColSize;
    for (let j = 0; j < xRowSize; j += 1) {
      for (let k = 0; k < yColSize; k += 1) {
        for (let l = 0; l < xColSize; l += 1) {
          const index = oOffset + j * yColSize + k;
          const xIndex = xOffset + j * xColSize + l;
          const yIndex = yOffset + l * yColSize + k;
          output[index] += x[xIndex] * y[yIndex];
        }
      }
    }
  }
  return output;
}

test.each([
  [1, 32],
  [32, 1],
  [32, 32],
])('test-batch-matmul', (xBatchSize: number, yBatchSize: number) => {
  const xShape = [xBatchSize, 16, 256];
  const x = Variable.rand('x', xShape);
  const yShape = [yBatchSize, 256, 64];
  const y = Variable.rand('y', yShape);
  const gpu = new GPU();

  const [matmulKernel, outputShape] = createBatchMatmulKernel(gpu, xShape, yShape);
  const z = matmulKernel(x.toArray(), y.toArray()) as number[];

  const refOutputShape = [Math.max(xBatchSize, yBatchSize), 16, 64];
  const refZ = refBatchMatmul(x.toArray(), y.toArray(), xShape, yShape);
  expect(outputShape).toEqual(refOutputShape);
  expectAllClose(z, refZ, 0.0001);
});

function refIm2Col(
  x: number[],
  shape: number[],
  outHeight: number,
  outWidth: number,
  kernelShape: number[],
  stride: number[],
): number[] {
  const [B, C, H, W] = shape;

  // (B, C, L, K)
  const y: number[] = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let k = 0; k < outHeight; k += 1) {
        for (let l = 0; l < outWidth; l += 1) {
          const baseIndex = i * (C * H * W) + j * (H * W) + k * stride[0] * W + l * stride[1];
          for (let m = 0; m < kernelShape[0]; m += 1) {
            for (let n = 0; n < kernelShape[1]; n += 1) {
              const index = baseIndex + m * W + n;
              y.push(x[index]);
            }
          }
        }
      }
    }
  }

  // (B, C, L, K) -> (B, C, K, L)
  const L = outHeight * outWidth;
  const K = kernelShape[0] * kernelShape[1];
  const transposedY: number[] = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let k = 0; k < K; k += 1) {
        for (let l = 0; l < L; l += 1) {
          const index = i * (C * L * K) + j * (L * K) + l * K + k;
          transposedY.push(y[index]);
        }
      }
    }
  }
  return transposedY;
}

test('test-im2col', () => {
  const shape = [32, 3, 28, 28];
  const x = Variable.rand('x', shape);
  const kernelShape = [16, 3, 2, 2];
  const stride = [2, 2];
  const gpu = new GPU();

  const outHeight = (28 - 2) / 2 + 1;
  const outWidth = outHeight;

  const [im2col, outputShape] = createIm2ColKernel(gpu, shape, [2, 2], stride, [0, 0]);
  const y = im2col(x.toArray()) as number[];

  const refOutputShape = [32, 3, 4, outHeight * outWidth];
  const refY = refIm2Col(
    x.toArray(),
    shape,
    outHeight,
    outWidth,
    [kernelShape[2], kernelShape[3]],
    stride,
  );
  expect(outputShape).toEqual(refOutputShape);
  expectAllClose(y, refY, 0.0001);
});
