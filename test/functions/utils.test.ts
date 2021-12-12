import { GPU } from 'gpu.js';
import Variable from '../../src/variable';
import { createMatmulKernel, createPadKernel, createIm2ColKernel } from '../../src/functions/utils';
import { expectAllClose } from '../testUtils';

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

test('test-matmul', () => {
  const xShape = [32, 256];
  const x = Variable.rand('x', xShape);
  const yShape = [256, 64];
  const y = Variable.rand('y', yShape);
  const gpu = new GPU();

  const [matmulKernel, outputShape] = createMatmulKernel(gpu, xShape, yShape);
  const z = matmulKernel(x.data, y.data);

  const refOutputShape = [32, 64];
  const refZ = refMatmul(x.data, y.data, xShape, yShape);
  expect(outputShape).toEqual(refOutputShape);
  expectAllClose(z, refZ, 0.0001);
});

function refPad(x: number[], shape: number[], pad: number[]): number[] {
  const [B, C, H, W] = shape;
  const [padH, padW] = pad;
  const y: number[] = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let k = 0; k < H + 2 * padH; k += 1) {
        for (let l = 0; l < W + 2 * padW; l += 1) {
          if (k < padH || k >= H + padH || l < padW || l >= W + padW) {
            y.push(0.0);
          } else {
            const index = i * (C * H * W) + j * (H * W) + (k - padH) * W + (l - padW);
            y.push(x[index]);
          }
        }
      }
    }
  }
  return y;
}

test('test-pad', () => {
  const shape = [32, 3, 28, 28];
  const x = Variable.rand('x', shape);
  const pad = [2, 2];
  const gpu = new GPU();

  const [padKernel, outputShape] = createPadKernel(gpu, shape, pad);
  const y = padKernel(x.data);

  const refOutputShape = [32, 3, 32, 32];
  const refY = refPad(x.data, shape, pad);
  expect(outputShape).toEqual(refOutputShape);
  expectAllClose(y, refY, 0.0001);
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

  // (B, C, L, K) -> (B, L, C, K)
  const L = outHeight * outWidth;
  const K = kernelShape[0] * kernelShape[1];
  const transposedY: number[] = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < L; j += 1) {
      for (let k = 0; k < C; k += 1) {
        for (let l = 0; l < K; l += 1) {
          const index = i * (C * L * K) + k * (L * K) + j * K + l;
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
  const kernelShape = [2, 2];
  const stride = [2, 2];
  const gpu = new GPU();

  const outHeight = (28 - 2) / 2 + 1;
  const outWidth = outHeight;

  const [im2col, outputShape] = createIm2ColKernel(gpu, shape, kernelShape, stride);
  const y = im2col(x.data);

  const refOutputShape = [32, outHeight * outWidth, 3, 4];
  const refY = refIm2Col(x.data, shape, outHeight, outWidth, kernelShape, stride);
  expect(outputShape).toEqual(refOutputShape);
  expectAllClose(y, refY, 0.0001);
});
