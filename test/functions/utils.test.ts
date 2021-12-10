import { GPU } from 'gpu.js';
import Variable from '../../src/variable';
import { createIm2ColKernel } from '../../src/functions/utils';
import { expectAllClose } from '../testUtils';

function refIm2Col(
  x: number[],
  shape: number[],
  outputShape: number[],
  outHeight: number,
  outWidth: number,
  kernelShape: number[],
  stride: number[],
): number[] {
  const [B, C, H, W] = shape;
  const [, L, , K] = outputShape;
  const [strideH, strideW] = stride;
  const y = [...Array(B * C * H * W)].map(() => 0.0);
  const transposedY = [...Array(B * C * H * W)].map(() => 0.0);

  // (B, C, H, W) -> (B, C, L, K)
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let k = 0; k < L; k += 1) {
        for (let l = 0; l < K; l += 1) {
          const bOffset = i * C * H * W;
          const cOffset = j * H * W;
          const lOffset = W * Math.floor(k / outWidth) * strideH + (k % outWidth) * strideW;
          const kOffset = W * Math.floor(l / strideW) + l % strideH;
          y[i * C * L * K + j + L * K + k + K + l] = x[bOffset + cOffset + lOffset + kOffset];
        }
      }
    }
  }

  // (B, C, L, K) -> (B, L, C, K)
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < L; j += 1) {
      for (let k = 0; k < C; k += 1) {
        for (let l = 0; l < K; l += 1) {
          transposedY[i * L * C * K + j * C * K + k * K + l] = y[i * C * L * K + j + L * K + k + K + l];
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
  const refY = refIm2Col(x.data, shape, refOutputShape, outHeight, outWidth, kernelShape, stride);
  expect(outputShape).toEqual(refOutputShape);
  expectAllClose(y, refY, 0.0001);
});
