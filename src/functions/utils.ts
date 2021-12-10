import { GPU } from 'gpu.js';

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
    const size = (shape[i + 2] - kernelShape[i]) / stride[i] + 1
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
    let tmp = dataSize;
    let pixelIndex = [];
    let cursor = i;
    for (let j = 0; j < shape.length - 2; j += 1) {
      tmp /= shape[j + 2];
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

  const outputShape = [shape[0], shape[1], L, K];
  const outputSize = shape[0] * shape[1] * L * K;
  const transposedShape = [shape[0], L, shape[1], K];

  // (B, C, H, W) -> (B, C, L, K)
  const kernel = gpu.createKernel(function (
    x: number[],
    outputShape: number[],
    l2pixMapping: number[],
    k2pixMapping: number[],
    dataSize: number,
  ): number {
    let outDataSize = 1;
    for (let i = 0; i < 2; i += 1) {
      outDataSize *= outputShape[i + 2];
    }

    const bcIndex = Math.floor(this.thread.x / outDataSize);
    const lkIndex = this.thread.x % outDataSize;
    const lIndex = Math.floor(lkIndex / outputShape[3]);
    const kIndex = lkIndex % outputShape[3];

    return x[dataSize * bcIndex + l2pixMapping[lIndex] + k2pixMapping[kIndex]];
  })
  .setOutput([outputSize]);

  // (B, C, L, K) -> (B, L, C, K)
  const transposeKernel = gpu.createKernel(function (
    x: number[],
    shape: number[],
  ): number {
    const [, C, L, K] = shape;
    let index = this.thread.x;
    const bIndex = Math.floor(index / (L * C * K));
    index -= bIndex * L * C * K;
    const lIndex = Math.floor(index / (C * K));
    index -= lIndex * C * K;
    const cIndex = Math.floor(index / K);
    index -= cIndex * K;
    const kIndex = index;
    return x[bIndex * (C * L * K) + cIndex * (L * K) + lIndex * K + kIndex];
  })
  .setOutput([outputSize]);

  function partialKernel(x: number[]): number[] {
    const col = kernel(x, outputShape, l2pixMapping, k2pixMapping, dataSize) as number[];
    return transposeKernel(col, outputShape) as number[];
  }

  return [partialKernel, transposedShape];
}
