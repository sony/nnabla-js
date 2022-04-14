// Copyright 2022 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';

/**
 * Returns ImageData object from Array.
 *
 * @remarks
 * The pixel data must be aranged in (C, H, W).
 *
 * @param array - The source pixel data.
 * @param channel - The channel size of the pixel data.
 * @param height - The image height.
 * @param width - The image width.
 * @param multiplier - The multiplier applied to each pixel. If not given, each pixel will be multiplied by 255.
 * @returns The ImageData object.
 *
 */
export function convertArrayToImage(
  array: number[],
  channel: number,
  height: number,
  width: number,
  multiplier: number | undefined,
): ImageData {
  const imageData = new ImageData(height, width);
  if (channel === 1) {
    // Gray to RGBA
    for (let i = 0; i < height * width; i += 1) {
      const pixel = array[i];
      for (let j = 0; j < 3; j += 1) {
        imageData.data[4 * i + j] = (multiplier || 255.0) * pixel;
      }
      imageData.data[4 * i + 3] = 255;
    }
  } else if (channel === 3) {
    for (let i = 0; i < height * width; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        const pixel = array[j * height * width + i];
        imageData.data[4 * i + j] = (multiplier || 255.0) * pixel;
      }
      imageData.data[4 * i + 3] = 255;
    }
  } else {
    throw Error('channel must be 1 or 3.');
  }
  return imageData;
}

/**
 * Returns Array object from ImageData.
 *
 * @remarks
 * The resulted array is aranged in (C, H, W).
 *
 * @param imageData - The source ImageData object.
 * @param channel - The channel size. If 1 is given, each pixel is averanged over RGB.
 * @param multiplier - The multiplier applied to each pixel. If not given, each pixel will be divided by 255.
 * @returns The Array object.
 *
 */
export function convertImageToArray(
  imageData: ImageData,
  channel: number,
  multiplier: number | undefined,
): number[] {
  const { height } = imageData;
  const { width } = imageData;
  const y = [];
  if (channel === 1) {
    for (let i = 0; i < height * width; i += 1) {
      const r = imageData.data[4 * i];
      const g = imageData.data[4 * i + 1];
      const b = imageData.data[4 * i + 2];
      y.push((r + g + b) / 3);
    }
  } else if (channel === 3) {
    for (let c = 0; c < 3; c += 1) {
      for (let i = 0; i < height * width; i += 1) {
        const index = 4 * i + c;
        y.push((multiplier || 1 / 255.0) * imageData.data[index]);
      }
    }
  } else {
    throw Error('channel must be 1 or 3.');
  }
  return y;
}

/**
 * Returns the kernel function that resizes images to the specified size.
 *
 * @param outShape - The target image size (C, H, W).
 * @param gpu - The GPU instance.
 * @param dynamicSize - The flag to accept the different input sizes.
 * @returns The kernel function that resizes the images.
 *
 * @example
 * ```
 * // create kernel
 * const resizeKernel = nnabla.ImageUtils.createResizeKernel([3, 224, 224], gpu, true);
 *
 * // resize image
 * const imageData = // source ImageData object.
 * const array = nnabla.ImageUtils.convertImageToArray(imageData, 3);
 * const resizedImage = resizeKernel(array, imageData.height, imageData.width);
 * console.log(resizedImage.length)  // 3x224x224
 * ```
 *
 */
export function createResizeKernel(
  outShape: number[],
  gpu: GPU,
  dynamicSize: boolean | undefined,
): IKernelRunShortcut {
  const [oC, oH, oW] = outShape;

  const kernel = gpu
    .createKernel(function (x: number[], iH: number, iW: number): number {
      const hScale = (this.constants.oH as number) / iH;
      const wScale = (this.constants.oW as number) / iW;
      const inSize = iH * iW;
      const outSize = (this.constants.oH as number) * (this.constants.oW as number);

      // output index
      const cIndex = Math.floor(this.thread.x / outSize);
      const hIndex = Math.floor((this.thread.x % outSize) / (this.constants.oW as number));
      const wIndex = this.thread.x % (this.constants.oW as number);

      // input index
      const inHIndex = Math.floor(hIndex / hScale);
      const inWIndex = Math.floor(wIndex / wScale);
      const inIndex = cIndex * inSize + inHIndex * iW + inWIndex;
      return x[inIndex];
    })
    .setConstants({ oC, oH, oW })
    .setOutput([oC * oH * oW])
    .setDynamicArguments(!!dynamicSize)
    .setPipeline(true);

  return kernel;
}

/**
 * Returns the asynchronous execution function that resizes images to the specified size.
 *
 * @param outShape - The target image size (C, H, W).
 * @param gpu - The GPU instance.
 * @param dynamicSize - The flag to accept the different input sizes.
 * @returns The kernel function that resizes the images.
 *
 * @example
 * ```
 * // create kernel
 * const resizeKernel = nnabla.ImageUtils.createResizeKernel([3, 224, 224], gpu, true);
 *
 * // resize image
 * const imageData = // source ImageData object.
 * const array = nnabla.ImageUtils.convertImageToArray(imageData, 3);
 * resizeKernel(array, imageData.height, imageData.width).then(resizedImage => {
 *   console.log(resizedImage.length)  // 3x224x224
 * });
 * ```
 *
 */
export function createAsyncResizeKernel(
  outShape: number[],
  gpu: GPU,
  dynamicSize: boolean | undefined,
): (x: number[], H: number, W: number) => Promise<Texture> {
  const kernel = createResizeKernel(outShape, gpu, dynamicSize);
  return (x: number[], H: number, W: number): Promise<Texture> =>
    new Promise<Texture>((resolve, reject) => {
      try {
        const output = kernel(x, H, W) as Texture;
        resolve(output);
      } catch (error) {
        reject(error);
      }
    });
}
