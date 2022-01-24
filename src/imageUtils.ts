import { GPU, IKernelRunShortcut } from 'gpu.js';

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

export function convertImageToArray(
  imageData: ImageData,
  channel: number,
  multiplier: number | undefined,
): number[] {
  const height = imageData.height;
  const width = imageData.width;
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

export function createResizeKernel(
  outShape: number[],
  gpu: GPU,
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
    .setPipeline(true);

  return kernel;
}
