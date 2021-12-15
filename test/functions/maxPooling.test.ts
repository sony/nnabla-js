import { MaxPoolingParameter, Shape } from '../../src/proto/nnabla_pb';
import MaxPooling from '../../src/functions/maxPooling';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function maxPoolingRef(
  x: number[],
  shape: number[],
  stride: number[],
  kernel: number[],
  outputShape: number[],
): number[] {
  const [B, C, H, W] = shape;
  const [sH, sW] = stride;
  const [kH, kW] = kernel;

  const y: number[] = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let k = 0; k < outputShape[2]; k += 1) {
        for (let l = 0; l < outputShape[3]; l += 1) {
          const bOffset = i * C * H * W;
          const cOffset = j * H * W;
          const index = bOffset + cOffset + k * sH * W + l * sW;
          let maxValue = -1000000;
          for (let m = 0; m < kH; m += 1) {
            for (let n = 0; n < kW; n += 1) {
              const value = x[index + m * W + n];
              if (value > maxValue) {
                maxValue = value;
              }
            }
          }
          y.push(maxValue);
        }
      }
    }
  }

  return y;
}

test('test-max-pooling', () => {
  const x = Variable.rand('x', [32, 3, 28, 28]);
  const y = Variable.rand('y', [32, 3, 13, 13]);

  const param = new MaxPoolingParameter();
  const pad = new Shape();
  pad.addDim(0);
  pad.addDim(0);
  param.setPad(pad);
  const stride = new Shape();
  stride.addDim(2);
  stride.addDim(2);
  param.setStride(stride);
  const kernel = new Shape();
  kernel.addDim(4);
  kernel.addDim(4);
  param.setKernel(kernel);

  const pooling = new MaxPooling(param);

  pooling.setup([x], [y]);
  pooling.forward([x], [y]);

  const yRef = maxPoolingRef(x.data, x.shape, [2, 2], [4, 4], y.shape);
  expectAllClose(y.data, yRef, 0.0001);
});
