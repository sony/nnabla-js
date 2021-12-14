import { ConvolutionParameter, Shape } from '../../src/proto/nnabla_pb';
import Convolution from '../../src/functions/convolution';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function convolutionRef(
  x: number[],
  w: number[],
  b: number[],
  xShape: number[],
  wShape: number[],
  stride: number[],
  outShape: number[],
): number[] {
  const [B, C, H, W] = xShape;
  const [wC, , wH, wW] = wShape;
  const [sH, sW] = stride;
  const output = [];

  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < wC; j += 1) {
      for (let l = 0; l < outShape[2]; l += 1) {
        for (let k = 0; k < outShape[3]; k += 1) {
          const batchOffset = i * C * H * W;
          const strideOffset = sH * W * l + sW * k;
          const channelOffset = j * C * wH * wW;
          let value = 0.0;
          for (let c = 0; c < C; c += 1) {
            for (let m = 0; m < wH; m += 1) {
              for (let n = 0; n < wW; n += 1) {
                const xIndex = batchOffset + strideOffset + c * W * H + m * W + n;
                const wIndex = channelOffset + c * wH * wW + m * wW + n;
                value += x[xIndex] * w[wIndex];
              }
            }
          }
          output.push(value);
        }
      }
    }
  }

  let cursor = 0;
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < wC; j += 1) {
      for (let k = 0; k < outShape[2]; k += 1) {
        for (let l = 0; l < outShape[3]; l += 1) {
          output[cursor] += b[j];
          cursor += 1;
        }
      }
    }
  }
  return output;
}

test('test-convolution', () => {
  const x = Variable.rand('x', [32, 3, 28, 28]);
  const w = Variable.rand('w', [16, 3, 4, 4]);
  const b = Variable.rand('b', [16]);
  const y = Variable.rand('y', [32, 16, 13, 13]);

  const param = new ConvolutionParameter();
  const pad = new Shape();
  pad.addDim(0);
  pad.addDim(0);
  param.setPad(pad);
  const stride = new Shape();
  stride.addDim(2);
  stride.addDim(2);
  param.setStride(stride);

  const conv = new Convolution(param);

  conv.setup([x, w, b], [y]);
  conv.forward([x, w, b], [y]);

  const yRef = convolutionRef(x.data, w.data, b.data, x.shape, w.shape, [2, 2], y.shape);
  expectAllClose(y.data, yRef, 0.0001);
});
