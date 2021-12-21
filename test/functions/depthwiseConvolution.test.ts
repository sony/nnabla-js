import { GPU } from 'gpu.js';
import { DepthwiseConvolutionParameter, Shape } from '../../src/proto/nnabla_pb';
import DepthwiseConvolution from '../../src/functions/depthwiseConvolution';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function depthwiseConvolutionRef(
  x: number[],
  w: number[],
  xShape: number[],
  wShape: number[],
  stride: number[],
  outShape: number[],
): number[] {
  const [B, C, H, W] = xShape;
  const [, wH, wW] = wShape;
  const [sH, sW] = stride;
  const output = [];

  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let l = 0; l < outShape[2]; l += 1) {
        for (let k = 0; k < outShape[3]; k += 1) {
          const batchOffset = i * C * H * W;
          const strideOffset = sH * W * l + sW * k;
          const channelOffset = j * H * W;
          let value = 0.0;
          for (let m = 0; m < wH; m += 1) {
            for (let n = 0; n < wW; n += 1) {
              const xIndex = batchOffset + channelOffset + strideOffset + m * W + n;
              const wIndex = j * wH * wW + m * wW + n;
              value += x[xIndex] * w[wIndex];
            }
          }
          output.push(value);
        }
      }
    }
  }
  return output;
}

test('test-depthwise-convolution', () => {
  const x = Variable.rand('x', [32, 3, 28, 28]);
  const w = Variable.rand('w', [3, 4, 4]);
  const y = Variable.rand('y', [32, 3, 13, 13]);

  const param = new DepthwiseConvolutionParameter();
  const pad = new Shape();
  pad.addDim(0);
  pad.addDim(0);
  param.setPad(pad);
  const stride = new Shape();
  stride.addDim(2);
  stride.addDim(2);
  param.setStride(stride);
  param.setMultiplier(1);

  const conv = new DepthwiseConvolution(param, new GPU());

  conv.setup([x, w], [y]);
  conv.forward([x, w], [y]);
  const yData = y.toArray();

  const yRef = depthwiseConvolutionRef(x.toArray(), w.toArray(), x.shape, w.shape, [2, 2], y.shape);
  expectAllClose(yData, yRef, 0.0001);
});
