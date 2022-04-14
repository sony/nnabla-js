// Copyright 2021,2022 Sony Group Corporation.
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

import { GPU } from 'gpu.js';
import {
  Function,
  AddScalarParameter,
  AffineParameter,
  ArangeParameter,
  AveragePoolingParameter,
  BatchNormalizationParameter,
  ConcatenateParameter,
  ConvolutionParameter,
  DeconvolutionParameter,
  DepthwiseConvolutionParameter,
  ELUParameter,
  LeakyReLUParameter,
  MaxPoolingParameter,
  MulScalarParameter,
  NmsDetection2dParameter,
  PowScalarParameter,
  RandnParameter,
  ReshapeParameter,
  SliceParameter,
  SoftmaxParameter,
  SplitParameter,
  TransposeParameter,
} from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { getOrThrow } from '../utils';
import Add2 from './add2';
import AddScalar from './addScalar';
import Affine from './affine';
import Arange from './arange';
import AveragePooling from './averagePooling';
import BatchNormalization from './batchNormalization';
import Concatenate from './concatenate';
import Convolution from './convolution';
import Deconvolution from './deconvolution';
import DepthwiseConvolution from './depthwiseConvolution';
import Div2 from './div2';
import ELU from './elu';
import Exp from './exp';
import LeakyReLU from './leakyRelu';
import MaxPooling from './maxPooling';
import Mul2 from './mul2';
import MulScalar from './mulScalar';
import NmsDetection2d from './nmsDetection2d';
import Pow2 from './pow2';
import PowScalar from './powScalar';
import Randn from './randn';
import ReLU from './relu';
import Reshape from './reshape';
import Slice from './slice';
import Sigmoid from './sigmoid';
import Softmax from './softmax';
import Split from './split';
import Sub2 from './sub2';
import Tanh from './tanh';
import Transpose from './transpose';

export default function buildFunctionImpl(func: Function, gpu: GPU): FunctionImpl {
  const functionType = func.getType();
  switch (functionType) {
    case 'Add2':
      return new Add2(gpu);
    case 'AddScalar':
      return new AddScalar(getOrThrow<AddScalarParameter>(func.getAddScalarParam()), gpu);
    case 'Affine':
      return new Affine(getOrThrow<AffineParameter>(func.getAffineParam()), gpu);
    case 'Arange':
      return new Arange(getOrThrow<ArangeParameter>(func.getArangeParam()), gpu);
    case 'AveragePooling':
      return new AveragePooling(
        getOrThrow<AveragePoolingParameter>(func.getAveragePoolingParam()),
        gpu,
      );
    case 'BatchNormalization':
      return new BatchNormalization(
        getOrThrow<BatchNormalizationParameter>(func.getBatchNormalizationParam()),
        gpu,
      );
    case 'Concatenate':
      return new Concatenate(getOrThrow<ConcatenateParameter>(func.getConcatenateParam()), gpu);
    case 'Convolution':
      return new Convolution(getOrThrow<ConvolutionParameter>(func.getConvolutionParam()), gpu);
    case 'Deconvolution':
      return new Deconvolution(
        getOrThrow<DeconvolutionParameter>(func.getDeconvolutionParam()),
        gpu,
      );
    case 'DepthwiseConvolution':
      return new DepthwiseConvolution(
        getOrThrow<DepthwiseConvolutionParameter>(func.getDepthwiseConvolutionParam()),
        gpu,
      );
    case 'Div2':
      return new Div2(gpu);
    case 'ELU':
      return new ELU(getOrThrow<ELUParameter>(func.getEluParam()), gpu);
    case 'Exp':
      return new Exp(gpu);
    case 'LeakyReLU':
      return new LeakyReLU(getOrThrow<LeakyReLUParameter>(func.getLeakyReluParam()), gpu);
    case 'MaxPooling':
      return new MaxPooling(getOrThrow<MaxPoolingParameter>(func.getMaxPoolingParam()), gpu);
    case 'Mul2':
      return new Mul2(gpu);
    case 'MulScalar':
      return new MulScalar(getOrThrow<MulScalarParameter>(func.getMulScalarParam()), gpu);
    case 'NmsDetection2d':
      return new NmsDetection2d(getOrThrow<NmsDetection2dParameter>(func.getNmsDetection2dParam()));
    case 'Pow2':
      return new Pow2(gpu);
    case 'PowScalar':
      return new PowScalar(getOrThrow<PowScalarParameter>(func.getPowScalarParam()), gpu);
    case 'Randn':
      return new Randn(getOrThrow<RandnParameter>(func.getRandnParam()), gpu);
    case 'ReLU':
      return new ReLU(gpu);
    case 'Reshape':
      return new Reshape(getOrThrow<ReshapeParameter>(func.getReshapeParam()), gpu);
    case 'Slice':
      return new Slice(getOrThrow<SliceParameter>(func.getSliceParam()), gpu);
    case 'Sigmoid':
      return new Sigmoid(gpu);
    case 'Softmax':
      return new Softmax(getOrThrow<SoftmaxParameter>(func.getSoftmaxParam()), gpu);
    case 'Split':
      return new Split(getOrThrow<SplitParameter>(func.getSplitParam()), gpu);
    case 'Sub2':
      return new Sub2(gpu);
    case 'Tanh':
      return new Tanh(gpu);
    case 'Transpose':
      return new Transpose(getOrThrow<TransposeParameter>(func.getTransposeParam()), gpu);
    default:
      throw Error(`${functionType} is not supported yet.`);
  }
}
