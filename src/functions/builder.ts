import { GPU } from 'gpu.js';
import {
  Function,
  AddScalarParameter,
  AffineParameter,
  AveragePoolingParameter,
  BatchNormalizationParameter,
  ConvolutionParameter,
  DeconvolutionParameter,
  DepthwiseConvolutionParameter,
  ELUParameter,
  MaxPoolingParameter,
  MulScalarParameter,
  PowScalarParameter,
  RandnParameter,
  ReshapeParameter,
  SliceParameter,
  SoftmaxParameter,
  TransposeParameter,
} from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { getOrThrow } from '../utils';
import Add2 from './add2';
import AddScalar from './addScalar';
import Affine from './affine';
import AveragePooling from './averagePooling';
import BatchNormalization from './batchNormalization';
import Convolution from './convolution';
import Deconvolution from './deconvolution';
import DepthwiseConvolution from './depthwiseConvolution';
import Div2 from './div2';
import ELU from './elu';
import MaxPooling from './maxPooling';
import Mul2 from './mul2';
import MulScalar from './mulScalar';
import Pow2 from './pow2';
import PowScalar from './powScalar';
import Randn from './randn';
import ReLU from './relu';
import Reshape from './reshape';
import Slice from './slice';
import Sigmoid from './sigmoid';
import Softmax from './softmax';
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
    case 'MaxPooling':
      return new MaxPooling(getOrThrow<MaxPoolingParameter>(func.getMaxPoolingParam()), gpu);
    case 'Mul2':
      return new Mul2(gpu);
    case 'MulScalar':
      return new MulScalar(getOrThrow<MulScalarParameter>(func.getMulScalarParam()), gpu);
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
