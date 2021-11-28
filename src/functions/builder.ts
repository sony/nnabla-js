import {
  Function,
  AffineParameter,
  MulScalarParameter,
  ReshapeParameter,
} from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { getOrThrow } from '../utils';
import Affine from './affine';
import MulScalar from './mulScalar';
import ReLU from './relu';
import Reshape from './reshape';

export default function buildFunctionImpl(func: Function): FunctionImpl {
  const functionType = func.getType();
  switch (functionType) {
    case 'Affine':
      return new Affine(getOrThrow<AffineParameter>(func.getAffineParam()));
    case 'MulScalar':
      return new MulScalar(getOrThrow<MulScalarParameter>(func.getMulScalarParam()));
    case 'ReLU':
      return new ReLU();
    case 'Reshape':
      return new Reshape(getOrThrow<ReshapeParameter>(func.getReshapeParam()));
    default:
      throw Error(`${functionType} is not supported yet.`);
  }
}
