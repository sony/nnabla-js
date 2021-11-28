import { Function, AffineParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { getOrThrow } from '../utils';
import Affine from './affine';
import ReLU from './relu';

export default function buildFunctionImpl(func: Function): FunctionImpl {
  const functionType = func.getType();
  switch (functionType) {
    case 'Affine':
      return new Affine(getOrThrow<AffineParameter>(func.getAffineParam()));
    case 'ReLU':
      return new ReLU();
    default:
      throw Error(`${functionType} is not supported yet.`);
  }
}
