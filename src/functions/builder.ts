import { IFunction, IAffineParameter } from '../nnabla_pb';
import FunctionImpl from './base';
import { getOrThrow } from '../utils';
import Affine from './affine';
import ReLU from './relu';

export default function buildFunctionImpl(func: IFunction): FunctionImpl {
  const functionType = getOrThrow<string>(func.type);
  switch (functionType) {
    case 'Affine':
      return new Affine(getOrThrow<IAffineParameter>(func.affineParam));
    case 'ReLU':
      return new ReLU();
    default:
      throw Error(`${functionType} is not supported yet.`);
  }
}
