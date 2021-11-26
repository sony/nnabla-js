import { IFunction } from '../nnabla_pb';
import FunctionImpl from './base';
import { getOrThrow } from '../utils';
import ReLU from './relu';

export default function buildFunctionImpl(func: IFunction): FunctionImpl {
  const functionType = getOrThrow<string>(func.type);
  switch (functionType) {
    case 'ReLU':
      return new ReLU();
    default:
      throw Error(`${functionType} is not supported yet.`);
  }
}
