import {
  Function,
  AddScalarParameter,
  AffineParameter,
  MulScalarParameter,
  PowScalarParameter,
  ReshapeParameter,
} from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { getOrThrow } from '../utils';
import Add2 from './add2';
import AddScalar from './addScalar';
import Affine from './affine';
import Div2 from './div2';
import Mul2 from './mul2';
import MulScalar from './mulScalar';
import Pow2 from './pow2';
import PowScalar from './powScalar';
import ReLU from './relu';
import Reshape from './reshape';
import Sub2 from './sub2';

export default function buildFunctionImpl(func: Function): FunctionImpl {
  const functionType = func.getType();
  switch (functionType) {
    case 'Add2':
      return new Add2();
    case 'AddScalar':
      return new AddScalar(getOrThrow<AddScalarParameter>(func.getAddScalarParam()));
    case 'Affine':
      return new Affine(getOrThrow<AffineParameter>(func.getAffineParam()));
    case 'Div2':
      return new Div2();
    case 'Mul2':
      return new Mul2();
    case 'MulScalar':
      return new MulScalar(getOrThrow<MulScalarParameter>(func.getMulScalarParam()));
    case 'Pow2':
      return new Pow2();
    case 'PowScalar':
      return new PowScalar(getOrThrow<PowScalarParameter>(func.getPowScalarParam()));
    case 'ReLU':
      return new ReLU();
    case 'Reshape':
      return new Reshape(getOrThrow<ReshapeParameter>(func.getReshapeParam()));
    case 'Sub2':
      return new Sub2();
    default:
      throw Error(`${functionType} is not supported yet.`);
  }
}
