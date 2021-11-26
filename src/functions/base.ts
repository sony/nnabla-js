/* eslint-disable */

import Variable from '../variable';

export default interface FunctionImpl {
  setup(inputs: Variable[], outputs: Variable[]): void;
  forward(inputs: Variable[], outputs: Variable[]): void;
}
