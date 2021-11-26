import { IFunction } from './nnabla_pb';
import { getOrThrow, getAsArrayOrThrow } from './utils';
import Variable from './variable';
import FunctionImpl from './functions/base';
import buildFunctionImpl from './functions/builder';
import VariableManager from './variableManager';

export default class Function {
  name: string;

  impl: FunctionImpl;

  inputs: Variable[];

  outputs: Variable[];

  constructor(name: string, impl: FunctionImpl, inputs: Variable[], outputs: Variable[]) {
    this.name = name;
    this.impl = impl;
    this.inputs = inputs;
    this.outputs = outputs;
  }

  forward(): void {
    this.impl.forward(this.inputs, this.outputs);
  }

  static fromProtoFunction(protoFunc: IFunction, variableManager: VariableManager): Function {
    const name = getOrThrow<string>(protoFunc.name);
    const inputNames = getAsArrayOrThrow<string>(protoFunc.input);
    const inputVariables = inputNames.map((vname) => variableManager.getVariable(vname));
    const outputNames = getAsArrayOrThrow<string>(protoFunc.output);
    const outputVariables = outputNames.map((vname) => variableManager.getVariable(vname));

    const impl = buildFunctionImpl(protoFunc);
    impl.setup(inputVariables, outputVariables);

    const func = new Function(name, impl, inputVariables, outputVariables);
    outputVariables.forEach((variable) => variable.setParent(func));
    return func;
  }
}
