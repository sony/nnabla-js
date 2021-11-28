import { Function as ProtoFunction } from './proto/nnabla_pb';
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

  static fromProtoFunction(protoFunc: ProtoFunction, variableManager: VariableManager): Function {
    const name = protoFunc.getName();
    const inputNames = protoFunc.getInputList();
    const inputVariables = inputNames.map((vname) => variableManager.getVariable(vname));
    const outputNames = protoFunc.getOutputList();
    const outputVariables = outputNames.map((vname) => variableManager.getVariable(vname));

    const impl = buildFunctionImpl(protoFunc);
    impl.setup(inputVariables, outputVariables);

    const func = new Function(name, impl, inputVariables, outputVariables);
    outputVariables.forEach((variable) => variable.setParent(func));
    return func;
  }
}
