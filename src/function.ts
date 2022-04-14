// Copyright 2021 Sony Group Corporation.
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

  static fromProtoFunction(
    protoFunc: ProtoFunction,
    variableManager: VariableManager,
    gpu: GPU,
  ): Function {
    const name = protoFunc.getName();
    const inputNames = protoFunc.getInputList();
    const inputVariables = inputNames.map((vname) => variableManager.getVariable(vname));
    const outputNames = protoFunc.getOutputList();
    const outputVariables = outputNames.map((vname) => variableManager.getVariable(vname));

    const impl = buildFunctionImpl(protoFunc, gpu);
    impl.setup(inputVariables, outputVariables);

    const func = new Function(name, impl, inputVariables, outputVariables);
    outputVariables.forEach((variable) => variable.setParent(func));
    return func;
  }
}
