// Copyright 2021,2022 Sony Group Corporation.
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
import { Network as ProtoNetwork } from './proto/nnabla_pb';
import Function from './function';
import Variable from './variable';
import VariableManager from './variableManager';

export default class Network {
  name: string;

  variables: { [key: string]: Variable };

  functions: { [key: string]: Function };

  constructor(
    name: string,
    variables: { [key: string]: Variable },
    functions: { [key: string]: Function },
  ) {
    this.name = name;
    this.variables = variables;
    this.functions = functions;
  }

  static fromProtoNetwork(
    network: ProtoNetwork,
    variableManager: VariableManager,
    gpu: GPU,
  ): Network {
    const name = network.getName();

    const variables = network.getVariableList();
    const variableMapping: { [key: string]: Variable } = {};
    for (const variable of variables) {
      const variableName = variable.getName();
      const variableType = variable.getType();
      if (variableType === 'Buffer' && !variableManager.hasVariable(variableName)) {
        variableManager.registerVariable(Variable.fromProtoVariable(variable));
      } else if (variableType === 'Parameter' && !variableManager.hasVariable(variableName)) {
        throw Error(`${variableName} should exist in VariableManager.`);
      }
      variableMapping[variableName] = variableManager.getVariable(variableName);
    }

    const functions = network.getFunctionList();
    const functionMapping: { [key: string]: Function } = {};
    for (const func of functions) {
      const functionName = func.getName();
      functionMapping[functionName] = Function.fromProtoFunction(func, variableManager, gpu);
    }

    return new Network(name, variableMapping, functionMapping);
  }

  getVariable(name: string): Variable {
    if (Object.prototype.hasOwnProperty.call(this.variables, name)) {
      return this.variables[name];
    }
    throw Error(`Variable ${name} does not exist in ${this.name}.`);
  }

  getFunction(name: string): Function {
    if (Object.prototype.hasOwnProperty.call(this.functions, name)) {
      return this.functions[name];
    }
    throw Error(`Function ${name} does not exist in ${this.name}`);
  }
}
