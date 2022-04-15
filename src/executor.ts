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

import { Executor as ProtoExecutor } from './proto/nnabla_pb';
import Function from './function';
import Network from './network';
import Variable from './variable';

export interface ForwardConfig {
  verbose?: boolean;
}

function forwardRecursively(leaf: Variable, visited: Function[], verbose: boolean): void {
  if (leaf.outputFrom === undefined) {
    return;
  }

  const func = leaf.outputFrom as Function;

  if (visited.includes(func)) {
    return;
  }

  for (const variable of func.inputs) {
    forwardRecursively(variable, visited, verbose);
  }

  leaf.outputFrom.forward();

  if (verbose) {
    console.log(`Visited ${leaf.name}`);
    console.log(leaf.toArray());
  }

  visited.push(func);
}

export class Executor {
  name: string;

  network: Network;

  inputNames: string[];

  outputNames: string[];

  constructor(name: string, network: Network, inputNames: string[], outputNames: string[]) {
    this.name = name;
    this.network = network;
    this.inputNames = inputNames;
    this.outputNames = outputNames;
  }

  static fromProtoExecutor(executor: ProtoExecutor, network: Network): Executor {
    const name = executor.getName();

    const inputNames = executor.getDataVariableList().map((v) => v.getVariableName());
    const outputNames = executor.getOutputVariableList().map((v) => v.getVariableName());

    return new Executor(name, network, inputNames, outputNames);
  }

  forward(
    inputs: { [key: string]: number[] },
    config?: ForwardConfig,
  ): { [key: string]: number[] } {
    let verbose = false;
    if (config !== undefined && config.verbose) {
      verbose = true;
    }

    // Set input data
    for (const inputKey of Object.keys(inputs)) {
      const variable = this.network.getVariable(inputKey);
      variable.setData(inputs[inputKey]);
    }

    // Perform forward propagation
    const outputVariables = this.outputNames.map((name) => this.network.getVariable(name));
    const visited: Function[] = [];
    for (const variable of outputVariables) {
      forwardRecursively(variable, visited, verbose);
    }

    // Get output data
    const output: { [key: string]: number[] } = {};
    for (const outputName of this.outputNames) {
      output[outputName] = Array.from(this.network.getVariable(outputName).toArray());
    }

    return output;
  }
}
