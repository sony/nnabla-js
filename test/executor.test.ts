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

import * as fs from 'fs';
import { GPU } from 'gpu.js';
import { unzipNNP } from '../src/nnp';
import Network from '../src/network';
import VariableManager from '../src/variableManager';
import { Executor } from '../src/executor';

test('test-executor-from-proto', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    unzipNNP(data).then((nnp) => {
      const gpu = new GPU();
      const variableManager = VariableManager.fromProtoParameters(nnp.parameters);
      const network = Network.fromProtoNetwork(nnp.networks[0], variableManager, gpu);
      const executor = Executor.fromProtoExecutor(nnp.executors[0], network);

      const inputs: { [key: string]: number[] } = {};
      for (const inputName of executor.inputNames) {
        const variable = network.getVariable(inputName);
        inputs[inputName] = [...Array(variable.size())].map(() => Math.random() * 2.0 - 1.0);
      }

      const output = executor.forward(inputs);

      for (const outputName of executor.outputNames) {
        const variable = network.getVariable(outputName);
        expect(output[outputName].length).toBe(variable.size());
      }

      done();
    });
  });
});
