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

const fs = require('fs');
const nnabla = require('../dist/index.js');

if (process.argv.length !== 4) {
  throw Error('Usage: node examples/benchmark.js <NNP file> <executor name>');
}

const nnpFile = process.argv[2];
const executorName = process.argv[3];

fs.readFile(nnpFile, (_, data) => {
  nnabla.NNP.fromNNPData(data).then((nnp) => {
    const executor = nnp.executors[executorName];
    let totalTime = 0.0;
    const inputs = {};
    for (let inputName of executor.inputNames) {
      const variable = executor.network.getVariable(inputName);
      inputs[inputName] = [...Array(variable.size())].map(() => Math.random() * 2.0 - 1.0);
    }
    for (let i = 0; i < 100; ++i) {
      const start = process.hrtime();
      nnp.forward(executorName, inputs);
      const end = process.hrtime(start);
      if (i > 0) {
        totalTime += end[0] + end[1] / 1000000000;
      }
    }

    console.log('Average execution time: %dms', totalTime / 99.0 * 1000.0);
  })
  .catch((err) => console.log(err));
});
