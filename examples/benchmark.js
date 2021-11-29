const fs = require('fs');
const NNP = require('../dist/index.js');

if (process.argv.length !== 4) {
  throw Error('Usage: node examples/benchmark.js <NNP file> <executor name>');
}

const nnpFile = process.argv[2];
const executorName = process.argv[3];

fs.readFile(nnpFile, (_, data) => {
  NNP.fromNNPData(data).then((nnp) => {
    const executor = nnp.executors[executorName];
    let totalTime = 0.0;
    const inputs = {};
    for (let inputName of executor.inputNames) {
      const variable = executor.network.getVariable(inputName);
      inputs[inputName] = [...Array(variable.size())].map(() => Math.random() * 2.0 - 1.0);
    }
    for (let i = 0; i < 10; ++i) {
      const start = process.hrtime();
      nnp.forward(executorName, inputs);
      const end = process.hrtime(start);
      totalTime += end[0] + end[1] / 1000000000;
    }

    console.log('Average execution time: %ds', totalTime / 10.0);
  });
});
