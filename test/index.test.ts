import * as fs from 'fs';
import NNP from '../src/index';

test('test-nnp', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    NNP.fromNNPData(data).then((nnp) => {
      const executorName = Object.keys(nnp.executors)[0];
      const executor = nnp.executors[executorName];
      const { network } = executor;

      const inputs: { [key: string]: number[] } = {};
      for (const inputName of executor.inputNames) {
        const variable = network.getVariable(inputName);
        inputs[inputName] = [...Array(variable.size())].map(() => Math.random() * 2.0 - 1.0);
      }

      const output = nnp.forward(executorName, inputs);

      for (const outputName of executor.outputNames) {
        const variable = network.getVariable(outputName);
        expect(output[outputName].length).toBe(variable.size());
      }

      done();
    });
  });
});
