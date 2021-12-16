import * as fs from 'fs';
import { GPU } from 'gpu.js';
import { unzipNNP, NNP } from '../src/nnp';

test('test-unzipNNP', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    unzipNNP(data).then((nnp) => {
      expect(nnp.version).toBe('0.1');
      expect(nnp.networks.length).toBe(1);
      expect(nnp.executors.length).toBe(1);
      done();
    });
  });
});

test('test-nnp', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    const gpu = new GPU();
    NNP.fromNNPData(data, gpu).then((nnp) => {
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

test('test-nnp-forwardAsync', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    const gpu = new GPU();
    NNP.fromNNPData(data, gpu).then((nnp) => {
      const executorName = Object.keys(nnp.executors)[0];
      const executor = nnp.executors[executorName];
      const { network } = executor;

      const inputs: { [key: string]: number[] } = {};
      for (const inputName of executor.inputNames) {
        const variable = network.getVariable(inputName);
        inputs[inputName] = [...Array(variable.size())].map(() => Math.random() * 2.0 - 1.0);
      }

      nnp.forwardAsync(executorName, inputs).then((output) => {
        for (const outputName of executor.outputNames) {
          const variable = network.getVariable(outputName);
          expect(output[outputName].length).toBe(variable.size());
        }
        done();
      });
    });
  });
});
