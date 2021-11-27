import * as fs from 'fs';
import { unzipNNP } from '../src/nnp';
import Network from '../src/network';
import VariableManager from '../src/variableManager';
import Executor from '../src/executor';

test('test-executor-from-proto', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    unzipNNP(data).then((nnp) => {
      const variableManager = VariableManager.fromProtoParameters(nnp.parameters);
      const network = Network.fromProtoNetwork(nnp.networks[0], variableManager);
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
