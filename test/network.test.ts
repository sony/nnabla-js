import * as fs from 'fs';
import { GPU } from 'gpu.js';
import { unzipNNP } from '../src/nnp';
import Network from '../src/network';
import VariableManager from '../src/variableManager';

test('test-network-from-proto', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    const gpu = new GPU();
    unzipNNP(data).then((nnp) => {
      const variableManager = VariableManager.fromProtoParameters(nnp.parameters);
      const network = Network.fromProtoNetwork(nnp.networks[0], variableManager, gpu);
      expect(Object.values(network.variables).length).toBeGreaterThan(1);
      expect(Object.values(network.functions).length).toBeGreaterThan(1);
      done();
    });
  });
});
