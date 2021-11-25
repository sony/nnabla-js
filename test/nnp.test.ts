import * as fs from 'fs';
import { unzipNNP } from '../src/nnp';

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
