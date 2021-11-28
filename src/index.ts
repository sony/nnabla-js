import Network from './network';
import Executor from './executor';
import { unzipNNP } from './nnp';
import VariableManager from './variableManager';
import { getOrThrow } from './utils';

export default class NNP {
  executors: { [key: string]: Executor };

  variableManager: VariableManager;

  constructor(executors: { [key: string]: Executor }, variableManager: VariableManager) {
    this.executors = executors;
    this.variableManager = variableManager;
  }

  static fromNNPData(data: Uint8Array): Promise<NNP> {
    return unzipNNP(data).then((nnp) => {
      const variableManager = VariableManager.fromProtoParameters(nnp.parameters);

      const networks: { [key: string]: Network } = {};
      for (const protoNetwork of nnp.networks) {
        const network = Network.fromProtoNetwork(protoNetwork, variableManager);
        networks[network.name] = network;
      }

      const executors: { [key: string]: Executor } = {};
      for (const protoExecutor of nnp.executors) {
        const networkName = getOrThrow<string>(protoExecutor.getNetworkName());
        const network = networks[networkName];
        const executor = Executor.fromProtoExecutor(protoExecutor, network);
        executors[executor.name] = executor;
      }

      return new NNP(executors, variableManager);
    });
  }

  forward(executorName: string, data: { [key: string]: number[] }): { [key: string]: number[] } {
    return this.executors[executorName].forward(data);
  }
}
