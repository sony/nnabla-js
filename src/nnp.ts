import { GPU } from 'gpu.js';
import JSZip from 'jszip';
import {
  Executor as ProtoExecutor,
  Network as ProtoNetwork,
  Parameter,
  NNablaProtoBuf,
} from './proto/nnabla_pb';
import decodePbtxt from './pbtxtDecoder';
import VariableManager from './variableManager';
import { getOrThrow } from './utils';
import { Executor, ForwardConfig } from './executor';
import Network from './network';

interface ProtoNNP {
  version: string;
  networks: ProtoNetwork[];
  parameters: Parameter[];
  executors: ProtoExecutor[];
}

/**
 * Unzips .nnp binary data.
 *
 * @param data - The .nnp binary data.
 * @returns The Promise object that returns unzipped structured object.
 *
 */
export function unzipNNP(data: Uint8Array): Promise<ProtoNNP> {
  return new JSZip().loadAsync(data).then(async (zip) => {
    // Extract version number
    let version: string = '';
    const versionPromise = zip
      .file('nnp_version.txt')
      ?.async('string')
      .then((text) => {
        version = text.trim();
      });

    // Extract network
    let networks: ProtoNetwork[] = [];
    let executors: ProtoExecutor[] = [];
    const networkPromise = zip
      .file('network.nntxt')
      ?.async('string')
      .then((text) => {
        const nnp = new NNablaProtoBuf();
        decodePbtxt(text, nnp);
        networks = nnp.getNetworkList();
        executors = nnp.getExecutorList();
      });

    // Extract parameters
    let parameters: Parameter[] = [];
    const paramPromise = zip
      .file('parameter.protobuf')
      ?.async('uint8array')
      .then((byteCode) => {
        const nnp = NNablaProtoBuf.deserializeBinary(byteCode);
        parameters = nnp.getParameterList();
      });

    await versionPromise;
    await networkPromise;
    await paramPromise;

    return {
      version,
      networks,
      parameters,
      executors,
    };
  });
}

export class NNP {
  executors: { [key: string]: Executor };

  variableManager: VariableManager;

  constructor(executors: { [key: string]: Executor }, variableManager: VariableManager) {
    this.executors = executors;
    this.variableManager = variableManager;
  }

  /**
   * Instantiates NNP object from .nnp binary data.
   *
   * @param data - The .nnp binary data.
   * @param gpu - The GPU instance. If not given, the new GPU instance will be created.
   * @returns The NNP object.
   *
   */
  static fromNNPData(data: Uint8Array, gpu: GPU | undefined): Promise<NNP> {
    return unzipNNP(data).then((nnp) => {
      const ctx = gpu === undefined ? new GPU() : gpu;
      const variableManager = VariableManager.fromProtoParameters(nnp.parameters);

      const networks: { [key: string]: Network } = {};
      for (const protoNetwork of nnp.networks) {
        const network = Network.fromProtoNetwork(protoNetwork, variableManager, ctx);
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

  /**
   * Performs forward propagation with the specified executor.
   *
   * @remarks
   * This method will block until the result is retrieved.
   * Please check forwardAsync for the asynchronous execution.
   *
   * @param executorName - The specified executor name.
   * @param data - The mapping of input variable data.
   * @param config - The config object.
   * @returns The mapping of output variable data.
   *
   */
  forward(
    executorName: string,
    data: { [key: string]: number[] },
    config?: ForwardConfig,
  ): { [key: string]: number[] } {
    return this.executors[executorName].forward(data, config);
  }

  /**
   * Asnchronously perform forward propagation with the specified executor.
   *
   * @param executorName - The specified executor name.
   * @param data - The mapping of input variable data.
   * @param config - The config object.
   * @returns The Promise object that returns the mapping of output variable data.
   *
   */
  forwardAsync(
    executorName: string,
    data: { [key: string]: number[] },
    config?: ForwardConfig,
  ): Promise<{ [key: string]: number[] }> {
    return new Promise((resolve, reject) => {
      try {
        const output = this.forward(executorName, data, config);
        resolve(output);
      } catch (error) {
        reject(error);
      }
    });
  }
}
