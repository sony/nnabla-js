import JSZip from 'jszip';
import { Executor, Network, Parameter, NNablaProtoBuf } from './proto/nnabla_pb';
import decodePbtxt from './pbtxtDecoder';

export interface ProtoNNP {
  version: string;
  networks: Network[];
  parameters: Parameter[];
  executors: Executor[];
}

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
    let networks: Network[] = [];
    let executors: Executor[] = [];
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
