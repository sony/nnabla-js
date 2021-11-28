import JSZip from 'jszip';
import { IExecutor, INetwork, IParameter, NNablaProtoBuf } from './nnabla_pb';
import decodePbtxt from './pbtxtDecoder';

export interface ProtoNNP {
  version: string;
  networks: INetwork[];
  parameters: IParameter[];
  executors: IExecutor[];
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
    let networks: INetwork[] = [];
    let executors: IExecutor[] = [];
    const networkPromise = zip
      .file('network.nntxt')
      ?.async('string')
      .then((text) => {
        const decodedObj = decodePbtxt(text);
        const nnp = NNablaProtoBuf.create(decodedObj);

        if (Array.isArray(nnp.network)) {
          networks = nnp.network;
        } else {
          networks = [nnp.network];
        }

        if (Array.isArray(nnp.executor)) {
          executors = nnp.executor;
        } else {
          executors = [nnp.executor];
        }
      });

    // Extract parameters
    let parameters: IParameter[] = [];
    const paramPromise = zip
      .file('parameter.protobuf')
      ?.async('uint8array')
      .then((byteCode) => {
        const nnp = NNablaProtoBuf.decode(byteCode);
        parameters = nnp.parameter;
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
