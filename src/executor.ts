import { IExecutor, IDataVariable, IOutputVariable } from './nnabla_pb';
import Function from './function';
import Network from './network';
import Variable from './variable';
import { getOrThrow, getAsArrayOrThrow } from './utils';

function forwardRecursively(leaf: Variable, visited: Function[]): void {
  if (leaf.outputFrom === undefined) {
    return;
  }

  const func = leaf.outputFrom as Function;

  if (visited.includes(func)) {
    return;
  }

  for (const variable of func.inputs) {
    forwardRecursively(variable, visited);
  }

  leaf.outputFrom.forward();
  visited.push(func);
}

export default class Executor {
  name: string;

  network: Network;

  inputNames: string[];

  outputNames: string[];

  constructor(name: string, network: Network, inputNames: string[], outputNames: string[]) {
    this.name = name;
    this.network = network;
    this.inputNames = inputNames;
    this.outputNames = outputNames;
  }

  static fromProtoExecutor(executor: IExecutor, network: Network): Executor {
    const name = getOrThrow<string>(executor.name);

    const inputNames = getAsArrayOrThrow<IDataVariable>(executor.dataVariable)
      .map((v) => getOrThrow(v.variableName));
    const outputNames = getAsArrayOrThrow<IOutputVariable>(executor.outputVariable)
      .map((v) => getOrThrow(v.variableName));

    return new Executor(name, network, inputNames, outputNames);
  }

  forward(inputs: {[key: string]: number[]}): {[key: string]: number[]} {
    // Set input data
    for (const inputKey of Object.keys(inputs)) {
      const variable = this.network.getVariable(inputKey);
      variable.setData(inputs[inputKey]);
    }

    // Perform forward propagation
    const outputVariables = this.outputNames.map((name) => this.network.getVariable(name));
    const visited: Function[] = [];
    for (const variable of outputVariables) {
      forwardRecursively(variable, visited);
    }

    // Get output data
    const output: {[key: string]: number[]} = {};
    for (const outputName of this.outputNames) {
      output[outputName] = Array.from(this.network.getVariable(outputName).data);
    }

    return output;
  }
}
