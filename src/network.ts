import { GPU } from 'gpu.js';
import { Network as ProtoNetwork } from './proto/nnabla_pb';
import Function from './function';
import Variable from './variable';
import VariableManager from './variableManager';

export default class Network {
  name: string;

  variables: { [key: string]: Variable };

  functions: { [key: string]: Function };

  constructor(
    name: string,
    variables: { [key: string]: Variable },
    functions: { [key: string]: Function },
  ) {
    this.name = name;
    this.variables = variables;
    this.functions = functions;
  }

  static fromProtoNetwork(
    network: ProtoNetwork,
    variableManager: VariableManager,
    gpu: GPU,
  ): Network {
    const name = network.getName();

    const variables = network.getVariableList();
    const variableMapping: { [key: string]: Variable } = {};
    for (const variable of variables) {
      const variableName = variable.getName();
      const variableType = variable.getType();
      if (variableType === 'Buffer' && !variableManager.hasVariable(variableName)) {
        variableManager.registerVariable(Variable.fromProtoVariable(variable));
      } else if (variableType === 'Parameter' && !variableManager.hasVariable(variableName)) {
        throw Error(`${variableName} should exist in VariableManager.`);
      }
      variableMapping[variableName] = variableManager.getVariable(variableName);
    }

    const functions = network.getFunctionList();
    const functionMapping: { [key: string]: Function } = {};
    for (const func of functions) {
      const functionName = func.getName();
      functionMapping[functionName] = Function.fromProtoFunction(func, variableManager, gpu);
    }

    return new Network(name, variableMapping, functionMapping);
  }

  getVariable(name: string): Variable {
    if (Object.prototype.hasOwnProperty.call(this.variables, name)) {
      return this.variables[name];
    }
    throw Error(`Variable ${name} does not exist in ${this.name}.`);
  }

  getFunction(name: string): Function {
    if (Object.prototype.hasOwnProperty.call(this.functions, name)) {
      return this.functions[name];
    }
    throw Error(`Function ${name} does not exist in ${this.name}`);
  }
}
