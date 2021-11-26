import { IFunction, INetwork, IVariable } from './nnabla_pb';
import Function from './function';
import Variable from './variable';
import VariableManager from './variableManager';
import { getOrThrow, getAsArrayOrThrow } from './utils';

export default class Network {
  name: string;

  variables: {[key: string]: Variable};

  functions: {[key: string]: Function};

  constructor(
    name: string,
    variables: {[key: string]: Variable},
    functions: {[key: string]: Function},
  ) {
    this.name = name;
    this.variables = variables;
    this.functions = functions;
  }

  static fromProtoNetwork(network: INetwork, variableManager: VariableManager): Network {
    const name = getOrThrow<string>(network.name);

    const variables = getAsArrayOrThrow<IVariable>(network.variable);
    const variableMapping: {[key: string]: Variable} = {};
    for (const variable of variables) {
      const variableName = getOrThrow<string>(variable.name);
      const variableType = getOrThrow<string>(variable.type);
      if (variableType === 'Buffer' && !variableManager.hasVariable(variableName)) {
        variableManager.registerVariable(Variable.fromProtoVariable(variable));
      } else if (variableType === 'Parameter' && !variableManager.hasVariable(variableName)) {
        throw Error(`${variableName} should exist in VariableManager.`);
      }
      variableMapping[variableName] = variableManager.getVariable(variableName);
    }

    const functions = getAsArrayOrThrow<IFunction>(network.function);
    const functionMapping: {[key: string]: Function} = {};
    for (const func of functions) {
      const functionName = getOrThrow<string>(func.name);
      functionMapping[functionName] = Function.fromProtoFunction(func, variableManager);
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
