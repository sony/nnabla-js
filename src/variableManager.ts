import { IParameter } from './nnabla_pb';
import Variable from './variable';

export default class VariableManager {
  variables: { [key: string]: Variable };

  constructor(variables: { [key: string]: Variable }) {
    this.variables = variables;
  }

  static fromProtoParameters(parameters: IParameter[]): VariableManager {
    const variables: { [key: string]: Variable } = {};
    for (const parameter of parameters) {
      const variable = Variable.fromProtoParameter(parameter);
      variables[variable.name] = variable;
    }
    return new VariableManager(variables);
  }

  hasVariable(name: string): boolean {
    return Object.prototype.hasOwnProperty.call(this.variables, name);
  }

  getVariable(name: string): Variable {
    if (this.hasVariable(name)) {
      return this.variables[name];
    }
    throw Error(`${name} does not exist.`);
  }

  registerVariable(variable: Variable): void {
    if (this.hasVariable(variable.name)) {
      throw Error(`${variable.name} already exists.`);
    } else {
      this.variables[variable.name] = variable;
    }
  }
}
