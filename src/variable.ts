import { IParameter, IVariable } from './nnabla_pb';
import { getOrThrow, getAsArrayOrThrow } from './utils';

interface IFunction {
  name: string;
  forward(): void;
}

export default class Variable {
  name: string;

  shape: number[];

  data: number[];

  outputFrom: IFunction|undefined;

  constructor(name: string, shape: number[], data: number[]) {
    this.name = name;
    this.shape = shape;
    this.data = data;
    this.outputFrom = undefined;
  }

  static fromProtoParameter(parameter: IParameter): Variable {
    const name = getOrThrow<string>(parameter.variableName);
    const shape = getAsArrayOrThrow<number>(parameter.shape?.dim as number[]);
    const data = getAsArrayOrThrow<number>(parameter.data);
    return new Variable(name, shape, data);
  }

  static fromProtoVariable(variable: IVariable): Variable {
    const name = getOrThrow<string>(variable.name);

    // -1 represents batch dimension
    const shape = getAsArrayOrThrow<number>(variable.shape?.dim as number[])
      .map((dim) => (dim === -1 ? 1 : dim));

    let size: number = 1;
    for (const dim of shape) {
      size *= dim as number;
    }
    const data = [...Array(size)].map(() => 0.0);

    return new Variable(name, shape, data);
  }

  static rand(name: string, shape: (number)[]): Variable {
    let size: number = 1;
    for (const dim of shape) {
      size *= dim as number;
    }
    const data = [...Array(size)].map(() => Math.random() * 2.0 - 1.0);
    return new Variable(name, shape, data);
  }

  setParent(func: IFunction): void {
    if (this.outputFrom !== undefined) {
      const parentName = this.outputFrom.name;
      throw Error(`${parentName} is already registered as parent.`);
    }
    this.outputFrom = func;
  }

  size(): number {
    let size: number = 1;
    for (const dim of this.shape) {
      size *= dim as number;
    }
    return size;
  }

  setData(data: number[]): void {
    if (data.length !== this.size()) {
      throw Error(`the data size does not match: execpted=${this.size()} actual=${data.length}`);
    }
    this.data = Array.from(data);
  }
}