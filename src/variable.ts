// Copyright 2021,2022 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { Texture, GPU } from 'gpu.js';
import { Parameter, Variable as ProtoVariable } from './proto/nnabla_pb';
import { getAsArrayOrThrow } from './utils';

interface IFunction {
  name: string;
  forward(): void;
}

function checkTexture(value: number[] | Texture): boolean {
  return Object.prototype.hasOwnProperty.call(value, 'texture');
}

export default class Variable {
  name: string;

  shape: number[];

  data: number[] | Texture;

  outputFrom: IFunction | undefined;

  constructor(name: string, shape: number[], data: number[]) {
    this.name = name;
    this.shape = shape;
    this.data = data;
    this.outputFrom = undefined;
  }

  static fromProtoParameter(parameter: Parameter): Variable {
    const name = parameter.getVariableName();
    const shape = getAsArrayOrThrow<number>(parameter.getShape()?.getDimList());
    const data = parameter.getDataList();
    return new Variable(name, shape, data);
  }

  static fromProtoVariable(variable: ProtoVariable): Variable {
    const name = variable.getName();

    // -1 represents batch dimension
    const shape = getAsArrayOrThrow<number>(variable.getShape()?.getDimList()).map((dim) =>
      dim === -1 ? 1 : dim,
    );

    let size: number = 1;
    for (const dim of shape) {
      size *= dim as number;
    }
    const data = [...Array(size)].map(() => 0.0);

    return new Variable(name, shape, data);
  }

  static rand(name: string, shape: number[]): Variable {
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

  setData(data: number[] | Texture): void {
    if (!checkTexture(data)) {
      const tData = data as number[];
      if (tData.length !== this.size()) {
        throw Error(`the data size does not match: execpted=${this.size()} actual=${tData.length}`);
      }
      this.data = Array.from(tData);
    } else {
      this.data = data;
    }
  }

  toArray(): number[] {
    if (checkTexture(this.data)) {
      return (this.data as Texture).toArray() as number[];
    }
    return this.data as number[];
  }

  isTexture(): boolean {
    return checkTexture(this.data);
  }

  cache(gpu: GPU): void {
    const kernel = gpu
      .createKernel(function (x: number[]): number {
        return x[this.thread.x];
      })
      .setOutput([this.size()])
      .setPipeline(true);
    this.data = kernel(this.data) as Texture;
  }
}
