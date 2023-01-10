// Copyright 2022 Sony Group Corporation.
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

importScripts("../dist/index.js");

const gpu = new GPU();
const resizeKernel = nnabla.ImageUtils.createAsyncResizeKernel([3, 224, 224], gpu, true);
let nnp = null;

function loadNNP(data) {
  return nnabla.NNP.fromNNPData(data, gpu).then(function(_nnp) {
    nnp = _nnp;
  });
}

self.addEventListener("message", (e) => {
  if (e.data.type === "load") {
    loadNNP(e.data.data).then(() => {
      self.postMessage({type: "loadCompleted"});
    })
  } else if (e.data.type === "execute") {
    const image = e.data.data;
    const array = nnabla.ImageUtils.convertImageToArray(image, 3, 1.0);
    resizeKernel(array, image.height, image.width).then((resizedArray) => {
      nnp.forwardAsync("runtime", {x0: resizedArray}).then((output) => {
        const resizedData = nnabla.ImageUtils.convertArrayToImage(resizedArray.toArray(), 3, 224, 224, 1.0);
        self.postMessage({
          type: "executeCompleted",
          data: output,
          resizedImage: resizedData,
        });
      })
    })
  }
});
