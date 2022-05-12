#!/bin/bash -eux
# Copyright 2022 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# download generated proto file
mkdir -p proto
wget https://nnabla.org/cpplib/1.28.0/nnabla.proto -O proto/nnabla.proto

PROTOC_GEN_TS_PATH="node_modules/.bin/protoc-gen-ts"
OUT_DIR="src"

# generate javascript file
protoc \
  --plugin="protoc-gen-ts=${PROTOC_GEN_TS_PATH}" \
  --js_out="import_style=commonjs,binary:${OUT_DIR}" \
  --ts_out="${OUT_DIR}" \
  proto/nnabla.proto
