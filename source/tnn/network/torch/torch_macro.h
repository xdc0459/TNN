// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the 
// specific language governing permissions and limitations under the License.

#ifndef TNN_SOURCE_TNN_NETWORK_TORCH_TORCH_MACRO_H_
#define TNN_SOURCE_TNN_NETWORK_TORCH_TORCH_MACRO_H_

#include "tnn/core/status.h"

namespace TNN_NS {

#define TORCH_THROW_ERROR(msg)        \
    std::stringstream ss{};           \
    ss << msg;                        \
    throw std::runtime_error(ss.str());

#define TORCH_CHECK_THROW_ERROR(status, ...)                               \
  if (status != TNN_OK){                                                   \
    TORCH_THROW_ERROR("Check " << #status << " Failed \n" << __VA_ARGS__); \
  }

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TORCH_TORCH_MACRO_H_
