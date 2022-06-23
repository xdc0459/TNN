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
#include <limits>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/interpreter/raw_buffer.h"
namespace TNN_NS {

float RANGE_BOUND = 0.6f;

bool NeedPerChannelQuantize(RawBuffer& raw_buffer, const int channel_size) {
    const int data_count = raw_buffer.GetDataCount();
    auto data            = raw_buffer.force_to<float*>();
    assert(data_count % channel_size == 0);
    std::vector<float> min_values(channel_size, std::numeric_limits<float>::max());
    std::vector<float> max_values(channel_size, std::numeric_limits<float>::min());
    int stride = data_count / channel_size;
    for (int i = 0; i < channel_size; ++i) {
        for (int j = 0; j < stride; ++j) {
            float value   = data[i * stride + j];
            min_values[i] = std::min(min_values[i], value);
        }
    }
    float sum_range = 0.0f;
    for (int i = 0; i < channel_size; ++i) {
        sum_range += max_values[i] - min_values[i];
    }
    float average_range = sum_range / channel_size;
    if (average_range > RANGE_BOUND) {
        LOGE("The range of weights overflowed.\n");
        return false;
    }
    return true;
}

bool NeedPerTensorQuantize(RawBuffer& raw_buffer) {
    const int data_count = raw_buffer.GetDataCount();
    auto data            = raw_buffer.force_to<float*>();
    float min            = std::numeric_limits<float>::max();
    float max            = std::numeric_limits<float>::min();
    for (int i = 0; i < data_count; ++i) {
        min = std::min(min, data[i]);
        max = std::max(max, data[i]);
    }
    float sum_range     = max - min;
    float average_range = sum_range / 1;
    if (average_range > RANGE_BOUND) {
        LOGE("The range of weights overflowed.\n");
        return false;
    }
    return true;
}
}  // namespace TNN_NS
