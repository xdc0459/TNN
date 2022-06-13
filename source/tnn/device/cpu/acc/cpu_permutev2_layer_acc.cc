// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/device/cpu/acc/cpu_permutev2_layer_acc.h"

#include "tnn/utils/dims_utils.h"
#include "tnn/utils/naive_compute.h"

namespace TNN_NS {

CpuPermuteV2LayerAcc::~CpuPermuteV2LayerAcc(){};

Status CpuPermuteV2LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = CpuLayerAcc::Init(context, param, resource, inputs, outputs);
    if (status != TNN_OK) {
        return status;
    }
    return TNN_OK;
}

Status CpuPermuteV2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuPermuteV2LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<PermuteV2LayerParam *>(param_);
    if (!param) {
        return Status(TNNERR_MODEL_ERR, "Error: PermuteV2LayerParam is empyt");
    }
    Blob *input_blob       = inputs[0];
    Blob *output_blob      = outputs[0];
    DataType data_type     = output_blob->GetBlobDesc().data_type;
    DimsVector input_dims  = input_blob->GetBlobDesc().dims;
    DimsVector output_dims = output_blob->GetBlobDesc().dims;
    const int output_count = DimsVectorUtils::Count(output_dims);

    std::vector<int> input_step;
    std::vector<int> output_step;
    int num_dims = int(input_dims.size());
    ASSERT(input_dims.size() == output_dims.size());
    for (int i = 0; i < input_dims.size(); ++i) {
        input_step.push_back(CpuPermuteV2LayerAcc::count(input_dims, i + 1));
        output_step.push_back(CpuPermuteV2LayerAcc::count(output_dims, i + 1));
    }
    //////////////////////////
    if (inputs[0]->GetBlobDesc().name == "x.38") {
    //if (1) {
        std::cout << "[Cpu PermuteV2 Fwd] Fwd 0, in0.name = " << inputs[0]->GetBlobDesc().name;
        std::cout << ", out.name = " << outputs[0]->GetBlobDesc().name << std::endl; 
        std::cout << "[Cpu PermuteV2 Fwd], input.shape = [";
        for (auto dim : inputs[0]->GetBlobDesc().dims)
            std::cout << dim << ",";
        std::cout << "], out.shape = [";
        for (auto dim : outputs[0]->GetBlobDesc().dims)
            std::cout << dim << ",";
        std::cout << "], param.orders = [ ";
        for (auto dim : param->orders)
            std::cout << dim << ",";
        std::cout << "], input_step = [ ";
        for (auto dim : input_step)
            std::cout << dim << ",";
        std::cout << "], output_step = [ ";
        for (auto dim : output_step)
            std::cout << dim << ",";
        std::cout << "]" << std::endl;
    }
    //////////////////////////

    if (data_type == DATA_TYPE_INT32 || data_type == DATA_TYPE_FLOAT) {
        // 32-bit data types.
        float *input_data  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_data = static_cast<float *>(output_blob->GetHandle().base);
        NaivePermute<float>(output_count, output_dims, input_data, param->orders, input_step, output_step, num_dims, output_data);
        //////////////////////////
        if (inputs[0]->GetBlobDesc().name == "x.38") {
            std::cout << "[Cpu PermuteV2 Fwd] in0.name = " << inputs[0]->GetBlobDesc().name;
            std::cout << ", out.name = " << outputs[0]->GetBlobDesc().name << std::endl; 
            std::cout << "[Cpu PermuteV2 Fwd] input_data.ptr=" << input_data << ", output_data.ptr=" << output_data;
            std::cout << ", in[0] = " << input_data[0] << ", in[1] = " << input_data[1];
            std::cout << ", out[0] = " << output_data[0] << ", out[1] = " << output_data[1] << std::endl;
        }
        //////////////////////////
    } else if (data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_HALF) {
        // 16-bit data types.
        fp16_t *input_data  = static_cast<fp16_t *>(input_blob->GetHandle().base);
        fp16_t *output_data = static_cast<fp16_t *>(output_blob->GetHandle().base);
        NaivePermute<fp16_t>(output_count, output_dims, input_data, param->orders, input_step, output_step, num_dims, output_data);
    } else {
        // 8-bit data types. DATA_TYPE_INT8
        int8_t *input_data  = static_cast<int8_t *>(input_blob->GetHandle().base);
        int8_t *output_data = static_cast<int8_t *>(output_blob->GetHandle().base);
        NaivePermute<int8_t>(output_count, output_dims, input_data, param->orders, input_step, output_step, num_dims, output_data);
    }
    return TNN_OK;
}

CpuTypeLayerAccRegister<TypeLayerAccCreator<CpuPermuteV2LayerAcc>> g_cpu_permutev2_layer_acc_register(LAYER_PERMUTEV2);

}  // namespace TNN_NS
