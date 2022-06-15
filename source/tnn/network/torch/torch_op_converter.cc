#include "tnn/network/torch/torch_op_converter.h"
// #include <ATen/native/quantized/cpu/conv_packed_params.h>

namespace TNN_NS {

// the function schema is defined in aten/src/ATen/native/native_functions.ymal
// Todo: tnn tensorrt plugin not fully support fp16, resource rawbuffer should be convert to fp32 to avoid init error

namespace conversion
{
std::map<std::string, std::shared_ptr<TorchOpConverter>>& GetGlobalTorchConvertMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<std::string, std::shared_ptr<TorchOpConverter>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<std::string, std::shared_ptr<TorchOpConverter>>); });
    return *creators;
}

#define ADD_INPUTS_AND_OUTPUTS                                                                                         \
    for (auto input : layer_info->inputs) {                                                                            \
        net_structure->blobs.insert(input);                                                                            \
    }                                                                                                                  \
    for (auto output : layer_info->outputs) {                                                                          \
        net_structure->blobs.insert(output);                                                                           \
    }

// aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor
class Conv1DTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CONVOLUTION_1D;
        layer_info->type_str = "Convolution1D";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<ConvLayerParam>();
        auto layer_res = new(ConvLayerResource);
        const auto weight = inputs[1];
        const auto bias = inputs[2];
        const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
        const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
        const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
        const auto group = getValue<int64_t>(inputs[6]);
        auto weight_buf = getValue(weight);
        auto shape = weight_buf.GetBufferDims();

        // set param accroding to real value, just test here
        layer_param->name = layer_info->name;
        layer_param->pad_type = -1;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        layer_param->kernels = {shape[2]};
        layer_param->dialations = {(int)dialation[0]};
        layer_param->strides = {(int)stride[0]};
        layer_param->group = group;
        layer_param->pads = {(int)padding[0], (int)padding[0]};
        layer_res->name = layer_info->name;
        layer_res->filter_handle = ConvertHalfHandle(weight_buf);

        if (toIValue(bias)->isTensor()) {
            layer_param->bias      = 1;
            layer_res->bias_handle = getValue(bias);
            layer_res->bias_handle = ConvertHalfHandle(layer_res->bias_handle);
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};



// func: conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
class Conv2DTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CONVOLUTION;
        layer_info->type_str = "Convolution";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<ConvLayerParam>();
        auto layer_res = new(ConvLayerResource);
        const auto weight = inputs[1];
        const auto bias = inputs[2];
        const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
        const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
        const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
        const auto group = getValue<int64_t>(inputs[6]);
        auto weight_buf = getValue(weight);
        auto shape = weight_buf.GetBufferDims();

        // set param accroding to real value, just test here
        layer_param->name = layer_info->name;
        layer_param->pad_type = -1;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        // order [w, h]
        layer_param->kernels = {shape[3], shape[2]};
        layer_param->dialations = {(int)dialation[1], (int)dialation[0]};
        layer_param->strides = {(int)stride[1], (int)stride[0]};
        layer_param->group = group;
        layer_param->pads = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
        layer_res->name = layer_info->name;
        layer_res->filter_handle = ConvertHalfHandle(weight_buf);

        if (toIValue(bias)->isTensor()) {
            layer_param->bias      = 1;
            layer_res->bias_handle = getValue(bias);
            layer_res->bias_handle = ConvertHalfHandle(layer_res->bias_handle);
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class Conv3DTorchConverter : public TorchOpConverter {
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CONVOLUTION_3D;
        layer_info->type_str = "Convolution3D";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<ConvLayerParam>();
        auto layer_res = new(ConvLayerResource);
        const auto weight = inputs[1];
        const auto bias = inputs[2];
        const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
        const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
        const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
        const auto group = getValue<int64_t>(inputs[6]);
        auto weight_buf = getValue(weight);
        auto shape = weight_buf.GetBufferDims();

        // set param accroding to real value, just test here
        layer_param->name = layer_info->name;
        layer_param->pad_type = -1;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        // order [w, h]
        layer_param->kernels = {shape[4], shape[3], shape[2]};
        layer_param->dialations = {(int)dialation[2], (int)dialation[1], (int)dialation[0]};
        layer_param->strides = {(int)stride[2], (int)stride[1], (int)stride[0]};
        layer_param->group = group;
        layer_param->pads = {(int)padding[2], (int)padding[2], (int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
        layer_res->name = layer_info->name;
        layer_res->filter_handle = ConvertHalfHandle(weight_buf);

        if (toIValue(bias)->isTensor()) {
            layer_param->bias      = 1;
            layer_res->bias_handle = getValue(bias);
            layer_res->bias_handle = ConvertHalfHandle(layer_res->bias_handle);
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: _convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, 
//                    int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
class _ConvTorchConverter : public TorchOpConverter {
public:
    //bool IsSupported(const torch::jit::Node *node) {
    //    const auto& inputs = GetEffectiveInputValues(node);
    //    const auto transposed = getValue<bool>(inputs[6]);
    //    return !transposed;
    //}

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        const auto& inputs = GetEffectiveInputValues(node);
        const auto transposed = getValue<bool>(inputs[6]);
        
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        const bool is_3d = getValue<std::vector<int64_t>>(inputs[4]).size() == 3;
        
        if (!is_3d) {
            layer_info->type = transposed ? LAYER_DECONVOLUTION : LAYER_CONVOLUTION;;
            layer_info->type_str = transposed ? "Deconvolution" : "Convolution";
            layer_info->name = node->output(0)->debugName();

            layer_info->inputs.push_back(inputs[0]->debugName());
            layer_info->outputs.push_back(node->output(0)->debugName());

            auto layer_param = std::make_shared<ConvLayerParam>();
            auto layer_res = new(ConvLayerResource);
            const auto weight = inputs[1];
            const auto bias = inputs[2];
            const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
            const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
            const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
            const auto output_pads = getValue<std::vector<int64_t>>(inputs[7]);
            const auto group = getValue<int64_t>(inputs[8]);
            // const auto transposed = getValue<bool>(inputs[6]);

            // if (transposed) {
            //     layer_info->type_str = LAYER_DECONVOLUTION;
            //     std::cout << "deconv" << std::endl;
            // }

            auto weight_buf = getValue(weight);
            auto shape = weight_buf.GetBufferDims();

            // set param accroding to real value, just test here
            layer_param->name = layer_info->name;
            if (output_pads.size()>0 && output_pads[0] != 0) {
                layer_param->pad_type = 3;
                layer_param->output_channel = shape[1] * group;
                layer_param->input_channel = shape[0] / group;
            } else {
                layer_param->pad_type = -1;
                layer_param->output_channel = shape[0];
                layer_param->input_channel = shape[1];
            }
            layer_param->kernels = {shape[3], shape[2]};
            layer_param->dialations = {(int)dialation[1], (int)dialation[0]};
            layer_param->strides = {(int)stride[1], (int)stride[0]};
            layer_param->pads = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
            layer_param->group = group;
            layer_res->name = layer_info->name;
            layer_res->filter_handle = ConvertHalfHandle(weight_buf);

            auto bias_buf = getValue(bias);
            if (bias_buf.GetBytesSize() != 0) {
                layer_param->bias = 1;
                layer_res->bias_handle = ConvertHalfHandle(bias_buf);
            }

            layer_info->param = layer_param;

            net_structure->layers.push_back(layer_info);
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

            ADD_INPUTS_AND_OUTPUTS;

            return TNN_OK;
        } else {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_CONVOLUTION_3D;
            layer_info->type_str = "Convolution3D";
            layer_info->name = node->output(0)->debugName();

            const auto& inputs = GetEffectiveInputValues(node);

            layer_info->inputs.push_back(inputs[0]->debugName());
            layer_info->outputs.push_back(node->output(0)->debugName());

            auto layer_param = std::make_shared<ConvLayerParam>();
            auto layer_res = new(ConvLayerResource);
            const auto weight = inputs[1];
            const auto bias = inputs[2];
            const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
            const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
            const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
            const auto group = getValue<int64_t>(inputs[8]);
            auto weight_buf = getValue(weight);
            auto shape = weight_buf.GetBufferDims();

            // set param accroding to real value, just test here
            layer_param->name = layer_info->name;
            layer_param->pad_type = -1;
            layer_param->output_channel = shape[0];
            layer_param->input_channel = shape[1];
            // order [w, h]
            layer_param->kernels = {shape[4], shape[3], shape[2]};
            layer_param->dialations = {(int)dialation[2], (int)dialation[1], (int)dialation[0]};
            layer_param->strides = {(int)stride[2], (int)stride[1], (int)stride[0]};
            layer_param->group = group;
            layer_param->pads = {(int)padding[2], (int)padding[2], (int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
            layer_res->name = layer_info->name;
            layer_res->filter_handle = ConvertHalfHandle(weight_buf);

            auto bias_buf = getValue(bias);
            if (bias_buf.GetBytesSize() != 0) {
                layer_param->bias = 1;
                layer_res->bias_handle = ConvertHalfHandle(bias_buf);
            }

                    layer_info->param = layer_param;

            net_structure->layers.push_back(layer_info);
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

            ADD_INPUTS_AND_OUTPUTS;

            return TNN_OK;
        }
    }
};


// new_zero, new_ones are not part of namespace at::aten.
// So we have to devide ConstantOfShapeConverter into two classes.
// aten::new_zeros/ones: Returns a Tensor of size size filled with 0. By default, the returned Tensor has the same torch.dtype and torch.device as this tensor.
// aten::new_zeros(const Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory);
// aten::new_ones(const Tensor & self, IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory);
class ConstantOfShapeZerosTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CONSTANT_OF_SHAPE;
        layer_info->type_str = "ConstantOfShape";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);
        const auto& outputs = node->outputs();
        layer_info->inputs.push_back(inputs[1]->debugName());  // size
        layer_info->outputs.push_back(outputs[0]->debugName());

        auto layer_param = std::make_shared<LayerParam>();
        auto layer_res = new(ConstantOfShapeLayerResource);

        auto dtype_kind = inputs[2]->type()->kind();
        if (dtype_kind==c10::TypeKind::NoneType) {
            int value = 0;
            RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
            value_buf.SetDataType(DATA_TYPE_INT32);
            layer_res->value = value_buf;
        } else if (dtype_kind == c10::TypeKind::ScalarTypeType || dtype_kind == c10::TypeKind::IntType) {
            int64_t dtype = getValue<int64_t>(inputs[2]);
            // libtorch/include/c10/core/ScalarType.h
            // 3:  int        Int
            // 4:  int64_t    Long
            // 5:  at::half   Half
            // 6:  float      Float
            // 7:  double     Double
            if (dtype==3 || dtype==4) {
                int value = 0;
                RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
                value_buf.SetDataType(DATA_TYPE_INT32);
                layer_res->value = value_buf;
            } else { // FLOAT, Half, DOUBLE not supported right now.
                float value = 0.0f;
                RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
                value_buf.SetDataType(DATA_TYPE_FLOAT);
                layer_res->value = value_buf;
            }
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};
class ConstantOfShapeOnesTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CONSTANT_OF_SHAPE;
        layer_info->type_str = "ConstantOfShape";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);
        const auto& outputs = node->outputs();
        layer_info->inputs.push_back(inputs[1]->debugName());  // size
        layer_info->outputs.push_back(outputs[0]->debugName());

        auto layer_param = std::make_shared<LayerParam>();
        auto layer_res = new(ConstantOfShapeLayerResource);

        auto dtype_kind = inputs[2]->type()->kind();
        if (dtype_kind==c10::TypeKind::NoneType) {
            int value = 1;
            RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
            value_buf.SetDataType(DATA_TYPE_INT32);
            layer_res->value = value_buf;
        } else if (dtype_kind == c10::TypeKind::ScalarTypeType || dtype_kind == c10::TypeKind::IntType) {
            int64_t dtype = getValue<int64_t>(inputs[2]);
            // libtorch/include/c10/core/ScalarType.h
            // 3:  int        Int
            // 4:  int64_t    Long
            // 5:  at::half   Half
            // 6:  float      Float
            // 7:  double     Double
            if (dtype==3 || dtype==4) {
                int value = 1;
                RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
                value_buf.SetDataType(DATA_TYPE_INT32);
                layer_res->value = value_buf;
            } else { // FLOAT, Half, DOUBLE not supported right now.
                float value = 1.0f;
                RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
                value_buf.SetDataType(DATA_TYPE_FLOAT);
                layer_res->value = value_buf;
            }
        }


        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)
class ContiguousTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        // IGNORE the OP
        return TNN_OK;
    }
};

// prim::device(Tensor in) -> Device
class DeviceTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        // IGNORE the OP
        return TNN_OK;
    }
};

// prim::dtype(Tensor in) -> int
class DtypeTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // only prim::dtype + aten::to.device right now.
        auto users = GetEffectiveOutputValue(node, 0)->uses();
        for (int i=0; i<users.size(); i++) {
            auto user_node = users[i].user;
            if (user_node->kind()!=at::aten::to) {
                return false;
            }
            if (toIValue(GetEffectiveInputValue(user_node,1)) && !toIValue(GetEffectiveInputValue(user_node,1)).value().isDevice()) {
                return false;
            }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        return TNN_OK;
    }
};

// func: max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
// func: adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
class PoolTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();

        auto is_3d = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node,1)).size() == 3;
        if (!is_3d) {
            layer_info->type = LAYER_POOLING;
            layer_info->type_str = "Pooling";
            layer_info->name = node->output(0)->debugName();

            const auto& inputs = GetEffectiveInputValues(node);

            layer_info->inputs.push_back(inputs[0]->debugName());
            layer_info->outputs.push_back(node->output(0)->debugName());

            auto layer_param = std::make_shared<PoolingLayerParam>();
            layer_param->name = layer_info->name;
            std::string op_type = node->kind().toUnqualString();

            if (op_type.find("adaptive") == std::string::npos) {
                const auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
                const auto stride = getValue<std::vector<int64_t>>(inputs[2]);
                const auto padding = getValue<std::vector<int64_t>>(inputs[3]);
                const auto dialation = getValue<std::vector<int64_t>>(inputs[4]);
                const auto ceil_mode = getValue<bool>(inputs[5]);
                
                layer_param->pad_type = -1;
                layer_param->kernels_params = {(int)kernel_size[1], (int)kernel_size[0]};
                layer_param->strides = {(int)stride[1], (int)stride[0]};
                layer_param->pads = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
                layer_param->kernel_indexs = {-1, -1};
                layer_param->kernels = {-1, -1};
                layer_param->output_shape = {-1, -1};
                layer_param->ceil_mode = ceil_mode;
            } else {
                const auto output_shape = getValue<std::vector<int64_t>>(inputs[1]);
                layer_param->is_adaptive_pool = 1;
                layer_param->output_shape = {(int)output_shape[1], (int)output_shape[0]};
                layer_param->kernels_params = {-1, -1};
                layer_param->strides = {1, 1};
                layer_param->pads = {0, 0, 0, 0};
                layer_param->kernel_indexs = {-1, -1};
                layer_param->kernels = {-1, -1};
            }

            layer_info->param = layer_param;

            net_structure->layers.push_back(layer_info);

            ADD_INPUTS_AND_OUTPUTS;

            return TNN_OK;
        } else {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_POOLING_3D;
            layer_info->type_str = "Pooling3D";
            layer_info->name = node->output(0)->debugName();

            const auto& inputs = GetEffectiveInputValues(node);

            layer_info->inputs.push_back(inputs[0]->debugName());
            layer_info->outputs.push_back(node->output(0)->debugName());

            auto layer_param = std::make_shared<PoolingLayerParam>();
            layer_param->name = layer_info->name;
            std::string op_type = node->kind().toUnqualString();

            if (op_type.find("adaptive") == std::string::npos) {
                const auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
                const auto stride = getValue<std::vector<int64_t>>(inputs[2]);
                const auto padding = getValue<std::vector<int64_t>>(inputs[3]);
                const auto dialation = getValue<std::vector<int64_t>>(inputs[4]);
                const auto ceil_mode = getValue<bool>(inputs[5]);
                
                layer_param->pad_type = -1;
                layer_param->kernels_params = {(int)kernel_size[2], (int)kernel_size[1], (int)kernel_size[0]};
                layer_param->strides = {(int)stride[2], (int)stride[1], (int)stride[0]};
                layer_param->pads = {(int)padding[2], (int)padding[2], (int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
                layer_param->kernel_indexs = {-1, -1, -1};
                layer_param->kernels = {-1, -1, -1};
                layer_param->output_shape = {-1, -1, -1};
                layer_param->ceil_mode = ceil_mode;
            } else {
                const auto output_shape = getValue<std::vector<int64_t>>(inputs[1]);
                layer_param->is_adaptive_pool = 1;
                layer_param->output_shape = {(int)output_shape[2], (int)output_shape[1], (int)output_shape[0]};
                layer_param->kernels_params = {-1, -1, -1};
                layer_param->strides = {1, 1, 1};
                layer_param->pads = {0, 0, 0, 0, 0, 0};
                layer_param->kernel_indexs = {-1, -1, -1};
                layer_param->kernels = {-1, -1, -1};
            }

            layer_info->param = layer_param;

            net_structure->layers.push_back(layer_info);

            ADD_INPUTS_AND_OUTPUTS;

            return TNN_OK;
        }
    }
};

// func: avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
class AvgPoolTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_POOLING;
        layer_info->type_str                  = "Pooling";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param  = std::make_shared<PoolingLayerParam>();
        layer_param->name = layer_info->name;

        auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
        auto stride      = getValue<std::vector<int64_t>>(inputs[2]);
        auto padding     = getValue<std::vector<int64_t>>(inputs[3]);
        auto ceil_mode   = getValue<bool>(inputs[4]);

        /*
         * When padding in AvgPool is not 0, the inference results of Pytorch and TNN are inconsistent.
         * Therefore, when converting, insert the Pad operator before AvgPool,
         * and set padding of AvgPool to 0 at the same time.

         * E.g.，
         * In AvgPool，kernel_size = 3, stride=1, padding=1
         * Input，
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         * the output of Pytorch，
         * 0.444, 0.667, 0.444
         * 0.667, 1.000, 0.667
         * 0.444, 0.667, 0.444
         * the output of TNN (Pad operator is not inserted)
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         */

        bool need_insert_pad = false;
        for (const auto &pad : padding) {
            need_insert_pad = (pad != 0);
        }

        if (need_insert_pad) {
            std::shared_ptr<LayerInfo> pad_layer_info = std::make_shared<LayerInfo>();
            pad_layer_info->type                      = LAYER_PAD;
            pad_layer_info->type_str                  = "Pad";
            pad_layer_info->name                      = layer_info->name + "_pad";

            pad_layer_info->inputs.push_back(layer_info->inputs[0]);
            pad_layer_info->outputs.push_back(pad_layer_info->name);
            layer_info->inputs[0] = pad_layer_info->outputs[0];

            auto pad_layer_param  = std::make_shared<PadLayerParam>();
            const int pad_h       = static_cast<int>(padding[0]);
            const int pad_w       = static_cast<int>(padding[1]);
            pad_layer_param->pads = {pad_w, pad_w, pad_h, pad_h, 0, 0, 0, 0};

            pad_layer_info->param = pad_layer_param;

            net_structure->layers.push_back(pad_layer_info);

            for (const auto &pad_input : pad_layer_info->inputs) {
                net_structure->blobs.insert(pad_input);
            }
            for (const auto &pad_output : pad_layer_info->outputs) {
                net_structure->blobs.insert(pad_output);
            }
        }

        layer_param->pool_type      = 1;
        layer_param->pad_type       = -1;
        layer_param->kernels_params = {(int)kernel_size[1], (int)kernel_size[0]};
        layer_param->strides        = {(int)stride[1], (int)stride[0]};
        layer_param->pads           = {0, 0, 0, 0};
        layer_param->kernel_indexs  = {-1, -1};
        layer_param->kernels        = {-1, -1};
        layer_param->output_shape   = {-1, -1};
        layer_param->ceil_mode      = ceil_mode;

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
class BinaryTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        switch (node->kind()) {
            case at::aten::add:
            case at::aten::add_:
                layer_info->type     = LAYER_ADD;
                layer_info->type_str = "Add";
                break;
            case at::aten::sub:
            case at::aten::sub_:
            case at::aten::rsub:
                layer_info->type     = LAYER_SUB;
                layer_info->type_str = "Sub";
                break;
            case at::aten::mul:
            case at::aten::mul_:
                layer_info->type     = LAYER_MUL;
                layer_info->type_str = "Mul";
                break;
            case at::aten::div:
            case at::aten::div_:
            case at::aten::floordiv:
                layer_info->type     = LAYER_DIV;
                layer_info->type_str = "Div";
                break;
            case at::aten::__and__:
                layer_info->type     = LAYER_AND;
                layer_info->type_str = "And";
                break;
            case at::aten::__or__:
                layer_info->type     = LAYER_OR;
                layer_info->type_str = "Or";
                break;
            case at::aten::__xor__:
                layer_info->type     = LAYER_XOR;
                layer_info->type_str = "Xor";
                break;
            case at::aten::eq:
                layer_info->type     = LAYER_EQUAL;
                layer_info->type_str = "Equal";
                break;
            default:
                LOGE("Unsupport layer type %s\n", node->kind().toUnqualString());
                ASSERT(0);
        }
        layer_info->name = node->output(0)->debugName();

        auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();

        const auto &inputs     = GetEffectiveInputValues(node);
        const auto input0_kind = inputs[0]->node()->kind();
        const auto input1_kind = inputs[1]->node()->kind();
        if (input0_kind == at::prim::Constant || input1_kind == at::prim::Constant) {
            const int weight_input_index    = input0_kind == at::prim::Constant ? 0 : 1;
            const int input_index           = input0_kind == at::prim::Constant ? 1 : 0;
            layer_param->weight_input_index = weight_input_index;
            layer_info->inputs.push_back(inputs[input_index]->debugName());

            auto layer_res            = new EltwiseLayerResource();
            auto element_buf          = getValue(inputs[weight_input_index]);
            layer_res->element_handle = ConvertHalfHandle(element_buf);
            layer_res->element_shape  = element_buf.GetBufferDims();

            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
            
            if (node->kind() == at::aten::rsub) {
                // rsub: swap input 1 and 2.
                layer_param->weight_input_index = 1 - layer_param->weight_input_index;
            }
        } else {
            layer_param->weight_input_index = -1;
            if (node->kind() != at::aten::rsub) {
                for (auto &input : GetEffectiveInputValues(node)) {
                    layer_info->inputs.push_back(input->debugName());
                    if (layer_info->inputs.size() == 2) {
                        break;
                    }
                }
            } else {
                // rsub: swap input 1 and 2.
                layer_info->inputs.push_back(inputs[1]->debugName());
                layer_info->inputs.push_back(inputs[0]->debugName());
            }
        }

        for (auto &output : node->outputs()) {
            layer_info->outputs.push_back(output->debugName());
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// Converter for all ELEMWISE OPs
class ElemwiseTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        switch (node->kind()) {
            case at::aten::abs:
                layer_info->type     = LAYER_ABS;
                layer_info->type_str = "Abs";
                break;
            case at::aten::acos:
                layer_info->type     = LAYER_ACOS;
                layer_info->type_str = "Acos";
                break;
            case at::aten::asin:
                layer_info->type     = LAYER_ASIN;
                layer_info->type_str = "Asin";
                break;
            case at::aten::atan:
                layer_info->type     = LAYER_ATAN;
                layer_info->type_str = "Atan";
                break;
            case at::aten::ceil:
                layer_info->type     = LAYER_CEIL;
                layer_info->type_str = "Ceil";
                break;
            case at::aten::cos:
                layer_info->type     = LAYER_COS;
                layer_info->type_str = "Cos";
                break;
            case at::aten::exp:
                layer_info->type     = LAYER_EXP;
                layer_info->type_str = "Exp";
                break;
            case at::aten::floor:
                layer_info->type     = LAYER_FLOOR;
                layer_info->type_str = "Floor";
                break;
            case at::aten::gelu:
                layer_info->type     = LAYER_GELU;
                layer_info->type_str = "GELU";
                break;
            case at::aten::log:
                layer_info->type     = LAYER_LOG;
                layer_info->type_str = "Log";
                break;
            case at::aten::neg:
                layer_info->type     = LAYER_NEG;
                layer_info->type_str = "Neg";
                break;
            case at::aten::__not__:
                layer_info->type     = LAYER_NOT;
                layer_info->type_str = "not";
                break;
            case at::aten::relu:
            case at::aten::relu_:
                layer_info->type     = LAYER_RELU;
                layer_info->type_str = "Relu";
                break;
            case at::aten::sigmoid:
            case at::aten::sigmoid_:
                layer_info->type     = LAYER_SIGMOID;
                layer_info->type_str = "Sigmoid";
                break;
            case at::aten::sign:
                layer_info->type     = LAYER_SIGN;
                layer_info->type_str = "Sign";
                break;
            case at::aten::sin:
                layer_info->type     = LAYER_SIN;
                layer_info->type_str = "Sin";
                break;
            case at::aten::sqrt:
                layer_info->type     = LAYER_SQRT;
                layer_info->type_str = "Sqrt";
                break;
            case at::aten::tan:
                layer_info->type     = LAYER_TAN;
                layer_info->type_str = "Tan";
                break;
            case at::aten::tanh:
                layer_info->type     = LAYER_TANH;
                layer_info->type_str = "Tanh";
                break;
            default:
                LOGE("Unsupport elemwise layer type %s\n", node->kind().toUnqualString());
                ASSERT(0);
        }
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node,0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());
        layer_info->param = std::make_shared<LayerParam>();

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};


// aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
// aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)
class ExpandTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type       = LAYER_EXPAND;
        layer_info->type_str   = "Expand";
        layer_info->name       = node->output(0)->debugName();

        const auto &inputs     = GetEffectiveInputValues(node);
        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());
        auto layer_param = std::make_shared<ExpandLayerParam>();

        if (toIValue(inputs[1])) {
            const auto shape_int64 = getValue<std::vector<int64_t>>(inputs[1]);
            std::vector<int> shape_int32;
            for (const auto &dim : shape_int64) {
                shape_int32.emplace_back(static_cast<int>(dim));
            }
            layer_param->shape = shape_int32;
        } else {
            // Try to Lower Dynamic Expand To Static
            // e.g:
            //   %68 : int[] = prim::ListConstruct(%bsz.11, %output_seq_len.1)
            //   %69 : Tensor = aten::expand(%67, %68, %7)
            //   %70 : int[] = prim::ListConstruct(%bsz.11, %self.conv_layers.0.layers.0.in_channels)
            //   %71 : Tensor = aten::reshape(%input_lengths.1, %70)
            //   %73 : Tensor = aten::expand(%71, %68, %7)
            // In Example Above, %68, %71 both has length 2, and the both of their inputs are %bsz.11
            // And in %70, %self.conv_layers.0.layers.0.in_channels is constant.
            bool expand_able_to_lower_to_static = false;
            auto in0_node = inputs[0]->node();
            auto in1_node = inputs[1]->node();
            if ((in0_node->kind() == at::aten::view || in0_node->kind() == at::aten::reshape) &&
                GetEffectiveInputValue(in0_node, 1)->node()->kind() == at::prim::ListConstruct &&
                in1_node->kind() == at::prim::ListConstruct) {
                auto in0_in1_node = GetEffectiveInputValue(in0_node, 1)->node();
                if (in0_in1_node->inputs().size() == in1_node->inputs().size()) {
                    bool all_dims_static = true;
                    std::vector<int> shape_int32(in0_in1_node->inputs().size(), 0);
                    for (int i=0; i<in0_in1_node->inputs().size(); i++) {
                        if (GetEffectiveInputValue(in0_in1_node, i)->debugName() == GetEffectiveInputValue(in1_node, i)->debugName()) {
                            shape_int32[i] = -1;
                            continue;
                        }
                        if (toIValue(GetEffectiveInputValue(in1_node, i))) {
                            shape_int32[i] = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(in1_node, i)));
                            continue;
                        }
                        all_dims_static = false;
                    }
                    if (all_dims_static) {
                        expand_able_to_lower_to_static = true;
                        layer_param->shape = shape_int32;
                    }
                }
            }

            if (!expand_able_to_lower_to_static) {
                layer_info->inputs.push_back(inputs[1]->debugName());
            }
        }
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        
        if (toIValue(inputs[0])) {
            net_resource->constant_map[GetEffectiveInputValue(node,0)->debugName()] = std::make_shared<RawBuffer>(getValue(GetEffectiveInputValue(node, 0)));
        }

        return TNN_OK;
    }
};

class ExpandasTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        if (GetEffectiveInputValues(node).size() == 2) {
            // only support "norm + clampmin + expandas + div"
            for (int i = 0; i < GetEffectiveOutputValue(node, 0)->uses().size(); i++) {
                if (GetEffectiveOutputValue(node, 0)->uses()[i].user->kind() != at::aten::div) {
                    return false;
                } else {
                    auto& converter = GetGlobalTorchConvertMap()["aten::div"];
                    if (!converter->IsSupported(GetEffectiveOutputValue(node, 0)->uses()[i].user)) {
                        return false;
                    }
                }
            }
            return true;
        } else {
            return false;
        }
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_EXPANDAS;
        layer_info->type_str                  = "Expandas";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->inputs.push_back(inputs[1]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        layer_info->param = std::make_shared<LayerParam>();

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class FlattenTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_FLATTENTORCH;
        layer_info->type_str = "FlattenTorch";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param       = std::make_shared<FlattenTorchLayerParam>();
        layer_param->start_dim = static_cast<int>(getValue<int64_t>(inputs[1]));
        layer_param->end_dim   = static_cast<int>(getValue<int64_t>(inputs[2]));
        layer_info->param      = layer_param;

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};


class GetItemTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        if (at::ListTypePtr list_type = GetEffectiveInputValue(node, 0)->type()->cast<at::ListType>()) {
            // GetItem of TensorType has been removed from net in TNN-Torch Optimize Passes.
            // Here we deal with aten::__getitem__ of int type only.
            if (list_type->getElementType()->kind()==at::TypeKind::IntType) {
                if (toIValue(GetEffectiveInputValue(node, 1))) {
                    return true;
                }
            }
        }
        return false;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                 = LAYER_GATHER;
        layer_info->type_str             = "Gather";
        layer_info->name                 = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param                 = std::make_shared<GatherLayerParam>();
        layer_param->axis                = 0;
        layer_param->indices_in_resource = true;
        layer_info->param                = layer_param;

        int indices        = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
        auto layer_res     = std::make_shared<GatherLayerResource>();
        auto indices_buf   = RawBuffer(sizeof(int), reinterpret_cast<char *>(&indices), {1});
        indices_buf.SetDataType(DATA_TYPE_INT32);
        layer_res->indices = indices_buf;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = layer_res;
        
        return TNN_OK;
    }
};

// https://arxiv.org/abs/1612.08083
// aten::glu(Tensor self, int dim=-1) -> Tensor
// SPLIT torch.nn.GLU into splitV + sigmoid + elemwise_mul
class GluTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::string split_out_name = node->output(0)->debugName() + "_splitv";
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_SPLITV;
            layer_info->type_str                  = "SplitV";
            layer_info->name                      = split_out_name;

            layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
            layer_info->outputs.push_back(split_out_name+"_0");
            layer_info->outputs.push_back(split_out_name+"_1");

            auto layer_param                      = std::make_shared<SplitVLayerParam>();
            layer_param->axis                     = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
            layer_param->is_split_specified       = false;
            layer_param->slices                   = std::vector<int>(2,-1);
            layer_info->param                     = layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_SIGMOID;
            layer_info->type_str                  = "Sigmoid";
            layer_info->name                      = node->output(0)->debugName() + "_sigmoid";

            layer_info->inputs.push_back(split_out_name + "_1");
            layer_info->outputs.push_back(node->output(0)->debugName() + "_sigmoid");
            layer_info->param = std::make_shared<LayerParam>();

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_MUL;
            layer_info->type_str                  = "Mul";
            layer_info->name                      = node->output(0)->debugName();

            layer_info->inputs.push_back(split_out_name + "_0");
            layer_info->inputs.push_back(node->output(0)->debugName() + "_sigmoid");
            layer_info->outputs.push_back(node->output(0)->debugName());

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->param = layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        return TNN_OK;
    }
};

// func: group_norm(const at::Tensor & input, int num_goups, const c10::optional<at::Tensor> & weight={}, const c10::optional<at::Tensor> & bias={}, double eps=1e-05);
class GroupNormTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_GROUP_NORM;
        layer_info->type_str                  = "GroupNorm";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm
        // Assume TorchScript is well-formed, weight, bias are present,
        // weight.shape, bias.shape = normalized_shape
        layer_info->inputs.push_back(inputs[0]->debugName()); // input
        layer_info->inputs.push_back(inputs[2]->debugName()); // weight
        layer_info->inputs.push_back(inputs[3]->debugName()); // bias
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        const auto num_groups   = getValue<int>(inputs[1]);
        const auto eps          = getValue<float>(inputs[4]);
        auto layer_param = std::make_shared<GroupNormLayerParam>();
        layer_param->group = num_groups;
        layer_param->eps = eps;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->constant_map[inputs[2]->debugName()] = std::make_shared<RawBuffer>(getValue(inputs[2])); // weight
        net_resource->constant_map[inputs[3]->debugName()] = std::make_shared<RawBuffer>(getValue(inputs[3])); // bias

        return TNN_OK;
    }
};

// Deal with Greater, Less, GreaterEqual, LessEqual
// aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
// aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
// aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
// aten::le.Tensor(Tensor self, Tensor other) -> Tensor
class GreaterLessTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        if (GetEffectiveInputValue(node, 0)->node()->kind() == at::prim::Constant ||
            GetEffectiveInputValue(node, 1)->node()->kind() == at::prim::Constant) {
            return false;
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            if (node->kind()==at::aten::gt || node->kind()==at::aten::le) {
                layer_info->type                  = LAYER_GREATER;
                layer_info->type_str              = "Greater";
            } else {
                layer_info->type                  = LAYER_LESS;
                layer_info->type_str              = "Less";
            }
            if (node->kind()==at::aten::gt || node->kind()==at::aten::lt) {
                layer_info->name                  = node->output(0)->debugName();
                layer_info->outputs.push_back(node->output(0)->debugName());
            } else { // node->kind()==at::aten::ge || node->kind()==at::aten::le
                layer_info->name                  = node->output(0)->debugName() + "greater_or_less";
                layer_info->outputs.push_back(node->output(0)->debugName() + "greater_or_less");
            }
            layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
            layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
            layer_info->param = std::make_shared<MultidirBroadcastLayerParam>();

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        if (node->kind()==at::aten::ge || node->kind()==at::aten::le) {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_NOT;
            layer_info->type_str                  = "Not";
            layer_info->name                      = node->output(0)->debugName();

            layer_info->inputs.push_back(node->output(0)->debugName() + "greater_or_less");
            layer_info->outputs.push_back(node->output(0)->debugName());
            layer_info->param = std::make_shared<LayerParam>();

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        return TNN_OK;
    }
};


// func: layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight={}, const c10::optional<at::Tensor> & bias={}, double eps=1e-05, bool cudnn_enable=true);
class LayerNormTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_LAYER_NORM;
        layer_info->type_str                  = "LayerNorm";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm
        // Assume TorchScript is well-formed, weight, bias are present,
        // weight.shape, bias.shape = normalized_shape
        layer_info->inputs.push_back(inputs[0]->debugName()); // input
        layer_info->inputs.push_back(inputs[2]->debugName()); // weight
        layer_info->inputs.push_back(inputs[3]->debugName()); // bias
        layer_info->outputs.push_back(node->output(0)->debugName());

        const auto normalized_shape = getValue<std::vector<int64_t>>(inputs[1]);
        const auto eps              = getValue<float>(inputs[4]);
        auto layer_param = std::make_shared<LayerNormLayerParam>();
        layer_param->reduce_dims_size = normalized_shape.size();
        layer_param->eps = eps;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->constant_map[inputs[2]->debugName()] = std::make_shared<RawBuffer>(getValue(inputs[2])); // weight
        net_resource->constant_map[inputs[3]->debugName()] = std::make_shared<RawBuffer>(getValue(inputs[3])); // bias

        return TNN_OK;
    }
};

// LayerNormDecomposed is used to split fused op to small ops.
// Fused op uses Plugin for implementation, which cannot use TensorRT Myelin optimization.
// Now, we just decompose aten::layer_norm.
class LayerNormDecomposedTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        const auto &inputs = GetEffectiveInputValues(node);
        const auto input   = inputs[0];
        const auto weight  = inputs[2];
        const auto bias    = inputs[3];
        const auto eps     = inputs[4];
        const auto normalized_shape = inputs[1];
        const auto pre_layer_name = node->output(0)->debugName();

        auto reduce_layer_param = std::make_shared<ReduceLayerParam>();
        auto axis = getValue<std::vector<int64_t>>(normalized_shape);
        for (int i = 1; i <= axis.size(); i++) {
            reduce_layer_param->axis.push_back(-i);
        }
        reduce_layer_param->keep_dims = true;

        // E[x]
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_REDUCE_MEAN;
            layer_info->type_str = "ReduceMean";
            layer_info->name = pre_layer_name + "_Avg";

            layer_info->inputs.push_back(input->debugName());
            layer_info->outputs.push_back(pre_layer_name + "_Avg");
            layer_info->param = reduce_layer_param;
            ADD_INPUTS_AND_OUTPUTS;
            net_structure->layers.push_back(layer_info);
        }

        // EX*EX
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_MUL;
            layer_info->type_str = "Mul";
            layer_info->name = pre_layer_name + "_EX2";

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->inputs.push_back(pre_layer_name + "_Avg");
            layer_info->inputs.push_back(pre_layer_name + "_Avg");
            layer_info->outputs.push_back(pre_layer_name + "_EX2");
            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;
        }

        // X*X
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_MUL;
            layer_info->type_str = "Mul";
            layer_info->name = pre_layer_name + "_X2";

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->inputs.push_back(input->debugName());
            layer_info->inputs.push_back(input->debugName());
            layer_info->outputs.push_back(pre_layer_name + "_X2");
            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;

        }

        // E[x^2]
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_REDUCE_MEAN;
            layer_info->type_str = "ReduceMean";
            layer_info->name = pre_layer_name + "_X2E";

            layer_info->inputs.push_back(pre_layer_name + "_X2");
            layer_info->outputs.push_back(pre_layer_name + "_X2E");
            layer_info->param = reduce_layer_param;
            ADD_INPUTS_AND_OUTPUTS;
            net_structure->layers.push_back(layer_info);
        }

        // E(X^2)-(E[x])^2
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_SUB;
            layer_info->type_str = "Sub";
            layer_info->name = pre_layer_name + "_Var";

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->inputs.push_back(pre_layer_name + "_X2E");
            layer_info->inputs.push_back(pre_layer_name + "_EX2");
            layer_info->outputs.push_back(pre_layer_name + "_Var");

            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;

        }

        // X-E[x]
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_SUB;
            layer_info->type_str = "Sub";
            layer_info->name = pre_layer_name + "_Sub";

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->inputs.push_back(input->debugName());
            layer_info->inputs.push_back(pre_layer_name + "_Avg");
            layer_info->outputs.push_back(pre_layer_name + "_Sub");

            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;
        }
        /* ONE method to compute VAR

        // POW(X-E(x))
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_POWER;
            layer_info->type_str                  = "Power";
            layer_info->name                      = pre_layer_name + "_Power";

            layer_info->inputs.push_back(pre_layer_name + "_Sub");
            layer_info->outputs.push_back(pre_layer_name + "_Power");
            auto layer_param         = std::make_shared<PowLayerParam>();
            layer_param->exponent = 2.0;
            layer_info->param = layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        // VAR: MEAN(POW(X-E(X)))
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_REDUCE_MEAN;
            layer_info->type_str = "ReduceMean";
            layer_info->name = pre_layer_name + "_Var";

            layer_info->inputs.push_back(pre_layer_name + "_Power");
            layer_info->outputs.push_back(pre_layer_name + "_Var");

            layer_info->param = reduce_layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);

        }
        */
        // VAR + eps
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_ADD;
            layer_info->type_str = "Add";
            layer_info->name = pre_layer_name + "_Add";

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = 1;
            auto layer_res            = new EltwiseLayerResource();
            auto element_buf          = getValue(eps);
            layer_res->element_handle = ConvertHalfHandle(element_buf);
            layer_res->element_shape  = element_buf.GetBufferDims();
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
            layer_info->inputs.push_back(pre_layer_name + "_Var");
            layer_info->outputs.push_back(pre_layer_name + "_Add");
            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;
        }

        // SQRT (VAR + eps)
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_SQRT;
            layer_info->type_str                  = "Sqrt";
            layer_info->name                      = pre_layer_name + "_Sqrt";

            layer_info->inputs.push_back(pre_layer_name + "_Add");
            layer_info->outputs.push_back(pre_layer_name + "_Sqrt");
            layer_info->param = std::make_shared<LayerParam>();
            ADD_INPUTS_AND_OUTPUTS;
            net_structure->layers.push_back(layer_info);
        }

        //div_out: (x - E[x]) / sqrt((var + eps))
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_DIV;
            layer_info->type_str = "Div";
            layer_info->name = pre_layer_name + "_Div";

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->inputs.push_back(pre_layer_name + "_Sub");
            layer_info->inputs.push_back(pre_layer_name + "_Sqrt");
            layer_info->outputs.push_back(pre_layer_name + "_Div");
            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;
        }

        if (!toIValue(weight) || !toIValue(bias)) {
            return TNN_OK;
        }

        // gamma * div_out
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_MUL;
            layer_info->type_str = "Mul";
            layer_info->name = pre_layer_name + "_Mul";

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->inputs.push_back(pre_layer_name + "_Div");
            layer_info->inputs.push_back(weight->debugName());
            layer_info->outputs.push_back(pre_layer_name + "_Mul");
            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;
        }

        // gamma * div_out + beta
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type     = LAYER_ADD;
            layer_info->type_str = "Add";
            layer_info->name = node->output(0)->debugName();

            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = -1;
            layer_info->inputs.push_back(pre_layer_name + "_Mul");
            layer_info->inputs.push_back(bias->debugName());
            layer_info->outputs.push_back(node->output(0)->debugName());

            layer_info->param = layer_param;
            net_structure->layers.push_back(layer_info);
            ADD_INPUTS_AND_OUTPUTS;
        }

        net_resource->constant_map[weight->debugName()] = std::make_shared<RawBuffer>(getValue(weight)); // weight
        net_resource->constant_map[bias->debugName()] = std::make_shared<RawBuffer>(getValue(bias)); // bias
        return TNN_OK;
    }
};

// func: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
class LinearTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        // aten::linear include batched cases, which is not supported by Inn
        // Convert aten::linear to matmul + add (with bias), matmul only (without bias)
        const auto& inputs = GetEffectiveInputValues(node);
        const auto weight = inputs[1];
        const auto bias = inputs[2];
        auto weight_buf = getValue(weight);
        auto bias_buf = getValue(bias);
        const auto data_type = weight_buf.GetDataType();
        const bool with_bias = bias_buf.GetBytesSize()!=0;
        std::string matmul_out_name = with_bias ? node->output(0)->debugName()+"_matmul" : node->output(0)->debugName();

        // Matmul layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_MATMUL;
            layer_info->type_str = "MatMul";
            layer_info->name = matmul_out_name;

            layer_info->inputs.push_back(inputs[0]->debugName());
            layer_info->outputs.push_back(matmul_out_name);

            const int dim0 = weight_buf.GetBufferDims()[0];
            const int dim1 = weight_buf.GetBufferDims()[1];
            RawBuffer transposed_weight_buf;
            // TODO: Naive 2D weight Transpose here, replace this one with a new faster one in the future.
            if (data_type==DATA_TYPE_HALF) {
                auto *weight_ptr = weight_buf.force_to<fp16_t *>();
                const int weight_byte_size = sizeof(fp16_t)*dim0*dim1;
                fp16_t *temp_weight_ptr = (fp16_t*)std::malloc(weight_byte_size);
                for (int i=0; i<dim0; i++) {
                    for (int j=0; j<dim1; j++) {
                        temp_weight_ptr[j*dim0+i] = weight_ptr[i*dim1+j];
                    }
                }
                std::memcpy(weight_ptr, temp_weight_ptr, weight_byte_size);
                transposed_weight_buf = RawBuffer(weight_byte_size, (char*)(weight_ptr), {dim1, dim0});
                transposed_weight_buf.SetDataType(DATA_TYPE_HALF);
                std::free(temp_weight_ptr);
            } else {
                // FLOAT
                auto *weight_ptr = weight_buf.force_to<float *>();
                const int weight_byte_size = sizeof(float)*dim0*dim1;
                float *temp_weight_ptr = (float*)std::malloc(weight_byte_size);
                for (int i=0; i<dim0; i++) {
                    for (int j=0; j<dim1; j++) {
                        temp_weight_ptr[j*dim0+i] = weight_ptr[i*dim1+j];
                    }
                }
                std::memcpy(weight_ptr, temp_weight_ptr, weight_byte_size);
                transposed_weight_buf = RawBuffer(weight_byte_size, (char*)(weight_ptr), {dim1, dim0});
                std::free(temp_weight_ptr);
            }

            auto layer_res = new MatMulLayerResource();
            layer_res->weight = transposed_weight_buf;

            auto layer_param = std::make_shared<MatMulLayerParam>();
            layer_param->weight_position = 1;
            layer_param->matrix_b_dims = transposed_weight_buf.GetBufferDims();
            layer_info->param = layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
        } // Matmul

        // add bias if needed.
        if (with_bias) {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_ADD;
            layer_info->type_str = "Add";
            layer_info->name = node->output(0)->debugName();

            // bias->node()->kind() == at::prim::Constant, weight here refers to "bias" of linear.
            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = 1;
            layer_info->param = layer_param;
            layer_info->inputs.push_back(matmul_out_name);
            layer_info->outputs.push_back(node->output(0)->debugName());

            auto layer_res = new EltwiseLayerResource();
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

            auto bias_buf = getValue(bias);
            layer_res->element_handle = bias_buf;
            layer_res->element_shape  = bias_buf.GetBufferDims();

            net_structure->layers.push_back(layer_info);

            ADD_INPUTS_AND_OUTPUTS;
        }

        return TNN_OK;
    }
};

// aten::bmm(Tensor self, Tensor mat2) -> Tensor (batched-GEMM)
// func: matmul(Tensor self, Tensor other) -> Tensor
class MatMulTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_MATMUL;
        layer_info->type_str = "MatMul";
        layer_info->name = node->output(0)->debugName();

        layer_info->outputs.push_back(node->output(0)->debugName());

        const auto& inputs     = GetEffectiveInputValues(node);
        const auto input0_kind = inputs[0]->node()->kind();
        const auto input1_kind = inputs[1]->node()->kind();
        
        auto layer_res         = new MatMulLayerResource();
        auto layer_param       = std::make_shared<MatMulLayerParam>();

        if (input0_kind == at::prim::Constant || input1_kind == at::prim::Constant) {
            const int weight_position      = input0_kind == at::prim::Constant ? 0 : 1;
            const int input_index          = input0_kind == at::prim::Constant ? 1 : 0;

            layer_info->inputs.push_back(inputs[input_index]->debugName());

            auto weight_buf                = getValue(inputs[weight_position]);
            layer_res->weight              = ConvertHalfHandle(weight_buf);

            layer_param->weight_position   = weight_position;
            if (input0_kind == at::prim::Constant) {
                layer_param->matrix_a_dims = weight_buf.GetBufferDims();
            } else {
                layer_param->matrix_b_dims = weight_buf.GetBufferDims();
            }
        } else {
            // No Constant.
            // by default, param.weight_position == -1, param.axis == 0.
            layer_info->inputs.push_back(inputs[0]->debugName());
            layer_info->inputs.push_back(inputs[1]->debugName());
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        return TNN_OK;
    }
};

class NormTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // only support "norm + clamp_min + expand_as + div"
        for (int i = 0; i < GetEffectiveOutputValue(node, 0)->uses().size(); i++) {
            if (GetEffectiveOutputValue(node, 0)->uses()[i].user->kind() != at::aten::clamp_min) {
                return false;
            } else {
                auto& converter = GetGlobalTorchConvertMap()["aten::clamp_min"];
                if (!converter->IsSupported(GetEffectiveOutputValue(node, 0)->uses()[i].user)) {
                    return false;
                }
            }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_NORM;
        layer_info->type_str                  = "Norm";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<NormLayerParam>();
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
class PermuteTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_PERMUTE;
        layer_info->type_str = "Permute";
        layer_info->name = node->output(0)->debugName();

        // https://pytorch.org/docs/stable/generated/torch.permute.html?highlight=permute#torch.permute
        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<PermuteLayerParam>();
        std::vector<int> permute_orders;
        for (auto dim : getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1))) {
            permute_orders.emplace_back(static_cast<int>(dim));
        }
        layer_param->orders = permute_orders;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        return TNN_OK;
    }
};

// func: hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!
class HardTanhTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CLIP;
        layer_info->type_str = "Clip";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<ClipLayerParam>();

        layer_param->min = getValue<float>(inputs[1]);
        layer_param->max = getValue<float>(inputs[2]);

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class HardSigmoidTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_HARDSIGMOID;
        layer_info->type_str = "HardSigmoid";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<HardSigmoidLayerParam>();

        layer_param->alpha = 1.0f / 6;
        layer_param->beta = 0.5f;

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class HardSwishTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_HARDSWISH;
        layer_info->type_str = "HardSwish";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<HardSwishLayerParam>();

        layer_param->alpha = 1.0f / 6;
        layer_param->beta = 0.5f;

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class IntTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        return TNN_OK;
    }
};

class BatchNormTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_BATCH_NORM;
        layer_info->type_str                  = "BatchNormCxx";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        const auto weight       = inputs[1];
        const auto bias         = inputs[2];
        const auto running_mean = inputs[3];
        const auto running_var  = inputs[4];
        const auto eps          = getValue<float>(inputs[7]);

        auto layer_param = std::make_shared<BatchNormLayerParam>();
        auto layer_res   = new BatchNormLayerResource();

        layer_param->eps = eps;

        {
            auto weight_buf = getValue(weight);
            auto bias_buf   = getValue(bias);
            auto mean_buf   = getValue(running_mean);
            auto var_buf    = getValue(running_var);

            auto fuseResource = [&](RawBuffer &gamma, RawBuffer &beta, RawBuffer &mean, RawBuffer &var,
                                    float eps) -> std::pair<RawBuffer, RawBuffer> {
                const int size       = gamma.GetDataCount();
                const auto data_type = gamma.GetDataType();
                auto gamma_fp32      = ConvertHalfHandle(gamma);
                auto beta_fp32       = ConvertHalfHandle(beta);
                auto mean_fp32       = ConvertHalfHandle(mean);
                auto var_fp32        = ConvertHalfHandle(var);
                auto *gamma_ptr      = gamma_fp32.force_to<float *>();
                auto *beta_ptr       = beta_fp32.force_to<float *>();
                auto *mean_ptr       = mean_fp32.force_to<float *>();
                auto *var_ptr        = var_fp32.force_to<float *>();

                auto scale      = std::shared_ptr<float>(new float[size], [](float *p) { delete[] p; });
                auto bias       = std::shared_ptr<float>(new float[size], [](float *p) { delete[] p; });
                auto *scale_ptr = scale.get();
                auto *bias_ptr  = bias.get();

                for (int i = 0; i < size; i++) {
                    double sqrt_var = 1.0 / std::sqrt(static_cast<double>(var_ptr[i] + eps));
                    bias_ptr[i]     = beta_ptr[i] - static_cast<float>(static_cast<double>(gamma_ptr[i]) *
                                                                   static_cast<double>(mean_ptr[i]) * sqrt_var);
                    scale_ptr[i]    = static_cast<float>(static_cast<double>(gamma_ptr[i]) * sqrt_var);
                }

                const int byte_size = size * sizeof(float);
                auto scale_buf_fp32 = RawBuffer(byte_size, reinterpret_cast<char *>(scale_ptr), gamma.GetBufferDims());
                auto bias_buf_fp32  = RawBuffer(byte_size, reinterpret_cast<char *>(bias_ptr), beta.GetBufferDims());
                // auto scale_buf      = data_type == DATA_TYPE_HALF ? ConvertFloatToHalf(scale_buf_fp32) : scale_buf_fp32;
                // auto bias_buf       = data_type == DATA_TYPE_HALF ? ConvertFloatToHalf(bias_buf_fp32) : bias_buf_fp32;

                return std::make_pair(scale_buf_fp32, bias_buf_fp32);
            };

            auto scaleAndBias = fuseResource(weight_buf, bias_buf, mean_buf, var_buf, eps);

            layer_res->name         = layer_info->name;
            layer_res->scale_handle = scaleAndBias.first;
            layer_res->bias_handle  = scaleAndBias.second;
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class ConcatTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_CONCAT;
        layer_info->type_str                  = "Concat";
        layer_info->name                      = node->output(0)->debugName();

        const auto inputs      = GetEffectiveInputValues(node);
        const auto tensor_list = inputs[0];
        for (const auto input : GetEffectiveInputValues(tensor_list->node())) {
            layer_info->inputs.push_back(input->debugName());
        }
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param  = std::make_shared<ConcatLayerParam>();
        layer_param->axis = static_cast<int>(getValue<int64_t>(inputs[1]));
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class UnsqueezeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_UNSQUEEZE;
        layer_info->type_str                  = "Unsqueeze";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<UnsqueezeLayerParam>();
        layer_param->axes = {static_cast<int>(getValue<int64_t>(inputs[1]))};

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class CloneTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_CLONE;
        layer_info->type_str                  = "Clone";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        layer_info->param = std::make_shared<LayerParam>();

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
// aten::select(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
// aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
class GatherTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // TensorType: ListConstruct + index
        if (node->kind()==at::aten::index) {
            const auto& inputs = GetEffectiveInputValues(node);
            auto in1_node = GetEffectiveInputValue(node, 1)->node();
            if (!toIValue(inputs[0])) {
                // aten::index only support input0 (Tensor::self) in Constant.
                // Other input type combination may be Added in the future.
                return false;
            }
            if (in1_node->kind() != at::prim::ListConstruct) {
                return false;
            }
            if (GetEffectiveInputValues(in1_node).size() != 1) {
                // Currently, OP at::aten::index only support one input tensor
                // That is, result of at::prim::ListConstruct should be Tensor?[] with only one element.
                return false;
            }
        }
        return true;
    }
    
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_GATHER;
        layer_info->type_str                  = "Gather";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<GatherLayerParam>();
        auto layer_res   = new GatherLayerResource();

        if (node->kind()==at::aten::index) {
            layer_info->inputs.push_back(GetEffectiveInputValue(GetEffectiveInputValue(node, 1)->node(), 0)->debugName());
            layer_param->axis                = 0;
            layer_param->data_in_resource    = true;
            layer_param->indices_in_resource = false;

            auto data_buf = getValue(GetEffectiveInputValue(node, 0));
            layer_res->data = data_buf;
        } else if (node->kind()==at::aten::select) {
            layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
            layer_param->axis                = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
            layer_param->data_in_resource    = false;
            layer_param->indices_in_resource = true;

            int index        = getValue<int64_t>(GetEffectiveInputValue(node, 2));
            auto indices_buf = RawBuffer(4, reinterpret_cast<char *>(&index), {});
            indices_buf.SetDataType(DATA_TYPE_INT32);
            layer_res->indices = indices_buf;
        } else { // node->kind()==at::aten::embedding
            layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName()); // indices
            layer_param->data_in_resource    = true;
            layer_param->indices_in_resource = false;

            auto weight_buf = getValue(GetEffectiveInputValue(node, 0));
            
            layer_res->data = weight_buf;
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        return TNN_OK;
    }
};

// func: aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
class LogSoftmaxTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_LOGSOFTMAX;
        layer_info->type_str = "LogSoftmax";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<LogSoftmaxLayerParam>();
        layer_param->axis = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
class StridedSliceTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        const auto &inputs = GetEffectiveInputValues(node);
        if ((inputs[2]->type()->kind() != c10::TypeKind::NoneType && !toIValue(inputs[2])) ||
            (inputs[3]->type()->kind() != c10::TypeKind::NoneType && !toIValue(inputs[3]))) {
            // StridedSliceV2 with dynamic begin or and is not supported yet.
            return false;
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_STRIDED_SLICE_V2;
        layer_info->type_str                  = "StridedSliceV2";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<StrideSliceV2LayerParam>();

        // Rule to set default values for param: start, end of aten::slice
        // is defined in pytorch/aten/src/ATen/TensorIndexing.h
        layer_param->axes    = {static_cast<int>(getValue<int64_t>(inputs[1]))};
        layer_param->strides = {static_cast<int>(getValue<int64_t>(inputs[4]))};

        if (inputs[2]->type()->kind() == c10::TypeKind::NoneType) {
            layer_param->begins = {layer_param->strides[0]<0 ? INT_MAX : 0};
        } else {
            layer_param->begins = {static_cast<int>(getValue<int64_t>(inputs[2]))};
        }
        if (inputs[3]->type()->kind() == c10::TypeKind::NoneType) {
            layer_param->ends = {layer_param->strides[0]<0 ? INT_MIN : INT_MAX};
        } else {
            auto end = getValue<int64_t>(inputs[3]);
            layer_param->ends = {end > INT_MAX? INT_MAX : static_cast<int>(end)};
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class SizeTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        if (GetEffectiveInputValues(node).size() == 2) {
            // aten::size(%in_tensor, %dim)
            for (int i = 0; i < GetEffectiveOutputValue(node, 0)->uses().size(); i++) {
                if (GetEffectiveOutputValue(node, 0)->uses()[i].user->kind() != at::prim::ListConstruct) {
                    //return false;
                    return true;
                } else {
                    auto& converter = GetGlobalTorchConvertMap()["prim::ListConstruct"];
                    if (!converter->IsSupported(GetEffectiveOutputValue(node, 0)->uses()[i].user)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        // generate shape layer
        {
            std::string shape_out_name = GetEffectiveInputValues(node).size() == 2 ? node->output(0)->debugName() + "_shape" : node->output(0)->debugName();

            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_SHAPE;
            layer_info->type_str                  = "Shape";
            layer_info->name                      = shape_out_name;

            layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
            layer_info->outputs.push_back(shape_out_name);

            layer_info->param = std::make_shared<LayerParam>();

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        if (GetEffectiveInputValues(node).size() == 2) {
            // generate gather layer
            {
                std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
                layer_info->type                      = LAYER_GATHER;
                layer_info->type_str                  = "Gather";
                layer_info->name                      = node->output(0)->debugName() + "_gather";

                layer_info->inputs.push_back(node->output(0)->debugName() + "_shape");
                layer_info->outputs.push_back(node->output(0)->debugName() + "_gather");

                auto layer_param                 = std::make_shared<GatherLayerParam>();
                layer_param->axis                = 0;
                layer_param->indices_in_resource = true;

                layer_info->param = layer_param;

                const auto indices = getValue(GetEffectiveInputValue(node, 1));
                auto layer_res     = std::make_shared<GatherLayerResource>();
                layer_res->indices = indices;

                ADD_INPUTS_AND_OUTPUTS;

                net_structure->layers.push_back(layer_info);
                net_resource->resource_map[layer_info->name] = layer_res;
            }

            // generate unsqueeze layer
            {
                std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
                layer_info->type                      = LAYER_UNSQUEEZE;
                layer_info->type_str                  = "Unsqueeze";
                layer_info->name                      = node->output(0)->debugName();

                layer_info->inputs.push_back(node->output(0)->debugName() + "_gather");
                layer_info->outputs.push_back(node->output(0)->debugName());

                auto layer_param  = std::make_shared<UnsqueezeLayerParam>();
                layer_param->axes = {0};

                layer_info->param = layer_param;

                ADD_INPUTS_AND_OUTPUTS;

                net_structure->layers.push_back(layer_info);
            }
        }

        return TNN_OK;
    }
};

// func: aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
//       aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor, NOT SUPPORTED NOW
//       dtype NOT SUPPORTED NOW.
class SoftmaxTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_SOFTMAX;
        layer_info->type_str = "SoftmaxCaffe";
        layer_info->name = node->output(0)->debugName();

        // https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html?highlight=softmax#torch.nn.Softmax
        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<SoftmaxLayerParam>();
        layer_param->axis = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]
// func: split.Tensor(Tensor(a -> *) self, int[] split_sections, int dim=0) -> Tensor(a)[]
class SplitTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        for (const auto& use : GetEffectiveOutputValue(node, 0)->uses()) {
            if (use.user->kind()!=c10::prim::ListUnpack && use.user->kind()!=at::aten::cat){
                return false;
            }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        auto unpack_node      = GetEffectiveOutputValue(node, 0)->uses()[0].user;
        const int output_size = unpack_node->outputs().size();
        const int axis        = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 2)));

        int split_size;
        int section_start_pos = 0;
        std::vector<int64_t> split_sections;
        bool is_split_sections = false;

        if (toIValue(GetEffectiveInputValue(node, 1)).value().isList()) {
            // NOTE: size of split sections == output_size of prim::ListUnpack
            is_split_sections = true;
            split_sections = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1));
        } else {
            split_size = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
        }

        for (int i = 0; i < output_size; i++) {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_STRIDED_SLICE_V2;
            layer_info->type_str                  = "StridedSliceV2";
            layer_info->name                      = unpack_node->output(i)->debugName();

            layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
            layer_info->outputs.push_back(unpack_node->output(i)->debugName());

            auto layer_param = std::make_shared<StrideSliceV2LayerParam>();
            if (is_split_sections) {
                layer_param->begins.push_back(section_start_pos);
                section_start_pos += static_cast<int>(split_sections[i]);
                layer_param->ends.push_back(section_start_pos);
            } else {
                layer_param->begins.push_back(i * split_size);
                layer_param->ends.push_back((i + 1) * split_size);
                if (i + 1 == output_size) {
                    layer_param->ends.back() = INT_MAX;
                }
            }
            layer_param->axes.push_back(axis);
            layer_param->strides.push_back(1);

            layer_info->param = layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        return TNN_OK;
    }
};

// func: aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
class SplitVTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        for (const auto& use : GetEffectiveOutputValue(node, 0)->uses()) {
            if (use.user->kind()!=c10::prim::ListUnpack){
                return false;
            }
        }
        if (!toIValue(GetEffectiveInputValue(node, 1))) {
            // int chunks should be prim::Constant.
            return false;
        }
        int chunks = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
        auto out_listunpack_node = GetEffectiveOutputValue(node, 0)->uses()[0].user;
        if (out_listunpack_node->outputs().size() != chunks) {
            // 1st input: chunks should be equal to number of outputs of the succeeding prim::ListUnpack
            return false;
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_SPLITV;
        layer_info->type_str                  = "SplitV";
        layer_info->name                      = node->output(0)->debugName();

        const auto& inputs = GetEffectiveInputValues(node);
        layer_info->inputs.push_back(inputs[0]->debugName());

        int num_outputs = static_cast<int>(getValue<int64_t>(inputs[1]));
        auto out_listunpack_node = GetEffectiveOutputValue(node, 0)->uses()[0].user;
        auto out_listunpack_outputs = out_listunpack_node->outputs();
        for (auto output : out_listunpack_outputs) {
            layer_info->outputs.push_back(output->debugName());
        }

        auto layer_param                      = std::make_shared<SplitVLayerParam>();
        layer_param->axis                     = static_cast<int>(getValue<int64_t>(inputs[2]));
        layer_param->is_split_specified       = false;
        layer_param->slices                   = std::vector<int>(num_outputs,-1);
        layer_info->param                     = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// Support:
// aten::type_as(Tensor self, Tensor other) -> Tensor
// aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
// aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
// aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
class ToTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        const auto& inputs = GetEffectiveInputValues(node);
        if (node->kind()==at::aten::to) {
            int dtype_idx = 1;
            if (toIValue(inputs[1]) && toIValue(inputs[1]).value().isDevice()) {
                dtype_idx = 2;
            }
            if (toIValue(inputs[dtype_idx])) {
                auto dtype_kind = inputs[dtype_idx]->type()->kind();
                if (dtype_kind == c10::TypeKind::ScalarTypeType || dtype_kind == c10::TypeKind::IntType) {
                    auto dtype = getValue<int64_t>(inputs[dtype_idx]);
                    // libtorch/include/c10/core/ScalarType.h
                    // 0:  uint8_t    Byte
                    // 1:  int8_t     Char
                    // 2:  int16_t    Short
                    // 3:  int        Int
                    // 4:  int64_t    Long
                    // 5:  at::half   Half
                    // 6:  float      Float
                    // 7:  double     Double
                    // 11: bool       Bool
                    return dtype==3 || dtype==4 || dtype==6 || dtype==11;
                } else {
                    return false;
                }
            } else {
                // dtype was not able to get. equal to CastTo, TypeAs
                if (inputs[dtype_idx]->node()->kind()==at::prim::dtype) {
                    return true;
                }
            }
        } else if (node->kind()==at::aten::type_as) {
            return true;
        }
        return false;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CAST;
        layer_info->type_str = "Cast";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<CastLayerParam>();

        if (node->kind()==at::aten::to) {
            int dtype_idx = 1;
            if (toIValue(GetEffectiveInputValue(node, 1)) && toIValue(GetEffectiveInputValue(node, 1)).value().isDevice()) {
                dtype_idx = 2;
            }
            if (toIValue(GetEffectiveInputValue(node, dtype_idx))) {
                int64_t dtype = getValue<int64_t>(GetEffectiveInputValue(node, dtype_idx));
                if (dtype==3 || dtype==4 || dtype==11) {
                    // Cast bool, int32 and int64 to int32.
                    layer_param->to = DATA_TYPE_INT32;
                } else {
                    // dtype == 6, aka float.
                    layer_param->to = DATA_TYPE_FLOAT;
                }
            } else {
                // input[dtype_ix] is result of prev prim::dtype, set input of prim::dtype as second input of CAST
                layer_info->inputs.push_back(GetEffectiveInputValue(GetEffectiveInputValue(node, dtype_idx)->node(), 0)->debugName());
            }
        } else { // node->kind()==at::aten::type_as
            // CastAs
            layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: view(Tensor(a) self, int[] size) -> Tensor(a)
class ReshapeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_RESHAPETORCH;
        layer_info->type_str                  = "ReshapeTorch";
        layer_info->name                      = node->output(0)->debugName();

        //for (const auto &input : GetEffectiveInputValues(node)) {
        //    layer_info->inputs.push_back(input->debugName());
        //}
        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<ReshapeLayerParam>();

        if (GetEffectiveInputValue(node, 1)->node()->kind() == at::prim::ListConstruct) {
            std::vector<int> reshape_dims;
            auto list_element_values = GetEffectiveInputValues(GetEffectiveInputValue(node, 1)->node());
            for (int i=0; i<list_element_values.size(); i++) {
                if (!toIValue(list_element_values[i])) {
                    reshape_dims.push_back(-1);
                } else {
                    reshape_dims.push_back(static_cast<int>(getValue<int64_t>(list_element_values[i])));
                }
            }
            if (std::count(reshape_dims.begin(), reshape_dims.end(), -1)>1) {
                layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
                layer_param->num_axes = 0;
            } else {
                layer_param->num_axes = reshape_dims.size();
                layer_param->shape = reshape_dims;
            }
        } else {
            if (!toIValue(GetEffectiveInputValue(node, 1))) {
                // reshpae param need to be calc in runtime
                layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
                layer_param->num_axes = 0;
            } else {
                const auto shapes     = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1));
                layer_param->num_axes = static_cast<int>(shapes.size());
                for (const auto &shape : shapes) {
                    layer_param->shape.emplace_back((int)shape);
                }
            }
        }
        
        // Rare Cases when input 0 is Constant.
        if (toIValue((GetEffectiveInputValue(node, 0)))) {
            auto in0_value = GetEffectiveInputValue(node, 0);
            if (in0_value->type()->kind() == c10::TypeKind::TensorType) {
                net_resource->constant_map[in0_value->debugName()] = std::make_shared<RawBuffer>(getValue(in0_value));
            }
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
class AddmmTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_INNER_PRODUCT;
        layer_info->type_str                  = "InnerProduct";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[1]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param  = std::make_shared<InnerProductLayerParam>();
        auto layer_res    = new (InnerProductLayerResource);
        const auto weight = inputs[2];
        const auto bias   = inputs[0];

        auto weight_buf = getValue(weight);
        auto shape      = weight_buf.GetBufferDims();
        weight_buf.Permute(shape[0], shape[1]);

        // set param accroding to real value, just test here
        layer_param->name       = layer_info->name;
        layer_param->num_output = shape[1];
        layer_param->axis       = 1;

        layer_res->name          = layer_info->name;
        layer_res->weight_handle = weight_buf;

        auto bias_buf = getValue(bias);
        if (bias_buf.GetBytesSize() != 0) {
            layer_param->has_bias  = 1;
            layer_res->bias_handle = bias_buf;
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class TransposeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_PERMUTEV2;
        layer_info->type_str = "PermuteV2";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<PermuteV2LayerParam>();
        layer_param->dim0 = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
        layer_param->dim1 = static_cast<int>(getValue<int64_t>(GetEffectiveInputValue(node, 2)));

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
// func: upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
// func: upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
// func: upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
class UpsampleTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // in this mode, upsample param dims will be calc runtime
        // Todo: trt shape tensor should expand hw tensor to nchw tensor
        if (!toIValue(GetEffectiveInputValue(node, 1))) {
            return false;
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_UPSAMPLE;
        layer_info->type_str = "Upsample";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<UpsampleLayerParam>();
        switch (node->kind()) {
            case at::aten::upsample_nearest2d:
                layer_param->mode = 1;
                if (GetEffectiveInputValues(node).size() == 3) {
                    if (GetEffectiveInputValue(node, 2)->type()->kind() == c10::TypeKind::NoneType) {
                        // scale is none, use dims
                        layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
                        layer_param->scales = {0.f, 0.f};
                    } else {
                        // scale is not none, use scale
                        auto scales = getValue<std::vector<double>>(GetEffectiveInputValue(node, 2));
                        layer_param->scales = {(float)scales[1], (float)scales[0]};
                    }
                } else if (GetEffectiveInputValues(node).size() == 4) {
                    if (!toIValue(GetEffectiveInputValue(node, 1))) {
                        layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName() + "_roi");
                        layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName() + "_scale");
                        layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
                        layer_param->scales = {0.f, 0.f};
                        // empty raw buffer just makes tnn not crash
                        net_resource->constant_map[layer_info->inputs[1]] = std::make_shared<RawBuffer>();
                        net_resource->constant_map[layer_info->inputs[2]] = std::make_shared<RawBuffer>();
                    } else {
                        layer_param->scales.push_back(getValue<float>(GetEffectiveInputValue(node, 3)));
                        layer_param->scales.push_back(getValue<float>(GetEffectiveInputValue(node, 2)));
                    }
                }

                break;
            case at::aten::upsample_bilinear2d:
                layer_param->mode = 2;
                if (GetEffectiveInputValues(node).size() == 4) {
                    layer_param->align_corners = getValue<bool>(GetEffectiveInputValue(node, 2));
                    if (GetEffectiveInputValue(node, 3)->type()->kind() == c10::TypeKind::NoneType) {
                        // scale is none, use dims
                        layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
                        layer_param->scales = {0.f, 0.f};
                    } else {
                        // scale is not none, use scale
                        auto scales = getValue<std::vector<double>>(GetEffectiveInputValue(node, 3));
                        layer_param->scales = {(float)scales[1], (float)scales[0]};
                    }
                } else if (GetEffectiveInputValues(node).size() == 5) {
                    layer_param->align_corners = getValue<bool>(GetEffectiveInputValue(node, 2));
                    if (GetEffectiveInputValue(node, 3)->type()->kind() == c10::TypeKind::NoneType) {
                        // scale is none, use dims
                        layer_info->inputs.push_back(GetEffectiveInputValue(node, 1)->debugName());
                        layer_param->scales = {0.f, 0.f};
                    } else {
                        layer_param->scales.push_back(getValue<float>(GetEffectiveInputValue(node, 4)));
                        layer_param->scales.push_back(getValue<float>(GetEffectiveInputValue(node, 3)));
                    }
                }

                break;
            default:
                break;
        } 

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: sum.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
// func: mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
class ReduceTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<ReduceLayerParam>();
        auto axis = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1));
        for(auto value : axis) {
            layer_param->axis.push_back(value);
        }
        layer_param->keep_dims = getValue<bool>(GetEffectiveInputValue(node, 2));

        switch (node->kind()) {
            case at::aten::mean:
                layer_info->type = LAYER_REDUCE_MEAN;
                layer_info->type_str = "ReduceMean";
                break;
            case at::aten::sum:
                layer_info->type = LAYER_REDUCE_SUM;
                layer_info->type_str = "ReduceSum";
                break;
            default: 
                break;
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};


// Value of start, end, step is int.
// aten::arange(end, dtype, layout, device, pin_memory) -> Tensor
// aten::arange(start, end, dtype, layout, device, pin_memory) -> Tensor
// aten::arange(start, end, step, dtype, layout, device, pin_memory) -> Tensor
class RangeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_RANGE;
        layer_info->type_str                  = "Range";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);
        const int num_inputs = inputs.size();
        layer_info->outputs.push_back(node->output(0)->debugName());
        auto layer_param  = std::make_shared<RangeLayerParam>();

        const auto dtype = inputs[num_inputs-4]->type()->kind();
        if (dtype==c10::TypeKind::NoneType || dtype==c10::TypeKind::FloatType) {
            layer_param->data_type = DATA_TYPE_FLOAT;
        } else if (dtype==c10::TypeKind::IntType) {
            layer_param->data_type = DATA_TYPE_INT32;
        }

        int cur_index = 0;
        if (num_inputs==5) {
            if (!toIValue(inputs[0])) {
                layer_info->inputs.push_back(inputs[0]->debugName());
                layer_param->limit_index = cur_index++;
            } else {
                layer_param->limit = {static_cast<int>(getValue<int64_t>(inputs[0]))};
            }
        } else { // num_inputs==6 or 7
            if (!toIValue(inputs[0])) {
                layer_info->inputs.push_back(inputs[0]->debugName());
                layer_param->start_index = cur_index++;
            } else {
                layer_param->start = {static_cast<int>(getValue<int64_t>(inputs[0]))};
            }
            if (!toIValue(inputs[1])) {
                layer_info->inputs.push_back(inputs[1]->debugName());
                layer_param->limit_index = cur_index++;
            } else {
                layer_param->limit = {static_cast<int>(getValue<int64_t>(inputs[1]))};
            }
        }

        if (num_inputs==7) {
            if (!toIValue(inputs[2])) {
                layer_info->inputs.push_back(inputs[2]->debugName());
                layer_param->delta_index = cur_index++;
            } else {
                layer_param->delta = {static_cast<int>(getValue<int64_t>(inputs[2]))};
            }
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// Only static Roll are supported.
// aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
class RollTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // Dynamic Roll Not Supported
        const auto& inputs = GetEffectiveInputValues(node);
        if (toIValue(inputs[0])) {
            // Does Not Support input 0 Constant.
            return false;
        }
        if (!toIValue(inputs[1])) {
            return false;
        }
        if (inputs[2]->type()->kind() == c10::TypeKind::NoneType && !toIValue(inputs[2])) {
            return false;
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_ROLL;
        layer_info->type_str                  = "Roll";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);
        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param  = std::make_shared<RollLayerParam>();
        std::vector<int64_t> shifts_int64 = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1));
        std::vector<int64_t> dims_int64;
        if (GetEffectiveInputValue(node, 2)->type()->kind() != c10::TypeKind::NoneType) {
            dims_int64 = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 2));
        } else {
            for (int i=0; i<shifts_int64.size(); i++) {
                dims_int64.push_back(i);
            }
        }
        std::vector<int> shifts;
        std::vector<int> dims;
        for (int i=0; i<shifts_int64.size(); i++) {
            shifts.push_back(static_cast<int>(shifts_int64[i]));
            dims.push_back(static_cast<int>(dims_int64[i]));
        }
        layer_param->shifts = shifts;
        layer_param->dims = dims;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};


// func: constant_pad_nd(const Tensor & self, IntArrayRef pad, const Scalar & value);
// func: reflection_pad2d(Tensor self, int[4] padding) -> Tensor
class PadTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // Do not support Pad > 3d.
        if (node->kind()==at::aten::constant_pad_nd) {
            int pad_size = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1)).size();
            if (pad_size!=2 && pad_size!=4 && pad_size!=6) {
                return false;
            }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_PAD;
        layer_info->type_str                  = "Pad";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<PadLayerParam>();
        if (node->kind()==at::aten::constant_pad_nd) {
            layer_param->type = 0;
            const auto pads   = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1));
            if (pads.size()==2) {
                layer_param->pads = {(int)(pads[0]), (int)(pads[1]), 0, 0, 0, 0};
            } else if (pads.size()==4) {
                layer_param->pads = {(int)(pads[0]), (int)(pads[1]), (int)(pads[2]), (int)(pads[3]), 0, 0};
            } else if (pads.size()==6) {
                layer_param->pads = {(int)(pads[0]), (int)(pads[1]), (int)(pads[2]), (int)(pads[3]), (int)(pads[4]), (int)(pads[5])};
            }
            if (!toIValue(GetEffectiveInputValue(node, 2))) {
                layer_param->value = 0.0f;
            } else {
                layer_param->value = getValue<float>(GetEffectiveInputValue(node, 2));
            }
        } else if (node->kind()==at::aten::reflection_pad2d) {
            layer_param->type = 1;
            const auto pads   = getValue<std::vector<int64_t>>(GetEffectiveInputValue(node, 1));
            layer_param->pads = {(int)(pads[2]), (int)(pads[3]), (int)(pads[0]), (int)(pads[1]), 0, 0};
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class ClampminTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // only support "norm + clampmin + expandas + div"
        for (int i = 0; i < GetEffectiveOutputValue(node, 0)->uses().size(); i++) {
            if (GetEffectiveOutputValue(node, 0)->uses()[i].user->kind() != at::aten::expand_as) {
                return false;
            } else {
                auto& converter = GetGlobalTorchConvertMap()["aten::expand_as"];
                if (!converter->IsSupported(GetEffectiveOutputValue(node, 0)->uses()[i].user)) {
                    return false;
                }
            }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_CLAMPMIN;
        layer_info->type_str                  = "Clampmin";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<ClampminLayerParam>();
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};


// func: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
class ClipTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_CLIP;
        layer_info->type_str                  = "Clip";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param       = std::make_shared<ClipLayerParam>();
        
        auto min_dtype_kind = GetEffectiveInputValue(node, 1)->type()->kind();
        auto max_dtype_kind = GetEffectiveInputValue(node, 2)->type()->kind();
        if (min_dtype_kind == c10::TypeKind::NoneType) {
            layer_param->min = -FLT_MAX;
        } else if (min_dtype_kind == c10::TypeKind::IntType) {
            layer_param->min = float(getValue<int64_t>(GetEffectiveInputValue(node, 1)));
        } else { // FloatType
            layer_param->min = getValue<float>(GetEffectiveInputValue(node, 1));
        }
        if (max_dtype_kind == c10::TypeKind::NoneType) {
            layer_param->max = FLT_MAX;
        } else if (max_dtype_kind == c10::TypeKind::IntType) {
            layer_param->max = float(getValue<int64_t>(GetEffectiveInputValue(node, 2)));
        } else { // FloatType
            layer_param->max = getValue<float>(GetEffectiveInputValue(node, 2));
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};


class PowerTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_POWER;
        layer_info->type_str                  = "Power";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param         = std::make_shared<PowLayerParam>();
        const auto exponent_type = GetEffectiveInputValue(node, 1)->type()->kind();
        switch (exponent_type) {
            case c10::TypeKind::IntType:
                layer_param->exponent = static_cast<float>(getValue<int>(GetEffectiveInputValue(node, 1)));
                break;
            case c10::TypeKind::FloatType:
                layer_param->exponent = getValue<float>(GetEffectiveInputValue(node, 1));
                break;
            default:
                break;
        }
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
class TopKTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_TOPK;
        layer_info->type_str                  = "TopK";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());
        layer_info->outputs.push_back(node->output(1)->debugName());

        auto layer_param = std::make_shared<TopKLayerParam>();

        layer_param->k       = static_cast<int>(getValue<int64_t>(inputs[1]));
        layer_param->axis    = static_cast<int>(getValue<int64_t>(inputs[2]));
        layer_param->largest = static_cast<int>(getValue<bool>(inputs[3]));
        layer_param->sorted  = static_cast<int>(getValue<bool>(inputs[4]));

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class NumToTensorTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        return TNN_OK;
    }
};

class ListTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // Supported Combinations:
        // IntType:    size + ListConstruct
        // TensorType: ListConstruct + cat
        // TensorType: ListConstruct + index
        if (GetEffectiveInputValues(node).size() == 0) {
            return false;
        }

        // When Judging Effective Input Value here, aten::Int and prim::NumToTensor should be
        // recoginzed as Effective Input, other OPs like aten::contiguous are still regarded as Non-Effective
        auto type = GetEffectiveInputValue(node, 0)->type();
        if (node->input(0)->node()->kind()==at::aten::Int || node->input(0)->node()->kind()==at::prim::NumToTensor) {
            type = node->input(0)->type();
        }

        if (type->kind() == c10::TypeKind::IntType) {
            auto user_type_str = GetEffectiveOutputValue(node, 0)->uses()[0].user->kind().toQualString();
            if (GetGlobalTorchConvertMap().count(user_type_str)) {
                auto& converter = GetGlobalTorchConvertMap()[user_type_str];
                if (converter->IsSupported(GetEffectiveOutputValue(node, 0)->uses()[0].user)) {
                    return true;
                }
            }
        } else if (type->kind() == c10::TypeKind::TensorType) {
            auto list_users = GetEffectiveOutputValue(node, 0)->uses();
            bool all_users_supported = true;
            for (const auto & use : list_users) {
                if (use.user->kind() != c10::aten::cat && use.user->kind() != at::aten::index) {
                    all_users_supported = false;
                }
                if (use.user->kind() == at::aten::index) {
                    // Currently, OP at::aten::index only support one input tensor
                    // That is, Tensor?[] with only one element.
                    if (GetEffectiveInputValues(node).size() != 1) {
                        all_users_supported = false;
                    }
                }
            }
            if (all_users_supported) {
                return true;
            }
        }

        return false;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        auto input_type = GetEffectiveInputValue(node, 0)->type();
        if (node->input(0)->node()->kind()==at::aten::Int || node->input(0)->node()->kind()==at::prim::NumToTensor) {
            input_type = node->input(0)->type();
        }

        if (input_type->kind() == c10::TypeKind::TensorType) {
            return TNN_OK;
        }

        if (input_type->kind() == c10::TypeKind::IntType) {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_CONCAT;
            layer_info->type_str                  = "Concat";
            layer_info->name                      = node->output(0)->debugName();

            const auto inputs      = GetEffectiveInputValues(node);
            const auto tensor_list = inputs[0];
            for (const auto input : inputs) {
                layer_info->inputs.push_back(input->debugName());
            }
            layer_info->outputs.push_back(node->output(0)->debugName());

            auto layer_param  = std::make_shared<ConcatLayerParam>();
            layer_param->axis = 0;
            layer_info->param = layer_param;

            for (const auto &input : inputs) {
                if (!toIValue(input)) continue;
                auto const_buf = getValue(input);
                if (const_buf.GetBytesSize() > 0) {
                    if (*(const_buf.force_to<int *>()) != INT_MAX) {
                        const_buf.SetBufferDims({1});
                        net_resource->constant_map[input->debugName()] = std::make_shared<RawBuffer>(const_buf);
                    }
                }
            }

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        return TNN_OK;
    }
};

class ListUnpackTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {

        torch::jit::Node* in0_node = GetEffectiveInputValue(node, 0)->node();
        if (in0_node->kind() == c10::aten::split || in0_node->kind() == c10::aten::chunk) {
            return true;
        } else if (in0_node->kind() == c10::aten::size) {
            if (GetEffectiveInputValues(in0_node).size() == 1) {
                // aten::size(%in_tensor), return a list representing shape of the Tensor.
                return true;
            }
        }
        return false;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        torch::jit::Node* in0_node = GetEffectiveInputValue(node, 0)->node();

        if (in0_node->kind() == c10::aten::size && GetEffectiveInputValues(in0_node).size() == 1) {
            const auto outputs = node->outputs();
            const int num_dims = outputs.size();
            
            for (int dim=0; dim<num_dims; dim++) {
                std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
                layer_info->type                 = LAYER_GATHER;
                layer_info->type_str             = "Gather";
                layer_info->name                 = outputs[dim]->debugName();

                layer_info->inputs.push_back(GetEffectiveInputValue(node, 0)->debugName());
                layer_info->outputs.push_back(outputs[dim]->debugName());

                auto layer_param                 = std::make_shared<GatherLayerParam>();
                layer_param->axis                = 0;
                layer_param->indices_in_resource = true;

                layer_info->param  = layer_param;

                int indices        = dim;
                auto layer_res     = std::make_shared<GatherLayerResource>();
                auto indices_buf   = RawBuffer(sizeof(int), reinterpret_cast<char *>(&indices), {1});
                indices_buf.SetDataType(DATA_TYPE_INT32);
                layer_res->indices = indices_buf;

                ADD_INPUTS_AND_OUTPUTS;

                net_structure->layers.push_back(layer_info);
                net_resource->resource_map[layer_info->name] = layer_res;
            }
        }
        
        return TNN_OK;
    }
};

class SqueezeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_SQUEEZE;
        layer_info->type_str                  = "Squeeze";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        layer_info->inputs.push_back(inputs[0]->debugName());
        layer_info->outputs.push_back(node->output(0)->debugName());

        auto layer_param = std::make_shared<SqueezeLayerParam>();

        layer_param->axes = {static_cast<int>(getValue<int64_t>(inputs[1]))};

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
// aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
class WhereTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_WHERE;
        layer_info->type_str                  = "Where";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = GetEffectiveInputValues(node);

        if (node->kind()==at::aten::masked_fill) {
            auto value_dtype = inputs[2]->type()->kind();
            if (!toIValue(inputs[2])) {
                layer_info->inputs.push_back(inputs[2]->debugName());  // value
                layer_info->inputs.push_back(inputs[0]->debugName());  // self
                layer_info->inputs.push_back(inputs[1]->debugName());  // mask
            } else {
                layer_info->inputs.push_back(inputs[0]->debugName());  // self
                layer_info->inputs.push_back(inputs[1]->debugName());  // mask
                auto layer_res = std::make_shared<WhereLayerResource>();
                //if (value_dtype==c10::TypeKind::IntType) {
                if (toIValue(inputs[2])->isInt()) {
                    int value = static_cast<int>(getValue<int64_t>(inputs[2]));
                    RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
                    value_buf.SetDataType(DATA_TYPE_INT32);
                    layer_res->x = value_buf;
                } else if (toIValue(inputs[2])->isDouble()) {
                    float value = getValue<float>(inputs[2]);
                    RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
                    layer_res->x = value_buf;
                } else { // RARE is Tensor, like: Float(requires_grad=0, device=cpu) = prim::Constant[value={-100000}]()
                    float value = static_cast<float*>(toIValue(inputs[2])->unsafeToTensorImpl()->data())[0];
                    value = std::max(-65503.0f, std::min(65503.0f, value)); // FP16 min, max.
                    RawBuffer value_buf = RawBuffer(4, reinterpret_cast<char *>(&value), {1});
                    layer_res->x = value_buf;
                }
                net_resource->resource_map[layer_info->name] = layer_res;
            }
        } else { // node->kind()==aten::where
            layer_info->inputs.push_back(inputs[1]->debugName());  // self
            layer_info->inputs.push_back(inputs[2]->debugName());  // other
            layer_info->inputs.push_back(inputs[0]->debugName());  // condition
        }
        layer_info->outputs.push_back(node->output(0)->debugName());

        layer_info->param = std::make_shared<LayerParam>();

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};







// class QuantConv2DTorchConverter : public TorchOpConverter {
// public:
//     Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
//         const auto& inputs = GetEffectiveInputValues(node);
//         auto weight = toIValue(inputs[1]).value();
//         std::cout << weight.isTuple() << std::endl;
//         std::cout << weight.isTensor() << std::endl;
//         std::cout << weight.isObject() << std::endl;
//         auto object = weight.toObject().get();
//         auto slots = object->slots();
//         for (auto &slot : slots) {
//             std::cout << slot.isCapsule() << std::endl;
//             auto conv_param = reinterpret_cast<ConvPackedParamsBase<2> *>(slot.toCapsule().get());
//             // c10::intrusive_ptr<ConvPackedParamsBase<2>> conv_param = slot.toCapsule();
//             std::cout << "get" << std::endl;
//         }

//         return TNN_OK;
//     }
// };




REGISTER_TORCH_OP_CONVERTER(Addmm, aten, addmm)
REGISTER_TORCH_OP_CONVERTER(AvgPool, aten, avg_pool2d)
REGISTER_TORCH_OP_CONVERTER(BatchNorm, aten, batch_norm)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add_)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, sub_)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, sub)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, rsub)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, mul_)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, mul)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, div_)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, div)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, floordiv)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, eq)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, __and__)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, __or__)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, __xor__)
REGISTER_TORCH_OP_CONVERTER(Clip, aten, clamp)
REGISTER_TORCH_OP_CONVERTER(Clampmin, aten, clamp_min)
REGISTER_TORCH_OP_CONVERTER(Clone, aten, clone)
REGISTER_TORCH_OP_CONVERTER(Concat, aten, cat)
REGISTER_TORCH_OP_CONVERTER(ConstantOfShapeZeros, aten, new_zeros)
REGISTER_TORCH_OP_CONVERTER(ConstantOfShapeOnes, aten, new_ones)
REGISTER_TORCH_OP_CONVERTER(Contiguous, aten, contiguous)
REGISTER_TORCH_OP_CONVERTER(Conv1D, aten, conv1d)
REGISTER_TORCH_OP_CONVERTER(Conv2D, aten, conv2d)
REGISTER_TORCH_OP_CONVERTER(Conv3D, aten, conv3d)
REGISTER_TORCH_OP_CONVERTER(_Conv, aten, _convolution)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, abs)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, acos)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, asin)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, atan)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, ceil)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, cos)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, exp)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, floor)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, gelu)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, log)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, neg)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, __not__)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, relu)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, relu_)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, sigmoid)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, sigmoid_)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, sign)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, sin)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, sqrt)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, tan)
REGISTER_TORCH_OP_CONVERTER(Elemwise, aten, tanh)
REGISTER_TORCH_OP_CONVERTER(Expand, aten, expand)
REGISTER_TORCH_OP_CONVERTER(Expandas, aten, expand_as)
REGISTER_TORCH_OP_CONVERTER(Flatten, aten, flatten)
REGISTER_TORCH_OP_CONVERTER(Gather, aten, embedding)
REGISTER_TORCH_OP_CONVERTER(Gather, aten, index)
REGISTER_TORCH_OP_CONVERTER(Gather, aten, select)
REGISTER_TORCH_OP_CONVERTER(GetItem, aten, __getitem__)
REGISTER_TORCH_OP_CONVERTER(Glu, aten, glu)
REGISTER_TORCH_OP_CONVERTER(GreaterLess, aten, gt)
REGISTER_TORCH_OP_CONVERTER(GreaterLess, aten, le)
REGISTER_TORCH_OP_CONVERTER(GreaterLess, aten, lt)
REGISTER_TORCH_OP_CONVERTER(GreaterLess, aten, ge)
REGISTER_TORCH_OP_CONVERTER(GroupNorm, aten, group_norm)
REGISTER_TORCH_OP_CONVERTER(HardTanh, aten, hardtanh_)
REGISTER_TORCH_OP_CONVERTER(HardSigmoid, aten, hardsigmoid_)
REGISTER_TORCH_OP_CONVERTER(HardSigmoid, aten, hardsigmoid)
REGISTER_TORCH_OP_CONVERTER(HardSwish, aten, hardswish_)
REGISTER_TORCH_OP_CONVERTER(Int, aten, Int)
REGISTER_TORCH_OP_CONVERTER(LayerNorm, aten, layer_norm)
//REGISTER_TORCH_OP_CONVERTER(LayerNormDecomposed, aten, layer_norm)
REGISTER_TORCH_OP_CONVERTER(Linear, aten, linear)
REGISTER_TORCH_OP_CONVERTER(LogSoftmax, aten, log_softmax)
REGISTER_TORCH_OP_CONVERTER(MatMul, aten, bmm)
REGISTER_TORCH_OP_CONVERTER(MatMul, aten, matmul)
REGISTER_TORCH_OP_CONVERTER(Norm, aten, norm)
REGISTER_TORCH_OP_CONVERTER(Pad, aten, constant_pad_nd)
REGISTER_TORCH_OP_CONVERTER(Pad, aten, reflection_pad2d)
REGISTER_TORCH_OP_CONVERTER(Permute, aten, permute)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, adaptive_avg_pool2d)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, adaptive_avg_pool3d)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, max_pool2d)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, max_pool3d)
REGISTER_TORCH_OP_CONVERTER(Power, aten, pow)
REGISTER_TORCH_OP_CONVERTER(Range, aten, arange)
REGISTER_TORCH_OP_CONVERTER(Reshape, aten, reshape)
REGISTER_TORCH_OP_CONVERTER(Reshape, aten, view)
REGISTER_TORCH_OP_CONVERTER(Roll, aten, roll)
REGISTER_TORCH_OP_CONVERTER(Size, aten, size)
REGISTER_TORCH_OP_CONVERTER(Softmax, aten, softmax)
REGISTER_TORCH_OP_CONVERTER(Squeeze, aten, squeeze)
REGISTER_TORCH_OP_CONVERTER(Split, aten, split)
REGISTER_TORCH_OP_CONVERTER(SplitV, aten, chunk)
REGISTER_TORCH_OP_CONVERTER(StridedSlice, aten, slice)
REGISTER_TORCH_OP_CONVERTER(To, aten, to)
REGISTER_TORCH_OP_CONVERTER(To, aten, type_as)
REGISTER_TORCH_OP_CONVERTER(TopK, aten, topk)
REGISTER_TORCH_OP_CONVERTER(Transpose, aten, transpose)
REGISTER_TORCH_OP_CONVERTER(Upsample, aten, upsample_bilinear2d)
REGISTER_TORCH_OP_CONVERTER(Upsample, aten, upsample_nearest2d)
REGISTER_TORCH_OP_CONVERTER(Unsqueeze, aten, unsqueeze)
REGISTER_TORCH_OP_CONVERTER(Where, aten, masked_fill)
REGISTER_TORCH_OP_CONVERTER(Where, aten, where)
REGISTER_TORCH_OP_CONVERTER(Reduce, aten, mean)
REGISTER_TORCH_OP_CONVERTER(Reduce, aten, sum)


REGISTER_TORCH_OP_CONVERTER(Device, prim, device)
REGISTER_TORCH_OP_CONVERTER(Dtype, prim, dtype)
REGISTER_TORCH_OP_CONVERTER(List, prim, ListConstruct)
REGISTER_TORCH_OP_CONVERTER(ListUnpack, prim, ListUnpack)
REGISTER_TORCH_OP_CONVERTER(NumToTensor, prim, NumToTensor)

// REGISTER_TORCH_OP_CONVERTER(QuantConv2D, quantized, conv2d)

} // namespace conversion
}
