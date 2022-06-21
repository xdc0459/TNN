#ifndef TNN_SOURCE_TNN_NETWORK_TNN_OPTIMIZE_H
#define TNN_SOURCE_TNN_NETWORK_TNN_OPTIMIZE_H

#include "torch/csrc/jit/ir/ir.h"
#include <torch/script.h>

#include "tnn/network/torch/torch_op_converter.h"

namespace TNN_NS {
    void TNNOptPass(std::shared_ptr<torch::jit::Graph>& graph, NetStructure* net_structure, NetResource* net_resource);
}  // namespace torch

#endif  // TNN_SOURCE_TNN_NETWORK_TNN_OPTIMIZE_H
