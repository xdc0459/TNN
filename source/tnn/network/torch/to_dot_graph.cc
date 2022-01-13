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

#include "to_dot_graph.h"

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

bool isDigit(const std::string& str)
{
    for (char const &c : str) {
       if (std::isdigit(c) == 0) return false;
    }
	return true;
}

std::string ToDotGraph(Graph& graph) {
  std::string dotgraph = "digraph G {\n";
  int node_index = 0;
  for (auto it = graph.block()->nodes().begin(), end = graph.block()->nodes().end();
       it != end;) {
    Node* cur = *it++;
	const std::string node_name = std::string(cur->kind().toQualString()) ;
	//prim::GetAttr
	//prim::Constant
	//123
    for (auto i: cur->inputs()) {
	    std::string input(i->debugName());
		std::string input_node_name(i->node()->kind().toQualString());
		if (node_name.find("prim::GetAttr") == std::string::npos &&
		    node_name.find("prim::Constant") == std::string::npos &&
		    input_node_name.find("prim::GetAttr") == std::string::npos &&
		    input_node_name.find("prim::Constant") == std::string::npos ) {
		    dotgraph += "\"" + input + "\"" + " -> " + "\"" + node_name + "---" + std::to_string(node_index) + "\";\n";
		}
	}
    for (auto i: cur->outputs()) {
	    std::string output(i->debugName());
		if (node_name.find("prim::GetAttr") == std::string::npos &&
		    node_name.find("prim::Constant") == std::string::npos) {
			//if ( (isDigit(output) && std::stoi(output) > 164) || false == isDigit(output))
		    dotgraph += "\"" + node_name + "---" + std::to_string(node_index) + "\"" + " -> " + "\"" + output + "\";\n";
	    }
	}
	node_index++;
  }
  dotgraph += "}\n";
  return dotgraph;
}

} // namespace jit
} // namespace torch
