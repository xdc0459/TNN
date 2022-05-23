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

#define UP_DIV(A, B) (((A) + (B) - 1) / B)

Texture2D<float4> input : register(t0);
Texture2D<float4> scale : register(t1);
Texture2D<float4> bias : register(t2);
RWTexture2D<float4> output : register(u0);

cbuffer Shapes: register( b0 )
{
    // NB
    // in a constant buffer, each element of an array must start on a 4-float boundary.
    // so we choose float4 for the ease of alignment with cpp

    // output dimension
    vector<int, 4> od;

};

[numthreads(4, 4, 1)]
void CSMain( uint3 DTid : SV_DispatchThreadID )
{
    int cw_idx = DTid.x;
    int hb_idx= DTid.y;

    if (cw_idx >= UP_DIV(od[1],4)*od[3] || hb_idx >= od[2]*od[0]) {
        return;
    }

    int width = od[3];
    int chan_blk_idx = cw_idx / width;

    float4 data = input[DTid.xy];
    int2 pos_scale = {chan_blk_idx, 0};
    float4 scale_ = scale[pos_scale];
    int2 pos_bias = {chan_blk_idx, 0};
    float4 bias_ = bias[pos_bias];
    data = mad(data, scale_, bias_);

    output[DTid.xy] = data;
}