#pragma once
// #include "../../include/moxSourceCluster/SharcHeap.hlsl"
#if defined(MOX_SHARC_HEAP_INCLUDED) && MOX_SHARC_HEAP_INCLUDED >= 0 && defined(MOX_SHADER_STRUCTURES)
namespace Sharc
{
[[vk::binding(    1   ,     MOX_SHARC_HEAP_INCLUDED     )]] RWStructuredBuffer<Sharc::TableInfo> tableInfoBuffer;
[[vk::binding(    0   ,     MOX_SHARC_HEAP_INCLUDED     )]] RWStructuredBuffer<float4> tableBuffer;


uint getIndex(const float3 P , const float3 N , const float d)
{
    uint index = 0;
    [loop]
    for(int i = MOX_SHARC_LAYER_COUNT - 1 ; i >= 0 ; i--)
    {
        if(d >= tableInfoBuffer[i].minDistance)
        {
            index = (uint)i;
            break;
        }
    }

    const Sharc::TableInfo info = tableInfoBuffer[index];

    const int3 cell = (int3)floor(P / info.cellSize);
    const uint3 id = (uint3)(cell & (int)(info.T - 1));

    const uint normalID = (N.z > 0.0f ? 4u : 0u) | (N.y > 0.0f ? 2u : 0u) | (N.x > 0.0f ? 1u : 0u);

    const uint cellID = (id.z * info.T + id.y) * info.T + id.x;
    return info.offset + cellID * 8u + normalID;      // offset is the first entry of the layer , 8 slots per cell
}
}
#endif
