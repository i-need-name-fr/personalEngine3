#pragma once
// #include "../../include/moxSourceCluster/SharcHeap.hlsl"
#if defined(MOX_SHARC_HEAP_INCLUDED) && MOX_SHARC_HEAP_INCLUDED >= 0 && defined(MOX_SHADER_STRUCTURES)
namespace Sharc
{
// the TEST switch : 0 - the cache gives the radiance as it is ( without the factor W of the reservoir ) , the same for RayGen and for SHaRCPass
static const bool SHARC_APPLY_W = false;

[[vk::binding(    1   ,     MOX_SHARC_HEAP_INCLUDED     )]] RWStructuredBuffer<Sharc::TableInfo> tableInfoBuffer;
[[vk::binding(    0   ,     MOX_SHARC_HEAP_INCLUDED     )]] RWStructuredBuffer<Sharc::Reservoir> tableBuffer;


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

float3 getGradRadiance(const float3 P , const float3 N , const float d)
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
    const uint T = info.T;

    const float3 q = P / info.cellSize;
    const float3 base = floor(q);
    const int3 cell = (int3)base;                  
    const float3 f = q - base;                    

    const uint normalID = (N.z > 0.0f ? 4u : 0u) | (N.y > 0.0f ? 2u : 0u) | (N.x > 0.0f ? 1u : 0u);
    float3 radiance = 0.0f;
    float weightSum = 0.0f;
    [unroll] for(int z = 0 ; z < 2 ; z++)
    [unroll] for(int y = 0 ; y < 2 ; y++)
    [unroll] for(int x = 0 ; x < 2 ; x++)
    {
        const uint3 corner = (uint3)((cell + int3(x , y , z)) & (int)(T - 1u));
        const float3 weights = lerp(1.0f - f , f , float3(x , y , z));
        const uint cellID = (corner.z * T + corner.y) * T + corner.x;
        const Sharc::Reservoir slotData = tableBuffer[cellID * 8u + normalID + info.offset];
        // an empty slot has no light yet
        if(slotData.valid == 0u) continue;
        const float w = weights.x * weights.y * weights.z / float(info.cellSize);
        float nW = saturate(dot(N , normalize(slotData.hitPoint - P)));
        nW = lerp(1.0f, nW, 0.5f);
        const float3 pRad = w * slotData.radiance * (SHARC_APPLY_W ? slotData.W : 1.0f) * nW;
        radiance += pRad;
        weightSum += w * nW;
    }

    return (weightSum > 1e-5f) ? radiance / weightSum : 0.0f;
}

}
#endif
