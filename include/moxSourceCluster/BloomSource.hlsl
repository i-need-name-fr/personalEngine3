#pragma once
// #include "../../include/moxSourceCluster/BloomSource.hlsl"
#if defined(MOX_BLOOM_SOURCE_INCLUDED) && MOX_BLOOM_SOURCE_INCLUDED >= 0 && defined(MOX_SHADER_STRUCTURES)

[[vk::binding(    0   ,    MOX_BLOOM_SOURCE_INCLUDED   )]] ConstantBuffer<BloomInfo> bloomBuffer;
[[vk::image_format("rgba16f")]]
[[vk::binding(    1   ,    MOX_BLOOM_SOURCE_INCLUDED   )]] RWTexture2DArray<float4> bloomImages;
[[vk::binding(    2   ,    MOX_BLOOM_SOURCE_INCLUDED   )]] SamplerState bloomSampler;
[[vk::binding(    3   ,    MOX_BLOOM_SOURCE_INCLUDED   )]] Texture2DArray bloomTexture;

// the level of the pyramid = the layer of the image . all the layers have the size of the layer 0 , so the level i of the size levelSize lives in the TOP LEFT corner of its layer .
// uv is the coordinate inside the LEVEL ( [0 , 1] of the level , not of the layer ) . the point is kept half a texel inside the level : the bilinear filter does not read the texels
// behind the border of the level ( the old data of the layer ) , it behaves like the clamp to edge
float3 bloomSampleLevel(const uint level , const uint2 levelSize , const float2 uv)
{
    uint layerWidth , layerHeight , layers;
    bloomImages.GetDimensions(layerWidth , layerHeight , layers);

    const float2 size = float2(levelSize);
    const float2 position = clamp(uv * size , float2(0.5f , 0.5f) , size - 0.5f);
    return bloomTexture.SampleLevel(bloomSampler , float3(position / float2(layerWidth , layerHeight) , float(level)) , 0.0f).rgb;
}
#endif
