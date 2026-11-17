#pragma once
// #include "../../include/moxSourceCluster/VolSource.hlsl"
#if defined (MOX_VOL_SOURCE_INCLUDED) && MOX_VOL_SOURCE_INCLUDED >= 0ul && defined (MOX_SHADER_STRUCTURES)

[[vk::binding(0 , MOX_VOL_SOURCE_INCLUDED)]] ConstantBuffer<FogVolume> fogBuffer;
[[vk::binding(1 , MOX_VOL_SOURCE_INCLUDED)]] Texture2DArray txFogImageArray;
[[vk::image_format("rgba16f")]]
[[vk::binding(2 , MOX_VOL_SOURCE_INCLUDED)]] RWTexture2DArray<float4> RWFogImageArray;

#endif
