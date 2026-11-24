#pragma once
// #include "../../include/moxSourceCluster/VolSource.hlsl"
#if defined (MOX_VOL_SOURCE_INCLUDED) && MOX_VOL_SOURCE_INCLUDED >= 0ul && defined (MOX_SHADER_STRUCTURES)

[[vk::binding(0 , MOX_VOL_SOURCE_INCLUDED)]] ConstantBuffer<FogVolume> fogBuffer;
[[vk::binding(1 , MOX_VOL_SOURCE_INCLUDED)]] Texture2DArray txFogImageArray;
[[vk::image_format("rgba16f")]]
[[vk::binding(2 , MOX_VOL_SOURCE_INCLUDED)]] RWTexture2DArray<float4> RWFogImageArray;
// the volume of the LAST frame ( the temporal accumulation of Volumentric )
[[vk::binding(3 , MOX_VOL_SOURCE_INCLUDED)]] Texture2DArray txFogHistory;

// the density ( the extinction sigmaT ) of the fog in the world point : it is NOT stored in the volume ( the alpha of the volume is the count of the accumulated frames ) , both Volumentric and PostProcess take it from here
float fogDensityAt(const float3 position){
    return fogBuffer.density * exp(-fogBuffer.heightFallOff * max(position.y - fogBuffer.fogBaseHeight , 0.0f));
}

#endif
