#pragma once
// #include "../../include/moxSourceCluster/AtmoScat.hlsl"
#if defined(MOX_SHADER_STRUCTURES) && defined(MOX_ATMO_SCAT_INCLUDED) && MOX_ATMO_SCAT_INCLUDED >= 0

[[vk::image_format("rgba16f")]]
[[vk::binding(  0   ,   MOX_ATMO_SCAT_INCLUDED  )]] RWTexture2D<float4> transmissionLUT;
[[vk::image_format("rgba16f")]] 
[[vk::binding(  1   ,   MOX_ATMO_SCAT_INCLUDED  )]] RWTexture2D<float4> multiscatterLUT;
[[vk::image_format("rgba16f")]]
[[vk::binding(  2   ,   MOX_ATMO_SCAT_INCLUDED  )]] RWTexture2D<float4> skyViewLUT;
[[vk::image_format("rgba16f")]]
[[vk::binding(  3   ,   MOX_ATMO_SCAT_INCLUDED  )]] RWTexture2DArray<float4> RWSkybox;
[[vk::binding(  4   ,   MOX_ATMO_SCAT_INCLUDED  )]] TextureCube skybox;
[[vk::image_format("rgba16f")]]
[[vk::binding(  5   ,   MOX_ATMO_SCAT_INCLUDED  )]] RWTexture2DArray<float4> RWIBL;
[[vk::binding(  6   ,   MOX_ATMO_SCAT_INCLUDED  )]] TextureCube IBL;
[[vk::binding(  7   ,   MOX_ATMO_SCAT_INCLUDED  )]] Texture2D txTransmissionLUT;
[[vk::binding(  8   ,   MOX_ATMO_SCAT_INCLUDED  )]] Texture2D txMultiScatterLUT;
[[vk::binding(  9   ,   MOX_ATMO_SCAT_INCLUDED  )]] Texture2D txSkyViewLUT;
[[vk::binding(  10   ,   MOX_ATMO_SCAT_INCLUDED  )]] SamplerState acSamp;
[[vk::binding(  11   ,   MOX_ATMO_SCAT_INCLUDED  )]] ConstantBuffer<SkyViewData> skyBuffer;


#endif