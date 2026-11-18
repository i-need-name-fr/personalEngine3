#pragma once
// #include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#if defined (MOX_SCENE_BUFFERS_INCLUDED) && MOX_SCENE_BUFFERS_INCLUDED >= 0ul && defined (MOX_SHADER_STRUCTURES)

[[vk::binding(0 , MOX_SCENE_BUFFERS_INCLUDED)]] ConstantBuffer<GeneralInfo> generalBuffer;
[[vk::binding(1 , MOX_SCENE_BUFFERS_INCLUDED)]] ConstantBuffer<GeneralFeatures> features;
[[vk::binding(2 , MOX_SCENE_BUFFERS_INCLUDED)]] ConstantBuffer<SunSource> sunBuffer;
[[vk::binding(3 , MOX_SCENE_BUFFERS_INCLUDED)]] RWStructuredBuffer<float4> SSAO_kernel;
#endif