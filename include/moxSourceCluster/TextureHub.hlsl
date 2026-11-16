#pragma once
#if defined(MOX_TEXTURE_HUB_INCLUDED) && MOX_TEXTURE_HUB_INCLUDED >= 0l

[[vk::binding(0 , MOX_TEXTURE_HUB_INCLUDED)]] RWStructuredBuffer<uint> txVirtualBuffer;
[[vk::binding(1 , MOX_TEXTURE_HUB_INCLUDED)]] SamplerState txSampler;
[[vk::binding(2 , MOX_TEXTURE_HUB_INCLUDED)]] Texture2D textureHub[];

#endif