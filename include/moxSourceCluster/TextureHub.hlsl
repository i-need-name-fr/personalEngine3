#pragma once
// #include "../../include/moxSourceCluster/TextureHub.hlsl"
#if defined(MOX_TEXTURE_HUB_INCLUDED) && MOX_TEXTURE_HUB_INCLUDED >= 0l

[[vk::binding(0 , MOX_TEXTURE_HUB_INCLUDED)]] RWStructuredBuffer<uint> txVirtualBuffer;
[[vk::binding(1 , MOX_TEXTURE_HUB_INCLUDED)]] SamplerState txSampler;
[[vk::binding(2 , MOX_TEXTURE_HUB_INCLUDED)]] Texture2D textureHub[];

    #ifdef MOX_VERTEX_HEAP_INCLUDED
        #ifndef MOX_CHECK_TRANS_HIT_INCLUDED
        #define MOX_CHECK_TRANS_HIT_INCLUDED

// return true if it passes
bool checkTransHit(const float2 barycentrics, const uint primitive , const uint instanceID)
{
    const InstanceData instance = instanceLayoutBuffer[instanceID];
    const LUT lut = LUT_Buffer[instance.LUT_index];
    // the materials without the test and without a map : everything passes
    if(lut.alphaMode == 0 || lut.diffuseID == NON_VALID_SLOT) return true;

    const MeshData mesh = meshBuffer[instance.meshDataIndex];
    const uint base = mesh.firstIndex + primitive * 3u;
    const uint i0 = mesh.firstVertex + indices[base + 0u];
    const uint i1 = mesh.firstVertex + indices[base + 1u];
    const uint i2 = mesh.firstVertex + indices[base + 2u];

    const float w0 = 1.0f - barycentrics.x - barycentrics.y;
    const float2 uv = vertices[i0].UV * w0 + vertices[i1].UV * barycentrics.x + vertices[i2].UV * barycentrics.y;

    const uint id = txVirtualBuffer[lut.diffuseID];
    const float alpha = textureHub[id].SampleLevel(txSampler , uv , 0).a;
    return alpha >= lut.alphaCutoff;
}

        #endif
    #endif

#endif