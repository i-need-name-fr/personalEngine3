#pragma once
//  #include "../../include/moxObject/VertexHeap.hlsl"
#if defined (MOX_VERTEX_HEAP_INCLUDED) && MOX_VERTEX_HEAP_INCLUDED >= 0 && defined(MOX_SHADER_STRUCTURES)

[[vk::binding(0 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(1 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<uint> indices;
[[vk::binding(2 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<LUT> LUT_Buffer;
[[vk::binding(3 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<MeshData> meshBuffer;
[[vk::binding(4 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<float4x4> instanceBuffer;
[[vk::binding(5 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<InstanceData> instanceLayoutBuffer;
[[vk::binding(6 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<uint> mVertices;
[[vk::binding(7 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<uint> mIndices;
[[vk::binding(8 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<ms::GeometryData> geometryBuffer;
[[vk::binding(9 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<ms::Meshlet> meshlets;
[[vk::binding(10 , MOX_VERTEX_HEAP_INCLUDED)]] RWStructuredBuffer<ms::MeshletBound> boundBuffer;
[[vk::binding(11 , MOX_VERTEX_HEAP_INCLUDED)]] RaytracingAccelerationStructure TLAS;


    #ifdef MOX_TEXTURE_HUB_INCLUDED
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