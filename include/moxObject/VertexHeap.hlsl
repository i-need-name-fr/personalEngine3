#pragma once

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

#endif