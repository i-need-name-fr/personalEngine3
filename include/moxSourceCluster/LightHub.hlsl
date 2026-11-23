#pragma once
// #include "../../include/moxSourceCluster/LightHub.hlsl"
#if defined(MOX_LIGHT_HUB_INCLUDED) && MOX_LIGHT_HUB_INCLUDED >= 0u && defined(MOX_SHADER_STRUCTURES)
[[vk::binding(0 , MOX_LIGHT_HUB_INCLUDED)]] RWStructuredBuffer<lg::LightSource> lightBuffer;
[[vk::binding(1 , MOX_LIGHT_HUB_INCLUDED)]] RWStructuredBuffer<lg::LightCell> lightCells;
[[vk::binding(2 , MOX_LIGHT_HUB_INCLUDED)]] ConstantBuffer<lg::GridCascadeInfo> gridInfo;
[[vk::binding(3 , MOX_LIGHT_HUB_INCLUDED)]] RWStructuredBuffer<uint> tileLightIndices;
[[vk::binding(4,  MOX_LIGHT_HUB_INCLUDED)]] RWStructuredBuffer<uint> lightCountPerTile;
[[vk::binding(5 , MOX_LIGHT_HUB_INCLUDED)]] ConstantBuffer<ForwardPlusInfo> forwardInfo;

inline float3 getGridOrigin(const float3 playerPosition) {
    const float3 snapped = floor(playerPosition / lg::GRID_CELL_SIZE) * lg::GRID_CELL_SIZE;
    return snapped - float3(lg::GRID_SIZE * 0.5f * lg::GRID_CELL_SIZE, lg::GRID_SIZE * 0.5f * lg::GRID_CELL_SIZE,lg::GRID_SIZE * 0.5f * lg::GRID_CELL_SIZE);
}

inline float3 translateToGridPos(const float3 position , const float3 playerPosition) {
    return position - getGridOrigin(playerPosition);
}

inline int32_t translatePosToCellIndex(const float3 gridPos) {
    const float3 cell = floor(gridPos / lg::GRID_CELL_SIZE);
    const float n = lg::GRID_SIZE;
    if(cell.x < 0.0f || cell.y < 0.0f || cell.z < 0.0f || cell.x >= n || cell.y >= n || cell.z >= n) return NON_VALID_SLOT;

    const int32_t count = int(lg::GRID_SIZE);
    return int32_t(cell.x) + count * (int32_t(cell.y) + count * int32_t(cell.z));
}

#ifdef MOX_SCENE_BUFFERS_INCLUDED
int findGridSlot(const float3 P){
    const float3 gridPos = translateToGridPos(P , generalBuffer.playerPosition);
    return translatePosToCellIndex(gridPos);
}

void findLightIndices(const float3 P , out uint lightIndices[lg::MAX_LIGHT_PER_CELL] ,out uint lightCount){
    lightCount = 0;
    const int gridID = findGridSlot(P);
    if(gridID == NON_VALID_SLOT) return;

    const lg::LightCell cell = lightCells[gridID];
    lightCount = min(cell.totalLightCount , (uint)lg::MAX_LIGHT_PER_CELL);
    lightIndices = cell.lightIndices;
}
#endif

void findLightIndices(uint2 currentPixel , out uint firstLight , out uint lightCount){
    firstLight = 0;
    lightCount = 0;

    uint2 tileCoord = currentPixel / forwardInfo.TILE_SIZE;
    uint tileIndex = tileCoord.x + tileCoord.y * forwardInfo.TILE_PER_ROW;

    // forward.hlsl's InterlockedAdd keeps counting past MAX_LIGHTS (it only stops WRITING there)
    lightCount = min(lightCountPerTile[tileIndex] , (uint)MAX_LIGHTS);
    firstLight = tileIndex * MAX_LIGHTS;
}

void diskBasis(float3 N, out float3 U, out float3 V)
{
    float s = (N.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (s + N.z);
    float b = N.x * N.y * a;
    U = float3(1.0f + s * N.x * N.x * a, s * b, -s * N.x);
    V = float3(b, s + N.y * N.y * a, -N.y);
}

// float3 pointU = center + U * radius;   // край диска в сторону U
// float3 pointV = center + V * radius;   // край диска в сторону V

// float r   = radius * sqrt(xi.x);
// float phi = 2.0f * PI * xi.y;
// float3 P  = center + U * (r * cos(phi)) + V * (r * sin(phi));

#endif