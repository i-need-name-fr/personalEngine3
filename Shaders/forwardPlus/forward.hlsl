#include "../../include/ShaderStructures/ShaderStructure.h"

#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"

#define MOX_LIGHT_HUB_INCLUDED 1
#include "../../include/moxSourceCluster/LightHub.hlsl"

float3 ScreenToViewDirection(float2 ndc , float4x4 invProj){
    float4 viewSpacePos = mul(invProj , float4(ndc , 1.0f , 1.0f));
    viewSpacePos /= viewSpacePos.w;
    return normalize(viewSpacePos.xyz);
}

float4 computePlane(float3 p0 , float3 p1 , float3 p2){
    float3 v1 = p1 - p0;
    float3 v2 = p2 - p0;
    float3 normal = normalize(cross(v1 , v2));
    float d = -dot(normal , p0);
    return float4(normal ,d);
}

struct TileFrustum{
    float4 planes[4];
};

TileFrustum buildTileFrustum(uint2 tileCoord , uint2 tileSize , float4x4 invProj){
    float2 screenSize = float2(generalBuffer.screenWidth , generalBuffer.screenHeight);

    // the tile in pixels , the last row / column of tiles is cut by the edge of the screen.
    // this used to be ( tileCoord + tileSize ) / screenSize - a tile index plus a tile size in pixels ,
    // not the pixel position of the tile at all
    const float2 pixelMin = float2(tileCoord * tileSize);
    const float2 pixelMax = min(pixelMin + float2(tileSize) , screenSize);

    // pixel ( 0 , 0 ) is the top left corner , but NDC y points up ( the engine flips it with the
    // negative viewport height ) : the BOTTOM edge of the tile ( pixelMax.y ) is the smaller NDC y
    float2 tileMinNDC = float2(pixelMin.x / screenSize.x * 2.0f - 1.0f , 1.0f - pixelMax.y / screenSize.y * 2.0f);
    float2 tileMaxNDC = float2(pixelMax.x / screenSize.x * 2.0f - 1.0f , 1.0f - pixelMin.y / screenSize.y * 2.0f);

    float3 corners[4];
    corners[0] = ScreenToViewDirection(float2(tileMinNDC.x, tileMinNDC.y), invProj);
    corners[1] = ScreenToViewDirection(float2(tileMaxNDC.x, tileMinNDC.y), invProj);
    corners[2] = ScreenToViewDirection(float2(tileMaxNDC.x, tileMaxNDC.y), invProj);
    corners[3] = ScreenToViewDirection(float2(tileMinNDC.x, tileMaxNDC.y), invProj);

    TileFrustum frustum;
    // каждая плоскость строится через ДВА соседних угла и начало координат (позиция камеры в view space = 0,0,0)
    frustum.planes[0] = computePlane(float3(0,0,0), corners[0], corners[3]);   // левая грань
    frustum.planes[1] = computePlane(float3(0,0,0), corners[2], corners[1]);   // правая грань
    frustum.planes[2] = computePlane(float3(0,0,0), corners[1], corners[0]);   // нижняя грань ( corners[0..1] - нижний край тайла )
    frustum.planes[3] = computePlane(float3(0,0,0), corners[3], corners[2]);   // верхняя грань ( corners[2..3] - верхний край тайла )
    return frustum;
}

bool SphereIntersectsTileFrustum(float3 lightPosViewSpace, float radius, TileFrustum frustum)
{
    if(lightPosViewSpace.z - radius > 0.0f)
        return false;

    [unroll]
    for (int i = 0; i < 4; i++)   // только 4 плоскости, не 6 - тайл не ограничен по глубине
    {
        float distance = dot(frustum.planes[i].xyz, lightPosViewSpace) + frustum.planes[i].w;
        if (distance < -radius)
            return false;
    }
    return true;
}

// the planes of the tile this workgroup works on , built once by its first thread
groupshared float4 tilePlanes[4];

[shader("compute")]
[numthreads(256 , 1 , 1)]
void main(uint3 groupId : SV_GroupID , uint localIndex : SV_GroupIndex){

    const uint tileIndex = groupId.x + groupId.y * forwardInfo.TILE_PER_ROW;

    // depends on the group only , so every thread of the group leaves before the barrier together
    if(groupId.x >= forwardInfo.TILE_PER_ROW || tileIndex >= forwardInfo.TILE_TOTAL_AMOUNT) return;

    if(localIndex == 0){
        const TileFrustum built = buildTileFrustum(groupId.xy , uint2(forwardInfo.TILE_SIZE , forwardInfo.TILE_SIZE) , generalBuffer.inverseProj);
        [unroll]
        for(uint p = 0 ; p < 4 ; p++){
            tilePlanes[p] = built.planes[p];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    TileFrustum frustum;
    [unroll]
    for(uint p = 0 ; p < 4 ; p++){
        frustum.planes[p] = tilePlanes[p];
    }

    for(uint lightIndex = localIndex ; lightIndex < forwardInfo.totalLights ; lightIndex += 256){

        const lg::LightSource light = lightBuffer[lightIndex];
        const float3 lightPosViewSpace = mul(generalBuffer.viewOnly , float4(light.position , 1.0f)).xyz;
        // the radius of the test is the radius of the light plus the fade of its shadow ( lightShadowReach ) : the shadow still counts there
        if(!SphereIntersectsTileFrustum(lightPosViewSpace , light.radius * (1.0f + max(forwardInfo.lightShadowFade , 0.0f)) , frustum)) continue;

        uint writeIndex;
        InterlockedAdd(lightCountPerTile[tileIndex] , 1 , writeIndex);
        // past MAX_LIGHTS this would write into the next tile's slots
        if(writeIndex < MAX_LIGHTS){
            tileLightIndices[tileIndex * MAX_LIGHTS + writeIndex] = lightIndex;
        }
    }
}