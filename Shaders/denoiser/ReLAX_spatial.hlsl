// ReLAX spatial for the specular , full size
//
// in  : tmpSpecularImage ( .rgb the temporal result , .a the frame count ) , specularDistance [ current ] ( the hit distance )
// out : specularImage [ current ] ( the filtered result , it is also the history of the next frame )
//       specularHistoryImage.r = the frame count , the temporal pass of the next frame reads it
//
// the spatial pass only READS tmpSpecularImage and only WRITES the images above , so the neighbour pixels never race

#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"

static const float MAX_ACCUM_FRAMES = 32.0f;
static const float MAX_RADIUS_PIXELS = 32.0f;
static const float DEPTH_SIGMA = 8.0f;
static const float LOBE_VOLUME = 0.5f;

float ComputeBlurRadius(float roughness , float hitDistance , float viewDistance , float historyCount) {
    const float alpha = roughness * roughness;
    const float tanHalfAngle = alpha * LOBE_VOLUME / max(1.0f - LOBE_VOLUME , 1e-4f);
    const float pixelAngle = 2.0f / (max(generalBuffer.projOnly[1][1] , 1e-3f) * float(max(generalBuffer.resolutionHeight , 1u)));
    const float spotShare = hitDistance / max(viewDistance + hitDistance , 1e-4f);
    const float baseRadius = tanHalfAngle * spotShare / max(pixelAngle , 1e-6f);
    const float historyFactor = saturate(1.0 - historyCount / MAX_ACCUM_FRAMES);
    return min(baseRadius * lerp(0.25, 1.0, historyFactor) , MAX_RADIUS_PIXELS);
}

float normalPowerForRoughness(float roughness){
    return lerp(256.0f , 16.0 , roughness);
}

uint randomNumberIn(inout uint seed , const uint totalCount){
    return uint(rand(seed) * float(totalCount));
}

[shader("compute")]
[numthreads(8, 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    specularImage.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    const float4 temporal = tmpSpecularImage[id.xy];

    if(pixelCheck[id.xy] == SKYBOX_PIXEL){
        // temporal wrote the count 0 for the sky : it goes to the history , so the next frame starts again
        specularHistoryImage[id.xy] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        return;
    }

    if(!generalBuffer.ReLAX){
        return;
    }

    float3 center = temporal.xyz;
    float historyCount = temporal.a;
    float hitDistance = specularDistance[id.xy];

    // the history is the temporal result in any case
    specularHistoryImage[id.xy] = float4(historyCount , 0.0f , 0.0f , 0.0f);

    if(hitDistance >= 999.0f){
        specularImage[id.xy] = float4(center , 1.0f);
        return;
    }

    float3 centerNormal = normalImage[id.xy].xyz;
    float centerDepth = depthImage[id.xy];
    float roughness = roughnessImage[id.xy];

    // the distance from the camera to the surface along the ray ( not the depth along the axis ) : the mirror is this far from the virtual camera
    const float viewDistance = length(positionImage[id.xy].xyz - generalBuffer.playerPosition);
    const float radius = ComputeBlurRadius(roughness , hitDistance , viewDistance , historyCount);

    float3 sum = center;
    float weight= 1.0f;

    [unroll]
    for(uint i = 0 ; i < 32 ; i++){
        uint seed = id.x * 3942u  + id.y * 32849u + 239u * i;
        const uint random = randomNumberIn(seed , 16);
        float2 offset = PoissonDisk[random] * radius;
        int2 samplePos = int2(id.xy) + int2(offset);
        if(any(samplePos < int2(0 , 0)) || any(samplePos >= int2(screenSize))) continue;

        float sDepth = depthImage[samplePos];
        float3 sNormal = normalImage[samplePos].xyz;

        float depthW = exp(-abs(centerDepth - sDepth) /centerDepth * DEPTH_SIGMA);
        float normalW = pow(saturate(dot(sNormal , centerNormal)) , normalPowerForRoughness(roughness));

        float w = depthW * normalW;
        sum += tmpSpecularImage[samplePos].xyz * w;
        weight += w;
    }

    specularImage[id.xy] = float4(sum / weight , 1.0f);
}
