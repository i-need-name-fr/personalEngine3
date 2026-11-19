// ReBLUR spatial , one shader for the diffuse and for the indirect ( push.indirect ) , full size
//
// in  : tmpDiffuseImage ( .rgb the temporal result , .a the frame count ) , diffuseDistance / indirectDistance [ current ] ( the hit distance )
// out : diffuseImage / indirectImage [ current ] ( the filtered result , it is also the history of the next frame )
//       diffuseHistoryImage : .r ( diffuse ) or .g ( indirect ) = the frame count , the temporal pass of the next frame reads it
//
// the spatial pass only READS tmpDiffuseImage and only WRITES the images above , so the neighbour pixels never race

static const float MAX_ACCUM_FRAMES = 32.0f;
static const float MAX_RADIUS_PIXELS = 32.0f;
static const float MAX_HIT_DISTANCE = 50.0f;
static const float DEPTH_SIGMA = 8.0f;
static const float NORMAL_POWER = 128.0f;

#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"

[[vk::push_constant]] struct {
    uint indirect;
} push;

float readHitDistance(uint2 p)
{
    if(push.indirect) return indirectDistance[p];
    return diffuseDistance[p];
}

void writeResult(uint2 p , float3 value , float count)
{
    if(push.indirect)
    {
        indirectImage[p] = float4(value , 1.0f);
        diffuseHistoryImage[p] = float4(diffuseHistoryImage[p].r , count , 0.0f , 0.0f);
    }
    else
    {
        diffuseImage[p] = float4(value , 1.0f);
        diffuseHistoryImage[p] = float4(count , diffuseHistoryImage[p].g , 0.0f , 0.0f);
    }
}

float ComputeBlurRadius(float normalizedHitDist, float historyCount) {
    float baseRadius = normalizedHitDist * MAX_RADIUS_PIXELS;
    // the longer the history , the smaller the spatial radius is needed
    float historyFactor = saturate(1.0 - historyCount / MAX_ACCUM_FRAMES);
    return baseRadius * lerp(0.25, 1.0, historyFactor); // the radius never falls to 0
}

uint randomNumberIn(inout uint seed , const uint totalCount){
    return uint(rand(seed) * float(totalCount));
}

float NormalizeHitDistance(float hitDistance, float viewZ, float roughness) {
    float scale = lerp(viewZ, viewZ * roughness, roughness); // roughness = 0 for ReBLUR
    return hitDistance / max(scale, 1e-4);
}

[shader("compute")]
[numthreads(8, 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    diffuseImage.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    const float4 temporal = tmpDiffuseImage[id.xy];

    if(pixelCheck[id.xy] == SKYBOX_PIXEL){
        // temporal wrote the count 0 for the sky : it goes to the history , so the next frame starts again
        writeResult(id.xy , temporal.xyz , 0.0f);
        return;
    }

    if(!generalBuffer.ReBLUR){
        return;
    }

    float3 center = temporal.xyz;
    float historyCount = temporal.a;
    float hitDistance = readHitDistance(id.xy);

    hitDistance = min(hitDistance , MAX_HIT_DISTANCE);

    // MAX_RADIUS_PIXELS was only a factor : a big normalized distance gave a radius of hundreds of pixels
    float radius = min(ComputeBlurRadius(NormalizeHitDistance(hitDistance , depthImage[id.xy] , 0) , historyCount ) , MAX_RADIUS_PIXELS);

    float3 centerNormal = normalImage[id.xy].xyz;
    float centerDepth = depthImage[id.xy];

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
        float normalW = pow(saturate(dot(sNormal , centerNormal)) , NORMAL_POWER);

        float w = depthW * normalW;
        sum += tmpDiffuseImage[samplePos].xyz * w;
        weight += w;
    }

    writeResult(id.xy , sum / weight , historyCount);
}
