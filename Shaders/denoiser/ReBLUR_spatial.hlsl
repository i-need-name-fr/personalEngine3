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
#include "../../include/moxSourceCluster/IndirectPixels.hlsl"

[[vk::push_constant]] struct {
    uint indirect;
} push;

// p is a pixel of the image that this pass denoises : of diffuseImage ( the full resolution ) or of indirectImage ( its own resolution ) .
// the G buffer has the full resolution : the pixel of it that p looks at ( the indirect pixel looks at one pixel of its block , the jitter of the frame , the same as the indirect pass and the temporal pass use )
uint2 gBufferPixel(const uint2 p)
{
    if(push.indirect) return indirectToDirectPixel(p , indirectJitter(p , generalBuffer.frameIndex));
    return p;
}

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
    if(push.indirect) indirectImage.GetDimensions(width, height);
    else diffuseImage.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    // the pixel of the G buffer of the centre , and how many pixels of this image fit into one pixel of the full resolution ( 1 for the diffuse )
    const uint2 gCenter = gBufferPixel(id.xy);
    const float resolutionScale = float(width) / float(directSize().x);

    const float4 temporal = tmpDiffuseImage[id.xy];

    if(pixelCheck[gCenter] == SKYBOX_PIXEL){
        // temporal wrote the count 0 for the sky : it goes to the history , so the next frame starts again .
        // the COLOUR of the sky is in diffuseImage ( RayGen wrote it there ) : it is not touched , writing temporal.xyz ( 0 ) here made the sky black
        if(push.indirect) diffuseHistoryImage[id.xy] = float4(diffuseHistoryImage[id.xy].r , 0.0f , 0.0f , 0.0f);
        else diffuseHistoryImage[id.xy] = float4(0.0f , diffuseHistoryImage[id.xy].g , 0.0f , 0.0f);
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
    // the radius is in the pixels of the full resolution , the offsets are in the pixels of THIS image : the indirect pixel is wider
    float radius = min(ComputeBlurRadius(NormalizeHitDistance(hitDistance , depthImage[gCenter] , 0) , historyCount ) , MAX_RADIUS_PIXELS) * resolutionScale;

    float3 centerNormal = normalImage[gCenter].xyz;
    float centerDepth = depthImage[gCenter];

    float3 sum = center;
    float weight= 1.0f;

    [unroll]
    for(uint i = 0 ; i < 32 ; i++){
        uint seed = id.x * 3942u  + id.y * 32849u + 239u * i;
        const uint random = randomNumberIn(seed , 16);
        float2 offset = PoissonDisk[random] * radius;
        int2 samplePos = int2(id.xy) + int2(offset);
        if(any(samplePos < int2(0 , 0)) || any(samplePos >= int2(screenSize))) continue;

        // the G buffer of the neighbour : the pixel that THAT pixel of the image looks at
        const uint2 gSample = gBufferPixel(uint2(samplePos));
        float sDepth = depthImage[gSample];
        float3 sNormal = normalImage[gSample].xyz;

        float depthW = exp(-abs(centerDepth - sDepth) /centerDepth * DEPTH_SIGMA);
        float normalW = pow(saturate(dot(sNormal , centerNormal)) , NORMAL_POWER);

        float w = depthW * normalW;
        sum += tmpDiffuseImage[samplePos].xyz * w;
        weight += w;
    }

    writeResult(id.xy , sum / weight , historyCount);
}
