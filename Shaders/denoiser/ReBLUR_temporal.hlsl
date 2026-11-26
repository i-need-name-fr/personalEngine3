static const float MAX_ACCUM_FRAMES = 64.0f;
static const float TRAIN_MAX_ACCUM_FRAMES = 512.0f;

#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#include "../../include/moxSourceCluster/IndirectPixels.hlsl"

[[vk::push_constant]] struct {
    uint indirect;
} push;

float3 readCurrent(uint2 p)
{
    if(push.indirect) return indirectImage[p].xyz;
    return diffuseImage[p].xyz;
}

float3 readHistory(uint2 p)
{
    if(push.indirect) return prevIndirectImage[p].xyz;
    return prevDiffuseImage[p].xyz;
}

float readHistoryCount(uint2 p)
{
    if(push.indirect) return diffuseHistoryImage[p].g;
    return diffuseHistoryImage[p].r;
}

static const float DISOCCLUSION_MIN_DISTANCE = 0.05f;
static const float DISOCCLUSION_RELATIVE = 0.02f;

bool checkDisocclusion(uint2 currentPixel , uint2 prevPixel){

    const float3 centerPosition = positionImage[currentPixel].xyz;
    const float3 centerNormal = normalImage[currentPixel].xyz;

    const float4 prevPositionSample = prevPositionImage[prevPixel];
    const float3 nNormal = prevNormalImage[prevPixel].xyz;

    // w == 0 marks a sky texel written by the miss shader
    if(prevPositionSample.w == 0.0f) return false;

    const float viewDistance = length(centerPosition - generalBuffer.playerPosition);
    const float planeDistance = abs(dot(prevPositionSample.xyz - centerPosition , centerNormal));
    if(planeDistance > max(DISOCCLUSION_MIN_DISTANCE , viewDistance * DISOCCLUSION_RELATIVE)) return false;

    const float normalDot = dot(centerNormal , nNormal);
    if(normalDot < 0.9f) return false;

    return true;
}

float relMSE(float3 a , float3 ref)
{
    const float3 d = a - ref;
    const float l = luminance(ref);
    return dot(d , d) / (l * l + 0.01f);   // знаменатель только от эталона
}


[shader("compute")]
[numthreads(8, 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;

    const bool indirect = push.indirect;
    if(!indirect)
    {
        diffuseImage.GetDimensions(width, height);
    }
    else
    {
        indirectImage.GetDimensions(width, height);
    }

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    uint2 gImageCoord;
    uint2 gSize;
    if(indirect)
    {
        gImageCoord = indirectToDirectPixel(id.xy , indirectJitter(id.xy , generalBuffer.frameIndex));
        gSize = directSize();
    }
    else
    {
        gImageCoord = id.xy;
        gSize = uint2(width , height);
    }

    if(pixelCheck[gImageCoord] == SKYBOX_PIXEL){
        tmpDiffuseImage[id.xy] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        rawSignalImage[id.xy] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        return;
    }

    if(!generalBuffer.ReBLUR){
        return;
    }

    float2 uv = (float2(id.xy) + 0.5f ) / float2(screenSize);
    float2 prevUV = uv + motionVector[gImageCoord];
    const uint2 prevCoord = uint2(round(clamp(prevUV * float2(screenSize) - 0.5f , float2(0.0f , 0.0f) , float2(screenSize) - 1.0f)));

    float2 gUV = (float2(gImageCoord) + 0.5f ) / float2(gSize);
    float2 gPrevUV = gUV + motionVector[gImageCoord];
    const uint2 gPrevCoord = uint2(round(clamp(gPrevUV * float2(gSize) - 0.5f , float2(0.0f , 0.0f) , float2(gSize) - 1.0f)));

    float3 currentSignal = readCurrent(id.xy);
    float pCount = readHistoryCount(prevCoord);
    float3 history = readHistory(prevCoord);
    bool valid = !(any(gPrevUV > float2(1.0f , 1.0f)) || any(gPrevUV < float2(0.0f , 0.0f))) && checkDisocclusion(gImageCoord , gPrevCoord);

    const bool currentBad = any(!isfinite(currentSignal));
    const bool historyBad = any(!isfinite(history)) || !isfinite(pCount) || pCount < 0.0f;
    if(currentBad) currentSignal = historyBad ? float3(0.0f , 0.0f , 0.0f) : history;
    if(historyBad) { history = currentSignal; pCount = 0.0f; }

    if(!valid){
        pCount = 0.0f;
        history = currentSignal;
    }

    float maxFrames = MAX_ACCUM_FRAMES;
    float realCount;
    if(features.NKPDTrain && generalBuffer.trainingIndex > 0u)
    {
        realCount = min(pCount + 1.0f , maxFrames);
        maxFrames = max(TRAIN_MAX_ACCUM_FRAMES , generalBuffer.trainingIndex + 1u);
        pCount = generalBuffer.trainingIndex;
    }
    else
    {
        pCount = min(pCount + 1.0f , maxFrames);
        realCount = pCount;
    }

    float alpha = 1.0f / pCount;
    float3 accumulated = lerp(history , currentSignal , alpha);
    accumulated = clamp(accumulated , float3(0.0f , 0.0f , 0.0f) , float3(60000.0f , 60000.0f , 60000.0f));
    currentSignal = clamp(currentSignal , float3(0.0f , 0.0f , 0.0f) , float3(60000.0f , 60000.0f , 60000.0f));
    tmpDiffuseImage[id.xy] = float4(accumulated , realCount);
    rawSignalImage[id.xy] = float4(currentSignal , 1.0f);
}
