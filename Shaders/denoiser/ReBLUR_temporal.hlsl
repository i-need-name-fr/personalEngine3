// ReBLUR temporal , one shader for the diffuse and for the indirect ( push.indirect ) , full size
//
// in  : diffuseImage / indirectImage [ current ] ( the raw signal ) , the same [ prev ] ( the filtered result of the last frame = the history )
//       diffuseHistoryImage : .r = the frame count of the diffuse , .g = the frame count of the indirect ( written by the spatial pass of the last frame )
// out : tmpDiffuseImage : .rgb = the accumulated signal , .a = the frame count of this frame ( the spatial pass reads it )
//
// no resource ( image ) is ever selected through a variable : SPIR-V can not do it , the PIXEL is selected in the functions below

static const float MAX_ACCUM_FRAMES = 64.0f;

#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"

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

bool checkDisocclusion(uint2 currentPixel , uint2 prevPixel){

    const float3 centerPosition = positionImage[currentPixel].xyz;
    const float3 centerNormal = normalImage[currentPixel].xyz;

    const float4 prevPositionSample = prevPositionImage[prevPixel];
    const float3 nNormal = prevNormalImage[prevPixel].xyz;

    // w == 0 marks a sky texel written by the miss shader
    if(prevPositionSample.w == 0.0f) return false;

    const float positionDiff = length(centerPosition - prevPositionSample.xyz);
    if(positionDiff > 0.05f) return false;

    const float normalDot = dot(centerNormal , nNormal);
    if(normalDot < 0.9f) return false;

    return true;
}

[shader("compute")]
[numthreads(8, 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){

    uint width , height;
    diffuseImage.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    if(pixelCheck[id.xy] == SKYBOX_PIXEL){
        // the spatial pass copies this count into the history
        tmpDiffuseImage[id.xy] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        return;
    }

    if(!generalBuffer.ReBLUR){
        return;
    }

    float2 uv = (float2(id.xy) + 0.5f ) / float2(screenSize);
    float2 prevUV = uv + motionVector[id.xy];
    // the pixel of the last frame : ROUNDED to the nearest one . the plain conversion to uint cut the number off ( always towards the top left ) , so every frame the history
    // moved half a pixel on average and a pixel that did not move at all ( mv = 1e-7 ) fell on its neighbour half of the time : the picture drifted and turned into soap
    const uint2 prevCoord = uint2(round(clamp(prevUV * float2(screenSize) - 0.5f , float2(0.0f , 0.0f) , float2(screenSize) - 1.0f)));

    float3 currentSignal = readCurrent(id.xy);
    float pCount = readHistoryCount(prevCoord);
    float3 history = readHistory(prevCoord);
    bool valid = !(any(prevUV > float2(1.0f , 1.0f)) || any(prevUV < float2(0.0f , 0.0f))) && checkDisocclusion(id.xy , prevCoord);

    if(!valid){
        pCount = 0.0f;
        history = currentSignal;
    }

    pCount = min(pCount + 1.0f , MAX_ACCUM_FRAMES);
    float alpha = 1.0f / pCount;
    float3 accumulated = lerp(history , currentSignal , alpha);
    tmpDiffuseImage[id.xy] = float4(accumulated , pCount);
}
