// ReLAX temporal for the specular , full size
//
// in  : specularImage [ current ] ( the raw signal ) , specularDistance [ current ] ( the hit distance ) , specularImage [ prev ] ( the filtered result of the last frame = the history )
//       specularHistoryImage.r = the frame count ( written by the spatial pass of the last frame )
// out : tmpSpecularImage : .rgb = the accumulated signal , .a = the frame count of this frame ( the spatial pass reads it )

#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"

static const float MAX_ACCUM_FRAMES = 64.0f;

float3 ComputeVirtualPosition(float3 worldPos, float3 normal, float3 viewDir, float hitDistance) {
    // the VIRTUAL position : the image of the reflected point behind the surface ( the ray of the camera goes on through the surface for the distance of the reflected ray ) .
    // reflect( reflectDir , normal ) is the ray of the camera again : the image of the hit point H = P + R * d in the plane of the surface is P - V * d .
    // for a flat mirror this point does not move with the camera : this is what the history is looked for by ( the real point H is not the one that the pixel shows )
    return worldPos - viewDir * hitDistance;
}

float2 ComputeVirtualMotionVector(float3 currentVirtualPos, float4x4 prevViewProj, float2 currentUV) {
    float4 prevClip = mul(prevViewProj, float4(currentVirtualPos, 1.0));
    // behind the previous camera : outside of the screen , the caller drops the history
    if(prevClip.w <= 0.0f) return currentUV - float2(-2.0f , -2.0f);

    float2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
    // the same flip as the other passes ( ndc y goes up , the uv goes down )
    prevUV.y = 1.0f - prevUV.y;
    return currentUV - prevUV;
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
[numthreads(8 ,8 ,1)]
void main(uint3 id : SV_DispatchThreadID){
    uint width , height;
    specularImage.GetDimensions(width, height);

    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    if(pixelCheck[id.xy] == SKYBOX_PIXEL ){
        // the spatial pass copies this count into the history
        tmpSpecularImage[id.xy] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        return;
    }

    if(!generalBuffer.ReLAX){
        return;
    }

    float2 uv = (float2(id.xy) + 0.5f ) / float2(screenSize);
    float3 current = specularImage[id.xy].xyz;
    // the input is cleaned before it can get into the history
    if(any(isnan(current)) || any(isinf(current))) current = float3(0.0f , 0.0f , 0.0f);
    float roughness = roughnessImage[id.xy];
    float3 normal = normalImage[id.xy].xyz;
    float3 V = viewImage[id.xy].xyz;

    float3 P = positionImage[id.xy].xyz;
    float hitDistance = specularDistance[id.xy];

    float3 virtualPos = ComputeVirtualPosition(P , normal , V ,hitDistance );
    float2 virtualUV = ComputeVirtualMotionVector(virtualPos , generalBuffer.prevViewProj , uv);
    float2 prevUV = uv - virtualUV;

    const uint2 prevCoord = uint2(round(clamp(prevUV * float2(screenSize) - 0.5f , float2(0.0f , 0.0f) , float2(screenSize) - 1.0f)));

    bool valid = all(prevUV >= float2(0.0f , 0.0f)) && all(prevUV <= float2(1.0f , 1.0f));
    valid = valid && checkDisocclusion(id.xy , prevCoord);

    float pCount = 0.0f;
    float3 history = current;

    if(valid){
        pCount = specularHistoryImage[prevCoord].r;
        history = prevSpecularImage[prevCoord].xyz;
        // a NaN / Inf in the history lives forever ( the output of this frame is the history of the next one ) : it is dropped , the pixel starts again
        if(any(isnan(history)) || any(isinf(history))){
            history = current;
            pCount = 0.0f;
        }
    }

    pCount = min(pCount + 1.0f , MAX_ACCUM_FRAMES);
    float alpha = 1.0f / pCount;
    float3 accumulated = lerp(history , current , alpha);
    tmpSpecularImage[id.xy] = float4(accumulated , pCount);
}
