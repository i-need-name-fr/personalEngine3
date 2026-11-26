#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#define MOX_BLOOM_SOURCE_INCLUDED 2
#include "../../include/moxSourceCluster/BloomSource.hlsl"

void writeColour(const uint2 pixel , const float4 colour)
{
    finalImage[pixel] = colour;
    guiImage[pixel] = colour;
}

// a linear value of a stage for the eyes : without the exposure , clamped , with the gamma ( a value of 0.05 is visible , a NaN / Inf is shown MAGENTA , a negative one BLUE )
float4 showLinear(const float3 value)
{
    if(any(isnan(value)) || any(isinf(value))) return float4(1.0f , 0.0f , 1.0f , 1.0f);
    if(any(value < 0.0f)) return float4(0.0f , 0.0f , 1.0f , 1.0f);
    return float4(pow(saturate(value) , 1.0f / 2.2f) , 1.0f);
}

const static float adaptationLight = 3.0f;
const static float adaptationNight = 1.5f;


float3 ACESFilm(float3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("compute")]
[numthreads(8, 8 ,1)]
void main(uint3 id : SV_DispatchThreadID , uint localIndex : SV_GroupIndex){
    uint width , height;
    DLSSImage.GetDimensions(width , height);

    if(id.x >= width || id.y >= height) return;

    // the output ( DLSSImage , finalImage ) has the size of the window , the g buffer and the rest have the size of the render : the pixel of the render that this pixel of the output covers
    const uint2 renderSize = uint2(generalBuffer.resolutionWidth , generalBuffer.resolutionHeight);
    const uint2 rid = min(uint2(float2(id.xy) * float2(renderSize) / float2(width , height)) , renderSize - 1u);

    if(features.currentMode == ALBEDO_MODE){
        // albedo image is surely rgba8 , since its diretly from textures
        writeColour(id.xy , float4( albedoImage[rid].xyz, 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == NORMAL_MODE){
        float3 normal = normalImage[rid].xyz;
        normal = normal * 0.5f + 0.5f;
        writeColour(id.xy , float4( normal, 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == METALLIC_MODE){
        const float metallic = metallicImage[rid].x;
        writeColour(id.xy , float4( metallic , metallic , metallic , 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == ROUGHNESS_MODE){
        const float rough = roughnessImage[rid].x;
        writeColour(id.xy , float4(rough , rough , rough , 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == DEPTH_MODE){
        // the depth of the g buffer ( the distance , meters ) : near is white
        const float depth = depthImage[rid];
        writeColour(id.xy , showLinear(float3(1.0f , 1.0f , 1.0f) * saturate(1.0f - depth / 100.0f)));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == DIFFUSE_MODE){
        writeColour(id.xy , showLinear(diffuseImage[rid].xyz * albedoImage[rid].xyz));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == SPECULAR_MODE){
        writeColour(id.xy , showLinear(specularImage[rid].xyz));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == HDR_RAW_MODE){
        writeColour(id.xy , showLinear(DLSSImage[id.xy].xyz));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == HISTORY_COUNT_MODE){
        const float2 counts = diffuseHistoryImage[rid].rg / 64.0f;
        writeColour(id.xy , float4(saturate(counts.x) , saturate(counts.y) , 0.0f , 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == HISTORY_TEST_MODE){
        const uint2 size = uint2(generalBuffer.resolutionWidth , generalBuffer.resolutionHeight);
        const float2 uvT = (float2(rid) + 0.5f) / float2(size);
        const float2 prevUVT = uvT + motionVector[rid];
        const uint2 prevCoordT = uint2(round(clamp(prevUVT * float2(size) - 0.5f , float2(0.0f , 0.0f) , float2(size) - 1.0f)));
        const bool onScreen = !(any(prevUVT > float2(1.0f , 1.0f)) || any(prevUVT < float2(0.0f , 0.0f)));
        const float4 prevPos = prevPositionImage[prevCoordT];
        const bool samePos = prevPos.w != 0.0f && length(positionImage[rid].xyz - prevPos.xyz) <= 0.05f;
        const bool sameNormal = dot(normalImage[rid].xyz , prevNormalImage[prevCoordT].xyz) >= 0.9f;
        writeColour(id.xy , float4(onScreen ? 1.0f : 0.0f , samePos ? 1.0f : 0.0f , sameNormal ? 1.0f : 0.0f , 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == OBJECT_AO_MODE){
        const float objectAO = objectAOImage[rid].x;
        writeColour(id.xy , showLinear(float3(objectAO , objectAO , objectAO)));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == AO_MODE){
        const float ao = fAmbientImage[rid].x;
        writeColour(id.xy , float4(ao , ao , ao , 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == SHADOW_MODE){
        const float shadow = fShadowImage[rid].x;
        writeColour(id.xy , (shadow > 1.001f) ? float4(1.0f , 0.0f , 0.0f , 1.0f) : float4(shadow , shadow , shadow , 1.0f));
        pixelCheck[rid] = UNDEFINED_PIXEL;
        return;
    }

    const float currentEV = EVBuffer[0];
    const float newEV = EVBuffer[0];
    const float exposure = exp2(newEV) * generalBuffer.exposure;      
    const float gamma = 2.2f;
    float3 bloom = bloomImages[uint3(rid , 0)].xyz * (features.bloomRequired ? bloomBuffer.intensity : 0.0f);
    if(any(!isfinite(bloom))) bloom = 0.0f;
    const float3 hdr = (DLSSImage[id.xy].xyz + bloom) * exposure;
    const float3 aces = ACESFilm(hdr);

    const float3 result = pow(aces , 1.0f / gamma);

    pixelCheck[rid] = UNDEFINED_PIXEL;
    writeColour(id.xy , float4(result ,1.0f));
}