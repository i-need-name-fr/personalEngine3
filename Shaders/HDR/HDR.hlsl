#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"

void writeColour(const uint2 pixel , const float4 colour)
{
    finalImage[pixel] = colour;
    guiImage[pixel] = colour;
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
    HDRImage.GetDimensions(width , height);

    if(id.x >= width || id.y >= height) return;

    if(features.currentMode == ALBEDO_MODE){
        // albedo image is surely rgba8 , since its diretly from textures
        writeColour(id.xy , float4( albedoImage[id.xy].xyz, 1.0f));
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == NORMAL_MODE){
        float3 normal = normalImage[id.xy].xyz;
        normal = normal * 0.5f + 0.5f;
        writeColour(id.xy , float4( normal, 1.0f));
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == METALLIC_MODE){
        const float metallic = metallicImage[id.xy].x;
        writeColour(id.xy , float4( metallic , metallic , metallic , 1.0f));
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == ROUGHNESS_MODE){
        const float rough = roughnessImage[id.xy].x;
        writeColour(id.xy , float4(rough , rough , rough , 1.0f));
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == AO_MODE){
        const float ao = fAmbientImage[id.xy].x;
        writeColour(id.xy , float4(ao , ao , ao , 1.0f));
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }else if(features.currentMode == SHADOW_MODE){
        const float shadow = fShadowImage[id.xy].x;
        writeColour(id.xy , (shadow > 1.001f) ? float4(1.0f , 0.0f , 0.0f , 1.0f) : float4(shadow , shadow , shadow , 1.0f));
        pixelCheck[id.xy] = UNDEFINED_PIXEL;
        return;
    }

    const float currentEV = EVBuffer[0];
    const float newEV = EVBuffer[0];
    const float exposure = exp2(newEV) * generalBuffer.exposure;      
    const float gamma = 2.2f;

    const float3 hdr = HDRImage[id.xy].xyz * exposure;
    const float3 aces = ACESFilm(hdr);

    const float3 result = pow(aces , 1.0f / gamma);

    pixelCheck[id.xy] = UNDEFINED_PIXEL;
    writeColour(id.xy , float4(result ,1.0f));
}