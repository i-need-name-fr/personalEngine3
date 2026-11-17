struct VsOutput{
    float4 clipPos  : SV_Position;
    [[vk::location(0)]] float3 fragPos                    : TEXCOORD0;
    [[vk::location(1)]] float3 normal                     : TEXCOORD1;
    [[vk::location(2)]] float3x3 TBN                      : TEXCOORD2;
    [[vk::location(5)]] float2 uv                         : TEXCOORD5;
    [[vk::location(6)]] nointerpolation uint LUTindex     : TEXCOORD6;
};

#include "../../include/ShaderStructures/ShaderStructure.h"

#define MOX_TEXTURE_HUB_INCLUDED 0
#include "../../include/moxSourceCluster/TextureHub.hlsl"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 2
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#define MOX_VERTEX_HEAP_INCLUDED 3
#include "../../include/moxObject/VertexHeap.hlsl"

struct PsOutput{
    float4 albedo     : SV_Target0;   // rgba16f
    float4 position   : SV_Target1;   // rgba32f
    float4 normal     : SV_Target2;   // rgba16f
    float  depth      : SV_Target3;   // r32f , linear
    float  roughness  : SV_Target4;   // r32f
    float  metallic   : SV_Target5;   // r32f
    float  pixelCheck : SV_Target6;   // r32f , COLOR_PIXEL where something was drawn
    float  objectAmbient : SV_Target7; // r32f , the ao map of the material ( 1 : there is no map ) , the direct light is multiplied by it
};


float2 calculateParallax(float2 currentUV , float3 V , int depthID){
    if(depthID == NON_VALID_SLOT) return currentUV;
    const uint physicalIndex = txVirtualBuffer[depthID];
    const float depth = textureHub[physicalIndex].SampleLevel(txSampler , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

bool isSpecularMode(const float roughness , const float metallic){ return (metallic >= 0.7f && roughness < 0.4f); }


[shader("pixel")]
PsOutput main(VsOutput input){

    PsOutput output;

    // depth & position
    output.position = float4(input.fragPos, 1.0f);
    output.depth = LinearizeDepth(input.clipPos.z , generalBuffer.playerNear , generalBuffer.playerFar);

    // current LUT
    LUT lut = LUT_Buffer[input.LUTindex];

    const float3 V = normalize(generalBuffer.playerPosition - input.fragPos);
    const float3x3 TBN = input.TBN;
    const float3 V_TBN = mul(TBN , V);
    // parallax
    const float2 uv = calculateParallax(input.uv , V_TBN , lut.depthID);
    // normal
    if(lut.normalID != NON_VALID_SLOT){
        const uint nID = txVirtualBuffer[lut.normalID];
        float3 n;
        n.xy = textureHub[nID].Sample(txSampler , uv).xy * 2.0f - 1.0f;
        n.z = sqrt(saturate(1.0f - dot(n.xy , n.xy)));
        n.y = (generalBuffer.normalMapOpenGL) ? n.y : -n.y;
        n = normalize(mul(n , TBN));
        output.normal = float4(n , 1.0f);
    }else{
        output.normal = float4(input.normal , 1.0f);
    }

    if(lut.diffuseID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.diffuseID];
        output.albedo = textureHub[id].Sample(txSampler , uv);
    }else{
        output.albedo = float4(lut.color , 1.0f);
    }

    if(lut.alphaMode != 0){
        if(output.albedo.a < lut.alphaCutoff) discard;
    }else{
        output.albedo.a = 1.0f;
    }

    if(lut.metallicTextureID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.metallicTextureID];
        output.metallic = textureHub[id].Sample(txSampler , uv).z;
    }else{
        output.metallic = lut.metallic;
    }

    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.roughnessTextureID];
        output.roughness = textureHub[id].Sample(txSampler , uv).y;
    }else{
        output.roughness = lut.roughness;
    }

    if(lut.armTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.armTextureID];
        const float3 arm = textureHub[id].Sample(txSampler, uv).xyz;
        output.roughness = arm.y;
        output.metallic = arm.z;
    }


    float ambientOcclusion = 1.0f;
    if(lut.aoTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.aoTextureID];
        ambientOcclusion = textureHub[id].Sample(txSampler , uv).x;
    }
    output.objectAmbient = ambientOcclusion;

    // a perfect mirror ( roughness 0 ) makes the GGX degenerate : the lower limit is for every path above ( the value , the maps , the ARM )
    output.roughness = max(output.roughness , 0.01f);
    if(isSpecularMode(output.roughness , output.metallic))
    {
        output.albedo = lerp(0.04 , output.albedo , output.metallic);
    }

    output.pixelCheck = COLOR_PIXEL;
    return output;
}
