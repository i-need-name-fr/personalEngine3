#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#define MOX_VOL_SOURCE_INCLUDED 3
#include "../../include/moxSourceCluster/VolSource.hlsl"
#define MOX_TEXTURE_HUB_INCLUDED 4
#include "../../include/moxSourceCluster/TextureHub.hlsl"

static const float FOG_SKY_DISTANCE = 20000.0f;

float3 ComputePixelRayDirection(uint2 pixelCoord, uint2 screenSize, float4x4 invViewProj , float3 playerPosition)
{
    float2 uv = (float2(pixelCoord) + 0.5) / float2(screenSize);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float4 farPoint = mul(invViewProj, float4(ndc, 1.0, 1.0));
    farPoint /= farPoint.w;

    return normalize(farPoint.xyz - playerPosition);
}

float fogSliceDepth(const float k , const float sliceCount){
    return FOG_NEAR * pow(FOG_FAR / FOG_NEAR , k / sliceCount);
}

float ComputeHeightFog(float3 origin , float3 direction , float rayLength){
    const float falloff = fogBuffer.heightFallOff;
    const float density = fogBuffer.density;
    const float y0 = origin.y - fogBuffer.fogBaseHeight;
    const float dy = direction.y;

    if(falloff < 1e-6f) return density * rayLength;

    if(abs(dy) < 1e-5f){
        return density * exp(-falloff * max(y0 , 0.0f)) * rayLength;
    }

    const float tCross = clamp(-y0 / dy , 0.0f , rayLength);

    float underStart , underEnd , aboveStart , aboveEnd;
    if(dy > 0.0f){
        underStart = 0.0f;   underEnd = tCross;
        aboveStart = tCross; aboveEnd = rayLength;
    }else{
        aboveStart = 0.0f;   aboveEnd = tCross;
        underStart = tCross; underEnd = rayLength;
    }

    const float yStart = y0 + dy * aboveStart;
    const float yEnd = y0 + dy * aboveEnd;

    float opticalDepth = density * max(underEnd - underStart , 0.0f);
    opticalDepth += density * (exp(-falloff * max(yStart , 0.0f)) - exp(-falloff * max(yEnd , 0.0f))) / (falloff * dy);
    return max(opticalDepth , 0.0f);
}

float3 applyFog(float3 color , float linearDepth , float3 rayDirection , float2 uv){
    const float cosAxis = max(dot(rayDirection , generalBuffer.playerDirection) , 1e-3f);
    const float rayScale = 1.0f / cosAxis;

    uint width , height , sliceCount , levels;
    txFogImageArray.GetDimensions(0 , width , height , sliceCount , levels);

    float transmittance = 1.0f;
    float3 scattered = float3(0.0f , 0.0f , 0.0f);

    const float volumeEnd = min(linearDepth , FOG_FAR);

    [loop]
    for(uint k = 0u ; k < sliceCount ; k++){
        const float near = fogSliceDepth(float(k) , float(sliceCount));
        if(near >= volumeEnd) break;
        const float far = min(fogSliceDepth(float(k + 1u) , float(sliceCount)) , volumeEnd);

        const float length = (far - near) * rayScale;

        const float4 slice = txFogImageArray.SampleLevel(txSampler , float3(uv , float(k)) , 0);
        const float sigmaT = slice.a;
        const float3 scatter = slice.rgb;

        const float stepTransmittance = exp(-sigmaT * length);
        const float3 stepScattered = (sigmaT > 1e-7f) ? (scatter - scatter * stepTransmittance) / sigmaT : scatter * length;

        scattered += transmittance * stepScattered;
        transmittance *= stepTransmittance;
    }

    float3 result = color * transmittance + scattered;

    if(linearDepth > FOG_FAR){
        const float3 origin = generalBuffer.playerPosition + rayDirection * (FOG_FAR * rayScale);
        const float rayLength = (linearDepth - FOG_FAR) * rayScale;

        const float opticalDepth = ComputeHeightFog(origin , rayDirection , rayLength);
        const float farTransmittance = exp(-opticalDepth);
        const float3 farScattered = fogBuffer.albedo * fogBuffer.color * (1.0f - farTransmittance);

        result = result * farTransmittance + farScattered;
    }
    return result;
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width , height;
    HDRImage.GetDimensions(width , height);
    uint2 screenSize = uint2(width , height);
    if(any(id.xy >= screenSize)) return;

    const float2 uv = (float2(id.xy) + 0.5f) / float2(screenSize);
    const float3 worldDir = ComputePixelRayDirection(id.xy , screenSize , generalBuffer.invViewProj , generalBuffer.playerPosition);

    if(pixelCheck[id.xy] == SKYBOX_PIXEL){
        float3 sky = diffuseImage[id.xy].xyz;
        if(features.volumentricFog){
            sky = applyFog(sky , FOG_SKY_DISTANCE , worldDir , uv);
        }
        HDRImage[id.xy] = float4(sky , diffuseImage[id.xy].w);
        return ;
    }

    const float3 diffuse = diffuseImage[id.xy].xyz;
    const float3 specular = specularImage[id.xy].xyz;
    const float3 albedo = max(albedoImage[id.xy].xyz , 0.02f);
    const float3 direct = diffuse * albedo;
    const float linearDepth = depthImage[id.xy].x;
    const float shadow = saturate(pow(max(fShadowImage[id.xy].x , 0.0001f) , features.shadowThickness));
    const float AO = pow(fAmbientImage[id.xy].x , features.AO_intens);
    const float3 indirect = indirectImage[id.xy].xyz * albedo;
    float3 result = direct * shadow * AO + specular + indirect;

    if(features.currentMode == INDIRECT_MODE){
        HDRImage[id.xy] = indirectImage[id.xy];
        return;
    }

    if(features.currentMode == DEPTH_MODE)
    {
        HDRImage[id.xy] = float4(direct , 1.0f);
        return;
    }

    if(features.volumentricFog){
        result = applyFog(result , linearDepth , worldDir , uv);
    }

    HDRImage[id.xy] = float4(result , 1.0f);
}