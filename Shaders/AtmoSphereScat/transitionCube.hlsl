#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_ATMO_SCAT_INCLUDED 0
#include "../../include/moxSourceCluster/AtmoScat.hlsl"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"

static const float3 GROUND_COLOR = float3(0.30f , 0.28f , 0.20f);
static const float GROUND_FADE_RANGE = 0.1f;

float3 GetCubemapDirection(uint face, float2 uv)
{
    float2 ndc = uv * 2.0 - 1.0; // [-1,1]
    float3 dir;
    switch (face)
    {
        case 0: dir = float3( 1.0, -ndc.y, -ndc.x); break; // +X
        case 1: dir = float3(-1.0, -ndc.y,  ndc.x); break; // -X
        case 2: dir = float3( ndc.x,  1.0,  ndc.y); break; // +Y
        case 3: dir = float3( ndc.x, -1.0, -ndc.y); break; // -Y
        case 4: dir = float3( ndc.x, -ndc.y,  1.0); break; // +Z
        default: dir = float3(-ndc.x, -ndc.y, -1.0); break; // -Z
    }
    return normalize(dir);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint face = DTid.z;
    if (DTid.x >= FACE_SIZE || DTid.y >= FACE_SIZE || face >= 6)
        return;

    float2 uv = (float2(DTid.xy) + 0.5) / float(FACE_SIZE);
    float3 dir = GetCubemapDirection(face, uv);
    const float sunAzimuth = atan2(skyBuffer.sunSinAzimuth , skyBuffer.sunCosAzimuth);
    float realAzimuth = atan2(dir.z, dir.x) - sunAzimuth;
    float elevation = asin(clamp(dir.y , -1.0 , 1.0));
    const float below = saturate(-elevation / GROUND_FADE_RANGE);
    const float groundBlend = below * below * (3.0 - 2.0 * below);
    elevation = abs(elevation);

    float u = frac(realAzimuth / (2.0 * PI) + 1.0);
    float l = sqrt(saturate(elevation / (PI * 0.5)));
    float v = l * 0.5 + 0.5;

    const float4 sky = txSkyViewLUT.SampleLevel(acSamp, float2(u, v), 0);
    RWSkybox[uint3(DTid.xy, face)] = float4(sky.rgb * lerp(float3(1.0 , 1.0 , 1.0) , GROUND_COLOR , groundBlend) , sky.a);
}