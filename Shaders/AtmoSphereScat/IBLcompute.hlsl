#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_ATMO_SCAT_INCLUDED 0
#include "../../include/moxSourceCluster/AtmoScat.hlsl"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"

static const uint IRRADIANCE_SIZE     = 32;
static const uint SAMPLE_COUNT_THETA  = 32;
static const uint SAMPLE_COUNT_PHI    = 64;

float3 CubemapTexelToDirection(float2 uv, uint faceIndex)
{
    float3 dir;
    if (faceIndex == 0)      dir = float3( 1.0,  -uv.y, -uv.x);
    else if (faceIndex == 1) dir = float3(-1.0,  -uv.y,  uv.x);
    else if (faceIndex == 2) dir = float3( uv.x,  1.0,   uv.y);
    else if (faceIndex == 3) dir = float3( uv.x, -1.0,  -uv.y);
    else if (faceIndex == 4) dir = float3( uv.x, -uv.y,  1.0);
    else                     dir = float3(-uv.x, -uv.y, -1.0);
    return normalize(dir);
}

[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){
    uint2 currentPixel = id.xy;
    if (id.x >= IRRADIANCE_SIZE || id.y >= IRRADIANCE_SIZE) return;

    float2 uv = (float2(id.xy) + 0.5f) / float(IRRADIANCE_SIZE) * 2.0f  - 1.0f;
    float3 N = CubemapTexelToDirection(uv , id.z);

    float3 up = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent   = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    float3 irradiance = float3(0.0f , 0.0f , 0.0f);
    uint sampleCount = 0;

    for(uint iTheta =  0 ; iTheta < SAMPLE_COUNT_THETA ;iTheta++){
        const float theta = (float(iTheta) / float(SAMPLE_COUNT_THETA)) * (0.5f * PI);
        const float cosTheta = cos(theta);
        const float sinTheta = sin(theta);
        for(uint iPhi = 0 ; iPhi < SAMPLE_COUNT_PHI ; iPhi++){
            float phi = (float(iPhi) / float(SAMPLE_COUNT_PHI)) * (2.0 * PI);
            float3 sampleDirTangent = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
            float3 sampleDir = sampleDirTangent.x * tangent
                              + sampleDirTangent.y * bitangent
                              + sampleDirTangent.z * N;

            float3 res = skybox.SampleLevel(acSamp , sampleDir , 0).rgb;
            irradiance += res * cosTheta * sinTheta;
            sampleCount++;
        }
    }

    irradiance = PI * irradiance / float(sampleCount);
    RWIBL[uint3(id.xyz)] = float4(irradiance , 1.0f);
}