#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_VOL_SOURCE_INCLUDED 0
#include "../../include/moxSourceCluster/VolSource.hlsl"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_LIGHT_HUB_INCLUDED 2
#include "../../include/moxSourceCluster/LightHub.hlsl"
#define MOX_TEXTURE_HUB_INCLUDED 3
#include "../../include/moxSourceCluster/TextureHub.hlsl"
#define MOX_VERTEX_HEAP_INCLUDED 4
#include "../../include/moxObject/VertexHeap.hlsl"


float sliceDepth(const float zf){
    return FOG_NEAR * pow(FOG_FAR / FOG_NEAR , zf / fogBuffer.gridSize );
}

// тип тут главное сделать так что Z будет Primary Образно говоря , поэтому здесь и деление такое странное 
float3 froxelWorldPos(uint3 id , uint3 fogSize){
    float2 uv = float2(float2(id.xy) + 0.5f) / float2(fogSize.xy);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    float4 v = mul(generalBuffer.inverseProj , float4(ndc , 1.0f , 1.0f));
    float3 vDis = v.xyz / v.w;
    vDis /= abs(vDis.z);
    float3 pos = vDis * sliceDepth(id.z + 0.5f );
    return mul(generalBuffer.inverseView , float4(pos , 1.0f)).xyz;
}

// в каком дальше направлении будет идти фотон через весь fog 
float phaseHG(const float cosTheta , const float g){
    float k = 1.0f + g * g - 2.0f * g * cosTheta;
    return (1.0f - g * g ) / (4.0f * PI * k * sqrt(k));
}


[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width , height , tempDepth;
    RWFogImageArray.GetDimensions(width , height , tempDepth);
    uint2 screenSize = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    uint3 fogSize = uint3(width , height , tempDepth);
    if(any(id >= fogSize)) return;

    float3 pos = froxelWorldPos(id , fogSize);
    float density = fogBuffer.density * exp(-fogBuffer.heightFallOff * max(pos.y - fogBuffer.fogBaseHeight , 0));   
    float sigmaT = density;
    float3 sigmaS = sigmaT * fogBuffer.albedo;

    const float g = fogBuffer.g;

    float3 V = normalize(pos - generalBuffer.playerPosition);
    float3 L_sun = -sunBuffer.direction;
    // the center of the froxel on the screen
    const uint2 pixel = min(uint2((float2(id.xy) + 0.5f) * float2(screenSize) / float2(fogSize.xy)) , screenSize - 1);
    // now the ray cast
    float vis = 1.0f;
    {
        RayDesc ray;
        ray.TMax = 200.0f;
        ray.TMin = 0.1f;
        ray.Origin = pos;
        ray.Direction = -normalize(sunBuffer.direction);

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
        rq.TraceRayInline(TLAS , RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH , 0x01 ,ray);
        while(rq.Proceed())
        {
            const uint instanceID = rq.CandidateInstanceID();
            const float2 bar = rq.CandidateTriangleBarycentrics();
            const uint prim = rq.CandidatePrimitiveIndex();
            if(!checkTransHit(bar , prim , instanceID)) continue;
            rq.CommitNonOpaqueTriangleHit();
        }

        if(rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            vis = 0.0f;
        }
    }
    float3 Lin = sunBuffer.diffuse  * phaseHG(dot(L_sun , V) , g) * vis;
    const float3 scattered = Lin + fogBuffer.color;
    RWFogImageArray[id] = float4(scattered * sigmaS , sigmaT);
}