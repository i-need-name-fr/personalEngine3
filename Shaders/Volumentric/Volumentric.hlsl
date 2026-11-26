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


// the number of the slices is the depth of the volume itself ( the same as PostProcess reads ) , fogBuffer.gridSize is not trusted
float sliceDepth(const float zf , const float sliceCount){
    return FOG_NEAR * pow(FOG_FAR / FOG_NEAR , zf / sliceCount);
}

// тип тут главное сделать так что Z будет Primary Образно говоря , поэтому здесь и деление такое странное
// jitter : the point of the froxel is moved inside the cell ( x , y , z in [0 , 1) ) , the centre of the cell is jitter = 0.5
float3 froxelWorldPos(uint3 id , uint3 fogSize , float3 jitter){
    float2 uv = (float2(id.xy) + jitter.xy) / float2(fogSize.xy);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    float4 v = mul(generalBuffer.inverseProj , float4(ndc , 1.0f , 1.0f));
    float3 vDis = v.xyz / v.w;
    vDis /= abs(vDis.z);
    float3 pos = vDis * sliceDepth(float(id.z) + jitter.z , float(fogSize.z));
    return mul(generalBuffer.inverseView , float4(pos , 1.0f)).xyz;
}

// PCG hash : 3 independent numbers in [0 , 1) for the froxel , they change every frame
float3 froxelNoise(uint3 id , uint frame){
    uint3 v = id * uint3(1664525u , 1013904223u , 2654435761u) + frame * 747796405u + 2891336453u;
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;  v.y += v.z * v.x;  v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;  v.y += v.z * v.x;  v.z += v.x * v.y;
    return float3(v & 0x00FFFFFFu) / 16777216.0f;
}

// в каком дальше направлении будет идти фотон через весь fog 
float phaseHG(const float cosTheta , const float g){
    float k = 1.0f + g * g - 2.0f * g * cosTheta;
    return (1.0f - g * g ) / (4.0f * PI * k * sqrt(k));
}

float calculateLightWindow(lg::LightSource light , float lightDistance)
{
    const float ratio = saturate(lightDistance / max(light.radius , 1e-4f));
    float window = saturate(1.0f - ratio * ratio * ratio * ratio);
    return window * window;
}

// the physical falloff times a window that reaches exactly 0 at lightRadius ( see lightAttenuation of RayGen )
float lightAttenuation(lg::LightSource light , float lightDistance)
{
    const float falloff = 1.0f / (1.0f + light.Linear * lightDistance + light.Quadratic * lightDistance * lightDistance);
    return falloff * calculateLightWindow(light , lightDistance);
}

// 1 : the way is free , 0 : something stands in it ( a transparent texel does not )
float calculateShadows(float3 position , float3 direction , float maxDistance)
{
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(TLAS , RAY_FLAG_NONE , 0x01 , ray);
    while(query.Proceed())
    {
        if(query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            if(checkTransHit(query.CandidateTriangleBarycentrics() , query.CandidatePrimitiveIndex() , query.CandidateInstanceID()))
            {
                query.CommitNonOpaqueTriangleHit();
            }
        }
    }

    return (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0f : 1.0f;
}

float3 sampleCone(const float3 axis , const float cosAlpha , inout uint seed)
{
    const float phi = 2.0f * PI * rand(seed);
    const float rand2 = rand(seed);

    const float cosTheta = lerp(cosAlpha , 1.0f , rand2);
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(axis , tangent , bitangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * axis;
}


[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width , height , tempDepth;
    RWFogImageArray.GetDimensions(width , height , tempDepth);
    uint2 screenSize = uint2(generalBuffer.resolutionWidth , generalBuffer.resolutionHeight);
    uint3 fogSize = uint3(width , height , tempDepth);
    if(any(id >= fogSize)) return;

    uint seed = initSeed(id.xy , generalBuffer.frameIndex);

    const float3 jitter = froxelNoise(id , generalBuffer.frameIndex);
    float3 pos = froxelWorldPos(id , fogSize , jitter);
    // the DUST ( features.dustMode ) : no fog , only the light of the sun and the lamps in the air . the density is the dust , the scattering is white ( no albedo of the fog ) , the phase is its own
    const bool dust = features.dustMode != 0u;
    float density = dust ? dustDensityAt(pos) : fogDensityAt(pos);
    float sigmaT = density;
    float3 sigmaS = dust ? float3(sigmaT , sigmaT , sigmaT) : sigmaT * fogBuffer.albedo;

    const float g = dust ? fogBuffer.dustG : fogBuffer.g;

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
    float3 Lin = sunBuffer.diffuse  * phaseHG(dot(L_sun , V) , g) * vis * fogBuffer.dustIntensity;

    uint lightCount;
    uint lightIndices[lg::MAX_LIGHT_PER_CELL];

    findLightIndices(pos , lightIndices , lightCount);
    
    if(lightCount > 0)
    {
        [loop]
        for(uint i = 0 ; i < lightCount ; i++)
        {
            const lg::LightSource light = lightBuffer[lightIndices[i]];
            float3 L = light.position - pos;
            const float d = length(L);
            L = normalize(L);

            const float shadow = calculateShadows(pos , L , d - 0.001f);
            const float att = lightAttenuation(light , d);
            const float3 diff = light.diffuse * shadow * att;
            Lin += diff *  phaseHG(dot(L , V) , g);
        }
    }

    // the glow of the sky is the fog ( the milk of the air ) : the dust has none , only the shafts
    float3 scattered = Lin * fogBuffer.dustIntensity;
    if(!dust)
    {
        const float SKY_VISIBLE_RANGE = 100.0f;
        const float SKY_VISIBLE_MIN = 0.1f;
        const float3 skyDirection = sampleCone(float3(0.0f , 1.0f , 0.0f) , 0.0f , seed);
        const float skyVisible = lerp(SKY_VISIBLE_MIN , 1.0f , calculateShadows(pos , skyDirection , SKY_VISIBLE_RANGE));
        scattered = Lin + fogBuffer.color * skyVisible;
    }
    float3 froxel = scattered * sigmaS;
    if(any(isnan(froxel)) || any(isinf(froxel))) froxel = float3(0.0f , 0.0f , 0.0f);

    static const float MAX_HISTORY_COUNT = 64.0f;
    float count = 1.0f;
    float4 history = float4(0.0f , 0.0f , 0.0f , 0.0f);
    {
        const float4 prevClip = mul(generalBuffer.prevViewProj , float4(pos , 1.0f));
        if(prevClip.w > FOG_NEAR && prevClip.w < FOG_FAR)
        {
            const float2 prevNdc = prevClip.xy / prevClip.w;
            const float2 prevUV = float2(prevNdc.x * 0.5f + 0.5f , 0.5f - prevNdc.y * 0.5f);
            const float layerCoord = log(prevClip.w / FOG_NEAR) / log(FOG_FAR / FOG_NEAR) * float(fogSize.z) - 0.5f;

            if(all(prevUV > 0.0f) && all(prevUV < 1.0f))
            {
                const float layerLow = clamp(floor(layerCoord) , 0.0f , float(fogSize.z - 1u));
                const float layerHigh = min(layerLow + 1.0f , float(fogSize.z - 1u));
                const float layerBlend = saturate(layerCoord - layerLow);
                history = lerp(txFogHistory.SampleLevel(txSampler , float3(prevUV , layerLow) , 0) , txFogHistory.SampleLevel(txSampler , float3(prevUV , layerHigh) , 0) , layerBlend);
                if(any(isnan(history)) || any(isinf(history))) history = float4(0.0f , 0.0f , 0.0f , 0.0f);

                if(history.a >= 0.5f) count = min(history.a + 1.0f , MAX_HISTORY_COUNT);
            }
        }
    }

    const float blend = 1.0f / count;
    RWFogImageArray[id] = float4(lerp(history.rgb , froxel , blend) , count);
}