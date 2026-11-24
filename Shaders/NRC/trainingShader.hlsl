#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_ATMO_SCAT_INCLUDED 1
#include "../../include/moxSourceCluster/AtmoScat.hlsl"
#define MOX_VERTEX_HEAP_INCLUDED 2
#include "../../include/moxObject/VertexHeap.hlsl"
#define MOX_TEXTURE_HUB_INCLUDED 3
#include "../../include/moxSourceCluster/TextureHub.hlsl"
#define MOX_LIGHT_HUB_INCLUDED 4
#include "../../include/moxSourceCluster/LightHub.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#define MOX_NRC_INCLUDED 5
#include "../../include/AI/NRC/NRC.hlsl"
using namespace AI::NRC;

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

float3 randomInSunRadius(inout uint seed , SunSource light , const float3 L)
{
    const float u = rand(seed) , v = rand(seed);
    const float z = 1.0f - 2 * u;
    const float phi = 2.0f * PI * v;
    const float r = sqrt(saturate(1.0f - z * z));
    const float x = cos(phi) * r;
    const float y = sin(phi) * r;
    float3 offset = light.sourceRadius * float3(x , y , z);
    if(dot(offset , L) > 0.0f) offset = -offset;
    return normalize(normalize(L) + offset);
}

// how much of the sky above the point is open ( 1 : all of it , 0 : closed )
float skyVisibilityAt(const float3 P , const float3 N , inout uint seed)
{
    const uint sampleCount = features.skyboxSampleCount;
    if(sampleCount == 0u) return 1.0f;

    float escaped = 0.0f;
    [loop]
    for(uint i = 0 ; i < sampleCount ; i++)
    {
        const float3 direction = sampleCone(float3(0.0f , 1.0f , 0.0f) , 0.5f , seed);
        if(calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , direction , 100.0f) == 1.0f) escaped += 1.0f;
    }
    return escaped / float(sampleCount);
}

float3 sampleIBL(const float3 N)
{
    return IBL.SampleLevel(txSampler , N , 0).xyz * sunBuffer.intensity;
}

// the sky that a ray sees when it leaves the scene
float3 sampleSky(const float3 direction)
{
    return skybox.SampleLevel(txSampler , direction , 2.0f).xyz * sunBuffer.intensity;
}

// the diffuse lobe WITHOUT the albedo : kD / PI ( kD = ( 1 - F ) * ( 1 - metallic ) ) . the albedo is multiplied once , when the labels are built from the end of the path
float3 diffuseWithoutAlbedo(const GeneralObjectInfo info , const float3 L)
{
    const float3 H = normalize(info.V + L);
    const float VdotH = max(dot(info.V , H) , 0.0001f);
    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
    const float3 F = F_fresnel(F0 , VdotH);
    return (1.0f - F) * (1.0f - info.metallic) / PI;
}

// the CLEAN direct light d_k of a vertex , the label of NRC : the light that reaches the point , WITHOUT the albedo of the point
float3 computeDirectAt(const GeneralObjectInfo data , inout uint seed)
{
    uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
    findLightIndices(data.P , lightIndices , lightCount);

    float3 result = 0.0f;
    if(lightCount)
    {
        // one lamp of the cell , chosen at random : the result is multiplied by the count
        const uint id = (uint)(rand(seed) * lightCount);
        lg::LightSource light = lightBuffer[lightIndices[id]];
        if(light.type == MOX_AREA_LIGHT_TYPE_DISK)
        {
            float3 U , V;
            diskBasis(light.direction , U , V);
            const float A = PI * light.sourceRadius * light.sourceRadius;
            const float pdf_A = 1.0f / max(A , 1e-3f);

            const float x1 = rand(seed);
            const float x2 = rand(seed);
            const float r = sqrt(x1) * light.sourceRadius;
            const float phi = 2.0f * PI * x2;
            const float3 lightPos = light.position + r * cos(phi) * U + r * sin(phi) * V;
            float3 L = lightPos - data.P;
            const float d = length(L);
            L = normalize(L);

            const float NdotL = saturate(dot(data.N , L));
            // a disk shines only to the side it faces
            const float DdotL = saturate(dot(-L , light.direction));
            const float shadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , d - 1e-3f);
            const float window = calculateLightWindow(light , d);

            result = diffuseWithoutAlbedo(data , L) * light.diffuse * shadow * window * NdotL * DdotL * lightCount / max(pdf_A * d * d , 0.0001f);
        }
        else
        {
            // the centre of the lamp , not a random point of its sphere ( the indirect light : a big sphere reached through the walls )
            float3 realL = light.position - data.P;
            const float dist = length(realL);
            realL = normalize(realL);

            const float NdotL = saturate(dot(data.N , realL));
            const float shadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , realL , dist);
            const float window = lightAttenuation(light , dist);

            result = NdotL * light.diffuse * diffuseWithoutAlbedo(data , realL) * shadow * window * lightCount;
        }
    }

    const float skyVisibility = skyVisibilityAt(data.P , data.N , seed);
    if(sunBuffer.intensity > 0.0f)
    {
        const float3 L = randomInSunRadius(seed , sunBuffer , -sunBuffer.direction);
        // under the open sky the shadow of the sun is traced , in the other case the sun gives nothing
        float sunShadow = 0.0f;
        if(skyVisibility > 0.0f) sunShadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , 1000.0f);
        result += diffuseWithoutAlbedo(data , L) * saturate(dot(data.N , L)) * sunBuffer.diffuse * sunShadow;
    }

    if(skyVisibility > 0.0f)
    {
        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , data.albedo , data.metallic);
        const float3 F = F_fresnel(F0 , max(dot(data.N , data.V) , 0.0001f));
        result += sampleIBL(data.N) * (1.0f - F) * (1.0f - data.metallic) * skyVisibility;
    }

    return result;
}

// ---------------------------------------------------------------- the hit
struct HitSurface
{
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
};

// the attributes of the triangle in the object space , interpolated by the barycentrics . the indices of a mesh are LOCAL ( vertex = firstVertex + index )
HitSurface interpolateHit(const uint meshID , const uint primitive , const float2 barycentrics)
{
    const MeshData mesh = meshBuffer[meshID];
    const uint base = mesh.firstIndex + primitive * 3u;

    const uint i0 = mesh.firstVertex + indices[base + 0u];
    const uint i1 = mesh.firstVertex + indices[base + 1u];
    const uint i2 = mesh.firstVertex + indices[base + 2u];

    const float w0 = 1.0f - barycentrics.x - barycentrics.y;
    const float w1 = barycentrics.x;
    const float w2 = barycentrics.y;

    HitSurface surface;
    surface.position  = vertices[i0].position  * w0 + vertices[i1].position  * w1 + vertices[i2].position  * w2;
    surface.normal    = vertices[i0].normal    * w0 + vertices[i1].normal    * w1 + vertices[i2].normal    * w2;
    surface.tangent   = vertices[i0].tangent   * w0 + vertices[i1].tangent   * w1 + vertices[i2].tangent   * w2;
    surface.bitangent = vertices[i0].bitangent * w0 + vertices[i1].bitangent * w1 + vertices[i2].bitangent * w2;
    surface.uv        = vertices[i0].UV        * w0 + vertices[i1].UV        * w1 + vertices[i2].UV        * w2;
    return surface;
}

// the same as the one of the fragment shader , but SampleLevel : a ray has no derivatives
float2 calculateParallaxAtHit(const float2 currentUV , const float3 V , const int depthID)
{
    if(depthID == NON_VALID_SLOT) return currentUV;
    const uint physicalIndex = txVirtualBuffer[depthID];
    const float depth = textureHub[physicalIndex].SampleLevel(txSampler , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

bool isSpecularMode(const float roughness , const float metallic) { return (metallic >= 0.7f && roughness < 0.4f); }

// the material of the committed hit : the same maps as the fragment shader of the G buffer reads . objectAO is the baked occlusion of the material ( 1 : there is no map ) : it darkens the direct light
GeneralObjectInfo readHit(const RayQuery<RAY_FLAG_NONE> rq , const float3 rayDirection , out float objectAO)
{
    const InstanceData instance = instanceLayoutBuffer[rq.CommittedInstanceID()];
    const uint meshID = instance.meshDataIndex;
    const LUT lut = LUT_Buffer[instance.LUT_index];

    const HitSurface surface = interpolateHit(meshID , rq.CommittedPrimitiveIndex() , rq.CommittedTriangleBarycentrics());
    const float4x4 model = instanceBuffer[meshBuffer[meshID].instanceIndex];

    const float3 N = normalize(mul(normalMatrix3x3((float3x3)model) , surface.normal));
    const float3 T = normalize(mul(model , float4(surface.tangent , 0.0f)).xyz);
    const float3 B = normalize(mul(model , float4(surface.bitangent , 0.0f)).xyz);
    const float3x3 TBN = float3x3(T , B , N);

    const float3 hitPoint = mul(model , float4(surface.position , 1.0f)).xyz;
    // the direction back to the point that the ray comes from
    const float3 V = -rayDirection;
    const float2 uv = calculateParallaxAtHit(surface.uv , mul(TBN , V) , lut.depthID);

    float3 normal = N;
    if(lut.normalID != NON_VALID_SLOT)
    {
        const uint nID = txVirtualBuffer[lut.normalID];
        float3 n;
        n.xy = textureHub[nID].SampleLevel(txSampler , uv , 0).xy * 2.0f - 1.0f;
        n.z = sqrt(saturate(1.0f - dot(n.xy , n.xy)));
        n.y = (generalBuffer.normalMapOpenGL != 0u) ? n.y : -n.y;
        normal = normalize(mul(n , TBN));
    }

    float3 albedo = lut.color;
    if(lut.diffuseID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.diffuseID];
        albedo = textureHub[id].SampleLevel(txSampler , uv , 0).xyz;
    }

    float metallic = lut.metallic;
    if(lut.metallicTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.metallicTextureID];
        metallic = textureHub[id].SampleLevel(txSampler , uv , 0).z;
    }

    float roughness = lut.roughness;
    if(lut.roughnessTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.roughnessTextureID];
        roughness = textureHub[id].SampleLevel(txSampler , uv , 0).y;
    }

    // the packed PBR map ( R occlusion , G roughness , B metallic )
    if(lut.armTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.armTextureID];
        const float3 arm = textureHub[id].SampleLevel(txSampler , uv , 0).xyz;
        roughness = arm.y;
        metallic = arm.z;
    }

    objectAO = 1.0f;
    if(lut.aoTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.aoTextureID];
        objectAO = textureHub[id].SampleLevel(txSampler , uv , 0).x;
    }

    // the normal looks at the ray ( a back face of a thin wall )
    if(dot(normal , V) < 0.0f) normal = -normal;

    GeneralObjectInfo info = (GeneralObjectInfo)0;
    info.P = hitPoint;
    info.N = normal;
    info.V = V;
    info.albedo = albedo;
    // the GGX of a perfect mirror ( roughness 0 ) is degenerate
    info.roughness = max(roughness , 0.1f);
    info.metallic = metallic;
    info.LUT_index = instance.LUT_index;
    return info;
}

// ---------------------------------------------------------------- the directions of the path
float3 sampleHemisphere(const float2 Xi)
{
    const float r = Xi.y;
    const float phi = 2.0f * PI * Xi.x;
    const float R = sqrt(r);
    return float3(cos(phi) * R , sin(phi) * R , sqrt(1.0f - r));
}

void sampleDiffuseDirection(const float2 Xi , const float3 N , out float3 L , out float pdf)
{
    float3 T , B;
    buildONB(N , T , B);
    const float3 local = sampleHemisphere(Xi);
    L = normalize(local.x * T + local.y * B + local.z * N);
    pdf = max(dot(N , L) , 0.0f) / PI;
}

float3 stretchView(const float3 V , const float a)
{
    return normalize(float3(a * V.x , V.y * a , V.z));
}

float3 sampleVNDF(float2 Xi , float3 V , float roughness)
{
    const float alpha = roughness * roughness;
    const float3 Vh = stretchView(V , alpha);

    const float3 T1 = (Vh.z < 0.9999f) ? normalize(cross(float3(0.0f , 0.0f , 1.0f) , Vh)) : float3(1.0f , 0.0f , 0.0f);
    const float3 T2 = cross(Vh , T1);

    const float r = Xi.x;
    const float R = sqrt(r);
    const float phi = 2.0f * PI * Xi.y;
    const float t1 = cos(phi) * R;
    float t2 = sin(phi) * R;
    const float s = 0.5 * (1 + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    const float z = sqrt(max(0.0f , 1.0f - t1 * t1 - t2 * t2));
    const float3 Nh = t1 * T1 + t2 * T2 + z * Vh;
    return normalize(float3(alpha * Nh.x , alpha * Nh.y , max(0.0f , Nh.z)));
}

float D_GGX_exact(float NdotH , float roughness)
{
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denum = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(denum * denum * PI , 1e-20f);
}

float G1_smithGGX(float NdotV , float alpha)
{
    const float a2 = alpha * alpha;
    return 2.0f * NdotV / (NdotV + sqrt(a2 + (1.0f - a2) * NdotV * NdotV));
}

float pdfVNDF(float3 N , float3 V , float3 L , float roughness)
{
    const float NdotV = dot(N , V);
    const float NdotL = dot(N , L);
    if(NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    const float3 H = normalize(V + L);
    const float NdotH = dot(N , H);
    const float alpha = roughness * roughness;
    const float safeNdotV = max(NdotV , 1e-4f);
    return G1_smithGGX(safeNdotV , alpha) * D_GGX_exact(NdotH , roughness) / (4.0f * safeNdotV);
}

void sampleSpecularDirection(const float2 Xi , const float3 N , const float3 V , const float roughness , out float3 L , out float pdf)
{
    float3 T , B;
    buildONB(N , T , B);
    const float3 Vlocal = float3(dot(V , T) , dot(V , B) , dot(V , N));

    L = N;
    pdf = 0.0f;
    // a view from behind the surface ( an interpolated normal ) : no valid sample
    if(Vlocal.z <= 0.0f) return;

    const float3 Hlocal = sampleVNDF(Xi , Vlocal , roughness);
    const float3 Llocal = normalize(reflect(-Vlocal , Hlocal));
    L = normalize(Llocal.x * T + Llocal.y * B + Llocal.z * N);
    pdf = pdfVNDF(float3(0.0f , 0.0f , 1.0f) , Vlocal , Llocal , roughness);
}

// ---------------------------------------------------------------- the path
// the path has MOX_NRC_MAX_BOUNCE_COUNT + 1 vertices x0 .. x3 : x0 .. x2 are the training samples , x3 is only asked from the net ( its answer closes the chain )
static const uint NRC_PATH_VERTICES = MOX_NRC_MAX_BOUNCE_COUNT + 1;

void getTrainingSamples(const RayDesc firstRay , inout uint seed , out TrainingSample samples[MOX_NRC_MAX_BOUNCE_COUNT])
{
    [unroll]
    for(uint i = 0 ; i < MOX_NRC_MAX_BOUNCE_COUNT ; i++)
    {
        samples[i] = (TrainingSample)0;
        samples[i].valid = MOX_NRC_NON_VALID_SAMPLE;
    }

    float3 positions[NRC_PATH_VERTICES];
    float3 normals[NRC_PATH_VERTICES];
    float3 direct[NRC_PATH_VERTICES];   // the pure direct light of the vertex ( no albedo , no throughput )
    float3 albedos[NRC_PATH_VERTICES];

    uint hitCount = 0;                  // how many vertices the path has
    float3 sky = 0.0f;                  // the light of the sky , if the path has ended with a miss
    bool missed = false;

    float pdf = 0.0f;
    float3 nextDir = 0.0f;
    RayDesc ray = firstRay;

    [unroll]
    for(uint bounce = 0 ; bounce < NRC_PATH_VERTICES ; bounce++)
    {
        if(bounce > 0)
        {
            // the ray goes out of the PREVIOUS vertex
            ray.Origin = offsetRay(positions[bounce - 1] , normals[bounce - 1] , generalBuffer.playerPosition);
            ray.TMax = 200.0f;
            ray.TMin = 0.01f;
            ray.Direction = nextDir;
        }

        // the mask 0x01 : the bodies of the lamps do not cover anything
        RayQuery<RAY_FLAG_NONE> rq;
        rq.TraceRayInline(TLAS , RAY_FLAG_NONE , 0x01 , ray);
        while(rq.Proceed())
        {
            if(rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            {
                // the holes of the leaves neither
                if(checkTransHit(rq.CandidateTriangleBarycentrics() , rq.CandidatePrimitiveIndex() , rq.CandidateInstanceID()))
                {
                    rq.CommitNonOpaqueTriangleHit();
                }
            }
        }

        if(rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            sky = sampleSky(ray.Direction);
            missed = true;
            break;
        }

        float objectAO;
        const GeneralObjectInfo info = readHit(rq , ray.Direction , objectAO);
        positions[bounce] = info.P;
        normals[bounce] = info.N;
        // the baked occlusion of the object darkens the light of the lamps and of the sun at the hit
        direct[bounce] = computeDirectAt(info , seed) * objectAO;
        albedos[bounce] = info.albedo;
        hitCount++;

        if(isSpecularMode(info.roughness , info.metallic))
        {
            sampleSpecularDirection(float2(randNoise() , randNoise()) , info.N , info.V , info.roughness , nextDir , pdf);
        }
        else
        {
            sampleDiffuseDirection(float2(randNoise() , randNoise()) , info.N , nextDir , pdf);
        }
    }

    if(hitCount == 0) return;

    float3 arriving = sky;
    if(!missed)
    {
        const uint last = NRC_PATH_VERTICES - 1;
        arriving = askNRC(positions[last] , normals[last]);
    }

    // from the end of the path to its start
    [unroll]
    for(int k = int(NRC_PATH_VERTICES) - 1 ; k >= 0 ; k--)
    {
        if(uint(k) >= hitCount) continue;

        if(k < int(MOX_NRC_MAX_BOUNCE_COUNT))
        {
            samples[k].position = positions[k];
            samples[k].normal = normals[k];
            samples[k].radiance = arriving;
            samples[k].valid = MOX_NRC_VALID_SAMPLE;
        }

        arriving = albedos[k] * (direct[k] + arriving);
    }
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint2 size = uint2(NRCGeneralData[0].width , NRCGeneralData[0].height);
    if(any(id.xy >= size)) return;

    TrainingSample samples[MOX_NRC_MAX_BOUNCE_COUNT];
    initNoise(id.xy , generalBuffer.frameIndex);
    uint seed = initSeed(id.xy , generalBuffer.frameIndex);

    // a random point of the cell of the training grid : the whole picture is covered over the frames
    const float2 uv = (float2(id.xy) + float2(rand(seed) , rand(seed))) / float2(size);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    const float4 viewPoint = mul(generalBuffer.inverseProj , float4(ndc , 1.0f , 1.0f));

    RayDesc ray;
    ray.Origin = generalBuffer.playerPosition;
    ray.Direction = normalize(mul((float3x3)generalBuffer.inverseView , viewPoint.xyz).xyz);
    ray.TMax = 1000.0f;
    ray.TMin = 0.001f;

    getTrainingSamples(ray , seed , samples);

    const uint buff = (id.x + id.y * size.x) * MOX_NRC_MAX_BOUNCE_COUNT;
    [unroll]
    for(uint i = 0 ; i < MOX_NRC_MAX_BOUNCE_COUNT ; i++)
    {
        trainingBuffer[buff + i] = samples[i];
    }

    // a new step of the training : the counter of Adam grows , the count of the valid samples starts again ( backstepShader counts them )
    if(all(id.xy == 0))
    {
        NRCGeneralData[0].stepCount += 1;
        NRCGeneralData[0].validCount = 0;
    }
}
