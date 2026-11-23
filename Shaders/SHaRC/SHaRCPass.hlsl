#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#define MOX_ATMO_SCAT_INCLUDED 2
#include "../../include/moxSourceCluster/AtmoScat.hlsl"
#define MOX_VERTEX_HEAP_INCLUDED 3
#include "../../include/moxObject/VertexHeap.hlsl"
#define MOX_TEXTURE_HUB_INCLUDED 4
#include "../../include/moxSourceCluster/TextureHub.hlsl"
#define MOX_LIGHT_HUB_INCLUDED 5
#include "../../include/moxSourceCluster/LightHub.hlsl"
#define MOX_SHARC_HEAP_INCLUDED 6
#include "../../include/moxSourceCluster/SharcHeap.hlsl"

uint pickRandomNumber(inout uint seed , uint totalCount){
    return uint(float(totalCount) * rand(seed));
}

struct [raypayload] HitRayPayLoad{
    bool   didHit;
    float3 worldPos;
    float3 normal;
    float3 albedo;
    float  roughness;
    float  metallic;
    // the LUT index of the surface that the ray starts from : the caller writes it , the closesthit compares it with the one of the hit
    uint   LUT_index;
    // true if the hit has the same LUT_index as the caller ( the closesthit writes it , the miss writes false )
    bool   sameLUT;
};
// all ReSTIR SSS

// TODO
// add here the NRC 

uint bufferIndex(uint2 pixel){
    const uint2 clamped = min(pixel , uint2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));
    return clamped.x + clamped.y * generalBuffer.screenWidth;
}

float3 ComputePixelRayDirection(uint2 pixelCoord, uint2 screenSize, float4x4 invViewProj , float3 playerPosition)
{
    float2 uv = (float2(pixelCoord) + 0.5) / float2(screenSize);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float4 farPoint = mul(invViewProj, float4(ndc, 1.0, 1.0));
    farPoint /= farPoint.w;

    return normalize(farPoint.xyz - playerPosition);   
}

float calcMultiScatterFactor(float3 F0 , float roughness){
    // average of the three channels, not their sum
    const float F0_avr = (F0.x + F0.y + F0.z) / 3.0f;
    const float Ess = MultiScatterRoughness.SampleLevel(txSampler , roughness, 0).x;
    return 1.0f + F0_avr * (1.0f / max(Ess , 0.001f) - 1.0f);
}


void computeLight(GeneralObjectInfo info , float3 L , out float3 diffuse , out float3 specular){
    const float roughness = info.roughness;
    const float metallic = info.metallic;
    const float3 V = info.V;
    const float3 N = info.N;
    const float3 albedo = info.albedo;
    const float3 H = normalize(V + L);
    const float NdotV = max(dot(N , V) , 0.0001f);
    const float VdotH = max(dot(V , H) , 0.0001f);
    const float NdotH = max(dot(N , H) , 0.0001f);
    const float NdotL = max(dot(N , L) , 0.0001f);

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f), albedo , metallic);
    const float D = D_GGX(NdotH , roughness);
    const float3 F = F_fresnel(F0 , VdotH);
    const float G = G_smith(NdotV , NdotL , roughness);

    float3 directSpecular = (D * G * F) / max(4.0f * NdotL * NdotV , 0.0001f);
    float multiScatter = calcMultiScatterFactor(F0 , roughness);
    specular = directSpecular * multiScatter;

    const float3 kD = (1.0f - F) * (1.0f - metallic);
    diffuse = kD * albedo / PI;
}

void computeLight(GeneralObjectInfo info , float3 L , out float3 specular){
    const float roughness = info.roughness;
    const float metallic = info.metallic;
    const float3 V = info.V;
    const float3 N = info.N;
    const float3 albedo = info.albedo;
    const float3 H = normalize(V + L);
    const float NdotV = max(dot(N , V) , 0.0001f);
    const float VdotH = max(dot(V , H) , 0.0001f);
    const float NdotH = max(dot(N , H) , 0.0001f);
    const float NdotL = max(dot(N , L) , 0.0001f);

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f), albedo , metallic);
    const float D = D_GGX(NdotH , roughness);
    const float3 F = F_fresnel(F0 , VdotH);
    const float G = G_smith(NdotV , NdotL , roughness);

    float3 directSpecular = (D * G * F) / max(4.0f * NdotL * NdotV , 0.0001f);
    float multiScatter = calcMultiScatterFactor(F0 , roughness);
    specular = directSpecular * multiScatter;
}

float calculateLightWindow(lg::LightSource light , float lightDistance)
{
    const float ratio = saturate(lightDistance / max(light.radius , 1e-4f));
    float window = saturate(1.0f - ratio * ratio * ratio * ratio);
    return window * window;
}

// the attenuation of a point light : the physical falloff times a window that reaches exactly 0 at lightRadius .
// without the window a light shone on EVERY pixel of the tiles it was listed in , far beyond its radius : the lit and the dark parts met
// on the edges of the Forward+ tiles ( hard vertical and horizontal borders ) and on the sphere of the radius
float lightAttenuation(lg::LightSource light , float lightDistance){
    const float falloff = 1.0f / (1.0f + light.Linear * lightDistance + light.Quadratic * lightDistance * lightDistance);
    const float ratio = saturate(lightDistance / max(light.radius , 1e-4f));
    float window = saturate(1.0f - ratio * ratio * ratio * ratio);
    window *= window;
    return falloff * window;
}


float calculateShadows(float3 position , float3 direction , float maxDistance){
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
            // the any hit : a transparent texel is not a blocker
            if(checkTransHit(query.CandidateTriangleBarycentrics() , query.CandidatePrimitiveIndex() , query.CandidateInstanceID()))
            {
                query.CommitNonOpaqueTriangleHit();
            }
        }
    }

    return (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0f : 1.0f;
}

float calculateShadowAtt(float3 position , float3 direction , float maxDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    RayQuery<RAY_FLAG_NONE> query;
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

    if(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return 1.0f;
    return saturate(query.CommittedRayT() / max(maxDistance , 1e-20f));
}

float calculateSunShadow(float3 position , float3 direction , float maxDistance , float fadeDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    if(fadeDistance <= 0.0f){
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(TLAS , RAY_FLAG_NONE , 0xFF , ray);
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

    // the nearest blocker ( see calculateShadowAtt ) : its distance is what the fade needs
    RayQuery<RAY_FLAG_NONE> query;
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

    if(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return 1.0f;
    return smoothstep(0.0f , fadeDistance , query.CommittedRayT());
}

struct HitSurface{
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
};

// the attributes of the triangle in the object space , interpolated by the barycentrics . the indices of a mesh are LOCAL ( the draw has
// vertexOffset = firstVertex , the BLAS has firstVertex ) , so the vertex is firstVertex + index
HitSurface interpolateHit(const uint meshID , const uint primitive , const float2 barycentrics){
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
float2 calculateParallaxAtHit(const float2 currentUV , const float3 V , const int depthID){
    if(depthID == NON_VALID_SLOT) return currentUV;
    const uint physicalIndex = txVirtualBuffer[depthID];
    const float depth = textureHub[physicalIndex].SampleLevel(txSampler , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

HitRayPayLoad getHitInfo(const uint instanceID , const uint primitive , const float2 barycentrics , const float3 rayDirection , const uint callerLUT = 0xFFFFFFFFu){
    // instanceID is the instanceCustomIndex of the TLAS instance = the index in the instance layouts ( CommittedInstanceID )
    const InstanceData instance = instanceLayoutBuffer[instanceID];
    const uint meshID = instance.meshDataIndex;
    const LUT lut = LUT_Buffer[instance.LUT_index];

    const HitSurface surface = interpolateHit(meshID , primitive , barycentrics);

    // the same model matrix as the vertex shader of the Deferred render uses
    const float4x4 model = instanceBuffer[meshBuffer[meshID].instanceIndex];

    // normalMatrix3x3 already is inverse( transpose( model ) ) ( see the vertex shader ) , the matrix is on the left
    const float3 N = normalize(mul(normalMatrix3x3((float3x3)model) , surface.normal));
    const float3 T = normalize(mul(model , float4(surface.tangent , 0.0f)).xyz);
    const float3 B = normalize(mul(model , float4(surface.bitangent , 0.0f)).xyz);
    const float3x3 TBN = float3x3(T , B , N);

    const float3 hitPoint = mul(model , float4(surface.position , 1.0f)).xyz;
    // the direction back to the point that the ray comes from
    const float3 V = -rayDirection;
    const float3 V_TBN = mul(TBN , V);

    const float2 uv = calculateParallaxAtHit(surface.uv , V_TBN , lut.depthID);

    float3 normal = N;
    if(lut.normalID != NON_VALID_SLOT){
        const uint nID = txVirtualBuffer[lut.normalID];
        float3 n;
        n.xy = textureHub[nID].SampleLevel(txSampler , uv , 0).xy * 2.0f - 1.0f;
        n.z = sqrt(saturate(1.0f - dot(n.xy , n.xy)));
        // OpenGL maps ( green up ) are used as they are with this TBN , the DirectX ones ( green down ) are flipped
        n.y = (generalBuffer.normalMapOpenGL != 0u) ? n.y : -n.y;
        normal = normalize(mul(n , TBN));
    }

    float3 albedo = lut.color;
    if(lut.diffuseID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.diffuseID];
        albedo = textureHub[id].SampleLevel(txSampler , uv , 0).xyz;
    }

    float metallic = lut.metallic;
    if(lut.metallicTextureID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.metallicTextureID];
        // RGBA8 map : a grey picture has the value in every channel , an ARM map has the metallic in B
        metallic = textureHub[id].SampleLevel(txSampler , uv , 0).z;
    }

    float roughness = lut.roughness;
    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.roughnessTextureID];
        // RGBA8 map : a grey picture has the value in every channel , an ARM map has the roughness in G
        roughness = textureHub[id].SampleLevel(txSampler , uv , 0).y;
    }

    // the packed PBR map ( R occlusion , G roughness , B metallic ) : the same as the deferred render reads it , the occlusion is not used yet
    if(lut.armTextureID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.armTextureID];
        const float3 arm = textureHub[id].SampleLevel(txSampler , uv , 0).xyz;
        roughness = arm.y;
        metallic = arm.z;
    }

    // the normal looks at the ray ( a back face of a thin wall )
    if(dot(normal , V) < 0.0f) normal = -normal;

    // every field is written : the caller reads all of them
    HitRayPayLoad hit;
    hit.didHit = true;
    hit.worldPos = hitPoint;
    hit.normal = normal;
    hit.albedo = albedo;
    hit.roughness = roughness;
    hit.metallic = metallic;
    hit.LUT_index = instance.LUT_index;
    hit.sameLUT = (instance.LUT_index == callerLUT);
    return hit;
}

float3 sampleCone(const float3 axis , const float cosAlpha , inout uint seed){
    const float phi = 2.0f * PI * rand(seed);
    const float rand2 = rand(seed);

    const float cosTheta = lerp(cosAlpha , 1.0f , rand2);
    const float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));
    const float3 localDir = {sinTheta * cos(phi) , sinTheta * sin(phi) , cosTheta};

    float3 tangent , bitangent;
    buildONB(axis , tangent , bitangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * axis;
}

float3 randomInLightRadius(inout uint seed  , lg::LightSource light , const float3 L){
    const float u = rand(seed) , v = rand(seed);
    const float z = 1.0f - 2 * u;
    const float phi = 2.0f * PI * v;
    const float r = sqrt(saturate(1.0f - z * z));
    const float x = cos(phi) * r;
    const float y = sin(phi) * r;
    const float R = light.sourceRadius;
    float3 offset = R * float3(x ,y ,z);
    const float facing = dot(offset , L);
    if(facing > 0.0f) offset = -offset;
    return light.position + offset;
}

float3 randomInSunRadius(inout uint seed  , SunSource light , const float3 L){
    const float u = rand(seed) , v = rand(seed);
    const float z = 1.0f - 2 * u;
    const float phi = 2.0f * PI * v;
    const float r = sqrt(saturate(1.0f - z * z));
    const float x = cos(phi) * r;
    const float y = sin(phi) * r;
    const float R = light.sourceRadius;
    float3 offset = R * float3(x ,y ,z);
    const float facing = dot(offset , L);
    if(facing > 0.0f) offset = -offset;
    return normalize(normalize(L) + offset);
}

float calculateShadowInSphere(inout uint seed , lg::LightSource light , const float3 P , const float3 N){
    float3 L = light.position - P;
    const float d = length(L);
    L = normalize(L);
    const float3 randomPos = randomInLightRadius(seed , light , L);
    const float3 toSample = randomPos - P;
    const float sampleDistance = length(toSample);
    const float3 dir = toSample / max(sampleDistance , 1e-4f);
    return calculateShadowAtt(offsetRay(P , N , generalBuffer.playerPosition) , dir , sampleDistance);
}

float calculateShadowToCenter(lg::LightSource light , const float3 P , const float3 N){
    const float3 toLight = light.position - P;
    const float distance = length(toLight);
    const float3 dir = toLight / max(distance , 1e-4f);
    return calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , dir , distance);
}



float3 stretchView(const float3 V , const float a){
    return normalize(float3(a  * V.x , V.y * a , V.z));
}

GeneralObjectInfo infoFromGBuffer(uint2 pixel){
    GeneralObjectInfo info;
    info.P = positionImage[pixel].xyz;
    info.LUT_index = uint(positionImage[pixel].a);
    info.N = normalImage[pixel].xyz;
    info.V = normalize(generalBuffer.playerPosition - info.P);
    info.albedo = albedoImage[pixel].xyz;
    info.roughness = roughnessImage[pixel];
    info.metallic = metallicImage[pixel];
    return info;
}

float3 sampleSkybox(const float3 P ,const float3 N){
    float3 result = 0.0f;
    if(features.skyboxSampleCount == 0u) return result;
    [loop]
    for(uint i = 0 ; i  < features.skyboxSampleCount ; i++){
        const float3 direction = randomInHemisphere(N);
        const float shadow = calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , direction , 5.0f);
        result += skybox.SampleLevel(txSampler , direction , 0).xyz * shadow;
    }
    result /= float(features.skyboxSampleCount);
    result = lerp(0.0f , result , sunBuffer.intensity);
    return result;
}

float3 sampleIBL(const float3 N){
    return IBL.SampleLevel(txSampler , N , 0).xyz * sunBuffer.intensity;
}

float skyVisibilityAt(const float3 P , const float3 N , inout uint seed)
{
    const uint sampleCount = features.skyboxSampleCount;
    // no rays : nothing says that the point is covered
    if(sampleCount == 0u) return 1.0f;

    float escaped = 0.0f;
    [loop]
    for(uint i = 0 ; i < sampleCount ; i++){
        const float3 direction = sampleCone(float3(0.0f , 1.0f , 0.0f), 0.5f , seed );
        const float shadow = calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , direction , 100.0f);
        if(shadow == 1.0f) escaped += 1.0f;
    }
    return escaped / float(sampleCount);
}

float3 sampleHemisphere(const float2 Xi){
    const float r = Xi.y;
    const float phi = 2.0f * PI * Xi.x;
    const float R = sqrt(r);
    const float t1 = cos(phi) * R;
    const float t2 = sin(phi) * R;
    const float z=  sqrt(1.0f - r);
    return float3(t1 , t2 , z);
}

float3 sampleVNDF(float2 Xi , float3 V , float roughness){
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
    const float z=  sqrt(max(0.0f , 1.0f - t1 * t1 - t2 * t2));
    const float3 Nh = t1 * T1 + t2 * T2 + z * Vh;
    return normalize(float3(alpha * Nh.x , alpha * Nh.y , max(0.0f , Nh.z)));
}

float D_GGX_exact(float NdotH , float roughness){
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denum = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    return a2 / max(denum * denum * PI , 1e-20f);
}

float G1_smithGGX(float NdotV , float alpha){
    const float a2 = alpha * alpha;
    return 2.0f * NdotV / (NdotV + sqrt(a2 + (1.0f - a2) * NdotV * NdotV));
}

float pdfVNDF(float3 N , float3 V , float3 L , float roughness){
    const float NdotV = dot(N , V);
    const float NdotL = dot(N , L);
    if(NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    const float3 H = normalize(V + L);
    const float NdotH = dot(N , H);
    const float alpha = roughness * roughness;
    const float safeNdotV = max(NdotV , 1e-4f);
    return G1_smithGGX(safeNdotV , alpha) * D_GGX_exact(NdotH , roughness) / (4.0f * safeNdotV);
}

float3 evalBRDF(const float3 N , const float3 V , const float3 L , const float roughness, const float3 F0){
    const float NdotV = dot(N , V);
    const float NdotL = dot(N , L);
    if(NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    const float3 H = normalize(V + L);
    const float NdotH = dot(N , H);
    const float VdotH = dot(V , H);
    const float alpha = roughness * roughness;
    // D without the cap of D_GGX ( the cap cut the highlight for roughness < 0.27 ) , G of both directions ( the shadowing of L is a part of it )
    const float D = D_GGX_exact(NdotH , roughness);
    const float G = G1_smithGGX(NdotV , alpha) * G1_smithGGX(NdotL , alpha);
    const float3 F = F_fresnel(F0 , VdotH);
    // f * cos : the NdotL of the cosine cancels the NdotL of the denominator
    return D * G * F / max(4.0f *  NdotV , 1e-20);
}

float3 evalDiffuse(const float3 albedo , const float3 N , const float3 L  ,const float metallic){
    const float3 color = albedo / PI * (1.0f - metallic);
    const float NdotL = saturate(dot(N , L));
    return color * NdotL;
}

float3 evalDiffuse(const float3 albedo , const float3 N ,const float metallic){
    const float3 color = albedo / PI * (1.0f - metallic);
    return color;
}

bool isSpecularMode(const float roughness , const float metallic){ return (metallic >= 0.7f && roughness < 0.4f); }

void sampleDiffuseDirection(const float2 Xi , const float3 N , out float3 L , out float pdf){
    float3 T , B;
    buildONB(N , T , B);
    const float3 local = sampleHemisphere(Xi);
    L = normalize(local.x * T + local.y * B + local.z * N);
    pdf = max(dot(N , L) , 0.0f) / PI;
}

void sampleSpecularDirection(const float2 Xi , const float3 N , const float3 V , const float roughness , out float3 L , out float pdf){
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
    // pdfVNDF works in any space as long as N , V and L are in the same one : the tangent one here ( it is 0 for L under the surface )
    pdf = pdfVNDF(float3(0.0f , 0.0f , 1.0f) , Vlocal , Llocal , roughness);
}

// indirect GI
// ReSTIR GI


float3 evalFcos(const GeneralObjectInfo info , const bool specularMode , const float3 L){
    if(specularMode){
        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
        return evalBRDF(info.N , info.V , L , info.roughness , F0) * calcMultiScatterFactor(F0 , info.roughness);
    }
    return evalDiffuse(info.albedo , info.N , L , info.metallic);
}


float3 computeDiffuseLightAt(HitRayPayLoad info, inout uint seed, const float roughness , const float xMetallic, const float3 x1Pos){
    // in order to literally minimaze the atmosphere effect , we actually need to lower the chances of it to appear *
    if(!info.didHit){
        return 0.0f;
    }
    uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];

    findLightIndices(info.worldPos , lightIndices , lightCount);

    GeneralObjectInfo data = (GeneralObjectInfo)0;
    data.albedo = info.albedo;
    data.metallic = info.metallic;
    data.roughness = info.roughness;
    data.N = info.normal;
    data.P = info.worldPos;
    
    const float3 saveAlbedo = max(info.albedo , 1e-20);

    float3 result = 0.0f ;
    if(lightCount){
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
            const float epsilon = 1e-3f;
            L = normalize(L);
            float3 diffuse, specular;
            computeLight(data ,L , diffuse , specular );
            const float NdotL = saturate(dot(data.N , L));  
            const float DdotL = saturate(dot(-L , light.direction));
            float shadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , d - epsilon);
            float window = calculateLightWindow(light , d);
            
            diffuse /= saveAlbedo;

            result = diffuse * light.diffuse * shadow * window * NdotL * DdotL * lightCount / max(pdf_A * d * d , 0.0001f);
        }
        else
        {
            float3 diffuse, specular;

            const float3 V = normalize(x1Pos - info.worldPos);
            data.V = V;
            float3 L = light.position - data.P;
            L = normalize(L);
            float3 lightPos = light.position;
            float3 realL = lightPos - info.worldPos;
            const float dist = length(realL);
            realL = normalize(realL);
            
            computeLight(data ,realL , diffuse , specular );

            const float NdotL = saturate(dot(data.N , realL));  
            float shadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , dist);
            float window = lightAttenuation(light, dist);
            diffuse /= saveAlbedo;
            result = NdotL * light.diffuse * (diffuse) * shadow  * window * lightCount;
        }
    }else{
        result =  0.01f;
    }
        const float skyVisibility = skyVisibilityAt(data.P , data.N , seed);
        if(sunBuffer.intensity > 0.0f){
            float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
            const float3 L = randomInSunRadius(seed , sunBuffer , -sunBuffer.direction);
            float sunShadow = 0.0f;
            if(skyVisibility > 0.0f){
                sunShadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , 1000.0f);
            }
            computeLight(data , L , sunDiffuse , sunSpecular);
            sunDiffuse /= saveAlbedo;
            result += (sunDiffuse) *  clamp(dot(data.N , L),  0.0f , 1.0f) * sunBuffer.diffuse  * sunShadow;
    }

    if(skyVisibility > 0.0f){
        const float3 skyV = normalize(x1Pos - data.P);
        const float3 skyF0 = lerp(float3(0.04f , 0.04f , 0.04f) , data.albedo , data.metallic);
        const float skyNdotV = max(dot(data.N , skyV) , 0.0001f);
        const float3 skyF = F_fresnel(skyF0 , skyNdotV);
        const float3 skyKD = (1.0f - skyF) * (1.0f - data.metallic);
        result += sampleIBL(data.N) * skyKD * skyVisibility;
    }

    {
        const uint id = Sharc::getIndex(data.P , data.N , length(generalBuffer.playerPosition - data.P));
        const float3 radiance = Sharc::tableBuffer[id].xyz;
        result += radiance;
    }

    result *= saveAlbedo;

    return result;
}

// hitDistance : how far the light comes from ( the distance to the hit , the same as diffuseDistance of the light pass ; the sky - SHARC_SKY_DISTANCE like in RayGen , where 10000 is the distance of a ray that left the scene )
static const float SHARC_SKY_DISTANCE = 10000.0f;

float3 radianceAlong(const GeneralObjectInfo info , const float3 L , inout uint seed , out float hitDistance)
{
    hitDistance = SHARC_SKY_DISTANCE;
    RayDesc ray;
    ray.TMax = 200.0f;
    ray.TMin = 0.02f;
    ray.Direction = L;
    ray.Origin = offsetRay(info.P , info.N , generalBuffer.playerPosition);

    RayQuery<RAY_FLAG_NONE> rq;
    rq.TraceRayInline(TLAS , RAY_FLAG_NONE , 0x01 , ray);
    while(rq.Proceed())
    {
        if(!checkTransHit(rq.CandidateTriangleBarycentrics() , rq.CandidatePrimitiveIndex() , rq.CandidateInstanceID()))
        {
            continue;
        }

        rq.CommitNonOpaqueTriangleHit();
    }

    if(rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        return skybox.SampleLevel(txSampler , L , 2.0f).xyz * sunBuffer.intensity;
    }

    // the direction is normalized , so the parameter of the ray is the distance in meters
    hitDistance = rq.CommittedRayT();
    const HitRayPayLoad load = getHitInfo(rq.CommittedInstanceID() , rq.CommittedPrimitiveIndex() , rq.CommittedTriangleBarycentrics() , L);
    return computeDiffuseLightAt(load , seed , info.roughness , info.metallic , info.P);
}

float3 getRadianceFrom(GeneralObjectInfo info , inout uint seed , out float hitDistance)
{
    float3 L;
    float pdf;
    hitDistance = SHARC_SKY_DISTANCE;

    sampleDiffuseDirection(float2(randNoise() , randNoise()) , info.N , L , pdf);
    if(pdf <= 0.0f) return 0.0f;

    return radianceAlong(info , L , seed , hitDistance);
}

static const float SHARC_MAX_RADIANCE = 50.0f;

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint2 imageResolution = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    // one thread is one tile of MOX_SHARC_RESOLUTION_DIVISOR x MOX_SHARC_RESOLUTION_DIVISOR pixels ( rounded up : the tiles at the right and the bottom edge are not lost )
    const uint2 size = (imageResolution + (MOX_SHARC_RESOLUTION_DIVISOR - 1u)) / MOX_SHARC_RESOLUTION_DIVISOR;
    if(any(id.xy >= size)) return;

    uint seed = initSeed(id.xy , generalBuffer.frameIndex);

    const uint xOffset = pickRandomNumber(seed , MOX_SHARC_RESOLUTION_DIVISOR);
    const uint yOffset = pickRandomNumber(seed , MOX_SHARC_RESOLUTION_DIVISOR);
    const uint2 pixel = min(id.xy * MOX_SHARC_RESOLUTION_DIVISOR + uint2(xOffset , yOffset) , imageResolution - 1u);

    initNoise(pixel , generalBuffer.frameIndex);

    // the sky has no surface : nothing to write
    if(pixelCheck[pixel] != COLOR_PIXEL) return;

    GeneralObjectInfo info = infoFromGBuffer(pixel);
    if(dot(info.N , info.N) < 0.5f) return;

    if(isSpecularMode(info.roughness , info.metallic)) return;

    const float d = length(info.P - generalBuffer.playerPosition);
    const uint slot = Sharc::getIndex(info.P , info.N , d);

    float newDistance;
    const float3 newRad = getRadianceFrom(info , seed , newDistance);
    if(!all(isfinite(newRad))) return;

    // rgb : the radiance , a ( alpha ) : diffuseDistance , the distance that the light came from ( the average of the last frames the same way ) . the same ratio 0.9 / 0.1 for both
    const float4 prev = Sharc::tableBuffer[slot];
    const float3 result = prev.xyz * 0.9f + min(newRad , float3(SHARC_MAX_RADIANCE , SHARC_MAX_RADIANCE , SHARC_MAX_RADIANCE)) * 0.1f;
    const float distanceResult = prev.w * 0.9f + min(newDistance , SHARC_SKY_DISTANCE) * 0.1f;
    Sharc::tableBuffer[slot] = float4(result , distanceResult);
}
