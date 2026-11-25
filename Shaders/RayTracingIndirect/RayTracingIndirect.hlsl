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
#define MOX_RESTIR_INCLUDED 6
#include "../../include/moxSourceCluster/SceneResTIR.hlsl"
#define MOX_SHARC_HEAP_INCLUDED 7
#include "../../include/moxSourceCluster/SharcHeap.hlsl"
#define MOX_NRC_INCLUDED 8
#include "../../include/AI/CoopVec.hlsl"
#include "../../include/AI/NRC/NRC.hlsl"

struct [raypayload] HitRayPayLoad{
    bool   didHit ;
    float3 worldPos ;
    float3 normal ;
    float3 albedo ;
    float  roughness : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float  metallic ;
    uint   LUT_index ;
    bool   sameLUT ;
};

float calculateLightWindow(lg::LightSource light , float lightDistance)
{
    const float ratio = saturate(lightDistance / max(light.radius , 1e-4f));
    float window = saturate(1.0f - ratio * ratio * ratio * ratio);
    return window * window;
}

float lightAttenuation(lg::LightSource light , float lightDistance)
{
    const float falloff = 1.0f / (1.0f + light.Linear * lightDistance + light.Quadratic * lightDistance * lightDistance);
    return falloff * calculateLightWindow(light , lightDistance);
}

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

// two pixels in this shader : indirectPixel is a pixel of indirectImage / the reservoirs ( any resolution , see MOX_INDIRECT_DIVISOR_RESOLUTION ) ,
// directPixel is a pixel of the G buffer ( the full resolution ) . they are never mixed : the sizes are always taken from GetDimensions ( IndirectPixels.hlsl )
#include "../../include/moxSourceCluster/IndirectPixels.hlsl"

GeneralObjectInfo infoFromGBuffer(uint2 directPixel){
    GeneralObjectInfo info = (GeneralObjectInfo)0;
    info.P = positionImage[directPixel].xyz;
    info.LUT_index = uint(max(positionImage[directPixel].a , 1.0f)) - 1u;
    info.N = normalImage[directPixel].xyz;
    info.V = normalize(generalBuffer.playerPosition - info.P);
    info.albedo = albedoImage[directPixel].xyz;
    info.roughness = roughnessImage[directPixel];
    info.metallic = metallicImage[directPixel];
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

float calcMultiScatterFactor(float3 F0 , float roughness){
    // average of the three channels, not their sum
    const float F0_avr = (F0.x + F0.y + F0.z) / 3.0f;
    const float Ess = MultiScatterRoughness.SampleLevel(txSampler , roughness, 0).x;
    return 1.0f + F0_avr * (1.0f / max(Ess , 0.001f) - 1.0f);
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

float3 evalFcos(const GeneralObjectInfo info , const bool specularMode , const float3 L){
    if(specularMode){
        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
        return evalBRDF(info.N , info.V , L , info.roughness , F0) * calcMultiScatterFactor(F0 , info.roughness);
    }
    return evalDiffuse(info.albedo , info.N , L , info.metallic);
}

uint bufferIndex(uint2 pixel){
    const uint2 clamped = min(pixel , uint2(generalBuffer.screenWidth - 1u , generalBuffer.screenHeight - 1u));
    return clamped.x + clamped.y * generalBuffer.screenWidth;
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

namespace ReSTIRGI{

    static const bool PREVIOUS_BOUNCE_ENABLED = true;

    float3 previousBounce(const GeneralObjectInfo data)
    {
        if(!PREVIOUS_BOUNCE_ENABLED) return 0.0f;
        const float4 clipPos = mul(generalBuffer.prevViewProj , float4(data.P , 1.0f));
        if(clipPos.w <= 0.0f) return 0.0f;

        float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f;
        prevUV.y = 1.0f - prevUV.y;
        if(any(prevUV < 0.0f) || any(prevUV >= 1.0f)) return 0.0f;

        // the G buffer has the full resolution , the reservoirs have the one of indirectImage
        const uint2 prevIndirect = uvToIndirectPixel(prevUV);
        // the pixel of the G buffer that this reservoir was made for ( the jitter of the last frame )
        const uint2 prevDirect = prevDirectPixelOf(prevIndirect);
        const uint2 iSize = indirectSize();

        // the same surface : the normal is about the same and the point is on the plane of the surface of that pixel ( the tolerance grows with the distance : a pixel is wider there )
        const float4 prevPosition = prevPositionImage[prevDirect];
        if(prevPosition.w == 0.0f) return 0.0f;
        const float3 prevNormal = prevNormalImage[prevDirect].xyz;
        if(dot(prevNormal , data.N) < 0.9f) return 0.0f;
        const float tolerance = max(0.05f , 0.01f * length(data.P - generalBuffer.playerPosition));
        if(abs(dot(data.P - prevPosition.xyz , prevNormal)) > tolerance) return 0.0f;

        const ReservoirGI prev = prevReSTIRGIBuffer[bufferIndex(prevIndirect , iSize.x , iSize.y)];
        if(prev.M == 0u) return 0.0f;

        const float3 L = normalize(prev.samplePos - data.P);
        const float3 bounce = prev.radiance * evalDiffuse(float3(1.0f , 1.0f , 1.0f) , data.N , L , data.metallic) * prev.W;
        // NaN is not taken , and a firefly of the last frame is limited ( it would come back through the loop )
        return sanitizeRadiance(bounce , RC::FEEDBACK_MAX_RADIANCE);
    }

    template<uint FLAGS>
    HitRayPayLoad readGIHit(RayQuery<FLAGS> rq , const float3 rayOrigin , const float3 rayDirection , const uint callerLUT)
    {
        HitRayPayLoad load = (HitRayPayLoad)0;
        load.LUT_index = callerLUT;

        if(rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            load.didHit    = false;
            load.albedo    = skybox.SampleLevel(txSampler , rayDirection , 0).xyz;
            load.normal    = -rayDirection;
            load.worldPos  = rayOrigin + rayDirection * 1000.0f;
            load.roughness = 1.0f;
            load.metallic  = 0.0f;
            load.sameLUT   = false;
            return load;
        }

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
            albedo = textureHub[txVirtualBuffer[lut.diffuseID]].SampleLevel(txSampler , uv , 0).xyz;
        }

        float metallic = lut.metallic;
        if(lut.metallicTextureID != NON_VALID_SLOT)
        {
            metallic = textureHub[txVirtualBuffer[lut.metallicTextureID]].SampleLevel(txSampler , uv , 0).z;
        }

        float roughness = lut.roughness;
        if(lut.roughnessTextureID != NON_VALID_SLOT)
        {
            roughness = textureHub[txVirtualBuffer[lut.roughnessTextureID]].SampleLevel(txSampler , uv , 0).y;
        }

        if(lut.armTextureID != NON_VALID_SLOT)
        {
            const float3 arm = textureHub[txVirtualBuffer[lut.armTextureID]].SampleLevel(txSampler , uv , 0).xyz;
            roughness = arm.y;
            metallic = arm.z;
        }

        // the normal looks at the ray ( a back face of a thin wall )
        if(dot(normal , V) < 0.0f) normal = -normal;

        load.didHit    = true;
        load.worldPos  = hitPoint;
        load.normal    = normal;
        load.albedo    = albedo;
        load.roughness = roughness;
        load.metallic  = metallic;
        load.sameLUT   = (instance.LUT_index == callerLUT);
        return load;
    }

    float3 computeDiffuseLightAt(HitRayPayLoad info, inout uint seed, const float roughness , const float xMetallic, const float3 x1Pos){
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
                // the centre of the lamp , not a random point of its sphere ( the indirect light : a big sphere reached through the walls )
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
        }

        const float skyVisibility = skyVisibilityAt(data.P , data.N , seed);
        if(sunBuffer.intensity > 0.0f){
            float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
            // now we also have to check if the shadow actually hits it , since if not, there is no point at computing the light , since its also the calcualtion
            const float3 L = randomInSunRadius(seed , sunBuffer , -sunBuffer.direction);
            // under the open sky the shadow of the sun is traced , in the other case the sun gives nothing ( 0 )
            float sunShadow = 0.0f;
            if(skyVisibility > 0.0f){
                sunShadow = calculateShadows(offsetRay(data.P , data.N , generalBuffer.playerPosition) , L , 1000.0f);
            }
            computeLight(data , L , sunDiffuse , sunSpecular);
            sunDiffuse /= saveAlbedo;
            result += (sunDiffuse) *  clamp(dot(data.N , L),  0.0f , 1.0f) * sunBuffer.diffuse  * sunShadow;
        }

        // the light of the sky : only under the open sky ( skyVisibility > 0 )
        if(skyVisibility > 0.0f){
            const float3 skyV = normalize(x1Pos - data.P);
            const float3 skyF0 = lerp(float3(0.04f , 0.04f , 0.04f) , data.albedo , data.metallic);
            const float skyNdotV = max(dot(data.N , skyV) , 0.0001f);
            const float3 skyF = F_fresnel(skyF0 , skyNdotV);
            const float3 skyKD = (1.0f - skyF) * (1.0f - data.metallic);
            result += sampleIBL(data.N) * skyKD * skyVisibility * info.albedo;
        }

        if(!features.NRC)
        {
            result += previousBounce(data);
        }
        else
        {
            result += AI::NRC::askNRC(data.P , data.N);
        }


        result *= saveAlbedo;

        return result;
    }

    float3 computeLightAt(HitRayPayLoad info, inout uint seed, const float roughness , const float xMetallic, const float3 x1Pos){
        if(!info.didHit) return 0.0f;
        return computeDiffuseLightAt(info , seed , roughness , xMetallic , x1Pos);
    }

    ReservoirGI generateReservoirGI(GeneralObjectInfo info , const uint2 indirectPixel , inout uint seed){
        // the DIFFUSE lobe only : the reflection is a separate ray ( traceRayReflection ) , the two lobes are summed in processPixel
        ReservoirGI res = (ReservoirGI)0;
        float3 L ; float pdf;
        sampleDiffuseDirection(float2(randNoise() , randNoise()) , info.N ,  L , pdf);

        if(pdf <= 0.0f) return res;
        RayDesc ray;
        ray.Origin = offsetRay(info.P , info.N , generalBuffer.playerPosition);
        ray.Direction = L;
        ray.TMax = 1000.0f;
        ray.TMin = 0.0001f;

        RayQuery<RAY_FLAG_NONE> rq;
        rq.TraceRayInline(TLAS , RAY_FLAG_NONE , 0x01 , ray);
        while(rq.Proceed())
        {
            // a transparent texel is not a hit
            if(checkTransHit(rq.CandidateTriangleBarycentrics() , rq.CandidatePrimitiveIndex() , rq.CandidateInstanceID()))
            {
                rq.CommitNonOpaqueTriangleHit();
            }
        }

        // the hit ( or the sky ) in the same format as the payload of GI_hit / GI_miss
        const HitRayPayLoad load = readGIHit(rq , ray.Origin , ray.Direction , info.LUT_index);

        if(load.didHit){

            const float3 brdf = computeLightAt(load,  seed, info.roughness , info.metallic , info.P);

            res.radiance = brdf;

            const float3 fcos = evalDiffuse(info.albedo , info.N , L , info.metallic);
            
            const float  p_hat = luminance(res.radiance * fcos);
            res.weightSum = p_hat / pdf;
            res.W = 1.0f / pdf;    

        }else{
        
            const float3 F0 = lerp(info.albedo , float3(0.04, 0.04 , 0.04) , (1.0f - info.metallic));
            const float3 V = normalize(generalBuffer.playerPosition - info.P);
            const float NdotV = max(dot(info.N , V) , 0.0001f);
            const float3 F = F_fresnel(F0 , NdotV);
            const float3 kD = (1.0f - F);
            res.radiance = sampleIBL(info.N)* kD;
            const float3 fcos = evalDiffuse(info.albedo , info.N , L , info.metallic);
            const float  p_hat = luminance(res.radiance * fcos);
            res.weightSum = p_hat / pdf;
            res.W = 1.0f / pdf;
        }

        res.sampleNormal = load.normal;
        res.samplePos = load.worldPos;
        res.roughness = load.roughness;
        res.M = 1;

        return res;
    }

    // prevDirectPixel : a pixel of the G buffer ( the full resolution )
    bool historyMatchesPosition(const float3 P , const uint2 prevDirectPixel){
        const float4 prevPosition = prevPositionImage[prevDirectPixel];
        if(prevPosition.w == 0.0f) return false;
        return length(P - prevPosition.xyz) <= 0.05f;
    }

    GeneralObjectInfo readPrevBuffer(GeneralObjectInfo current , uint2 prevDirectPixel)
    {
        GeneralObjectInfo a = current;

        a.P = prevPositionImage[prevDirectPixel].xyz;
        a.N = prevNormalImage[prevDirectPixel].xyz;

        return a;
    }

    float targetFunction(ReservoirGI a , GeneralObjectInfo info)
    {
        float3 fcos = 0.0f;
        const float3 L = normalize(a.samplePos - info.P);
        if(isSpecularMode(info.roughness , info.metallic))
        {
            const float3 F0 = lerp(0.04 , info.albedo , info.metallic);
            fcos = evalBRDF(info.N , info.V , L , info.roughness , F0);
        }
        else
        {
            fcos = evalDiffuse(info.albedo , info.N , L , info.metallic);
        }
        return luminance(fcos * a.radiance);
    }

    // prevIndirectPixel : the reservoir of the last frame ( the resolution of indirectImage ) , prevDirectPixel : the G buffer of the last frame ( the full resolution )
    ReservoirGI temporalReuse_GI( inout uint seed ,ReservoirGI current , uint2 prevIndirectPixel , uint2 prevDirectPixel , GeneralObjectInfo info){
        ReservoirGI r = (ReservoirGI)0;
        const uint2 iSize = indirectSize();
        ReservoirGI prevGI = prevReSTIRGIBuffer[bufferIndex(prevIndirectPixel , iSize.x , iSize.y)];

        const uint M_CAP = 30u;
        prevGI.M = min(prevGI.M , M_CAP * max(current.M , 1u));
        const GeneralObjectInfo prev = readPrevBuffer(info , prevDirectPixel);
        // now to this ReSTIR GI we are going to add MIS , it will be counted as new kinda
        // however it will be posted as a personalEngine3 new ReSTIR GI 
        float3 Q = prevGI.samplePos;
        float3 Q_normal = prevGI.sampleNormal;

        float3 dirFromQ = Q - prev.P;
        float distN = length(dirFromQ);
        float cosN = abs(dot(normalize(-dirFromQ), Q_normal));
        float3 dirFromMine = Q - info.P;
        float distMine = length(dirFromMine);
        float cosMine = abs(dot(normalize(-dirFromMine),  Q_normal));
        float jacobian = (cosMine * distN * distN) / max(cosN * distMine * distMine , 1e-20);

        jacobian = clamp(jacobian , 0.0f , 10.0f);

        prevGI.W *= jacobian;

        const float p_hat_A_x_a = targetFunction(current , info);
        const float p_hat_B_x_a = targetFunction(current , prev);
        const float p_hat_A_x_b = targetFunction(prevGI , info);
        const float p_hat_B_x_b = targetFunction(prevGI , prev);

        const float m_a = float(current.M) * p_hat_A_x_a / max(float(current.M) * p_hat_A_x_a + float(prevGI.M) * p_hat_B_x_a , 1e-20f);
        const float m_b = float(prevGI.M) * p_hat_B_x_b / max(float(prevGI.M) * p_hat_B_x_b + float(current.M) * p_hat_A_x_b , 1e-20f);

        const float w_c = m_a * p_hat_A_x_a * current.W;
        const float w_p = m_b * p_hat_A_x_b * prevGI.W;

        r.update(current , w_c , current.M , rand(seed));
        r.update(prevGI , w_p , prevGI.M , rand(seed));

        const float p_hat = targetFunction(r , info);
        r.W = (p_hat > 0.0f && current.M > 0u) ? r.weightSum / p_hat : 0.0f;
        if(!isfinite(r.W)) r.W = 0.0f;
        return r;
    }

    uint pickRandomNumber(inout uint seed , uint totalCount){
        return uint(float(totalCount) * rand(seed));
    }


    ReservoirGI ReSTIR_GI_generation(inout uint seed , GeneralObjectInfo info , const uint2 indirectPixel){
        ReservoirGI res = (ReservoirGI)0;
        if(!features.ReSTIRGI){
            res.W = 1.0f;
            res.radiance = 1.0f;
            return res;
        }

        res = generateReservoirGI(info , indirectPixel , seed);

        if(features.ReSTIRGI_temporal){
            float4 clipPos = mul(generalBuffer.prevViewProj , float4(info.P , 1.0f));
            float3 pos = clipPos.xyz / clipPos.w;
            float2 uv = pos.xy * 0.5f + 0.5f;
            uv.y = 1.0f - uv.y;
            const bool validUV = (uv.x >= 0.0f && uv.x < 1.0f && uv.y >= 0.0f && uv.y < 1.0f);
            if(validUV){
                const uint2 prevIndirect = uvToIndirectPixel(uv);
                // the reservoir of prevIndirect was made for this pixel of the G buffer ( the jitter of the last frame ) : its position and normal are the ones the jacobian needs
                const uint2 prevDirect = prevDirectPixelOf(prevIndirect);
                if(historyMatchesPosition(info.P , prevDirect)){
                    res = temporalReuse_GI(seed , res , prevIndirect , prevDirect , info);
                }
            }
        }
        return res;
    }
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width , height;
    indirectImage.GetDimensions(width , height);
    if(any(id.xy >= uint2(width , height))) return;

    uint seed = initSeed(id.xy , generalBuffer.frameIndex);
    initNoise(id.xy , generalBuffer.frameIndex);

    const uint2 directPixel = indirectToDirectPixel(id.xy , indirectJitter(id.xy , generalBuffer.frameIndex));

    GeneralObjectInfo info = (GeneralObjectInfo)0;
    float3 totalIndirect = 0.0f;
    float dHit = 10000.0f;
    ReservoirGI gi = (ReservoirGI)0;

    if(pixelCheck[directPixel] == COLOR_PIXEL)
    {
        dHit = 0.0f;
        info = infoFromGBuffer(directPixel);

        if(features.ReSTIRGI)
        {
            gi = ReSTIRGI::ReSTIR_GI_generation(seed , info , id.xy);
            if(gi.M > 0)
            {
                const float3 L = normalize(gi.samplePos - info.P);
                dHit = length(gi.samplePos - info.P);
                totalIndirect = gi.radiance * gi.W * evalDiffuse(info.albedo , info.N , L , info.metallic);
            }
        }
        else if(features.SHaRC)
        {
            const float d = length(generalBuffer.playerPosition - info.P);
            const uint index = Sharc::getIndex(info.P , info.N , d);
            const Sharc::Reservoir sharc = Sharc::tableBuffer[index];
            dHit = (sharc.valid != 0u) ? max(length(sharc.hitPoint - info.P) , 1.0f) : 10000.0f;
            const float3 rad = Sharc::getGradRadiance(info.P , info.N , d);
            totalIndirect = rad * info.albedo * (1.0f - info.metallic);
        }
        else if(features.NRCFull)
        {
            totalIndirect = AI::NRC::askNRC(info.P , info.N) * info.albedo * (1.0f - info.metallic);
            dHit = 10.0f;
        }
    }

    const float3 albedoSafe = max(info.albedo , 0.02f);
    indirectImage[id.xy] = float4(totalIndirect / albedoSafe , 1.0f);
    indirectDistance[id.xy] = min(max(dHit , 2.0f) , 10000.0f);
    ReSTIRGIBuffer[bufferIndex(id.xy , width , height)] = gi;
}
