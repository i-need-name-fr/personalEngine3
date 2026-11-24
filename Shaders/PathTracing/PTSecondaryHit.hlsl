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
#define MOX_NRC_INCLUDED 8
#include "../../include/AI/CoopVec.hlsl"
#include "../../include/AI/NRC/NRC.hlsl"

struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};


struct [raypayload] RayPayLoad_primary_
{
    uint seed : write(caller , closesthit , miss) : read(caller , closesthit , miss);
};

struct [raypayload] RayPayLoad_secondary_
{
    float3 throughput : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    float3 accumulatedColor : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    float3 hitPoint : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    float3 normal : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    float hitDistance : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    uint depth : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    uint seed : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    bool firstBounce : write(caller , closesthit , miss) : read(caller , closesthit , miss);
    bool didHit  : write(caller , closesthit , miss) : read(caller , closesthit , miss);
};

typedef RayPayLoad_primary_ RayPayLoad;
typedef RayPayLoad_secondary_ RayPayLoadSecondary;

#define RT_SHADOW_MISS 0
#define RT_SHADOW_HIT 0
#define RT_GI_HIT 1
#define RT_GI_MISS 1
#define RT_REFLECTION_HIT 2
#define RT_REFLECTION_MISS 2
#define RT_SECONDARY_BOUNCE_HIT 3
#define RT_SECONDARY_BOUNCE_MISS 3

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


// ---------------------------------------------------------------- the sun disc
static const float SUN_ANGULAR_RADIUS = 0.00465f;
static const float SUN_DISC_BRIGHTNESS = 100.0f;
static const float SUN_DISC_MAX = 60000.0f;

float3 sunTransmittanceToCamera(){
    const float u = skyBuffer.sunCosZenith * 0.5f + 0.5f;
    const float v = saturate(skyBuffer.cameraHeight / (ATMOSPHERE_RADIUS - PLANET_RADIUS));
    return txTransmissionLUT.SampleLevel(txSampler , float2(u , v) , 0).rgb;
}

float3 AddSunDisc(float3 skyColor , float3 rayDir){
    const float lightLength = length(sunBuffer.direction);
    // the light is the direction where the light goes : the sun is on the other side
    if(lightLength < 1e-5f) return skyColor;
    const float3 toSun = -sunBuffer.direction / lightLength;

    // nothing under the horizon
    if(rayDir.y < 0.0f) return skyColor;

    // the angle of one pixel from the projection ( projOnly[1][1] is 1 / tan( fov / 2 ) )
    const float pixelAngle = 2.0f / (max(generalBuffer.projOnly[1][1] , 1e-3f) * float(max(generalBuffer.screenHeight , 1u)));
    const float cosAngle = dot(rayDir , toSun);
    if(cosAngle < cos(SUN_ANGULAR_RADIUS + pixelAngle)) return skyColor;

    const float angle = acos(clamp(cosAngle , -1.0f , 1.0f));
    const float coverage = saturate((SUN_ANGULAR_RADIUS + 0.5f * pixelAngle - angle) / pixelAngle);

    const float3 disc = min(sunBuffer.diffuse * SUN_DISC_BRIGHTNESS * sunTransmittanceToCamera() , float3(SUN_DISC_MAX , SUN_DISC_MAX , SUN_DISC_MAX));
    return skyColor + disc * coverage;
}

// shadow calc
float calculateShadows(float3 position , float3 direction , float maxDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    ShadowRayPayLoad shadow;
    shadow.inShadow = true;
    shadow.depth = maxDistance;
    TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH , 0x01 , RT_SHADOW_HIT , 0 , RT_SHADOW_MISS , ray, shadow);
    const float data = (shadow.inShadow )? 0.0f  : 1.0f;
    return data;
}

// this funcntion also 
float calculateShadowAtt(float3 position , float3 direction , float maxDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    ShadowRayPayLoad shadow;
    shadow.inShadow = true;
    shadow.depth = maxDistance;
    TraceRay(TLAS, RAY_FLAG_NONE , 0x01 , RT_SHADOW_HIT , 0 , RT_SHADOW_MISS , ray, shadow);
    float data = 1.0f;
    if(!shadow.inShadow) return data;
    const float r = saturate(shadow.depth / max(maxDistance , 1e-20f));
    return r;
}

float calculateSunShadow(float3 position , float3 direction , float maxDistance , float fadeDistance){
    RayDesc ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.TMax = maxDistance;
    ray.TMin = 0.001f;

    ShadowRayPayLoad shadow;
    shadow.inShadow = true;
    shadow.depth = maxDistance;

    if(fadeDistance <= 0.0f){
        TraceRay(TLAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH , 0xFF , RT_SHADOW_HIT , 0 , RT_SHADOW_MISS , ray, shadow);
        return shadow.inShadow ? 0.0f : 1.0f;
    }

    // no ACCEPT_FIRST_HIT : the closest hit shader runs for the NEAREST accepted hit ( the lamp bodies are ignored by the any hit ) , and its depth is what is needed
    TraceRay(TLAS, RAY_FLAG_NONE , 0x01 , RT_SHADOW_HIT , 0 , RT_SHADOW_MISS , ray, shadow);
    if(!shadow.inShadow) return 1.0f;
    return smoothstep(0.0f , fadeDistance , shadow.depth);
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
    // L goes FROM the point TO the lamp , the visible half of the sphere is the one that looks back at the point : the offsets that look the way of L ( dot > 0 )
    // are on the far side of the lamp , they are turned over ( it was dot < 0 : the far half was chosen )
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
    // L goes FROM the point TO the lamp , the visible half of the sphere is the one that looks back at the point : the offsets that look the way of L ( dot > 0 )
    // are on the far side of the lamp , they are turned over ( it was dot < 0 : the far half was chosen )
    const float facing = dot(offset , L);
    if(facing > 0.0f) offset = -offset;
    // the sun is at infinity : its "sphere" is a disc of the angular radius R on the unit sphere of directions , so the result is a DIRECTION ( the centre L plus the offset , normalized ) ,
    // not the offset alone ( it was a vector of length ~0.005 : dot(N , L) and the light went to 0 )
    return normalize(normalize(L) + offset);
}

float calculateShadowInSphere(inout uint seed , lg::LightSource light , const float3 P , const float3 N){
    float3 L = light.position - P;
    const float d = length(L);
    L = normalize(L);
    const float3 randomPos = randomInLightRadius(seed , light , L);
    // the ray ends at the chosen point of the sphere , not at its centre ( offsetRay wants the position of the camera as the third argument , not a distance )
    const float3 toSample = randomPos - P;
    const float sampleDistance = length(toSample);
    const float3 dir = toSample / max(sampleDistance , 1e-4f);
    return calculateShadowAtt(offsetRay(P , N , generalBuffer.playerPosition) , dir , sampleDistance);
}

// the shadow of a lamp for the INDIRECT light ( the lamps that light the hits of the GI , Radiance Cascade , the reflections and the bounces of the path tracer ) : the ray goes to the
// CENTRE of the lamp , not to a random point of its sphere . with a big sphere the random point is on the other side of a wall , and the light of a lamp under the floor leaked through
float calculateShadowToCenter(lg::LightSource light , const float3 P , const float3 N){
    const float3 toLight = light.position - P;
    const float distance = length(toLight);
    const float3 dir = toLight / max(distance , 1e-4f);
    // the FULL shadow ( 0 or 1 ) : calculateShadowAtt gives the visibility r = depth / distance , a wall near the lamp then let a part of the light through
    return calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , dir , distance);
}



float3 stretchView(const float3 V , const float a){
    return normalize(float3(a  * V.x , V.y * a , V.z));
}

GeneralObjectInfo infoFromGBuffer(uint2 pixel){
    GeneralObjectInfo info;
    info.P = positionImage[pixel].xyz;
    info.N = normalImage[pixel].xyz;
    info.V = normalize(generalBuffer.playerPosition - info.P);
    info.albedo = albedoImage[pixel].xyz;
    info.roughness = roughnessImage[pixel];
    info.metallic = metallicImage[pixel];
    return info;
}

float3 sampleSkybox(const float3 P ,const float3 N){
    float3 result = 0.0f;
    // 0 samples : no sky light ( and no division by 0 below )
    if(features.skyboxSampleCount == 0u) return result;
    [loop]
    for(uint i = 0 ; i  < features.skyboxSampleCount ; i++){
        const float3 direction = randomInHemisphere(N);
        const float shadow = calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , direction , 5.0f);
        result += IBL.SampleLevel(txSampler , direction , 0).xyz * shadow;
    }
    result /= float(features.skyboxSampleCount);
    result = lerp(0.0f , result , sunBuffer.intensity);
    return result;
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

bool isSpecularMode(const float roughness , const float metallic){ return (metallic >= 0.7f && roughness < 0.4f); }

float3 evalFcos(const GeneralObjectInfo info , const bool specularMode , const float3 L){
    if(specularMode){
        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
        return evalBRDF(info.N , info.V , L , info.roughness , F0) * calcMultiScatterFactor(F0 , info.roughness);
    }
    return evalDiffuse(info.albedo , info.N , L , info.metallic);
}

// how much of the sky the point sees : 0 ( covered , a room ) ... 1 ( open ) . the share of the rays of the cone around the up direction that left the scene . it is a WEIGHT , not a yes / no :
// with a few rays the yes / no flickers at the border of a cover , the weight changes in steps

static const float INDIRECT_FALLOFF_DISTANCE = 8.0f;

float indirectDistanceWeight(float hitDistance){
    const float ratio = hitDistance / INDIRECT_FALLOFF_DISTANCE;
    return 1.0f / (1.0f + ratio * ratio);
}
static const float MAX_LIGHT_CONE_SIN = 0.05f;


bool checkDisocclusion(uint2 pixel , uint2 prevPixel){

    const float3 centerPosition = positionImage[pixel].xyz;
    const float3 centerNormal = normalImage[pixel].xyz;

    const float4 prevPositionSample = prevPositionImage[prevPixel];
    const float3 nNormal = prevNormalImage[prevPixel].xyz;   

    // w == 0 marks a sky texel written by the miss shader
    if(prevPositionSample.w == 0.0f) return false;

    const float positionDiff = length(centerPosition - prevPositionSample.xyz);
    if(positionDiff > 0.05f) return false;

    const float normalDot = dot(centerNormal , nNormal);
    if(normalDot < 0.9f) return false;
    
    return true;
}

struct HitSurface{
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
};

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

// the surface that the ray hit : the position , the normal ( with the normal map ) , the albedo , the roughness and the metallic from the textures of its LUT
GeneralObjectInfo readHitData(const float2 barycentrics , out float objectAO){
    const InstanceData instance = instanceLayoutBuffer[InstanceID()];
    const uint meshID = instance.meshDataIndex;
    const LUT lut = LUT_Buffer[instance.LUT_index];

    const HitSurface surface = interpolateHit(meshID , PrimitiveIndex() , barycentrics);

    // the same model matrix as the vertex shader of the Deferred render uses
    const float4x4 model = instanceBuffer[meshBuffer[meshID].instanceIndex];

    // normalMatrix3x3 already is inverse( transpose( model ) ) ( see the vertex shader ) , the matrix is on the left
    const float3 N = normalize(mul(normalMatrix3x3((float3x3)model) , surface.normal));
    const float3 T = normalize(mul(model , float4(surface.tangent , 0.0f)).xyz);
    const float3 B = normalize(mul(model , float4(surface.bitangent , 0.0f)).xyz);
    const float3x3 TBN = float3x3(T , B , N);

    const float3 hitPoint = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    // the direction back to the point that the ray comes from
    const float3 V = -WorldRayDirection();
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

    // the packed PBR map ( R occlusion , G roughness , B metallic ) : the occlusion is not used yet
    if(lut.armTextureID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.armTextureID];
        const float3 arm = textureHub[id].SampleLevel(txSampler , uv , 0).xyz;
        roughness = arm.y;
        metallic = arm.z;
    }

    // the ao map of the material ( the red channel ) : 1 where there is no map . objectAmbientImage gets it , and the DIRECT light is multiplied by it
    objectAO = 1.0f;
    if(lut.aoTextureID != NON_VALID_SLOT){
        const uint id = txVirtualBuffer[lut.aoTextureID];
        objectAO = textureHub[id].SampleLevel(txSampler , uv , 0).x;
    }

    // the normal looks at the ray ( a back face , or a normal map that tilts the normal away at a groove ) : the same as GI_hit.hlsl and getHitInfo of SHaRC do . without it
    // offsetRay moved the origin of the next ray / of the shadow rays to the WRONG side of the surface
    if(dot(normal , V) < 0.0f) normal = -normal;

    GeneralObjectInfo info = (GeneralObjectInfo)0;
    info.albedo = albedo;
    info.metallic = metallic;
    // the GGX of a perfect mirror ( roughness 0 ) is degenerate : D is 0 almost everywhere , the surface came out black
    info.roughness = max(roughness , 0.1f);
    info.N = normal;
    info.V = V;
    info.P = hitPoint;
    return info;
}

// how much of the sky the point sees : 0 ( covered , a room ) ... 1 ( open ) . the share of the rays of the cone around the up direction that left the scene . under the open sky ( > 0 )
// the sun is traced , in the other case it gives nothing
float skyVisibilityAt(const float3 P , const float3 N , inout uint seed)
{
    const uint sampleCount = features.skyboxSampleCount;
    // no rays : nothing says that the point is covered
    if(sampleCount == 0u) return 1.0f;

    float escaped = 0.0f;
    [loop]
    for(uint i = 0 ; i < sampleCount ; i++){
        const float3 direction = sampleCone(float3(0.0f , 1.0f , 0.0f), 0.5f , seed );
        const float shadow = calculateShadows(offsetRay(P , N , generalBuffer.playerPosition) , direction , 10.0f);
        if(shadow == 1.0f) escaped += 1.0f;
    }
    return escaped / float(sampleCount);
}

uint pickRandomNumber(inout uint seed , uint max)
{
    return uint(rand(seed) * float(max));
}

float powerHeuristic(const float a , const float b )
{
    const float a2 = a * a;
    const float b2 = b * b;
    return a2 / max(a2 + b2,  1e-20);
}

float3 sampleIBL(const float3 N){
    return IBL.SampleLevel(txSampler , N , 0).xyz * sunBuffer.intensity;
}

void processSecondaryBounce(inout RayPayLoadSecondary load , in float2 barycentrics, bool tale)
{
    float objectAO =1.0f;
    GeneralObjectInfo info = readHitData(barycentrics , objectAO);
    if(tale)
    {
        if(features.NRC) 
        {
            load.accumulatedColor += load.throughput * AI::NRC::askNRC(info.P , info.N) * info.albedo * (1.0f - info.metallic);
        }
        return;
    }
    uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
    findLightIndices(info.P , lightIndices , lightCount);
    const float3 safeAlbedo = max(info.albedo / PI , 1e-20);
    const float skyVisibility = skyVisibilityAt(info.P , info.N , load.seed);
    if(sunBuffer.intensity > 0.0f)
    {
        const float3 diffuse = sunBuffer.diffuse;
        const float3 L = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
        const float NdotL = saturate(dot(info.N , L));
        float shadow = 0.0f;
        if(skyVisibility > 0.0f){
            shadow = calculateShadows(offsetRay(info.P , info.N , generalBuffer.playerPosition) , L , 1000.0f);
        }
        float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
        computeLight(info , L , sunDiffuse , sunSpecular);
        load.accumulatedColor += load.throughput * sunDiffuse * diffuse * shadow * NdotL * objectAO;
        const float3 skyV = normalize(WorldRayDirection());
        const float3 skyF0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
        const float skyNdotV = max(dot(info.N , skyV) , 0.0001f);
        const float3 skyF = F_fresnel(skyF0 , skyNdotV);
        const float3 skyKD = (1.0f - skyF) * (1.0f - info.metallic);
        load.accumulatedColor += load.throughput * sampleIBL(skyV) * skyKD * info.albedo * (1.0f - info.metallic);
    }

    // now lets do the lights
    if(lightCount)
    {
        // we pick a random light
        const uint id = pickRandomNumber(load.seed , lightCount);
        const lg::LightSource light = lightBuffer[lightIndices[id]];
        const float pdfLight = 1.0f / lightCount;
        // the centre of the lamp , not a random point of its sphere ( a big sphere is reached through the walls )
        const float3 lightPos = light.position;
        float3 L = lightPos - info.P;
        const float d = length(L);
        L = normalize(L);
        const float cosTheta = saturate(dot(L , info.N));
        if(cosTheta > 0.0f)
        {
            // the ray ends at the lamp : what is behind it does not cover it
            const float shadow = calculateShadows(offsetRay(info.P , info.N , generalBuffer.playerPosition) , L , d);
            if(shadow == 1.0f)
            {
                float3 diffuse = 0.0f, specular = 0.0f;
                computeLight(info , L , diffuse , specular);
                load.accumulatedColor += load.throughput * diffuse * lightAttenuation(light , d) * lightCount * light.diffuse * cosTheta * objectAO;
            }
        }
    }

    const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
    const float3 Fview = F_fresnel(F0 , max(dot(info.N , info.V) , 0.0001f));
    const float specularWeight = luminance(Fview);
    const float diffuseWeight = luminance(info.albedo * (1.0f - info.metallic) * (1.0f - Fview));
    const float specularChance = (diffuseWeight <= 1e-6f) ? 1.0f : clamp(specularWeight / max(specularWeight + diffuseWeight , 1e-6f) , 0.05f , 0.95f);

    const float lobeChance = 1.0f;
    float pdf = 0.0f; float3 nextDir = 0.0f; float3 fcos = 0.0f;
    if(isSpecularMode(info.roughness , info.metallic))
    {
        sampleSpecularDirection(float2(randNoise() , randNoise()) , info.N, info.V , info.roughness , nextDir , pdf);
        if(pdf > 0.0f) fcos = evalBRDF(info.N , info.V , nextDir , info.roughness , F0);
    }
    else
    {
        sampleDiffuseDirection(float2(randNoise() , randNoise()) , info.N , nextDir , pdf);
        if(pdf > 0.0f) fcos = evalDiffuse(info.albedo , info.N , nextDir , info.metallic);
    }

    if(pdf > 0.0f && lobeChance > 0.0f)
    {
        // the weight of the step : f * cos / ( the pdf of the direction * the chance of the lobe that gave it )
            load.throughput *= fcos / (pdf * lobeChance);

            // the distance to the first hit is what the denoiser of the specular needs , it is stored before the path can end
            if(load.firstBounce)
            {
                load.firstBounce = false;
                load.hitDistance = RayTCurrent();
                load.hitPoint = info.P;
                load.didHit = true;
                load.normal = info.N;
            }

            // now the chances ( the roulette by the weight of the path ) : the floor keeps the division away from 0
            const float chance = clamp(max(load.throughput.x , max(load.throughput.y , load.throughput.z)) , 0.05f , 0.95f);
            if(chance < rand(load.seed)) return;
            load.throughput /= chance;

            RayDesc ray;
            ray.Origin = offsetRay(info.P , info.N , generalBuffer.playerPosition);
            ray.Direction = nextDir;
            ray.TMax = 100.0f;
            ray.TMin = 0.001f;

            // 0x01 : the same mask as the rays of the primary hit , the geometry of the lamps ( 0x02 ) is not a surface for the path
            TraceRay(TLAS , RAY_FLAG_NONE , 0x01 , RT_SECONDARY_BOUNCE_HIT , 0 , RT_SECONDARY_BOUNCE_MISS , ray , load);
    }
}


[shader("closesthit")]
void main(inout RayPayLoadSecondary load , in BuiltInTriangleIntersectionAttributes attr)
{   
    const bool tale = load.depth >= MAX_PT_RAY_CAST;
    load.depth++;

    initNoise(DispatchRaysIndex().xy , generalBuffer.frameIndex);

    processSecondaryBounce(load , attr.barycentrics , tale);
}