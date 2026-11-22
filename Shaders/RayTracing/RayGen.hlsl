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

#define RT_SHADOW_MISS 0
#define RT_SHADOW_HIT 0
#define RT_GI_HIT 1
#define RT_GI_MISS 1
#define RT_REFLECTION_HIT 2
#define RT_REFLECTION_MISS 2

struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

struct [raypayload] HitRayPayLoad{
    bool   didHit : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 worldPos : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 normal : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float3 albedo : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float  roughness : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    float  metallic : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    // the LUT index of the surface that the ray starts from : the caller writes it , the closesthit compares it with the one of the hit
    uint   LUT_index : write(caller,closesthit , miss) : read(caller,closesthit , miss);
    // true if the hit has the same LUT_index as the caller ( the closesthit writes it , the miss writes false )
    bool   sameLUT : write(caller,closesthit , miss) : read(caller,closesthit , miss);
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


// ---------------------------------------------------------------- the sun disc
static const float SUN_ANGULAR_RADIUS = 1.0f;
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
    // no ACCEPT_FIRST_HIT : the depth has to be the one of the NEAREST blocker ( the first accepted hit can be a farther one ) , as in calculateSunShadow
    TraceRay(TLAS, RAY_FLAG_NONE , 0x01 , RT_SHADOW_HIT , 0 , RT_SHADOW_MISS , ray, shadow);
    float data = 1.0f;
    if(!shadow.inShadow) return data;
    // VISIBILITY of the lamp along this ray : r = depth / distance to the lamp . a blocker at the pixel ( r -> 0 ) gives 0 ( the full shadow ) , a blocker at the lamp
    // ( r -> 1 ) gives 1 ( no shadow ) . it was 1 - r : the contact shadow was the brightest one and the far one the darkest
    const float r = saturate(shadow.depth / max(maxDistance , 1e-20f));
    return r;
}

// the shadow of the sun with a fade : how much of the sun reaches the pixel along this ray . 1 - nothing covers it , 0 - covered at once . the shadow gets weaker
// with the distance from the pixel to what covers it ( the nearest hit along the ray to the sun , it is the depth that shadowHit.hlsl writes ) : it is fully gone at
// fadeDistance ( smoothstep , features.sunShadowFade , in meters ) . fadeDistance 0 - no fade , the same as calculateShadows
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
    // 0 samples : no sky light ( and no division by 0 below )
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

// how much of the sky the point sees : 0 ( covered , a room ) ... 1 ( open ) . the share of the rays of the cone around the up direction that left the scene . it is a WEIGHT , not a yes / no :
// with a few rays the yes / no flickers at the border of a cover , the weight changes in steps
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

namespace ReSTIRGI{





    // next task is to fix the sun shadows
    // MULTI BOUNCE : the indirect light that the hit point itself gets , taken from the reservoir of ReSTIR GI of the LAST frame . the pixel of the last frame that shows this point is
    // found by prevViewProj , and its reservoir is used only when it is the same surface ( the normal and the plane ) . DIFFUSE ONLY : the radiance of the stored sample goes through
    // the diffuse lobe of the hit ( without the albedo : computeLightAt multiplies it back at the end ) . there is no specular path : the spikes came from it
    float3 previousBounce(const GeneralObjectInfo data)
    {
        const float4 clipPos = mul(generalBuffer.prevViewProj , float4(data.P , 1.0f));
        if(clipPos.w <= 0.0f) return 0.0f;

        float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f;
        prevUV.y = 1.0f - prevUV.y;
        if(any(prevUV < 0.0f) || any(prevUV >= 1.0f)) return 0.0f;

        const uint2 screenSize = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
        const uint2 prevPixel = min(uint2(prevUV * float2(screenSize)) , screenSize - 1u);

        // the same surface : the normal is about the same and the point is on the plane of the surface of that pixel ( the tolerance grows with the distance : a pixel is wider there )
        const float4 prevPosition = prevPositionImage[prevPixel];
        if(prevPosition.w == 0.0f) return 0.0f;
        const float3 prevNormal = prevNormalImage[prevPixel].xyz;
        if(dot(prevNormal , data.N) < 0.9f) return 0.0f;
        const float tolerance = max(0.05f , 0.01f * length(data.P - generalBuffer.playerPosition));
        if(abs(dot(data.P - prevPosition.xyz , prevNormal)) > tolerance) return 0.0f;

        const ReservoirGI prev = prevReSTIRGIBuffer[bufferIndex(prevPixel)];
        if(prev.M == 0u) return 0.0f;

        const float3 L = normalize(prev.samplePos - data.P);
        const float3 bounce = prev.radiance * evalDiffuse(float3(1.0f , 1.0f , 1.0f) , data.N , L , data.metallic) * prev.W;
        // NaN is not taken , and a firefly of the last frame is limited ( it would come back through the loop )
        return sanitizeRadiance(bounce , RC::FEEDBACK_MAX_RADIANCE);
    }

    // the DIFFUSE light of the hit ( the specular of the lamps and of the sun is not here : a metal has no diffuse , so it is 0 for it , the reflection ray below gives its colour )

// void diskBasis(float3 N, out float3 U, out float3 V)
// {
//     float s = (N.z >= 0.0f) ? 1.0f : -1.0f;
//     float a = -1.0f / (s + N.z);
//     float b = N.x * N.y * a;
//     U = float3(1.0f + s * N.x * N.x * a, s * b, -s * N.x);
//     V = float3(b, s + N.y * N.y * a, -N.y);
// }

// // float3 pointU = center + U * radius;   // край диска в сторону U
// // float3 pointV = center + V * radius;   // край диска в сторону V

// // float r   = radius * sqrt(xi.x);
// // float phi = 2.0f * PI * xi.y;
// // float3 P  = center + U * (r * cos(phi)) + V * (r * sin(phi));

    float3 computeDiffuseLightAt(HitRayPayLoad info, inout uint seed, const float roughness , const float xMetallic, const float3 x1Pos){
        // in order to literally minimaze the atmosphere effect , we actually need to lower the chances of it to appear *
        if(!info.didHit){
            return 0.0f;
        }
        uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
        uint2 pixel = DispatchRaysIndex().xy;

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
        }else{
            result =  0.01f;
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
            result += sampleSkybox(data.P , data.N) * skyKD;
        }

        // the multi bounce : the light that this point gets itself ( the reservoir of the last frame ) , diffuse only
        result += previousBounce(data);

        result *= saveAlbedo;

        return result;
    }

    float3 computeLightAt(HitRayPayLoad info, inout uint seed, const float roughness , const float xMetallic, const float3 x1Pos){
        if(!info.didHit) return 0.0f;
        return computeDiffuseLightAt(info , seed , roughness , xMetallic , x1Pos);
    }

    ReservoirGI generateReservoirGI(GeneralObjectInfo info , const uint2 pixel ,inout uint seed){
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

        HitRayPayLoad load = (HitRayPayLoad)0;

        TraceRay(TLAS , RAY_FLAG_NONE , 0x01 , RT_GI_HIT , 0 , RT_GI_MISS , ray , load);

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
            res.radiance = sampleSkybox(info.P , info.N) * kD;
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

    bool historyMatchesPosition(const float3 P , const uint2 prevPixel){
        const float4 prevPosition = prevPositionImage[prevPixel];
        if(prevPosition.w == 0.0f) return false;
        return length(P - prevPosition.xyz) <= 0.05f;
    }

    GeneralObjectInfo readPrevBuffer(GeneralObjectInfo current , uint2 prevPixel)
    {
        GeneralObjectInfo a = current;

        a.P = prevPositionImage[prevPixel].xyz;
        a.N = prevNormalImage[prevPixel].xyz;

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

    ReservoirGI temporalReuse_GI( inout uint seed ,ReservoirGI current , uint2 prevCoord , GeneralObjectInfo info){
        ReservoirGI r = (ReservoirGI)0;

        ReservoirGI prevGI = prevReSTIRGIBuffer[bufferIndex(prevCoord)];

        const uint M_CAP = 30u;
        prevGI.M = min(prevGI.M , M_CAP * max(current.M , 1u));
        const GeneralObjectInfo prev = readPrevBuffer(info , prevCoord);
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
        r.W = (p_hat > 1e-3f && current.M > 0u) ? r.weightSum / p_hat :    0.0f;
        // if(!isfinite(r.W)) r.W = 0.0f;
        return r;
    }

    uint pickRandomNumber(inout uint seed , uint totalCount){
        return uint(float(totalCount) * rand(seed));
    }


    ReservoirGI ReSTIR_GI_generation(inout uint seed , GeneralObjectInfo info){
        ReservoirGI res = (ReservoirGI)0;
        if(!features.ReSTIRGI){
            res.W = 1.0f;
            res.radiance = 1.0f;
            return res;
        }

        uint2 pixel = DispatchRaysIndex().xy;

        res = generateReservoirGI(info , pixel  , seed);

        if(features.ReSTIRGI_temporal){
            uint2 size = DispatchRaysDimensions().xy;
            float4 clipPos = mul(generalBuffer.prevViewProj , float4(info.P , 1.0f));
            float3 pos = clipPos.xyz / clipPos.w;
            float2 uv = pos.xy * 0.5f + 0.5f;
            uv.y = 1.0f - uv.y;
            // the check is made BEFORE the conversion to uint : a negative float does not become a pixel outside of the screen , and an unsigned
            // value is never < 0 ( the old check was always true )
            const bool validUV = (uv.x >= 0.0f && uv.x < 1.0f && uv.y >= 0.0f && uv.y < 1.0f);
            if(validUV){
                const uint2 prevPixel = min(uint2(uv * float2(size)) , size - 1u);
                // only the history of the same surface is reused ( the position of the last frame is close to this one )
                if(historyMatchesPosition(info.P , prevPixel)){
                    res = temporalReuse_GI(seed , res , prevPixel ,info);
                }
            }
        }
        return res;
    }
}
float gaussianProfile(const float sigma , const float r)
{
    const float s2 = max(sigma * sigma , 1e-8f);
    return exp(-(r * r) / (2.0f * s2)) / (2.0f * PI * s2);
}


namespace ReSTIRSSS
{

    ReservoirSS generateCandidate(GeneralObjectInfo info , inout uint seed)
    {
        const float u1 = clamp(rand(seed) , 0.0f , 0.999f);
        const float u2 = rand(seed);
        const LUT lut = LUT_Buffer[info.LUT_index];
        const float sigma = lut.sigma;
        const float r = sigma * sqrt(max(-2.0f * log(1.0f - u1) , 0.0f));
        const float maxR = float(ReservoirSS::maxR);
        ReservoirSS a = (ReservoirSS)0;
        a.valid = 0;
        if(r >= maxR) return a;      
        const float h = sqrt(maxR * maxR - r * r);
        float3 T , B;
        buildONB(info.N , T , B);
        const float phi = 2.0f * PI * u2;

        RayDesc ray;
        ray.Origin = info.P + r * (cos(phi) * T + sin(phi) * B) + info.N * h;
        ray.Direction = -info.N;
        ray.TMax = 2 * h;
        ray.TMin = 0.0001f;

        HitRayPayLoad load = (HitRayPayLoad)0;

        TraceRay(TLAS , RAY_FLAG_NONE , 0x01 , RT_GI_HIT , 0 , RT_GI_MISS , ray , load);


        if(load.didHit)
        {
            a.valid = 1;
            a.position = load.worldPos;
            a.roughness = load.roughness;
            a.metallic = load.metallic;
            a.M = 1;
            a.weightSum = 0.0f; 
            a.normal = load.normal;
        }
        else
        {
            a.valid = 0;
        }

        return a;
    }

    ReservoirSS temporalReuse(ReservoirSS a , const float p_hat_c , const uint2 prev , inout uint seed , const float sigma , GeneralObjectInfo info)
    {
        ReservoirSS b = prevReSTIRSSBuffer[bufferIndex(prev)];
        if(!b.valid) return a;
        b.M = min(b.M , ReservoirSS::maxM);

        const float w_c = a.W * float(a.M) * p_hat_c;
        const float g_b = gaussianProfile(sigma , length(b.position - info.P));
        const float w_p = b.W * float(b.M) * luminance(b.radiance * g_b);

        ReservoirSS c = b;              
        c.M = b.M;
        c.weightSum = w_p;              
        c.update(a , w_c , a.M , rand(seed));   

        const float g_sel = gaussianProfile(sigma , length(c.position - info.P));
        c.W = c.weightSum / (float(c.M) * max(luminance(c.radiance * g_sel) , 1e-8f));
        c.valid = 1;
        return c;
    }
}

namespace Reflection
{
    float3 radianceAt(const HitRayPayLoad hit , inout uint seed , const uint2 pixel , const float3 V)
    {
        GeneralObjectInfo info = (GeneralObjectInfo)0;
        info.P = hit.worldPos;
        info.N = hit.normal;
        info.V = V;
        info.albedo = hit.albedo;
        info.metallic = hit.metallic;
        info.roughness = hit.roughness;

        const float3 rayOrigin = offsetRay(info.P , info.N , generalBuffer.playerPosition);

        float3 totalDiffuse = 0.0f;
        float3 totalSpecular = 0.0f;
        float3 totalIndirect = 0.0f;

        // one lamp of the cell of the hit , the weight is lightCount ( the chance of the lamp is 1 / lightCount )
        uint lightCount , lightIndices[lg::MAX_LIGHT_PER_CELL];
        findLightIndices(info.P , lightIndices , lightCount);
        if(lightCount > 0u)
        {
            const uint lampSlot = min(uint(rand(seed) * float(lightCount)) , lightCount - 1u);
            const lg::LightSource light = lightBuffer[lightIndices[lampSlot]];
            float3 L = light.position - info.P;
            const float d = length(L);
            L /= max(d , 1e-4f);
            const float NdotL = saturate(dot(info.N , L));
            const float att = lightAttenuation(light , d);
            if(NdotL > 0.0f && att > 0.0f)
            {
                const float shadow = calculateShadowToCenter(light , info.P , info.N);
                if(shadow > 0.0f)
                {
                    float3 diffuse = 0.0f , specular = 0.0f;
                    computeLight(info , L , diffuse , specular);
                    const float3 energy = light.diffuse * att * NdotL * shadow * float(lightCount);
                    totalDiffuse += diffuse * energy;
                    totalSpecular += specular * energy;
                }
            }
        }

        // the sun
        const float skyVisibility = skyVisibilityAt(info.P , info.N , seed);
        if(sunBuffer.intensity > 0.0f && skyVisibility > 0.0f)
        {
            const float3 L = randomInSunRadius(seed , sunBuffer , -sunBuffer.direction);
            const float sunNdotL = saturate(dot(info.N , L));
            if(sunNdotL > 0.0f)
            {
                const float sunShadow = calculateSunShadow(rayOrigin , L , 1000.0f , 200.0f);
                float3 sunDiffuse = 0.0f , sunSpecular = 0.0f;
                computeLight(info , L , sunDiffuse , sunSpecular);
                totalDiffuse += sunDiffuse * sunBuffer.diffuse * sunNdotL * sunShadow;
                totalSpecular += sunSpecular * sunBuffer.diffuse * sunNdotL * sunShadow;
            }
        }

        // the indirect light of the hit : one GI ray , the diffuse lobe only ( a metal has no diffuse ) . ( Radiance Cascade and NRC of pe2 are not in this shader yet )
        if((1.0f - info.metallic) > 0.3f)
        {
            if(features.ReSTIRGI)
            {
                const ReservoirGI gi = ReSTIRGI::generateReservoirGI(info , pixel , seed);
                if(gi.M > 0u)
                {
                    const float3 L = normalize(gi.samplePos - info.P);
                    totalIndirect = gi.radiance * evalDiffuse(info.albedo , info.N , L , info.metallic) * gi.W;
                }
            }
        }

        // NaN / Inf would go through the denoiser into the history and live there
        if(!all(isfinite(totalIndirect))) totalIndirect = 0.0f;
        if(!all(isfinite(totalDiffuse))) totalDiffuse = 0.0f;
        if(!all(isfinite(totalSpecular))) totalSpecular = 0.0f;

        return totalDiffuse + totalSpecular + totalIndirect;
    }

    // the radiance that the ray brings from direction : the sky if it leaves the scene , the light of the hit if not . hitDistance is for the radius of the filter of ReLAX
    float3 trace(const float3 direction , const float3 position , inout uint seed , const uint2 pixel , out float hitDistance)
    {
        HitRayPayLoad hit = (HitRayPayLoad)0;

        RayDesc ray;
        ray.Origin = position;
        ray.Direction = direction;
        ray.TMax = 1000.0f;
        ray.TMin = 0.001f;

        TraceRay(TLAS , RAY_FLAG_NONE , 0xFF , RT_REFLECTION_HIT , 0 , RT_REFLECTION_MISS , ray , hit);

        // the miss shader puts the point 1000 away : the distance is limited like in pe2 ( 900 )
        hitDistance = min(length(hit.worldPos - position) , 900.0f);
        // the miss shader gives the radiance of the sky ( without the sun disc ) in albedo
        if(!hit.didHit) return hit.albedo;

        return radianceAt(hit , seed , pixel , -direction);
    }
}

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

struct LightData{
    float3 diffuse;
    float3 specular;
    float3 indirect;
    float shadow;
    float diffuseHitDistance;
    float specularHitDistance;
};

struct PixelData{
    float3 viewDir;
    float2 motionVector;
    float pixelType;
};

struct [raypayload] RayPayLoad{
    uint depth : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    uint seed : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    PixelData pixelInfo : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    LightData lightInfo : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
    bool fromReflection : read(closesthit , miss, caller) : write(closesthit , miss , caller) ;
};

float2 skyMotionVector(const float3 dir , const uint2 pixel){
    const float2 size = float2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    const float2 uv = (float2(pixel) + 0.5f) / size;
    const float4 clip = mul(generalBuffer.prevViewProj , float4(dir , 0.0f));
    if(clip.w <= 1e-6f) return float2(0.0f , 0.0f);
    float2 prevUV = (clip.xy / clip.w) * 0.5f + 0.5f;
    prevUV.y = 1.0f - prevUV.y;
    return prevUV - uv;
}

static const float MOTION_TOLERANCE = 8.0f;
static const float MAX_LIGHT_CONE_SIN = 0.05f;

[shader("raygeneration")]
void main()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 size = DispatchRaysDimensions().xy;
    float2 uv = (float2(pixel) + 0.5f) / float2(size);
    
    RayPayLoad load = (RayPayLoad)0;
    ReservoirGI gi = (ReservoirGI)0;
    ReservoirPT pt = (ReservoirPT)0;
    ReservoirSS ss = (ReservoirSS)0;

    ss.valid = 0;

    load.seed = initSeed(pixel , generalBuffer.frameIndex);
    initNoise(pixel , generalBuffer.frameIndex);

    if(pixelCheck[pixel] != COLOR_PIXEL)
    {
        float2 ndc = uv * 2.0 - 1.0f;
        ndc.y = -ndc.y;

        float4 pos = mul(generalBuffer.inverseProj , float4(ndc , 1.0f , 1.0f));
        float3 worldDir = mul((float3x3)generalBuffer.inverseView , pos.xyz).xyz;
        const float3 color = skybox.SampleLevel(txSampler , worldDir , 0).xyz;
        const float3 result = AddSunDisc(color , worldDir);

        // write here
        load.lightInfo.diffuse = result;
        load.lightInfo.specular = float3(0.0f ,0.0f , 0.0f);
        load.lightInfo.indirect = float3(0.0f ,0.0f , 0.0f);
        load.pixelInfo.pixelType = SKYBOX_PIXEL;
        load.lightInfo.shadow = 1.0f;
        load.lightInfo.diffuseHitDistance = 10000.0f;
        load.lightInfo.specularHitDistance = 10000.0f;

        load.pixelInfo.motionVector = skyMotionVector(worldDir , pixel);
        load.pixelInfo.viewDir = worldDir;
    }   
    else
    {
        initNoise(pixel , generalBuffer.frameIndex);
        GeneralObjectInfo info = infoFromGBuffer(pixel);

        float3 finalDiffuse = 0.0f;
        float3 finalSpecular = 0.0f;
        float3 finalIndirect = 0.0f;
        float finalShadow = 0.0f;
        float tdiffuseDistance = 0.0f;
        float tspecularDistance = 0.0f;

        float4 clipPos = mul(generalBuffer.prevViewProj , float4(info.P , 1.0f));
        float3 pos = clipPos.xyz / clipPos.w;
        float2 prevUV = pos.xy * 0.5f + 0.5f;
        prevUV.y = 1.0f - prevUV.y;
        const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
        uint2 prevPixel = min(uint2(max(prevUV * float2(size) , 0.0f)) , size - 1u);                


        uint tileLightCount , tileLightOffset;
        findLightIndices(pixel , tileLightOffset , tileLightCount);
        float3 totalDiffuse = 0.0f;
        float3 totalSpecular = 0.0f;   
        float totalShadow = 0.0f;
        float3 totalIndirect = 0.0f;
        float sunShadow = 1.0f;
        float lightShadow = 1.0f;
        float sunLuminance = 0.0f;
        float lightLuminance = 0.0f;

        const bool needSpecular = isSpecularMode(info.roughness , info.metallic);
        const bool needDiffuse  = ((1.0f - info.metallic) > 0.3f);

        const float3 F0 = lerp(float3(0.04f , 0.04f , 0.04f) , info.albedo , info.metallic);
        const float3 rayOrigin = offsetRay(info.P , info.N , generalBuffer.playerPosition);

        const float2 motion = prevUV - uv;
        const float motionDist = length(motion * float2(generalBuffer.screenWidth , generalBuffer.screenHeight));
        const float motionW = smoothstep(0.0f , MOTION_TOLERANCE , motionDist);

        const float objectAmbient = pow(objectAOImage[pixel].x , features.AO_intens);
        float3 reflectedSpecular = 0.0f;
        const float NdotV = saturate(dot(info.N , info.V));  
        const float3 F_o = F_fresnel(F0 , NdotV);
        // REFLECTION ( the specular lobe is one ray , the light of its hit is computed in Reflection:: )
        if(needSpecular)
        {
            float3 L; float pdf;
            sampleSpecularDirection(float2(randNoise() , randNoise()) , info.N , info.V , info.roughness , L , pdf);
            if(pdf > 0.0f)
            {
                const float NdotV = saturate(dot(info.V , info.N));
                float reflectionDistance;
                const float3 radiance = Reflection::trace(L , rayOrigin , load.seed , pixel , reflectionDistance);
                const float3 weight = F_fresnel(F0 , NdotV);
                float3 reflectedColor = radiance * weight;

                // the lobe of a metal is narrow : one NaN / Inf goes through the denoiser into the history and lives there ( the whole specular goes black )
                if(any(isnan(reflectedColor)) || any(isinf(reflectedColor))) reflectedColor = 0.0f;
                reflectedSpecular += reflectedColor;
                tspecularDistance = reflectionDistance;
            }
        }

        // GI here

        if(features.ReSTIRGI)
        {
            gi = ReSTIRGI::ReSTIR_GI_generation(load.seed , info);
        }

        // direct
        float indirectLum = 0.0f;

        if(features.ReSTIRSSS) ss = ReSTIRSSS::generateCandidate(info , load.seed);
        
        if(ss.valid)
        {
            const float c = 1.0f - 2.0f * LUT_Buffer[info.LUT_index].c;

            GeneralObjectInfo infoSS = info;
            infoSS.N = ss.normal;
            infoSS.P = ss.position;
            infoSS.roughness = ss.roughness;
            infoSS.metallic = ss.metallic;
            infoSS.V = normalize(generalBuffer.playerPosition - infoSS.P);

            const float skyVisibility = skyVisibilityAt(infoSS.P , infoSS.N , load.seed);
            if(sunBuffer.intensity > 0.0f && skyVisibility > 0.0f){
                // sun
                // the direction of the sun is the direction of the LIGHT : the vector to the sun is the opposite one , and it has to be a unit vector
                const float3 L = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
                const float sunNdotL = saturate(dot(infoSS.N , L)); 
                const float3 diff = sunBuffer.diffuse;
                float3 specular = 0.0f;
                // the weight of the sun in the shadow : the less sky the point sees , the less the sun counts there
                sunLuminance = luminance(sunBuffer.diffuse) * smoothstep(0.0f, 0.7f , dot(L , infoSS.N)) * skyVisibility;
                computeLight(info , L , specular);
                // a surface that looks away from the sun is not lit by it whatever is in the way : no rays ( the weight of the sun is 0 there too )
                if(sunNdotL > 0.0f){
                    const float3 dir = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
                    float rawShadow = calculateSunShadow(offsetRay(infoSS.P , infoSS.N , generalBuffer.playerPosition) , dir , 1000.0f , 200.0f);
                    sunShadow = (sunLuminance * rawShadow + indirectLum) / max(indirectLum + sunLuminance , 1e-20);
                    // if(needSpecular) sunShadow = 1.0f;
                }
                // however we are doing different diffuse here
                const float3 F_i = F_fresnel(F0 , sunNdotL);

                const float3 diffuse = (1.0f - F_o) * (1.0f - F_i) / ( PI * c) * info.albedo * sunShadow * diff * sunNdotL; 

                specular = lerp(0.0f, specular , sunShadow);
                totalDiffuse += diffuse;
                totalSpecular += specular * diff * sunNdotL;
            }else sunShadow = 1.0f;


            if(tileLightCount > 0){

                float3 totalLightSpecular = 0.0f;
                float3 totalLightDiffuse = 0.0f;

                float weightSum = 0.0f;
                float visibleSum = 0.0f;
                [loop]
                for(uint i = 0 ; i < tileLightCount ;i++){
                    const uint lightID = tileLightIndices[i + tileLightOffset];
                    lg::LightSource light = lightBuffer[lightID];

                    if(light.type == MOX_AREA_LIGHT_TYPE_DISK)
                    {
                        float3 U , V;
                        diskBasis(light.direction , U , V);
                        const float A = PI * light.sourceRadius * light.sourceRadius;
                        const float pdf_A = 1.0f / max(A , 1e-3f);

                        const float x1 = rand(load.seed);
                        const float x2 = rand(load.seed);
                        const float r = sqrt(x1) * light.sourceRadius;
                        const float phi = 2.0f * PI * x2;
                        const float3 lightPos = light.position + r * cos(phi) * U + r * sin(phi) * V;

                        float3 L = lightPos - infoSS.P;
                        const float d = length(L);
                        L = normalize(L);
                        const float epsilon = 1e-3f;

                        float3 L_nonSS = lightPos - info.P;
                        const float d_nonSS = length(L_nonSS);
                        L_nonSS = normalize(L_nonSS);

                        float3 specular;

                        computeLight(info ,L_nonSS , specular );

                        const float NdotL = saturate(dot(infoSS.N , L));  
                        const float DdotL = saturate(dot(-L , light.direction));

                        float window = calculateLightWindow(light , d);

                        const float sinAlpha = min(light.radius / max(d , light.radius) , MAX_LIGHT_CONE_SIN);
                        const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);

                        const float reach = lightShadowReach(d , light.radius , features.lightShadowFade);
                        // the weight is the weight of the PIXEL ( x_o ) : the distance , the direction , the normal and the reach of info , not of infoSS
                        const float reach_o = lightShadowReach(d_nonSS , light.radius , features.lightShadowFade);
                        const float falloff_o = 1.0f / (1.0f + light.Linear * d_nonSS + light.Quadratic * d_nonSS * d_nonSS);
                        const float weight = luminance(light.diffuse) * falloff_o * saturate(dot(info.N , L_nonSS)) * reach_o;

                        const float3 coneSamp = sampleCone(L , cosAlpha , load.seed);

                        float shadow = calculateShadows(offsetRay(infoSS.P , infoSS.N , generalBuffer.playerPosition) , L , d - 0.01f);
                        shadow = lerp(1.0f , shadow , reach);

                        // result = diffuse * light.diffuse * shadow * window * NdotL * DdotL * lightCount / max(pdf_A * d * d , 0.0001f);
                        const float3 lightEm = light.diffuse * shadow * DdotL * NdotL * window /  max(pdf_A * d * d , 0.0001f);
                        const float3 F_i = F_fresnel(F0 , NdotL);

                        const float3 diffuse = (1.0f - F_o) * (1.0f - F_i) / ( PI * c) * info.albedo * lightEm;
                        const float NdotL_o = saturate(dot(info.N , L_nonSS));
                        const float DdotL_o = saturate(dot(-L_nonSS , light.direction));
                        const float window_o = calculateLightWindow(light , d_nonSS);

                        float shadow_o = calculateShadows( offsetRay(info.P , info.N , generalBuffer.playerPosition) , L_nonSS , d_nonSS - 0.01f );
                        shadow_o = lerp(1.0f , shadow_o , reach_o);
                        specular *= light.diffuse * shadow_o * DdotL_o * NdotL_o * window_o / max(pdf_A * d_nonSS * d_nonSS , 0.0001f);

                        weightSum += weight;
                        visibleSum += weight * shadow_o;
                        lightLuminance += weight;

                        totalLightDiffuse += diffuse;
                        totalLightSpecular += specular;
                    }
                    else
                    {
                        const float3 realL = normalize(light.position - info.P);
                        const float3 lightPos = randomInLightRadius(load.seed , light , realL);
                        float3 L = lightPos - infoSS.P;
                        const float d = length(L);
                        L = normalize(L);
                        const float epsilon = 1e-3f;

                        float3 L_nonSS = lightPos - info.P;
                        const float d_nonSS = length(L_nonSS);
                        L_nonSS = normalize(L_nonSS);

                        float3 specular;

                        computeLight(info ,L_nonSS , specular );

                        const float NdotL = saturate(dot(infoSS.N , L));  
                        const float DdotL = saturate(dot(-L , light.direction));

                        float window = calculateLightWindow(light , d);

                        const float sinAlpha = min(light.radius / max(d , light.radius) , MAX_LIGHT_CONE_SIN);
                        const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);

                        const float reach = lightShadowReach(d , light.radius , features.lightShadowFade);
                        // the weight is the weight of the PIXEL ( x_o ) : the distance , the direction , the normal and the reach of info , not of infoSS
                        const float reach_o = lightShadowReach(d_nonSS , light.radius , features.lightShadowFade);
                        const float falloff_o = 1.0f / (1.0f + light.Linear * d_nonSS + light.Quadratic * d_nonSS * d_nonSS);
                        const float weight = luminance(light.diffuse) * falloff_o * saturate(dot(info.N , L_nonSS)) * reach_o;

                        const float3 coneSamp = sampleCone(L , cosAlpha , load.seed);

                        float shadow = calculateShadows(offsetRay(infoSS.P , infoSS.N , generalBuffer.playerPosition) , L , d - 0.01f);
                        shadow = lerp(1.0f , shadow , reach);

                        // result = diffuse * light.diffuse * shadow * window * NdotL * DdotL * lightCount / max(pdf_A * d * d , 0.0001f);
                        const float att = lightAttenuation(light , d);
                        const float3 lightEmission = light.diffuse * att * NdotL * shadow;
                        const float3 F_i = F_fresnel(F0 , NdotL);

                        const float3 diffuse = (1.0f - F_o) * (1.0f - F_i) / ( PI * c) * info.albedo * lightEmission;

                        const float NdotL_o = saturate(dot(info.N , L_nonSS));
                        const float DdotL_o = saturate(dot(-L_nonSS , light.direction));
                        const float window_o = calculateLightWindow(light , d_nonSS);

                        float shadow_o = calculateShadows( offsetRay(info.P , info.N , generalBuffer.playerPosition) , L_nonSS , d_nonSS - 0.01f );
                        shadow_o = lerp(1.0f , shadow_o , reach_o);
                        const float att_o = lightAttenuation(light , d_nonSS);
                        const float3 lightEmission_o = light.diffuse * att_o * NdotL_o;
                        specular *= lightEmission_o * shadow_o;

                        weightSum += weight;
                        visibleSum += weight * shadow_o;
                        lightLuminance += weight;

                        totalLightDiffuse += diffuse;
                        totalLightSpecular += specular;
                    }
                }
                if(weightSum > 0.0f) lightShadow = visibleSum / weightSum;
                // if(needSpecular) lightShadow = 1.0f;

                totalDiffuse += totalLightDiffuse;
                totalSpecular += totalLightSpecular;


            }else lightShadow = 1.0f;
    
        }
        else
        {
            const float skyVisibility = skyVisibilityAt(info.P , info.N , load.seed);
            if(sunBuffer.intensity > 0.0f && skyVisibility > 0.0f){
                // sun
                // the direction of the sun is the direction of the LIGHT : the vector to the sun is the opposite one , and it has to be a unit vector
                const float3 L = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
                const float sunNdotL = saturate(dot(info.N , L));
                const float3 diff = sunBuffer.diffuse;
                float3 diffuse = 0.0f;
                float3 specular = 0.0f;
                // the weight of the sun in the shadow : the less sky the point sees , the less the sun counts there
                sunLuminance = luminance(sunBuffer.diffuse) * smoothstep(0.0f, 0.7f , dot(L , info.N)) * skyVisibility;
                computeLight(info , L , diffuse , specular);
                // a surface that looks away from the sun is not lit by it whatever is in the way : no rays ( the weight of the sun is 0 there too )
                if(sunNdotL > 0.0f){
                    sunShadow =0.0f;
                    const float sunCosAlpha = cos(sunBuffer.sourceRadius);
                    const float3 sunOrigin = pos + L * 0.01f;
                    [unroll]
                    for(uint i = 0 ; i < 1 ; i++){
                        const float3 dir = randomInSunRadius(load.seed , sunBuffer , -sunBuffer.direction);
                        float rawShadow = calculateSunShadow(offsetRay(info.P , info.N , generalBuffer.playerPosition) , dir , 1000.0f , 200.0f);
                        rawShadow = (sunLuminance * rawShadow + indirectLum) / max(indirectLum + sunLuminance , 1e-20);
                        sunShadow += rawShadow;
                    }

                    sunShadow /= float(1);
                    // if(needSpecular) sunShadow = 1.0f;
                }

                specular = lerp(0.0f, specular , sunShadow);
                diffuse = lerp(0.0f , diffuse , sunShadow);
                totalDiffuse += diffuse * diff * sunNdotL;
                totalSpecular += specular * diff * sunNdotL;
            }else sunShadow = 1.0f;

            // light here should be chosen by the forward + , not world grid light poses

            {
                // lights
                if(tileLightCount > 0){

                    float3 totalLightSpecular = 0.0f;
                    float3 totalLightDiffuse = 0.0f;

                    float weightSum = 0.0f;
                    float visibleSum = 0.0f;
                    [loop]
                    for(uint i = 0 ; i < tileLightCount ;i++){
                        const uint lightID = tileLightIndices[i + tileLightOffset];
                        lg::LightSource light = lightBuffer[lightID];

                        if(light.type == MOX_AREA_LIGHT_TYPE_DISK)
                        {
                            float3 U , V;
                            diskBasis(light.direction , U , V);
                            const float A = PI * light.sourceRadius * light.sourceRadius;
                            const float pdf_A = 1.0f / max(A , 1e-3f);

                            const float x1 = rand(load.seed);
                            const float x2 = rand(load.seed);
                            const float r = sqrt(x1) * light.sourceRadius;
                            const float phi = 2.0f * PI * x2;
                            const float3 lightPos = light.position + r * cos(phi) * U + r * sin(phi) * V;
                            float3 L = lightPos - info.P;
                            const float d = length(L);
                            const float epsilon = 1e-3f;
                            L = normalize(L);
                            float3 diffuse, specular;
                            computeLight(info ,L , diffuse , specular );
                            const float NdotL = saturate(dot(info.N , L));  
                            const float DdotL = saturate(dot(-L , light.direction));
                            float window = calculateLightWindow(light , d);
                            const float sinAlpha = min(light.radius / max(d , light.radius) , MAX_LIGHT_CONE_SIN);
                            const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);
                            const float reach = lightShadowReach(d , light.radius , features.lightShadowFade);
                            const float falloff = 1.0f / (1.0f + light.Linear * d + light.Quadratic * d * d);
                            const float weight = luminance(light.diffuse) * falloff * saturate(dot(info.N , L)) * reach;
                            const float3 coneSamp = sampleCone(L , cosAlpha , load.seed);
                            float shadow = calculateShadows(offsetRay(info.P , info.N , generalBuffer.playerPosition) , L , d - 0.01f);
                            shadow = lerp(1.0f , shadow , reach);
                            // result = diffuse * light.diffuse * shadow * window * NdotL * DdotL * lightCount / max(pdf_A * d * d , 0.0001f);
                            diffuse *= light.diffuse * shadow * DdotL * NdotL * window /  max(pdf_A * d * d , 0.0001f);
                            specular *= light.diffuse * shadow * DdotL * NdotL * window /  max(pdf_A * d * d , 0.0001f);

                            weightSum += weight;
                            visibleSum += weight * shadow;
                            lightLuminance += weight;

                            totalLightDiffuse += diffuse;
                            totalLightSpecular += specular;
                        }
                        else
                        {
                            float3 L = light.position - info.P;
                            const float realDistance = length(L);
                            L = normalize(L);
                            const float3 lightPos = randomInLightRadius(load.seed , light , L);
                            float3 dir = lightPos - info.P;
                            const float d = length(dir);
                            dir = normalize(dir);
                            float shadow = 0.0f;
                            const float reach = lightShadowReach(d , light.radius , features.lightShadowFade);
                            const float falloff = 1.0f / (1.0f + light.Linear * d + light.Quadratic * d * d);
                            const float weight = luminance(light.diffuse) * falloff * saturate(dot(info.N , dir)) * reach;
                            const float NdotL = dot(info.N , dir);
                            if(reach <= 0.0f || weight <= 0.0f || NdotL <= 0.0f) continue;
                            const float sinAlpha = min(light.radius / max(d , light.radius) , MAX_LIGHT_CONE_SIN);
                            const float cosAlpha = sqrt(1.0f - sinAlpha * sinAlpha);
                            const float att = lightAttenuation(light , d);
                            const float3 lightEmission = light.diffuse * att * NdotL;
                            float3 diffuse = 0.0f , specular = 0.0f;
                            computeLight(info , dir , diffuse, specular);

                            float shadowForEmission = 0.0f;
                            [unroll]
                            for(uint j = 0 ; j < 1 ;j++){
                                float currentShadow = calculateShadowInSphere(load.seed , light , info.P , info.N);
                                const float3 coneSamp = sampleCone(L , cosAlpha , load.seed);
                                shadowForEmission += calculateShadows(offsetRay(info.P , info.N , generalBuffer.playerPosition) , coneSamp , realDistance);
                                currentShadow = (weight * currentShadow + indirectLum) / max(indirectLum + weight , 1e-20);
                                shadow += clamp(currentShadow , 0.0f , 1.0f);
                            }
                            shadow /= float(1);
                            shadowForEmission /= float(1);

                            shadow = lerp(1.0f , shadow , reach);

                            specular *= lerp(0.0f , lightEmission , shadowForEmission) ;
                            diffuse *=  lerp(0.0f , lightEmission , shadowForEmission);

                            diffuse = lerp(0.0f, diffuse , shadowForEmission);
                            specular = lerp(0.0f , specular , shadowForEmission);
                
                            weightSum += weight;
                            visibleSum += weight * shadow;
                            lightLuminance += weight;

                            totalLightDiffuse += diffuse;
                            totalLightSpecular += specular;
                        }
                    }
                    if(weightSum > 0.0f) lightShadow = visibleSum / weightSum;
                    // if(needSpecular) lightShadow = 1.0f;

                    totalDiffuse += totalLightDiffuse;
                    totalSpecular += totalLightSpecular;


                }else lightShadow = 1.0f;
            }

        }

        const float totalWeight = sunLuminance + lightLuminance;
        totalShadow = (totalWeight > 1e-6f) ? (sunLuminance * sunShadow + lightLuminance * lightShadow) / totalWeight : 1.0f;
        
        if(validUV){
            const float prevShadowData = prevShadowImage[prevPixel];
            const float prevShadow = prevShadowData;
            float count = 32.0f;
            const float diff = abs(prevShadow - totalShadow) * motionW;
            count = min(count + 1u , MAX_SHADOW_FRAME);
            const float baseAlpha = 1.0f / float(count);
            const float alpha = lerp(baseAlpha , 1.0f , pow(diff , 2.0f));
            totalShadow = lerp(prevShadow , totalShadow , alpha);
        }

        const float3 albedoSafe = max(info.albedo , 0.02f);
        float3 diffuseOut = (totalDiffuse * objectAmbient) / albedoSafe;

        if(ss.valid && validUV)
        {
            ss.radiance = diffuseOut;
            const float sigma = LUT_Buffer[info.LUT_index].sigma;
            float r2 = length(info.P - ss.position);
            float g = gaussianProfile(sigma , r2);
            const float p_hat = luminance(ss.radiance * g);
            const float pdf = max(g , 1e-6f);
            ss.W = 1.0f / pdf;
            ss.weightSum = p_hat / pdf;
            ss = ReSTIRSSS::temporalReuse(ss ,p_hat, prevPixel , load.seed , sigma , info);
            const float c = 1.0f - 2.0f * LUT_Buffer[info.LUT_index].c;
            r2 = length(info.P - ss.position);
            g = gaussianProfile(sigma , r2);
            diffuseOut = ss.W * ss.radiance * g;
        }

        load.lightInfo.diffuseHitDistance = min(tdiffuseDistance , 10000.0f);
        if(!load.fromReflection) load.lightInfo.specularHitDistance = min(tspecularDistance , 10000.0f);
        if(!load.fromReflection) load.lightInfo.specularHitDistance = min(tspecularDistance , 10000.0f);
        load.pixelInfo.viewDir = info.V;
        load.lightInfo.shadow = totalShadow;
        load.lightInfo.diffuse = (diffuseOut);
        load.lightInfo.indirect = totalIndirect / albedoSafe;
        load.lightInfo.specular = (totalSpecular * objectAmbient + reflectedSpecular);
        load.pixelInfo.motionVector = motion;
        load.pixelInfo.pixelType = COLOR_PIXEL;
    }

    diffuseImage[pixel]      = float4(load.lightInfo.diffuse , 1.0f);
    diffuseDistance[pixel]   = load.lightInfo.diffuseHitDistance;
    specularImage[pixel]     = float4(load.lightInfo.specular , 1.0f);
    specularDistance[pixel]  = load.lightInfo.specularHitDistance;
    indirectImage[pixel]     = float4(load.lightInfo.indirect , 1.0f);
    indirectDistance[pixel]  = load.lightInfo.diffuseHitDistance;
    shadowImage[pixel]       = load.lightInfo.shadow;
    viewImage[pixel]         = float4(load.pixelInfo.viewDir , 1.0f);
    motionVector[pixel]      = load.pixelInfo.motionVector;
    pixelCheck[pixel]        = load.pixelInfo.pixelType;

    const uint reservoirID = bufferIndex(pixel);
    ReSTIRGIBuffer[reservoirID] = gi;
    ReSTIRPTBuffer[reservoirID] = pt;
    ReSTIRSSBuffer[reservoirID] = ss;
}