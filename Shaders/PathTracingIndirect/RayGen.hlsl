// ReSTIR PT of the INDIRECT light : its own pass ( ray tracing pipeline , the recursion of the secondary bounces needs it ) with its own resolution ( see IndirectPixels.hlsl ) .
// the hit and the miss shaders of the pipeline are the ones of Shaders/PathTracing , the sbt is the same
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
#include "../../include/moxSourceCluster/IndirectPixels.hlsl"

#define RT_SHADOW_MISS 0
#define RT_SHADOW_HIT 0
#define RT_GI_HIT 1
#define RT_GI_MISS 1
#define RT_REFLECTION_HIT 2
#define RT_REFLECTION_MISS 2
#define RT_SECONDARY_BOUNCE_HIT 3
#define RT_SECONDARY_BOUNCE_MISS 3


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

typedef RayPayLoad_secondary_ RayPayLoadSecondary;


// the ray of the visibility ( ReSTIR PT temporal reuse )
float calcMultiScatterFactor(float3 F0 , float roughness){
    // average of the three channels, not their sum
    const float F0_avr = (F0.x + F0.y + F0.z) / 3.0f;
    const float Ess = MultiScatterRoughness.SampleLevel(txSampler , roughness, 0).x;
    return 1.0f + F0_avr * (1.0f / max(Ess , 0.001f) - 1.0f);
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

float3 stretchView(const float3 V , const float a){
    return normalize(float3(a  * V.x , V.y * a , V.z));
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

RayPayLoadSecondary makeBounce(const uint seed)
{
    RayPayLoadSecondary bounce = (RayPayLoadSecondary)0;
    bounce.throughput = 1.0f;
    bounce.accumulatedColor = 0.0f;
    bounce.hitDistance = 0.0f;
    bounce.firstBounce = true;
    bounce.depth = 1u;
    bounce.seed = seed;
    return bounce;
}



namespace ReSTIRPT
{
    float3 shiftedDirection(const ReservoirPT r , const float3 P)
    {
        return (r.normal.w == MOX_RESTIR_PT_HIT_COMMITTED) ? normalize(r.position - P) : r.nextDir;
    }

    float targetFunction(const ReservoirPT r , GeneralObjectInfo info , const float3 L)
    {
        float3 fcos;
        if(isSpecularMode(info.roughness , info.metallic))
        {
            const float3 F0 = lerp(0.04 , info.albedo , info.metallic);
            fcos = evalBRDF(info.N , info.V , L , info.roughness , F0);
        }
        else
        {
            fcos = evalDiffuse(info.albedo , info.N , L , info.metallic);
        }
        static const float PT_TARGET_MAX_CHANNEL = 1.0f;
        const float3 v = r.radiance * fcos;
        return (PT_TARGET_MAX_CHANNEL > 0.5f) ? max(v.x , max(v.y , v.z)) : luminance(v);
    }

    // the same , the direction is the one from x1 of info to x2 of the sample ( the shift )
    float targetFunction(const ReservoirPT r , GeneralObjectInfo info)
    {
        return targetFunction(r , info , shiftedDirection(r , info.P));
    }

    // prevDirectCoord : a pixel of the G buffer of the last frame ( the full resolution )
    GeneralObjectInfo readPrevBuffer(const GeneralObjectInfo current , const uint2 prevDirectCoord)
    {
        GeneralObjectInfo infoB = current;
        infoB.P = prevPositionImage[prevDirectCoord].xyz;
        infoB.N = prevNormalImage[prevDirectCoord].xyz;
        return infoB;
    }

    ReservoirPT generateCandidate(GeneralObjectInfo info , inout uint seed)
    {
        ReservoirPT r = (ReservoirPT)0;
        
        float pdf = 0.0f;
        float3 L = 0.0f;
        float3 fcos = 0.0f;

        r.normal.w = MOX_RESTIR_PT_NON_HIT_COMMITTED;

        const bool spec = isSpecularMode(info.roughness , info.metallic);
        if(!spec)
        {
            sampleDiffuseDirection(float2(randNoise() , randNoise()), info.N , L , pdf);
            if(pdf > 0.0f)
            {
                RayDesc ray;
                ray.Origin = offsetRay(info.P , info.N , generalBuffer.playerPosition);
                ray.Direction = L;
                ray.TMax = 1000.0f;
                ray.TMin = 0.001f;
                RayPayLoadSecondary bounce = makeBounce(seed);
                TraceRay(TLAS , RAY_FLAG_NONE , 0x01 , RT_SECONDARY_BOUNCE_HIT , 0 , RT_SECONDARY_BOUNCE_MISS , ray , bounce);
                fcos = evalDiffuse(info.albedo, info.N , L , info.metallic);
                r.radiance = bounce.accumulatedColor;
                r.hD = min(990ul , bounce.hitDistance);
                r.normal.w = (bounce.didHit ? MOX_RESTIR_PT_HIT_COMMITTED : MOX_RESTIR_PT_NON_HIT_COMMITTED);
                r.normal.xyz = bounce.normal.xyz;
                r.position = bounce.hitPoint;
            }
            else
            {
                r.radiance = 0.0f;
                r.hD = 990ul;
            }
            if(pdf <= 0.0f) return r;
        }
        else
        {
            sampleSpecularDirection( float2(randNoise() , randNoise()) ,info.N , info.V , info.roughness , L ,pdf);
            if(pdf > 0.0f)
            {
                float3 F0 = lerp(0.04 , info.albedo , info.metallic);
                RayDesc ray;
                ray.Origin = offsetRay(info.P , info.N , generalBuffer.playerPosition);
                ray.Direction = L;
                ray.TMax = 1000.0f;
                ray.TMin = 0.001f;
                RayPayLoadSecondary bounce = makeBounce(seed);
                TraceRay(TLAS , RAY_FLAG_NONE , 0x01 , RT_SECONDARY_BOUNCE_HIT , 0 , RT_SECONDARY_BOUNCE_MISS , ray , bounce);
                fcos = evalBRDF(info.N , info.V , L , info.roughness , F0);
                r.radiance = bounce.accumulatedColor;
                r.hD = min(990ul , bounce.hitDistance);
                r.normal.w = (bounce.didHit ? MOX_RESTIR_PT_HIT_COMMITTED : MOX_RESTIR_PT_NON_HIT_COMMITTED);
                r.normal.xyz = bounce.normal.xyz;
                r.position = bounce.hitPoint;
            }
            else
            {
                r.radiance = 0.0f;
                r.hD = 990ul;
            }
            if(pdf <= 0.0f) return r;

        }

        r.nextDir = L;
        r.M = 1ul;
        float  p_hat = targetFunction(r , info);
        r.weightSum = p_hat / pdf;
        r.W = 1.0f / pdf;    

        return r;
    }

    static const uint PT_COMBINE_M_CAP = 20u;
    static const float PT_P_HAT_MIN = 1e-3f;
    static const float PT_W_CAP_FACTOR = 10.0f;
    static const float PT_SHIFT_RATIO_MAX = 4.0f;

    ReservoirPT combineWithJacobian(ReservoirPT a , ReservoirPT b , GeneralObjectInfo main , const float3 nP , inout uint seed , const uint2 prevDirectCoord)
    {
        ReservoirPT output = b;
        const float candidateW = a.W;

        float jacobian = 1.0f;
        // a miss is only a direction : the distance is infinite , nothing changes in the jacobian
        if(b.normal.w == MOX_RESTIR_PT_HIT_COMMITTED)
        {
            const float3 Q = b.position;
            const float3 Q_normal = b.normal.xyz;

            const float3 dirFromQ = Q - nP;
            const float distN = length(dirFromQ);
            const float cosN = abs(dot(normalize(-dirFromQ) , Q_normal));
            const float3 dirFromMine = Q - main.P;
            const float distMine = length(dirFromMine);
            const float cosMine = abs(dot(normalize(-dirFromMine) , Q_normal));
            jacobian = (cosMine * distN * distN) / max(cosN * distMine * distMine , 1e-20);
            jacobian = clamp(jacobian , 0.0f , 10.0f);
        }
        output.M = min(b.M , PT_COMBINE_M_CAP);
        output.W *= clamp(jacobian , 0.0f , 10.0f);
        const float p_hat_old = targetFunction(b , main , b.nextDir);
        output.nextDir = shiftedDirection(output , main.P);

        const float p_hat_old_T = targetFunction(output , main);

        const float p_hat_new = targetFunction(a , main);
        const float3 newDir = shiftedDirection(a , main.P); 
        const float3 old_new_direction = a.nextDir;
        a.nextDir = newDir;
        const float p_hat_new_T = targetFunction(a , main);
        a.nextDir = old_new_direction;

#ifdef MOX_MIS_INCLUDED

        GeneralObjectInfo info2 = readPrevBuffer(main , prevDirectCoord);

        const float p_hat_A_x_a = p_hat_new;
        const float p_hat_B_x_a = targetFunction(a , info2);
        const float p_hat_B_x_b = targetFunction(output , info2);
        const float p_hat_A_x_b = p_hat_old_T;

        // the denominator is 0 when no technique could give the sample ( both p_hat are 0 ) : without the guard it is 0 / 0 = NaN
        const float m_a = float(a.M) * p_hat_A_x_a / max(float(a.M) * p_hat_A_x_a + float(output.M) * p_hat_B_x_a , 1e-20f);
        const float m_b = float(output.M) * p_hat_B_x_b / max(float(output.M) * p_hat_B_x_b + float(a.M) * p_hat_A_x_b , 1e-20f);

        const float w_c = m_a * p_hat_new_T * a.W;
        const float w_p = m_b * p_hat_old_T * output.W;
        ReservoirPT r = (ReservoirPT)0;

        r.update(a , w_c , a.M , rand(seed));
        r.update(output , w_p , output.M , rand(seed));

        const float p_hat = targetFunction(r , main);
        r.W = (p_hat > PT_P_HAT_MIN && a.M > 0u) ? r.weightSum / (float(r.M) * p_hat) :    0.0f;
        if(!isfinite(r.W)) r.W = 0.0f;
        return r;
#else
        if(PT_SHIFT_RATIO_MAX > 0.0f && p_hat_old_T > PT_P_HAT_MIN && p_hat_old_T > PT_SHIFT_RATIO_MAX * p_hat_old) return a;
        const float weight = clamp(p_hat_old_T * output.W * float(output.M), 0.0f , a.W * 2.0f);;

        a.update(output , weight , output.M , rand(seed));

        const float p_hat_chosen = targetFunction(a , main);
        a.W = (p_hat_chosen > PT_P_HAT_MIN && a.M > 0u) ? a.weightSum / (float(a.M) * p_hat_chosen) :    0.0f;
        if(PT_W_CAP_FACTOR > 0.0f && candidateW > 0.0f) a.W = min(a.W , PT_W_CAP_FACTOR * candidateW);
#endif
        return a;
    }

    // the jacobian of moving x2 ( Q ) from the x1 of the neighbour ( nP ) to ours ( P ) : the same formula as in combineWithJacobian
    float shiftJacobian(const float3 Q , const float3 Q_normal , const float3 nP , const float3 P)
    {
        const float3 dirFromQ = Q - nP;
        const float distN = length(dirFromQ);
        const float cosN = abs(dot(normalize(-dirFromQ) , Q_normal));
        const float3 dirFromMine = Q - P;
        const float distMine = length(dirFromMine);
        const float cosMine = abs(dot(normalize(-dirFromMine) , Q_normal));
        const float jacobian = (cosMine * distN * distN) / max(cosN * distMine * distMine , 1e-20);
        return clamp(jacobian , 0.0f , 10.0f);
    }

    // prevIndirectCoord : the reservoir of the last frame ( the resolution of indirectImage ) , prevDirectCoord : the pixel of the G buffer that it was made for ( the full resolution )
    ReservoirPT temporalReuse(ReservoirPT c , GeneralObjectInfo info , uint2 prevIndirectCoord , uint2 prevDirectCoord , inout uint seed)
    {
        const uint2 iSize = indirectSize();
        ReservoirPT p = prevReSTIRPTBuffer[bufferIndex(prevIndirectCoord , iSize.x , iSize.y)];

        if(p.M == 0u || p.normal.w == MOX_RESTIR_PT_NON_HIT_COMMITTED) return c;
        if(!(p.W > 0.0f) || isinf(p.W) || any(isnan(p.radiance)) || any(isinf(p.radiance))) return c;

        p.M = min(p.M , PT_COMBINE_M_CAP * max(c.M , 1u));


        ReservoirPT r = (ReservoirPT)0;
        {
            const float p_hat = targetFunction(c , info);
            r.update(c , p_hat * c.W * float(c.M) , c.M , rand(seed));
        }
        {
            // x2 stays , it is seen from our x1 now : W of the old sample gets the jacobian of this move ( x1 of the old pixel is the position of the last frame )
            p.W *= shiftJacobian(p.position , p.normal.xyz , prevPositionImage[prevDirectCoord].xyz , info.P);
            p.nextDir = shiftedDirection(p , info.P);
            const float p_hat = targetFunction(p , info);
            r.update(p , p_hat * p.W * float(p.M) , p.M , rand(seed));
        }

        const float p_hat_chosen = targetFunction(r , info);
        r.W = (p_hat_chosen > 1e-30f && r.M > 0u) ? r.weightSum / (float(r.M) * p_hat_chosen) : 0.0f;
        return r;
    }
    // the version with the ray of visibility and the jacobian ( combineWithJacobian )
    ReservoirPT temporalReuseJacobian(ReservoirPT c , GeneralObjectInfo info, uint2 prevIndirectCoord , uint2 prevDirectCoord , inout uint seed)
    {
        const uint2 iSize = indirectSize();
        ReservoirPT p = prevReSTIRPTBuffer[bufferIndex(prevIndirectCoord , iSize.x , iSize.y)];

        // now we want to find out if the prev reservoir is even suitable
        // we will check it simply to throwing our ray to it
        if(p.normal.w == MOX_RESTIR_PT_NON_HIT_COMMITTED) return c; // if the prev reservoir doesnt have a radiance , we wont check it 
        // now we get the direction we will check
        if(p.M == 0) return c;
        float3 direction = p.position - info.P;
        const float epsilon = 0.02f;
        const float d = length(direction) - epsilon;
        direction = normalize(direction);
        
        const float shadow = calculateShadows(offsetRay(info.P, info.N, generalBuffer.playerPosition) , direction , d);
        
        if(shadow != 1.0f) return c; // since we dont want to check reservoir that is not suitable for us 
        const float3 pP = prevPositionImage[prevDirectCoord].xyz;
        return combineWithJacobian(c , p ,info , pP , seed , prevDirectCoord);
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

GeneralObjectInfo infoFromGBuffer(uint2 directPixel)
{
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

[shader("raygeneration")]
void main()
{
    const uint2 indirectPixel = DispatchRaysIndex().xy;
    const uint2 iSize = indirectSize();
    if(any(indirectPixel >= iSize)) return;

    uint seed = initSeed(indirectPixel , generalBuffer.frameIndex);
    initNoise(indirectPixel , generalBuffer.frameIndex);

    const uint2 directPixel = indirectToDirectPixel(indirectPixel , indirectJitter(indirectPixel , generalBuffer.frameIndex));

    GeneralObjectInfo info = (GeneralObjectInfo)0;
    float3 totalIndirect = 0.0f;
    float dHit = 10000.0f;
    ReservoirPT pt = (ReservoirPT)0;

    // not a pixel of a surface ( the sky ) : there is no indirect result to calculate
    if(pixelCheck[directPixel] == COLOR_PIXEL)
    {
        dHit = 0.0f;
        info = infoFromGBuffer(directPixel);

        if(features.PT)
        {
            // a candidate , then the reuse of the sample of the last frame
            pt = ReSTIRPT::generateCandidate(info , seed);

            const float4 clipPos = mul(generalBuffer.prevViewProj , float4(info.P , 1.0f));
            if(clipPos.w > 0.0f)
            {
                float2 prevUV = (clipPos.xy / clipPos.w) * 0.5f + 0.5f;
                prevUV.y = 1.0f - prevUV.y;
                const bool validUV = (prevUV.x >= 0.0f && prevUV.x < 1.0f && prevUV.y >= 0.0f && prevUV.y < 1.0f);
                if(validUV)
                {
                    // the reservoir of the last frame is at the indirect resolution : the pixel of the G buffer that it was made for comes from the jitter of the last frame
                    const uint2 prevIndirect = uvToIndirectPixel(prevUV);
                    const uint2 prevDirect = prevDirectPixelOf(prevIndirect);
                    if(checkDisocclusion(directPixel , prevDirect))
                    {
                        pt = ReSTIRPT::temporalReuseJacobian(pt , info , prevIndirect , prevDirect , seed);
                    }
                }
            }

            if(pt.M > 0u)
            {
                totalIndirect = pt.radiance * pt.W * evalDiffuse(info.albedo , info.N , pt.nextDir , info.metallic);
                dHit = pt.hD;
            }
        }
    }

    // the same as RayTracingIndirect : the albedo is applied again after the denoiser ( HDR / PostProcess multiply by albedoImage ) , the distance of the indirect light is never smaller than 2 m ( the radius of the blur )
    const float3 albedoSafe = max(info.albedo , 0.02f);
    indirectImage[indirectPixel] = float4(totalIndirect / albedoSafe , 1.0f);
    indirectDistance[indirectPixel] = min(max(dHit , 2.0f) , 10000.0f);
    ReSTIRPTBuffer[bufferIndex(indirectPixel , iSize.x , iSize.y)] = pt;
}
