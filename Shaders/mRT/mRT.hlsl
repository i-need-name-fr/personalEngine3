#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#define MOX_VERTEX_HEAP_INCLUDED 2
#include "../../include/moxObject/VertexHeap.hlsl"
#define MOX_TEXTURE_HUB_INCLUDED 3
#include "../../include/moxSourceCluster/TextureHub.hlsl"

// mRT : the G buffer by RAY TRACING instead of the mesh shaders . one primary ray per pixel ( RayQuery ) , the same images as the fragment shader of the Deferred render writes ( position + the LUT index + 1 ,
// normal , albedo , the linear depth , roughness , metallic , the ao map of the material , pixelCheck ) . the pass runs AFTER the Deferred render and overwrites its G buffer ( the depth attachment of the raster
// pass stays : DLSS , HiZ and the culling still use it ) . if a bug disappears with mRT ON , it was in the raster G buffer ( culling , missing meshlets ) ; if it stays , it is not
//
// the material of the hit is read the same way as getHitInfo of SHaRCPass.hlsl does it ( the same maps , the same parallax )

bool isSpecularMode(const float roughness , const float metallic){ return (metallic >= 0.7f && roughness < 0.4f); }

struct HitSurface{
    float3 position;
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
    surface.position  = vertices[i0].position  * w0 + vertices[i1].position  * w1 + vertices[i2].position  * w2;
    surface.normal    = vertices[i0].normal    * w0 + vertices[i1].normal    * w1 + vertices[i2].normal    * w2;
    surface.tangent   = vertices[i0].tangent   * w0 + vertices[i1].tangent   * w1 + vertices[i2].tangent   * w2;
    surface.bitangent = vertices[i0].bitangent * w0 + vertices[i1].bitangent * w1 + vertices[i2].bitangent * w2;
    surface.uv        = vertices[i0].UV        * w0 + vertices[i1].UV        * w1 + vertices[i2].UV        * w2;
    return surface;
}

float2 calculateParallaxAtHit(const float2 currentUV , const float3 V , const int depthID){
    if(depthID == NON_VALID_SLOT) return currentUV;
    const uint physicalIndex = txVirtualBuffer[depthID];
    const float depth = textureHub[physicalIndex].SampleLevel(txSampler , currentUV , 0).x;
    const float height = 1.0f - depth;
    const float height_scale = 0.001f;
    const float2 p = (V.xy) / V.z * height * height_scale;
    return currentUV - float2(p.x , -p.y);
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint2 size = uint2(generalBuffer.screenWidth , generalBuffer.screenHeight);
    if(any(id.xy >= size)) return;
    const uint2 pixel = id.xy;

    // the direction of the primary ray : the same way as the sky pixels of RayGen get it
    const float2 uv = (float2(pixel) + 0.5f) / float2(size);
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    const float4 farPoint = mul(generalBuffer.inverseProj , float4(ndc , 1.0f , 1.0f));
    const float3 rayDir = normalize(mul((float3x3)generalBuffer.inverseView , farPoint.xyz).xyz);

    RayDesc ray;
    ray.Origin = generalBuffer.playerPosition;
    ray.Direction = rayDir;
    ray.TMin = generalBuffer.playerNear;
    ray.TMax = generalBuffer.playerFar;

    // every instance : the lamps ( mask 2 ) are drawn by the raster pass too
    RayQuery<RAY_FLAG_NONE> rq;
    rq.TraceRayInline(TLAS , RAY_FLAG_NONE , 0xFF , ray);
    while(rq.Proceed())
    {
        if(rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            // a transparent texel is not a hit
            if(checkTransHit(rq.CandidateTriangleBarycentrics() , rq.CandidatePrimitiveIndex() , rq.CandidateInstanceID()))
            {
                rq.CommitNonOpaqueTriangleHit();
            }
        }
    }

    if(rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        // the sky : the state of the cleared images ( RayGen takes every pixel that is not COLOR_PIXEL for the sky )
        positionImage[pixel] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        normalImage[pixel] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        albedoImage[pixel] = float4(0.0f , 0.0f , 0.0f , 0.0f);
        depthImage[pixel] = 0.0f;
        roughnessImage[pixel] = 0.0f;
        metallicImage[pixel] = 0.0f;
        objectAOImage[pixel] = 1.0f;
        pixelCheck[pixel] = UNDEFINED_PIXEL;
        return;
    }

    const uint instanceID = rq.CommittedInstanceID();
    const InstanceData instance = instanceLayoutBuffer[instanceID];
    const uint meshID = instance.meshDataIndex;
    const LUT lut = LUT_Buffer[instance.LUT_index];

    const HitSurface surface = interpolateHit(meshID , rq.CommittedPrimitiveIndex() , rq.CommittedTriangleBarycentrics());
    const float4x4 model = instanceBuffer[meshBuffer[meshID].instanceIndex];

    // the same matrices as the vertex shader of the Deferred render ( see getHitInfo of SHaRCPass.hlsl )
    const float3 N = normalize(mul(normalMatrix3x3((float3x3)model) , surface.normal));
    const float3 T = normalize(mul(model , float4(surface.tangent , 0.0f)).xyz);
    const float3 B = normalize(mul(model , float4(surface.bitangent , 0.0f)).xyz);
    const float3x3 TBN = float3x3(T , B , N);

    const float3 hitPoint = mul(model , float4(surface.position , 1.0f)).xyz;
    const float3 V = -rayDir;
    const float3 V_TBN = mul(TBN , V);

    const float2 texUV = calculateParallaxAtHit(surface.uv , V_TBN , lut.depthID);

    float3 normal = N;
    if(lut.normalID != NON_VALID_SLOT){
        const uint nID = txVirtualBuffer[lut.normalID];
        float3 n;
        n.xy = textureHub[nID].SampleLevel(txSampler , texUV , 0).xy * 2.0f - 1.0f;
        n.z = sqrt(saturate(1.0f - dot(n.xy , n.xy)));
        n.y = (generalBuffer.normalMapOpenGL != 0u) ? n.y : -n.y;
        normal = normalize(mul(n , TBN));
    }

    float4 albedo = float4(lut.color , 1.0f);
    if(lut.diffuseID != NON_VALID_SLOT){
        const uint dID = txVirtualBuffer[lut.diffuseID];
        albedo = textureHub[dID].SampleLevel(txSampler , texUV , 0);
    }
    albedo.a = 1.0f;

    float metallic = lut.metallic;
    if(lut.metallicTextureID != NON_VALID_SLOT){
        const uint mID = txVirtualBuffer[lut.metallicTextureID];
        metallic = textureHub[mID].SampleLevel(txSampler , texUV , 0).z;
    }

    float roughness = lut.roughness;
    if(lut.roughnessTextureID != NON_VALID_SLOT){
        const uint rID = txVirtualBuffer[lut.roughnessTextureID];
        roughness = textureHub[rID].SampleLevel(txSampler , texUV , 0).y;
    }

    if(lut.armTextureID != NON_VALID_SLOT){
        const uint aID = txVirtualBuffer[lut.armTextureID];
        const float3 arm = textureHub[aID].SampleLevel(txSampler , texUV , 0).xyz;
        roughness = arm.y;
        metallic = arm.z;
    }

    float ambientOcclusion = 1.0f;
    if(lut.aoTextureID != NON_VALID_SLOT){
        const uint oID = txVirtualBuffer[lut.aoTextureID];
        ambientOcclusion = textureHub[oID].SampleLevel(txSampler , texUV , 0).x;
    }

    roughness = max(roughness , 0.01f);
    if(isSpecularMode(roughness , metallic)){
        albedo = lerp(0.04f , albedo , metallic);
    }

    // the normal looks at the viewer ( a back face of a thin wall )
    if(dot(normal , V) < 0.0f) normal = -normal;

    // the linear depth : the same function and the same matrix as the raster pass ( SV_Position.z is the z / w of the clip position )
    const float4 clip = mul(generalBuffer.viewProj , float4(hitPoint , 1.0f));
    const float linearDepth = LinearizeDepth(clip.z / clip.w , generalBuffer.playerNear , generalBuffer.playerFar);

    positionImage[pixel] = float4(hitPoint , float(instance.LUT_index + 1u));
    normalImage[pixel] = float4(normal , 1.0f);
    albedoImage[pixel] = albedo;
    depthImage[pixel] = linearDepth;
    roughnessImage[pixel] = roughness;
    metallicImage[pixel] = metallic;
    objectAOImage[pixel] = ambientOcclusion;
    pixelCheck[pixel] = COLOR_PIXEL;
}
