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

struct HitSurface{
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

[shader("closesthit")]
void main(inout HitRayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    // InstanceID() is the instanceCustomIndex of the TLAS instance = the index in the instance layouts
    const InstanceData instance = instanceLayoutBuffer[InstanceID()];
    const uint meshID = instance.meshDataIndex;
    const LUT lut = LUT_Buffer[instance.LUT_index];

    const HitSurface surface = interpolateHit(meshID , PrimitiveIndex() , attr.barycentrics);

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
        // OpenGL maps ( green up ) are used as they are with this TBN , the DirectX ones ( green down ) are flipped . ( it was the other way round : every map was inverted )
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


    // the map of the ambient occlusion : not used yet
    if(lut.aoTextureID != NON_VALID_SLOT){
    }

    // every field of the payload is written : the caller reads all of them
    if(dot(normal , -WorldRayDirection()) < 0.0f) normal = -normal;
    load.didHit = true;
    load.worldPos = hitPoint;
    load.normal = normal;
    load.albedo = albedo;
    load.roughness = roughness;
    load.metallic = metallic;
    // the LUT_index of the caller is already in the payload
    load.sameLUT = (instance.LUT_index == load.LUT_index);
}
