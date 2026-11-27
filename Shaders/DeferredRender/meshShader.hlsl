// the descriptor sets ( the same in task , mesh and fragment )
#include "../../include/ShaderStructures/ShaderStructure.h"

#define MOX_TEXTURE_HUB_INCLUDED 0
#include "../../include/moxSourceCluster/TextureHub.hlsl"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 2
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#define MOX_VERTEX_HEAP_INCLUDED 3
#include "../../include/moxObject/VertexHeap.hlsl"

struct VsOutput{
    float4 clipPos  : SV_Position;
    [[vk::location(0)]] float3 fragPos                    : TEXCOORD0;
    [[vk::location(1)]] float3 normal                     : TEXCOORD1;
    [[vk::location(2)]] float3x3 TBN                      : TEXCOORD2;
    [[vk::location(5)]] float2 uv                         : TEXCOORD5;
    [[vk::location(6)]] nointerpolation uint LUTindex     : TEXCOORD6;
};

struct VsInput{
    // general vertex
    float3 position  ;
    float3 normal    ;
    float3 tangent   ;
    float3 bitangent ;
    float2 uv        ;

};

VsOutput processVertex(uint instanceID , const VsInput input)
{
    VsOutput output;

    InstanceData currentInstance = instanceLayoutBuffer[instanceID];
    uint meshID = currentInstance.meshDataIndex;
    uint instanceBufferIndex = meshBuffer[meshID].instanceIndex;
    float4x4 model = instanceBuffer[instanceBufferIndex];
    uint LUT_index = currentInstance.LUT_index;

    output.fragPos = mul(model , float4(input.position , 1.0f)).xyz;
    output.normal = normalize(mul(normalMatrix3x3((float3x3)model) , input.normal));
    output.LUTindex = LUT_index;
    output.uv = input.uv;
    const float3 T = normalize(mul(model , float4(input.tangent , 0.0f)).xyz);
    const float3 B = normalize(mul(model , float4(input.bitangent , 0.0f)).xyz);
    output.TBN = float3x3(T , B , output.normal);
    output.clipPos = mul(generalBuffer.viewProj , float4(output.fragPos , 1.0f));
    return output;
}

static const uint TASK_GROUP_SIZE = 32;

// the size of the OUTPUT of the mesh shader ( the arrays of the vertices , the indices and the primitives , sClip ) . it is NOT the size of the meshlet ( ms::MAX_BOUND_* is for building of the meshlets in Entity ) :
// the tessellation makes more vertices and triangles than the meshlet has . LOD 4 of a meshlet of 12 triangles : 12 * 15 = 180 vertices , 12 * 16 = 192 triangles .
// 192 vertices * ~128 bytes + 192 primitives stay under the 32 KB of the output ( maxMeshOutputMemorySize )
static const uint MESH_OUT_VERTICES = 192;
static const uint MESH_OUT_PRIMITIVES = 192;

struct Payload
{
    uint layoutIndex;
    uint32_t LOD;
    uint meshletID[TASK_GROUP_SIZE];
};

struct PrimOutput
{
    bool cull : SV_CullPrimitive;
};

groupshared float4 sClip[MESH_OUT_VERTICES];

bool tooSmall(const float4 c0 ,  const float4 c1 , const float4 c2)
{
    if(c0.w < 0.0f) return false;
    if(c1.w < 0.0f) return false;
    if(c2.w < 0.0f) return false;

    const float2 p0 = c0.xy / c0.w;
    const float2 p1 = c1.xy / c1.w;
    const float2 p2 = c2.xy / c2.w;

    const float2 texelSize = float2(float(generalBuffer.resolutionWidth) , float(generalBuffer.resolutionHeight));

    const float2 uv1 = (p0 * 0.5f + 0.5f) * texelSize;
    const float2 uv2 = (p1 * 0.5f + 0.5f) * texelSize;
    const float2 uv3 = (p2 * 0.5f + 0.5f) * texelSize;

    float2 boxMax = -1e9f;
    float2 boxMin =  1e9f;
    
    float2 uvs[3] = {uv1 , uv2 , uv3};

    [unroll]
    for(uint i = 0 ; i < 3 ; i++)
    {
        boxMax.x = max(boxMax.x , uvs[i].x);
        boxMax.y = max(boxMax.y , uvs[i].y);

        boxMin.x = min(boxMin.x , uvs[i].x);
        boxMin.y = min(boxMin.y , uvs[i].y);
    }

    return any(round(boxMin) == round(boxMax));
}

bool cullTriangle(const float4 c0 , const float4 c1 , const float4 c2)
{   
    if(c0.w < 0.0f) return false;
    if(c1.w < 0.0f) return false;
    if(c2.w < 0.0f) return false;

    if(tooSmall(c0 , c1 , c2)) return true;

    const float det =
        c0.x * (c1.y * c2.w - c2.y * c1.w)
        - c0.y * (c1.x * c2.w - c2.x * c1.w)
        + c0.w * (c1.x * c2.y - c2.x * c1.y);
    return det < 0.0f;
}

uint32_t getIndicesCount(const uint32_t LOD)
{
    return (2 * LOD - 1);
}
uint verticesPerTriangle(uint lod)  { return (lod + 1) * (lod + 2) / 2; }   // 3, 6, 10, 15 для LOD 1..4
uint trianglesPerTriangle(uint lod) { return lod * lod; } 

#define MESH_ROUGHNESS_TEXTURES

float calcRoughness(const uint instanceID , const float2 uv , const float mip)
{
    const LUT lut = LUT_Buffer[instanceLayoutBuffer[instanceID].LUT_index];

    // DEBUG ( the search of the device lost ) : NO textures here , only the value of the material . the texture part is below , it is switched off by MESH_ROUGHNESS_TEXTURES
    float roughness = lut.roughness;
#if defined(MESH_ROUGHNESS_TEXTURES)
    if(lut.roughnessTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.roughnessTextureID];
        roughness = textureHub[id].SampleLevel(txSampler , uv , mip).y;
    }

    if(lut.armTextureID != NON_VALID_SLOT)
    {
        const uint id = txVirtualBuffer[lut.armTextureID];
        roughness = textureHub[id].SampleLevel(txSampler , uv , mip).y;
    }
#endif
    return max(roughness , 0.01f);
}

float hash31(float3 p)
{
    p = frac(p * 0.3183099f + 0.1f);
    p *= 17.0f;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float valueNoise(const float3 x)
{
    const float3 i = floor(x);
    float3 f = frac(x);
    f = f * f * (3.0f - 2.0f * f);
    return lerp(lerp(lerp(hash31(i + float3(0.0f , 0.0f , 0.0f)) , hash31(i + float3(1.0f , 0.0f , 0.0f)) , f.x) ,
                     lerp(hash31(i + float3(0.0f , 1.0f , 0.0f)) , hash31(i + float3(1.0f , 1.0f , 0.0f)) , f.x) , f.y) ,
                lerp(lerp(hash31(i + float3(0.0f , 0.0f , 1.0f)) , hash31(i + float3(1.0f , 0.0f , 1.0f)) , f.x) ,
                     lerp(hash31(i + float3(0.0f , 1.0f , 1.0f)) , hash31(i + float3(1.0f , 1.0f , 1.0f)) , f.x) , f.y) , f.z);
}

static const float DISPLACEMENT_SCALE = 0.0089f;   // the amplitude as a part of the RADIUS of the box of the mesh ( the radius is in the WORLD space , see Entity.hpp ) : 0.3 %
static const float DISPLACEMENT_FREQUENCY = 20.0f;

// the height of the displacement in the point : -1 .. 1 ( it depends only on the position )
float displacementHeight(const float3 p)
{
    return valueNoise(p * DISPLACEMENT_FREQUENCY) * 2.0f - 1.0f;
}

void displacedFrame(const float3 position , const float amplitude , const float3 normal , const float3 tangent , const float3 bitangent , out float3 outNormal , out float3 outTangent , out float3 outBitangent)
{
    const float e = 0.05f / DISPLACEMENT_FREQUENCY;
    const float h0 = displacementHeight(position);
    const float3 grad = float3(displacementHeight(position + float3(e , 0.0f , 0.0f)) - h0 ,
                               displacementHeight(position + float3(0.0f , e , 0.0f)) - h0 ,
                               displacementHeight(position + float3(0.0f , 0.0f , e)) - h0) * (amplitude / e);
    const float3 tangentialGrad = grad - normal * dot(grad , normal);
    outNormal = normalize(normal - tangentialGrad);

    // the tangent is made perpendicular to the new normal ( Gram Schmidt ) , the bitangent keeps the handedness of the original frame
    const float handedness = (dot(cross(normal , tangent) , bitangent) >= 0.0f) ? 1.0f : -1.0f;
    outTangent = normalize(tangent - outNormal * dot(outNormal , tangent));
    outBitangent = cross(outNormal , outTangent) * handedness;
}

// add Tesselation
// will be only 4 levels and hardcoded 
// the dislocation of position will be based of roughness
uint vIndex(uint i , uint j , uint n) { return i * (n + 1) - (i * (i - 1)) / 2 + j; }

[shader("mesh")]
[numthreads(128 , 1 , 1)]
[outputtopology("triangle")]
void main(uint3 id : SV_DispatchThreadID , uint gtid : SV_GroupThreadID, uint gid : SV_GroupID , in payload Payload load, out vertices VsOutput output[MESH_OUT_VERTICES] , out indices uint3 ind[MESH_OUT_PRIMITIVES] , out primitives PrimOutput prims[MESH_OUT_PRIMITIVES])
{
    if(gid < TASK_GROUP_SIZE) 
    {   
        initNoise(uint2(id.xy)  , generalBuffer.frameIndex);

        const ms::Meshlet m = meshlets[load.meshletID[gid]];

        // Entity.hpp : m.indexCount = meshopt triangle_count , the NUMBER OF THE TRIANGLES ( not of the indices )
        uint32_t T = m.indexCount;
        // triangle count in this stage contains how much traingles does the meshlet has
        // LOD 1 : the meshlet as it is ( its own vertices are shared by the triangles ) . LOD > 1 : every triangle makes its own pack of vertices and triangles
        const uint vCount = (load.LOD == 1) ? m.vertexCount : T * verticesPerTriangle(load.LOD);
        const uint pCount = T * trianglesPerTriangle(load.LOD);

        SetMeshOutputCounts(vCount, pCount);
        const MeshData mesh = meshBuffer[instanceLayoutBuffer[load.layoutIndex].meshDataIndex];
        if(load.LOD == 1)
        {
            if(gtid < m.vertexCount)
            {
                // now we process each vertex of the meshlet
                VsInput input = (VsInput)0;
                const Vertex v = vertices[mVertices[m.firstVertex + gtid] + mesh.firstVertex];
                const float roughness  = calcRoughness(load.layoutIndex , v.UV , 0.0f);
                const float4x4 displacementModel = instanceBuffer[mesh.instanceIndex];
                const float displacementScale = max(length(float3(displacementModel[0][0] , displacementModel[1][0] , displacementModel[2][0])) ,
                                                    max(length(float3(displacementModel[0][1] , displacementModel[1][1] , displacementModel[2][1])) ,
                                                        length(float3(displacementModel[0][2] , displacementModel[1][2] , displacementModel[2][2]))));
                const float displacementWorld = mesh.boxRadius * DISPLACEMENT_SCALE / max(displacementScale , 1e-6f);

                const float amplitude = roughness * displacementWorld;
                const float offset = displacementHeight(v.position);   // -1 .. 1 , depends only on the position

                input.position = v.position + v.normal * (offset * amplitude);
                input.normal = v.normal;
                input.bitangent = v.bitangent;
                input.tangent = v.tangent;
                input.uv = v.UV;
                const VsOutput processed = processVertex(load.layoutIndex , input);
                output[gtid] = processed;
                sClip[gtid] = processed.clipPos;
            }
            GroupMemoryBarrierWithGroupSync();
            if(gtid < m.indexCount)
            {
                const uint i0 = mIndices[m.firstIndex + gtid * 3 ];
                const uint i1 = mIndices[m.firstIndex + gtid * 3 +1];
                const uint i2 = mIndices[m.firstIndex + gtid * 3 +2];
                ind[gtid] = uint3(i0 , i1 , i2);
                prims[gtid].cull = cullTriangle(sClip[i0] , sClip[i1] , sClip[i2]);
                // prims[gtid].cull = 0;
                
            }
        }
        else
        {
            if(gtid < m.indexCount)
            {
                // the body of the former processVerticesLOD : it is INLINED here . the arrays of the output of the mesh shader can not be passed as inout into a function : HLSL copies them in and out ( every thread writes all the 192 vertices back )
                Vertex inputVertices[3];

                [unroll]
                for(uint k = 0 ; k < 3 ; k++)
                {
                    const uint localIndex = mIndices[m.firstIndex + gtid * 3 + k];
                    inputVertices[k] = vertices[mVertices[m.firstVertex + localIndex] + mesh.firstVertex];
                }

                const uint verticesCount = verticesPerTriangle(load.LOD);
                uint32_t slot = 0;
                const uint trianglesCount = trianglesPerTriangle(load.LOD);   // n²
                const uint pBase = gtid * trianglesCount;     
                const float n = float(load.LOD);

                // the amplitude is set in the WORLD units ( a part of the radius of the box , it is in the world space ) , but the displacement is made in the space of the model : divide by the scale of the instance
                const float4x4 displacementModel = instanceBuffer[mesh.instanceIndex];
                const float displacementScale = max(length(float3(displacementModel[0][0] , displacementModel[1][0] , displacementModel[2][0])) ,
                                                    max(length(float3(displacementModel[0][1] , displacementModel[1][1] , displacementModel[2][1])) ,
                                                        length(float3(displacementModel[0][2] , displacementModel[1][2] , displacementModel[2][2]))));
                const float displacementWorld = mesh.boxRadius * DISPLACEMENT_SCALE / max(displacementScale , 1e-6f);
                [loop]
                for(float i = 0.0f ; i <= n ; i += 1.0f)
                {
                    [loop]
                    for(float j = 0.0f ; j <= n - i ; j += 1.0f)
                    {
                        const float wA = (n - i - j) / n;
                        const float wB = i / n;
                        const float wC = j / n;

                        Vertex v;
                        v.position = inputVertices[0].position * wA + inputVertices[1].position * wB + inputVertices[2].position * wC;
                        v.normal = normalize(inputVertices[0].normal * wA + inputVertices[1].normal * wB + inputVertices[2].normal * wC);
                        v.tangent = normalize(inputVertices[0].tangent * wA + inputVertices[1].tangent * wB + inputVertices[2].tangent * wC);
                        v.bitangent = normalize(inputVertices[0].bitangent * wA + inputVertices[1].bitangent * wB + inputVertices[2].bitangent * wC);
                        v.UV = inputVertices[0].UV * wA + inputVertices[1].UV * wB + inputVertices[2].UV * wC;

                        const float roughness  = calcRoughness(load.layoutIndex , v.UV , 0.0f);

                        const float amplitude = roughness * displacementWorld;
                        const float offset = displacementHeight(v.position);   // -1 .. 1 , depends only on the position

                        VsInput input = (VsInput)0;
                        input.position = v.position + v.normal * (offset * amplitude);
                        displacedFrame(v.position , amplitude , v.normal , v.tangent , v.bitangent , input.normal , input.tangent , input.bitangent);
                        input.uv = v.UV;
                        const VsOutput processed = processVertex(load.layoutIndex , input);
                        output[gtid * verticesCount + slot] = processed;
                        sClip[gtid * verticesCount + slot] = processed.clipPos;
                        slot++;
                    }
                }

                const uint N = load.LOD;
                const uint vBase = gtid * verticesCount;
                uint tria = 0;
                [loop]
                for(uint i = 0 ; i < N ; i++)
                {
                    [loop]
                    for(uint j = 0 ; j < N - i ; j++)
                    {
                        const uint a = vBase + vIndex(i     , j     , N);
                        const uint b = vBase + vIndex(i + 1 , j     , N);
                        const uint c = vBase + vIndex(i     , j + 1 , N);
                        ind[pBase + tria] = uint3(a , b , c);
                        prims[pBase + tria].cull = cullTriangle(sClip[a] , sClip[b] , sClip[c]);
                        tria++;

                        if(j + 1 < N - i)                       // «вниз» есть, пока i + j < N - 1
                        {
                            const uint d = vBase + vIndex(i + 1 , j + 1 , N);
                            ind[pBase + tria] = uint3(b , d , c);
                            prims[pBase + tria].cull = cullTriangle(sClip[b] , sClip[d] , sClip[c]);
                            tria++;
                        }
                    }
                }
            }
        }
    }
    // 
}