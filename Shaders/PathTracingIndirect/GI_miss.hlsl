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

[shader("miss")]
void main(inout HitRayPayLoad payload){
    const float3 dir = WorldRayDirection();
    payload.didHit    = false;
    payload.albedo    = skybox.SampleLevel(txSampler , dir , 0).xyz;  // радианс неба
    payload.normal    = -dir;                                              // «поверхность» смотрит на нас
    payload.worldPos  = WorldRayOrigin() + dir * 1000.0f;                  // далёкая точка вдоль луча
    payload.roughness = 1.0f;                                              // небо переносимо
    payload.metallic  = 0.0f;
    payload.sameLUT   = false;                                             // nothing was hit
}
