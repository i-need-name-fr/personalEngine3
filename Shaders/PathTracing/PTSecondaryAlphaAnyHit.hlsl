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

// the any hit of a hit group of the geometry : the pixels of a leaf under the cutoff of the alpha test do not exist for the ray ( the hit is ignored ) , everything else is
// accepted as it is ( the instances that are not tested are opaque for the ray , this shader does not run for them , except the bodies of the lamps )
[shader("anyhit")]
void main(inout RayPayLoad_secondary_ payload , in BuiltInTriangleIntersectionAttributes attr){
    if(!checkTransHit(attr.barycentrics , PrimitiveIndex() , InstanceID())){
        IgnoreHit();
    }
}
