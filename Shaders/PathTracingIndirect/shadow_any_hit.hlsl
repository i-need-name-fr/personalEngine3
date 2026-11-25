struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    // the distance from the origin of the ray to what stopped it ( a hit ) , the caller sets it to the length of the ray before the trace ( nothing in the way )
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

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


[shader("anyhit")]
void main(inout ShadowRayPayLoad payload , in BuiltInTriangleIntersectionAttributes attr)
{
    if(!checkTransHit(attr.barycentrics , PrimitiveIndex() , InstanceID()))
    {
        IgnoreHit();
    }
    else
    {
        AcceptHitAndEndSearch();
    }
}
