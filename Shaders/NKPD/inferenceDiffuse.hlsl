// NKPD of the DIRECT ( diffuse ) light : one thread = one pixel of diffuseImage ( the full resolution ) . the passes : step 1 ( the temporal one , reads the raw signal of diffuseImage ) then 2 and more ( read the result of the previous pass from tmpDiffuseImage ) .
// in : the raw signal in diffuseImage [ current ] , the history in diffuseImage [ prev ] , diffuseHistoryImage .r = the counter of the diffuse light
// out : the LAST pass writes diffuseImage [ current ] ( the denoised signal : it is also the history of the next frame ) , the others write tmpDiffuseImage . the pass 1 writes diffuseHistoryImage .r
#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#include "../../include/moxSourceCluster/IndirectPixels.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#define MOX_NKPD_INCLUDED 2
#include "../../include/AI/NKPD/NKPD.hlsl"
using namespace AI::NKPD;

[[vk::push_constant]] struct 
{
    uint stepSize;
    uint last;
} push;

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint2 size = directSize();
    if(any(id.xy >= size)) return;

    if(!features.NKPD) return;

    const bool first = push.stepSize <= 1u;

    if(pixelCheck[gPixel(id.xy , false)] != COLOR_PIXEL)
    {
        if(first) diffuseHistoryImage[id.xy] = float4(0.0f , diffuseHistoryImage[id.xy].g , 0.0f , 0.0f);
        return;
    }

    NKPDOptimal info = buildNKPDInfo(id.xy , false);
    float count = 0.0f;
    const float3 result = askNKPD(info , count , float(push.stepSize));

    // the counter of the history is written only by the pass 1
    if(first)
    {
        const float diffuseCount = info.validUV ? min(count + 1.0f , MAX_ACCUM_FRAMES) : 1.0f;
        diffuseHistoryImage[id.xy] = float4(diffuseCount , diffuseHistoryImage[id.xy].g , 0.0f , 0.0f);
    }

    if(push.last != 0u) diffuseImage[id.xy] = float4(result , 1.0f);
    else tmpDiffuseImage[id.xy] = float4(result , 1.0f);
}
