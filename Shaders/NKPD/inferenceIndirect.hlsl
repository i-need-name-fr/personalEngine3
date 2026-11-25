// NKPD of the INDIRECT light : one thread = one pixel of indirectImage ( its own resolution , see IndirectPixels.hlsl ) . the net sees the grid of the indirect pixels ( the neighbours , the history , the counters ) ,
// the G buffer is the full resolution : the pixel that the indirect pixel looks at is the jitter of the frame ( gPixel ) . the diffuse signal is denoised in PostProcess ( the full resolution )
// in : the raw signal in indirectImage [ current ] , the history in indirectImage [ prev ] , diffuseHistoryImage .g = the counter of the indirect light
// out : indirectImage [ current ] ( the denoised signal : it is also the history of the next frame ) , diffuseHistoryImage .g
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

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint2 size = indirectSize();
    if(any(id.xy >= size)) return;

    if(!features.NKPD) return;

    if(pixelCheck[gPixel(id.xy , true)] != COLOR_PIXEL)
    {
        diffuseHistoryImage[id.xy] = float4(diffuseHistoryImage[id.xy].r , 0.0f , 0.0f , 0.0f);
        return;
    }

    NKPDOptimal info = buildNKPDInfo(id.xy , true);
    float count = 0.0f;
    const float3 result = askNKPD(info , count);
    const float indirectCount = info.validUV ? min(count + 1.0f , MAX_ACCUM_FRAMES) : 1.0f;

    indirectImage[id.xy] = float4(result , 1.0f);
    diffuseHistoryImage[id.xy] = float4(diffuseHistoryImage[id.xy].r , indirectCount , 0.0f , 0.0f);
}
