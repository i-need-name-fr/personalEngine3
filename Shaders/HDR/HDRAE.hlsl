#include "../../include/ShaderStructures/ShaderStructure.h"

#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"

uint32_t computeBit(const float3 radiance)
{
    const float lum = luminance(radiance);
    if(!(lum >= exp2(minLogRange))) return 0; 
    if(!isfinite(lum)) return 0;
    const float t = saturate((log2(lum) - minLogRange) / lumRange); 
    return min(uint(t * 254.0f + 1.0f), 255u);
}

[shader("compute")]
[numthreads(8 , 8 ,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint32_t width , height;
    HDRImage.GetDimensions(width , height);
    uint2 size = uint2(width , height);
    if(any(id.xy >= size)) return;
    const uint32_t bit = computeBit(HDRImage[id.xy].xyz);
    InterlockedAdd(histoGram[bit] , 1);
}