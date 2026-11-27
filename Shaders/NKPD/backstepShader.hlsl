// the input of the net in this shader is the signal BEFORE the spatial pass of ReBLUR ( see readInput ) , the ground truth is the result of ReBLUR
#define NKPD_TRAINING
#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 2
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#include "../../include/moxSourceCluster/IndirectPixels.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#define MOX_NKPD_INCLUDED 0
#include "../../include/AI/NKPD/NKPD.hlsl"

using namespace AI::NKPD;

[[vk::push_constant]] struct {
    uint indirect;
    uint stepSize;
} push;

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    // the grid of the signal : the full resolution for the diffuse , the resolution of indirectImage for the indirect light
    const uint2 size = gridSize(push.indirect);
    const uint2 tiles = (size + MOX_NKPD_RESOLUTION_DIVISOR - 1u) / MOX_NKPD_RESOLUTION_DIVISOR;
    const bool insideTile = all(id.xy < tiles);

    uint seed = initSeed(id.xy , generalBuffer.frameIndex);
    const uint2 offset = uint2(randomNumberIn(seed , MOX_NKPD_RESOLUTION_DIVISOR) , randomNumberIn(seed , MOX_NKPD_RESOLUTION_DIVISOR));
    const uint2 pixel = min(id.xy * MOX_NKPD_RESOLUTION_DIVISOR + offset , size - 1u);

    // the sky has no ground truth
    const bool valid = insideTile && (pixelCheck[gPixel(pixel , push.indirect)] == COLOR_PIXEL);

    const NKPDOptimal info = buildNKPDInfo(pixel , push.indirect);
    const float3 ideal = readIdeal(pixel , push.indirect);

    if(valid)
    {
        InterlockedAdd(NKPDGeneral[0].validCount , 1);
    }
    float3 result;
    processBackStep(ideal , info , valid , result , push.stepSize);
}
