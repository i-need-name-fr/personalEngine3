#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#define MOX_NRC_INCLUDED 1
#include "../../include/AI/NRC/NRC.hlsl"
using namespace AI::NRC;

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint2 size = uint2(NRCGeneralData[0].width , NRCGeneralData[0].height);
    const bool inside = all(id.xy < size);
    const uint2 clamped = min(id.xy , size - 1u);
    const uint bufferID = (clamped.x + clamped.y * size.x) * MOX_NRC_MAX_BOUNCE_COUNT;

    [unroll]
    for(uint i = 0 ; i < MOX_NRC_MAX_BOUNCE_COUNT ; i++)
    {
        const TrainingSample sample = trainingBuffer[bufferID + i];
        const bool valid = inside && sample.valid == MOX_NRC_VALID_SAMPLE;
        if(valid)
        {
            InterlockedAdd(NRCGeneralData[0].validCount , 1);
        }
        processBackstep(sample , valid);
    }
}
