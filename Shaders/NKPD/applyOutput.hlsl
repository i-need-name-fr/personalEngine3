#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 2
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#include "../../include/moxSourceCluster/IndirectPixels.hlsl"
#define MOX_NKPD_INCLUDED 0
#include "../../include/AI/NKPD/NKPD.hlsl"
using namespace AI::NKPD;

// the OPTIMIZER ( Adam ) of the output layer : id.x is the number of the parameter ( the matrix first , then the bias ) .
// the matrix is updated in the row major fp32 staging buffer ( the host converts it into the optimal layouts after the passes ) , the bias goes to the weight buffer as fp16
// the dispatch : x = ( 16 * 64 + 16 = 1040 + 127 ) / 128 = 9 groups , y = z = 1
[shader("compute")]
[numthreads(128 , 1 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    recordAdam_output(id);
}
