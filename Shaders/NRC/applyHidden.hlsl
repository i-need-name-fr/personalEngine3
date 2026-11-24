#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#define MOX_NRC_INCLUDED 1
#include "../../include/AI/NRC/NRC.hlsl"
using namespace AI::NRC;

// the OPTIMIZER ( Adam ) of the layer 1 ( the hidden one , 16 -> 64 ) : id.x is the number of the parameter ( the matrix first , then the bias ) .
// the matrix is updated in the row major fp32 staging buffer ( the host converts it into the optimal layouts after the passes ) , the bias goes to the weight buffer as fp16
// the dispatch : x = ( rows * columns + rows + 127 ) / 128 = 9 groups , y = z = 1
[shader("compute")]
[numthreads(128 , 1 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    recordAdam_hidden(id);
}
