#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#define MOX_NRC_INCLUDED 1
#include "../../include/AI/NRC/NRC.hlsl"
using namespace AI::NRC;

// the OPTIMIZER ( Adam ) of the HASH GRID : id.x is the number of the parameter inside a level , id.y is the LEVEL ( 0 .. MOX_NRC_MAX_HASH_GRID_SIZE - 1 ) : one dispatch updates all the levels
// the dispatch : x = ( the biggest T * T * T * 2 + 127 ) / 128 , y = MOX_NRC_MAX_HASH_GRID_SIZE , z = 1 . a thread over the size of its level returns
[shader("compute")]
[numthreads(128 , 1 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if(id.y >= MOX_NRC_MAX_HASH_GRID_SIZE) return;
    recordAdam_hash(id , id.y);
}
