#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 1
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 2
#include "../../include/moxSourceCluster/SceneImages.hlsl"
#include "../../include/AI/CoopVec.hlsl"
#define MOX_NKPD_INCLUDED 0
#include "../../include/AI/NKPD/NKPD.hlsl"
using namespace AI::NKPD;

// a new step of the training : the counter of Adam grows , the count of the valid samples starts again ( backstepShader counts them ) .
// it is a SEPARATE dispatch before the first backstep of the frame : inside of backstepShader the reset would race with the InterlockedAdd of the other threads ,
// and backstep is dispatched twice a frame ( diffuse , indirect ) , so the second one would reset the count of the first
// the dispatch : 1 x 1 x 1
[shader("compute")]
[numthreads(1 , 1 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    NKPDGeneral[0].stepCount += 1;
    NKPDGeneral[0].validCount = 0;
}
