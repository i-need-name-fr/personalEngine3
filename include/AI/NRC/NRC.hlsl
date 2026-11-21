#pragma once
#if defined(MOX_NRC_INCLUDED) && MOX_NRC_INCLUDED >= 0ul
 #if defined(MOX_SHADER_STRUCTURES)
namespace AI
{
namespace NRC
{
[[vk::binding(  0   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<AI::Parameter> hashGridBuffer;
[[vk::binding(  1   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<AI::NRC::HashGridInfo> HashGridInfos;
[[vk::binding(  2   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<AI::NRC::TrainingSample> trainingBuffer;
[[vk::binding(  3   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<AI::NRC::GeneralData> NRCGeneralData;

[[vk::binding(  4   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> inputGradBuffer;
[[vk::binding(  5   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> inputWeightBuffer;
[[vk::binding(  6   ,   MOX_NRC_INCLUDED    )]] ConstantBuffer<AI::OffsetData> inputOffsetBuffer;
[[vk::binding(  7   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> inputStagingGradBuffer;
[[vk::binding(  8   ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> inputStagingWeightBuffer;
[[vk::binding(  9   ,   MOX_NRC_INCLUDED    )]] ConstantBuffer<AI::ptrData> inputPtrBuffer;
[[vk::binding(  10  ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<AI::Parameter> inputParamBuffer;

[[vk::binding(  11  ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> outputGradBuffer;
[[vk::binding(  12  ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> outputWeightBuffer;
[[vk::binding(  13  ,   MOX_NRC_INCLUDED    )]] ConstantBuffer<AI::OffsetData> outputOffsetBuffer;
[[vk::binding(  14  ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> outputStagingGradBuffer;
[[vk::binding(  15  ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<float> outputStagingWeightBuffer;
[[vk::binding(  16  ,   MOX_NRC_INCLUDED    )]] ConstantBuffer<AI::ptrData> outputPtrBuffer;
[[vk::binding(  17  ,   MOX_NRC_INCLUDED    )]] RWStructuredBuffer<AI::Parameter> outputParamBuffer;
}
}
 #endif
#endif