#include "../../shaderStructures/shaderStructure.h"

[[vk::binding(  0   ,   0    )]] RWStructuredBuffer<float> gradBuffer;
[[vk::binding(  1   ,   0    )]] RWStructuredBuffer<float> weightBuffer;
[[vk::binding(  2  ,   0    )]] ConstantBuffer<AI::OffsetData> offsetBuffer;
[[vk::binding(  3  ,   0    )]] RWStructuredBuffer<float> stagingGradBuffer;
[[vk::binding(  4  ,   0    )]] RWStructuredBuffer<float> stagingWeightBuffer;
[[vk::binding(  5  ,   0    )]] ConstantBuffer<AI::ptrData> ptrBuffer;
[[vk::binding(  6  ,   0    )]] RWStructuredBuffer<AI::Parameter> paramBuffer;

[[vk::binding(  0   ,   1    )]] RWStructuredBuffer<AI::NKPD::GeneralData> NKPDGeneral;

[[vk::push_constant]] struct 
{
    uint32_t M;   // rows of the layer ( the size of the bias )
    uint32_t K;   // columns of the layer
    float lr;     // the learning rate
} push;

uint adamMatrixSize() { return push.M * push.K; }
uint adamTotalSize() { return push.M * push.K + push.M; }

using namespace AI::NKPD;
using namespace AI;

void adamStep(inout Parameter p , const float g , const float lr)
{
    static const float b1 = 0.9f;
    static const float b2 = 0.999f;
    p.m = p.m * b1 + (1.0f - b1) * g;
    p.v = p.v * b2 + (1.0f - b2) * g * g;
    const float m = p.m / (1.0f - pow(b1 , float(NKPDGeneral[0].stepCount)));
    const float v = p.v / (1.0f - pow(b2 , float(NKPDGeneral[0].stepCount)));
    const float epsilon = 1e-8f;
    p.w -= lr * m / (sqrt(v) + epsilon);
}

struct NKPDBlob { uint data[1 << 20]; };
typedef vk::BufferPointer<NKPDBlob , 64> BufPtr;

float readHalf(const BufPtr buffer , const uint halfIndex)
{
    const uint word = buffer.Get().data[halfIndex >> 1];
    return f16tof32(word >> ((halfIndex & 1) * 16));
}

void writeHalf(const BufPtr buffer , const uint halfIndex , const float value)
{
    const uint shift = (halfIndex & 1) * 16;
    InterlockedAnd(buffer.Get().data[halfIndex >> 1] , ~(0xFFFFu << shift));
    InterlockedOr(buffer.Get().data[halfIndex >> 1] , f32tof16(value) << shift);
}


void adam(const uint id)
{
    const uint validCount = NKPDGeneral[0].validCount;
    if(validCount == 0) return;
    const uint pos = id;
    const uint matrixSize = adamMatrixSize();
    if(pos >= adamTotalSize()) return;
    float g = 0.0f;

    const BufPtr ptrW = BufPtr(ptrBuffer.pWeightBuffer);
    const BufPtr ptrG = BufPtr(ptrBuffer.pGradBuffer);

    if(pos < matrixSize)
    {
        g = stagingGradBuffer[pos] / (float(validCount) * 1.0f / 16.0f);
    }
    else
    {
        const uint b = pos - matrixSize;
        g = readHalf(ptrG , offsetBuffer.BIAS_GRAD_OFFSET / 2 + b) / (float(validCount) * 1.0f / 16.0f);
    }

    AI::Parameter p = paramBuffer[pos];
    if(!isfinite(g)) return;
    adamStep(p , g , push.lr);

    paramBuffer[pos] = p;

    if(pos < matrixSize)
    {
        stagingWeightBuffer[pos] = p.w;
    }
    else
    {
        writeHalf(ptrW , offsetBuffer.BIAS_OFFSET / 2 + (pos - matrixSize) , p.w);
    }
}


[shader("compute")]
[numthreads(256 , 1 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if(id.x >= adamTotalSize()) return;
    adam(id.x);
}