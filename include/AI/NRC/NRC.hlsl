#pragma once
// #include "../../include/AI/NRC/NRC.hlsl"

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

void adamStep(inout Parameter p , const float g , const float lr)
{
    static const float b1 = 0.9f;
    static const float b2 = 0.999f;
    p.m = p.m * b1 + (1.0f - b1) * g;
    p.v = p.v * b2 + (1.0f - b2) * g * g;
    const float m = p.m / (1.0f - pow(b1 , float(NRCGeneralData[0].stepCount)));
    const float v = p.v / (1.0f - pow(b2 , float(NRCGeneralData[0].stepCount)));
    const float epsilon = 1e-8f;
    p.w -= lr * m / (sqrt(v) + epsilon);
}


#if defined(MOX_COOP_VECTOR)

    void harmonicN(const float3 N ,inout float h[5])
    {
        h[0] = N.x * N.y;
        h[1] = N.z * N.y;
        h[2] = N.z * N.z * 3  - 1.0f;
        h[3] = N.x * N.z;
        h[4] = N.x * N.x - N.y * N.y;
    }

    static const int NRC_F16 = 0;
    static const int NRC_LAYOUT_ROW_MAJOR = 0;
    static const int NRC_LAYOUT_COLUMN_MAJOR = 1;
    static const int NRC_LAYOUT_INFERENCING_OPTIMAL = 2;
    static const int NRC_LAYOUT_TRAINING_OPTIMAL = 3;

    static const float NRC_LR_MLP = 1e-3f;
    static const float NRC_LR_GRID[4] = { 1e-3f , 1e-3f , 5e-4f , 2.5e-4f };

    struct NrcBlob { uint data[1 << 20]; };
    typedef vk::BufferPointer<NrcBlob , 64> NrcPtr;

    float readHalf(const NrcPtr buffer , const uint halfIndex)
    {
        const uint word = buffer.Get().data[halfIndex >> 1];
        return f16tof32(word >> ((halfIndex & 1) * 16));
    }

    void writeHalf(const NrcPtr buffer , const uint halfIndex , const float value)
    {
        const uint shift = (halfIndex & 1) * 16;
        InterlockedAnd(buffer.Get().data[halfIndex >> 1] , ~(0xFFFFu << shift));
        InterlockedOr(buffer.Get().data[halfIndex >> 1] , f32tof16(value) << shift);
    }

    float derivativeSoftplus(const float x)
    {
        return 1.0f /(1.0f + exp(-x));
    }

    float softplus(float x)
    {
        return log(1.0f + exp(x));
    }

    float3 forwardNRC(const float3 P , const float3 N , out Coop16 x , out Coop64 h , out Coop64 z1)
    {
        HashGridResult hresult;

        [unroll(8)]
        for(uint i = 0 ; i < 8 ; i++)
        {
            hresult.data[i] = 0.0f;
        }

        [unroll]
        for(uint level = 0 ; level < MOX_NRC_MAX_HASH_GRID_SIZE ; level++)
        {
            const HashGridInfo levelData = HashGridInfos[level];
            const float size = levelData.cellSize;
            const int T = int(levelData.T);
            const int offset = int(levelData.offset);

            const float3 q = P / size;
            int3 base = int3(floor(q));
            const float3 f = q - float3(base);
            int3 id = base % T;
            if(id.x < 0) id.x += T;
            if(id.y < 0) id.y += T;
            if(id.z < 0) id.z += T;

            float totalW1 = 0.0f;
            float totalW2 = 0.0f;

            [unroll]
            for(int z = 0 ; z < 2 ; z++)
            [unroll]
            for(int y = 0 ; y < 2 ; y++)
            [unroll]
            for(int x = 0 ; x < 2 ; x++)
            {
                const int3 index = (id + int3(x , y , z)) % T;
                const int3 angle = int3(x , y , z);
                const float3 weights = lerp(1.0f - f , f , float3(angle));

                int hashIndex = index.z * T * T + index.y * T + index.x;
                hashIndex *= 2;
                hashIndex += offset;

                // since our F = 2
                const float w1 = hashGridBuffer[hashIndex].w;
                const float w2 = hashGridBuffer[hashIndex + 1].w;

                const float sumWeight = weights.x * weights.y * weights.z;

                totalW1 += sumWeight * w1;
                totalW2 += sumWeight * w2;
            }

            hresult.data[level * 2] = totalW1;
            hresult.data[level * 2 + 1] = totalW2;
        }

        const float3 inputN = N * 0.5 + 0.5;
        float harmonic[5];
        harmonicN(N , harmonic);

        Coop16 input = coop::construct16(float16_t(hresult.data[0]) , float16_t(hresult.data[1]) , float16_t(hresult.data[2]),
                                         float16_t(hresult.data[3]) , float16_t(hresult.data[4]) , float16_t(hresult.data[5]),
                                         float16_t(hresult.data[6]) , float16_t(hresult.data[7]) , float16_t(inputN.x) ,
                                         float16_t(inputN.y) , float16_t(inputN.z) , float16_t(harmonic[0]) , float16_t(harmonic[1]) ,
                                         float16_t(harmonic[2]) , float16_t(harmonic[3]) , float16_t(harmonic[4]));
        x = input;

        // layer 1 : z = W1 * x + b1 ( 16 -> 64 )
        const NrcPtr layer1 = NrcPtr(inputPtrBuffer.pWeightBuffer);
        static const int M1 = coop::Size<Coop64>::value;
        static const int K1 = coop::Size<Coop16>::value;
        Coop64 hiddenOutput = coop::matMulAdd<Coop64>(input , NRC_F16 ,
                                                      layer1.Get().data , int(inputOffsetBuffer.INFERENCE_W_OFFSET) , NRC_F16 ,
                                                      layer1.Get().data , int(inputOffsetBuffer.BIAS_OFFSET) , NRC_F16 ,
                                                      M1 , K1 , NRC_LAYOUT_INFERENCING_OPTIMAL , false , K1 * 2 , 0);

        z1 = hiddenOutput;
        hiddenOutput = coop::relu(hiddenOutput);
        h = hiddenOutput;

        // layer 2 : o = W2 * h + b2 ( 64 -> 16 )
        const NrcPtr layer2 = NrcPtr(outputPtrBuffer.pWeightBuffer);
        static const int M2 = coop::Size<Coop16>::value;
        static const int K2 = coop::Size<Coop64>::value;
        Coop16 outputVector = coop::matMulAdd<Coop16>(hiddenOutput , NRC_F16 ,
                                                      layer2.Get().data , int(outputOffsetBuffer.INFERENCE_W_OFFSET) , NRC_F16 ,
                                                      layer2.Get().data , int(outputOffsetBuffer.BIAS_OFFSET) , NRC_F16 ,
                                                      M2 , K2 , NRC_LAYOUT_INFERENCING_OPTIMAL , false , K2 * 2 , 0);

        const float r = float(coop::extract(outputVector , 0));
        const float g = float(coop::extract(outputVector , 1));
        const float b = float(coop::extract(outputVector , 2));

        return float3(r , g , b);
    }

    float3 askNRC(const float3 P , const float3 N)
    {
        Coop16 x;
        Coop64 h;
        Coop64 z1;
        const float3 answer = forwardNRC(P , N , x , h , z1);

        const float3 result = float3(softplus(answer.x) , softplus(answer.y) ,softplus(answer.z));
        return result;
    }


    void backStepHashGrid(const float3 P ,const Coop16 data)
    {
        [unroll]
        for(uint level = 0 ; level < MOX_NRC_MAX_HASH_GRID_SIZE ; level++)
        {

            const NRC::HashGridInfo levelData = HashGridInfos[level];
            const float size = levelData.cellSize;
            const int T = int(levelData.T);
            const int offset = int(levelData.offset);

            const float3 q = P / size;
            int3 base = int3(floor(q));
            const float3 f = q - float3(base);
            int3 id = base % T;
            if(id.x < 0) id.x += T;
            if(id.y < 0) id.y += T;
            if(id.z < 0) id.z += T;


            [unroll]
            for(int z = 0 ; z < 2 ; z++)
            [unroll]
            for(int y = 0 ; y < 2 ; y++)
            [unroll]
            for(int x = 0 ; x < 2 ; x++)
            {
                const int3 index = (id + int3(x , y , z)) % T;
                const int3 angle = int3(x , y , z);
                const float3 weights = lerp(1.0f - f , f , float3(angle));
                // z = 1
                // y = 0
                // x = 0
                // 4 spot - 5 spot
                int hashIndex = index.z * T * T + index.y * T + index.x;
                hashIndex *= 2;
                hashIndex += offset;
                const float sumWeight = weights.x * weights.y * weights.z;

                // since our F = 2
                atomicAddFloat(hashGridBuffer[hashIndex].grad , 1 , 0 , float(coop::extractDynamic(data , level * 2)) * sumWeight);
                atomicAddFloat(hashGridBuffer[hashIndex + 1].grad , 1 , 0 , float(coop::extractDynamic(data , level * 2 + 1)) * sumWeight);
            }
        }
    }

    void processBackstep(TrainingSample s , const bool valid)
    {
        Coop16 x; 
        Coop64 h;
        Coop64 z1;

        const float3 NRC_answer = forwardNRC(s.position , s.normal , x,  h , z1);
        const float3 NRC_result = float3(softplus(NRC_answer.x) , softplus(NRC_answer.y) , softplus(NRC_answer.z));
        const float3 dL_dprev = 2.0f * (NRC_result - s.radiance) / (NRC_result * NRC_result + 1e-3f);
        const float3 dL_dout = dL_dprev * (valid ? 1.0f / 16.0f : 0.0f);
        Coop16 firstLayerInput = coop::construct16(float16_t(dL_dout.x) * float16_t(derivativeSoftplus(NRC_answer.x)), float16_t(dL_dout.y) * float16_t(derivativeSoftplus(NRC_answer.y)) , float16_t(dL_dout.z) *float16_t(derivativeSoftplus(NRC_answer.z))
                                                    , 0.0f , 0.0f , 0.0f , 0.0 , 0.0, 0.0, 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0);

        const NrcPtr outGrad = NrcPtr(outputPtrBuffer.pGradBuffer);
        const NrcPtr outWeights = NrcPtr(outputPtrBuffer.pWeightBuffer);
        const NrcPtr inGrad = NrcPtr(inputPtrBuffer.pGradBuffer);
        const NrcPtr inWeights = NrcPtr(inputPtrBuffer.pWeightBuffer);

        coop::outerProduct(outGrad.Get().data , 0 , firstLayerInput , h , NRC_LAYOUT_TRAINING_OPTIMAL , NRC_F16 , 64 * 2);
        // Out matMul(In input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);
        Coop64 dl_dh = coop::matMul<Coop64>(firstLayerInput , NRC_F16 , outWeights.Get().data , int(outputOffsetBuffer.TRAINING_W_OFFSET) , NRC_F16 , 
                                    64 , 16 , NRC_LAYOUT_TRAINING_OPTIMAL , true , 64 * 2 , 0);

        Coop64 gate = coop::reluGate(z1);
        Coop64 dz1 = coop::mul(dl_dh , gate);
        Coop16 inputCoop = coop::matMul<Coop16>(dz1 , NRC_F16 , inWeights.Get().data , int(inputOffsetBuffer.TRAINING_W_OFFSET) , NRC_F16 , 16 , 64 , NRC_LAYOUT_TRAINING_OPTIMAL , true , 16 * 2 , 0);
        coop::outerProduct(inGrad.Get().data , 0 , dz1,  x , NRC_LAYOUT_TRAINING_OPTIMAL , NRC_F16 , 16 * 2);

        coop::reduceSum(outGrad.Get().data , outputOffsetBuffer.BIAS_GRAD_OFFSET , firstLayerInput);
        coop::reduceSum(inGrad.Get().data , inputOffsetBuffer.BIAS_GRAD_OFFSET , dz1);

        if(valid) backStepHashGrid(s.position, inputCoop);
    }   

    void recordAdam_hidden(const uint3 id) 
    {
        const uint validCount = NRCGeneralData[0].validCount;
        if(validCount == 0) return;
        const uint pos = id.x;
        const uint totalSize = 64 * 16 + 64; //
        if(pos >= totalSize) return;
        float g = 0.0f;

        NrcPtr ptrW = NrcPtr(inputPtrBuffer.pWeightBuffer);
        NrcPtr ptrG = NrcPtr(inputPtrBuffer.pGradBuffer);

        if(pos < 64 * 16)
        {
            g = inputStagingGradBuffer[pos] / (float(validCount) * 1.0f / 16.0f);
        }
        else
        {
            // the bias : its gradient is fp16 inside hiddenGrads after the TRAINING_OPTIMAL matrix
            const uint b = pos - 64 * 16;
            g = readHalf(ptrG , inputOffsetBuffer.BIAS_GRAD_OFFSET / 2 + b) / (float(validCount) * 1.0f/16);
        }



        AI::Parameter p = inputParamBuffer[pos];
        if(!isfinite(g)) return;
        adamStep(p , g , NRC_LR_MLP);

        inputParamBuffer[pos] = p;

        if(pos < 64 * 16)
        {
            inputStagingWeightBuffer[pos] = p.w;
        }
        else
        {
            writeHalf(ptrW , inputOffsetBuffer.BIAS_OFFSET / 2 + (pos - 64 * 16) , p.w);
        }
    }

    void recordAdam_output(const uint3 id)
    {
        const uint validCount = NRCGeneralData[0].validCount;
        if(validCount == 0) return;
        const uint pos = id.x;
        const uint totalSize = 16 * 64 + 16;   // the matrix ( 16 rows x 64 columns ) and the bias ( 16 )
        if(pos >= totalSize) return;
        float g = 0.0f;

        NrcPtr ptrW = NrcPtr(outputPtrBuffer.pWeightBuffer);
        NrcPtr ptrG = NrcPtr(outputPtrBuffer.pGradBuffer);

        if(pos < 16 * 64)
        {
            g = outputStagingGradBuffer[pos] / (float(validCount) * 1.0f/16);
        }
        else
        {
            const uint b = pos - 16 * 64;
            g = readHalf(ptrG , outputOffsetBuffer.BIAS_GRAD_OFFSET / 2 + b) / (float(validCount) * 1.0f/16);
        }



        AI::Parameter p = outputParamBuffer[pos];
        if(!isfinite(g)) return;

        adamStep(p , g , NRC_LR_MLP);

        outputParamBuffer[pos] = p;
        if(pos < 16 * 64)
        {
            outputStagingWeightBuffer[pos] = p.w;
        }
        else
        {
            writeHalf(ptrW , outputOffsetBuffer.BIAS_OFFSET / 2 + (pos - 16 * 64) , p.w);
        }
    }

    void recordAdam_hash(const uint3 id , const uint layer)
    {
        const uint validCount = NRCGeneralData[0].validCount;
        const HashGridInfo levelData = HashGridInfos[layer];
        const int T = int(levelData.T);
        const int offset = int(levelData.offset);
        if(validCount == 0) return;
        uint pos = id.x;
        const uint totalSize = T * T * T * 2; //
        if(pos >= totalSize) return;
        pos += offset;
        AI::Parameter p = hashGridBuffer[pos];
        if(p.grad == 0.0f) return;
        const float g = p.grad / (float(validCount) * 1.0f/16);
        if(!isfinite(g))
        {
            hashGridBuffer[pos].grad = 0.0f;
            return;
        }

        adamStep(p , g , NRC_LR_GRID[min(layer , 3u)]);
        p.grad = 0;
        hashGridBuffer[pos] = p;
    }

#endif

}
}
 #endif
#endif