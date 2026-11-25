#pragma once
// #include "../../include/AI/NKPD/NKPD.hlsl"

#if defined(MOX_NKPD_INCLUDED) && MOX_NKPD_INCLUDED >= 0ul
 #if defined(MOX_SHADER_STRUCTURES)
namespace AI
{
namespace NKPD
{
[[vk::binding(  0   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<AI::NKPD::GeneralData> NKPDGeneral;

// the input layer : 192 -> 64 ( 16 neighbours x 12 numbers )
[[vk::binding(  1   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> inputGradBuffer;
[[vk::binding(  2   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> inputWeightBuffer;
[[vk::binding(  3   ,   MOX_NKPD_INCLUDED    )]] ConstantBuffer<AI::OffsetData> inputOffsetBuffer;
[[vk::binding(  4   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> inputStagingGradBuffer;
[[vk::binding(  5   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> inputStagingWeightBuffer;
[[vk::binding(  6   ,   MOX_NKPD_INCLUDED    )]] ConstantBuffer<AI::ptrData> inputPtrBuffer;
[[vk::binding(  7   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<AI::Parameter> inputParamBuffer;

// the hidden layer : 64 -> 64
[[vk::binding(  8   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> hiddenGradBuffer;
[[vk::binding(  9   ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> hiddenWeightBuffer;
[[vk::binding(  10  ,   MOX_NKPD_INCLUDED    )]] ConstantBuffer<AI::OffsetData> hiddenOffsetBuffer;
[[vk::binding(  11  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> hiddenStagingGradBuffer;
[[vk::binding(  12  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> hiddenStagingWeightBuffer;
[[vk::binding(  13  ,   MOX_NKPD_INCLUDED    )]] ConstantBuffer<AI::ptrData> hiddenPtrBuffer;
[[vk::binding(  14  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<AI::Parameter> hiddenParamBuffer;

// the output layer : 64 -> 16 ( one weight per neighbour )
[[vk::binding(  15  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> outputGradBuffer;
[[vk::binding(  16  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> outputWeightBuffer;
[[vk::binding(  17  ,   MOX_NKPD_INCLUDED    )]] ConstantBuffer<AI::OffsetData> outputOffsetBuffer;
[[vk::binding(  18  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> outputStagingGradBuffer;
[[vk::binding(  19  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<float> outputStagingWeightBuffer;
[[vk::binding(  20  ,   MOX_NKPD_INCLUDED    )]] ConstantBuffer<AI::ptrData> outputPtrBuffer;
[[vk::binding(  21  ,   MOX_NKPD_INCLUDED    )]] RWStructuredBuffer<AI::Parameter> outputParamBuffer;


struct NKPDOptimal
{
    bool validUV;
    bool useIndirect;
    uint2 prevCoord;
    uint2 currentCoord;
};

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

static const int NP_F16 = 0;
static const int NP_LAYOUT_ROW_MAJOR = 0;
static const int NP_LAYOUT_COLUMN_MAJOR = 1;
static const int NP_LAYOUT_INFERENCING_OPTIMAL = 2;
static const int NP_LAYOUT_TRAINING_OPTIMAL = 3;

#if defined(MOX_COOP_VECTOR) && defined(MOX_SCENE_IMAGES_INCLUDED)

    // the matrix operations of the cooperative vector take a POINTER ( ptrData holds the addresses of the weight and gradient buffers ) , the same idea as NrcPtr
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


    float derivativeSoftplus(const float x)
    {
        return 1.0f /(1.0f + exp(-x));
    }

    float softplus(float x)
    {
        return log(1.0f + exp(x));
    }

    static const uint32_t PIXEL_COUNT = 16;
    // the numbers of ONE neighbour : nInput 3 , prevInput 3 , normalW , depthW , hit , colorDiff 3
    static const uint FEATURE_COUNT = 12;
    // the length of the history ( the counter and 1 / alpha of the temporal part ) : the same as the temporal pass of ReBLUR
    static const float MAX_ACCUM_FRAMES = 64.0f;
    // the history after which the blur radius is the smallest : the same as the spatial pass of ReBLUR ( it is NOT the length of the history )
    static const float RADIUS_HISTORY_FRAMES = 32.0f;
    static const float MAX_RADIUS_PIXELS = 32.0f;
    static const float MAX_HIT_DISTANCE = 50.0f;
    static const float DEPTH_SIGMA = 8.0f;
    static const float NORMAL_POWER = 128.0f;

    float ComputeBlurRadius(float normalizedHitDist, float historyCount) {
        float baseRadius = normalizedHitDist * MAX_RADIUS_PIXELS;
        float historyFactor = saturate(1.0 - historyCount / RADIUS_HISTORY_FRAMES);
        return baseRadius * lerp(0.25, 1.0, historyFactor); // the radius never falls to 0
    }

    uint randomNumberIn(inout uint seed , const uint totalCount){
        return uint(rand(seed) * float(totalCount));
    }

    float NormalizeHitDistance(float hitDistance, float viewZ, float roughness) {
        float scale = lerp(viewZ, viewZ * roughness, roughness); // roughness = 0 for ReBLUR
        return hitDistance / max(scale, 1e-4);
    }

    float3 readInput(const uint2 p , const bool useIndirect)
    {
#if defined(NKPD_TRAINING)
        // the training shader : diffuseImage / indirectImage are the ground truth ( the result of ReBLUR ) , the net gets the signal before its spatial pass
        // tmpDiffuseImage keeps only the LAST temporal pass of ReBLUR , that is the indirect one
        return tmpDiffuseImage[p].xyz;
#endif   
        if(useIndirect)
        {
            return indirectImage[p].xyz;
        }
        else
        {
            return diffuseImage[p].xyz;
        }
    }

    float3 readPrevInput(const uint2 p , const bool useIndirect)
    {
        if(useIndirect) return prevIndirectImage[p].xyz;
        return prevDiffuseImage[p].xyz;
    }

    float readHitDistance(const uint2 p , const bool useIndirect)
    {
        if(useIndirect) return indirectDistance[p];
        return diffuseDistance[p];
    }

    float readHistoryCount(const uint2 p , const bool useIndirect)
    {
        if(useIndirect) return diffuseHistoryImage[p].g;
        return diffuseHistoryImage[p].r;
    }

    // does the pixel of the last frame show the same surface : the same check as the temporal pass of ReBLUR
    bool checkDisocclusion(const uint2 currentPixel , const uint2 prevPixel)
    {
        const float3 centerPosition = positionImage[currentPixel].xyz;
        const float3 centerNormal = normalImage[currentPixel].xyz;

        const float4 prevPositionSample = prevPositionImage[prevPixel];
        const float3 nNormal = prevNormalImage[prevPixel].xyz;

        // w == 0 marks a sky texel written by the miss shader
        if(prevPositionSample.w == 0.0f) return false;

        const float positionDiff = length(centerPosition - prevPositionSample.xyz);
        if(positionDiff > 0.05f) return false;

        const float normalDot = dot(centerNormal , nNormal);
        if(normalDot < 0.9f) return false;

        return true;
    }

    float3 ColorDiff(float3 n, float3 c)
    {
        return (n - c) / (n + c + 0.01f);   // каждый канал отдельно
    }

    // diffuse.xyz or indirect                      3
    // prev diffuse.xyz  or indirect                6
    // normal check diff                            7
    // depth diff                                   8
    // hit distance ,  color diff                   9

    #define NKPD_H(k) ((float16_t)data[k])
    #define NKPD_ROW(b) NKPD_H((b) + 0) , NKPD_H((b) + 1) , NKPD_H((b) + 2) , NKPD_H((b) + 3) , NKPD_H((b) + 4) , NKPD_H((b) + 5) , NKPD_H((b) + 6) , NKPD_H((b) + 7) ,                     NKPD_H((b) + 8) , NKPD_H((b) + 9) , NKPD_H((b) + 10) , NKPD_H((b) + 11) , NKPD_H((b) + 12) , NKPD_H((b) + 13) , NKPD_H((b) + 14) , NKPD_H((b) + 15)
    Coop192 makeFirstLayerVector(const float data[FEATURE_COUNT * PIXEL_COUNT])
    {
        return coop::construct192(NKPD_ROW(0) , NKPD_ROW(16) , NKPD_ROW(32) , NKPD_ROW(48) , NKPD_ROW(64) , NKPD_ROW(80) ,
                                NKPD_ROW(96) , NKPD_ROW(112) , NKPD_ROW(128) , NKPD_ROW(144) , NKPD_ROW(160) , NKPD_ROW(176));
    }
    #undef NKPD_ROW
    #undef NKPD_H

    void fillFirstLayerVector(NKPDOptimal info , out float weights[PIXEL_COUNT] , out Coop192 firstLayerVector , out float3 radiances[PIXEL_COUNT])
    {
        uint2 id = info.currentCoord;
        uint2 pid = info.prevCoord;
        const float3 center = readInput(id , info.useIndirect);
        const float history = readHistoryCount(id , info.useIndirect);
        const float3 cN = normalImage[id].xyz;
        const float cD = depthImage[id];
        float distance = readHitDistance(id , info.useIndirect);
        uint width , height;
        depthImage.GetDimensions(width , height);
        distance = min(distance , MAX_HIT_DISTANCE);
        float radius = min(ComputeBlurRadius(NormalizeHitDistance(distance , cD , 0) , history ) , MAX_RADIUS_PIXELS);

        // the centre is accumulated the same way as the neighbours ( below ) , so the colour difference compares accumulated with accumulated .
        // it is the same formula as the temporal part of forwardNKPD
        const float centerAlpha = info.validUV ? 1.0f / min(history + 1.0f , MAX_ACCUM_FRAMES) : 1.0f;
        const float3 centerAccumulated = lerp(readPrevInput(pid , info.useIndirect) , center , centerAlpha);

        static const uint stride = FEATURE_COUNT;
        float data[FEATURE_COUNT * PIXEL_COUNT];

        uint patternSeed = initSeed(id.xy , generalBuffer.frameIndex);
        const float patternAngle = rand(patternSeed) * 2.0f * PI;
        const float2 patternRotation = float2(cos(patternAngle) , sin(patternAngle));

        [unroll]
        for(uint i = 0 ; i < PIXEL_COUNT ; i++)
        {
            const float2 poisson = PoissonDisk[i];
            const float2 turned = float2(poisson.x * patternRotation.x - poisson.y * patternRotation.y , poisson.x * patternRotation.y + poisson.y * patternRotation.x);
            float2 miss = turned * radius;
            int2 samplePos = int2(id.xy) + int2(miss);
            if(any(samplePos < int2(0 , 0)) || any(samplePos >= int2(width , height)))
            {
                weights[i] = 0.0f;
                radiances[i] = 0.0f;

                data[i * stride] = 0.0f;
                data[i * stride + 1] = 0.0f;
                data[i * stride + 2] = 0.0f;
                data[i * stride + 3] = 0.0f;
                data[i * stride + 4] = 0.0f;
                data[i * stride + 5] = 0.0f;
                data[i * stride + 6] = 0.0f;
                data[i * stride + 7] = 0.0f;
                data[i * stride + 8] = 0.0f;
                data[i * stride + 9] = 0.0f;
                data[i * stride + 10] = 0.0f;
                data[i * stride + 11] = 0.0f;

                continue;
            }

            if(pixelCheck[samplePos] != COLOR_PIXEL)        
            {
                weights[i] = 0.0f;
                radiances[i] = 0.0f;
                data[i * stride] = 0.0f;
                data[i * stride + 1] = 0.0f;
                data[i * stride + 2] = 0.0f;
                data[i * stride + 3] = 0.0f;
                data[i * stride + 4] = 0.0f;
                data[i * stride + 5] = 0.0f;
                data[i * stride + 6] = 0.0f;
                data[i * stride + 7] = 0.0f;
                data[i * stride + 8] = 0.0f;
                data[i * stride + 9] = 0.0f;
                data[i * stride + 10] = 0.0f;
                data[i * stride + 11] = 0.0f;

                continue;
            }
            const float3 P = positionImage[samplePos].xyz;
            float4 clipPos = mul(generalBuffer.prevViewProj , float4(P , 1.0f));
            float3 pos = clipPos.xyz / clipPos.w;
            float2 prevUV = pos.xy * 0.5f + 0.5f;
            prevUV.y = 1.0f - prevUV.y;
            const bool validUV = (prevUV.x >= 0.0f && prevUV.x <= 1.0f && prevUV.y >= 0.0f && prevUV.y <= 1.0f);
            uint2 prevPixel = min(uint2(max(prevUV * float2(uint2(width , height)) , 0.0f)) , uint2(width , height) - 1u);      

            float3 nInput = readInput(samplePos , info.useIndirect);
            const float3 prevInput = readPrevInput(prevPixel , info.useIndirect);
            const float pHistory = readHistoryCount(prevPixel , info.useIndirect);

            // the position and the normal of the last frame have to match , the screen check alone gives ghosts on the edges of the objects
            const bool nValid = validUV && checkDisocclusion(samplePos , prevPixel);

            float alpha = 1.0f;
            if(nValid)
            {
                alpha = 1.0f / min(pHistory + 1.0f , MAX_ACCUM_FRAMES);
            }

            nInput = lerp(prevInput , nInput , alpha);

            float sDepth = depthImage[samplePos];
            float3 sNormal = normalImage[samplePos].xyz;

            float3 colorDiff = ColorDiff(nInput , centerAccumulated);

            float depthW = exp(-abs(cD - sDepth) /cD * DEPTH_SIGMA);
            float normalW = pow(saturate(dot(sNormal , cN)) , NORMAL_POWER);

            const float hit = readHitDistance(samplePos , info.useIndirect);

            weights[i] = 1.0f;
            data[i * stride] = nInput.x;
            data[i * stride + 1] = nInput.y;
            data[i * stride + 2] = nInput.z;
            data[i * stride + 3] = prevInput.x;
            data[i * stride + 4] = prevInput.y;
            data[i * stride + 5] = prevInput.z;
            data[i * stride + 6] = normalW;
            data[i * stride + 7] = depthW;
            data[i * stride + 8] = hit;
            data[i * stride + 9] = colorDiff.x;
            data[i * stride + 10] = colorDiff.y;
            data[i * stride + 11] = colorDiff.z;
            radiances[i] = nInput;
        }

        firstLayerVector = makeFirstLayerVector(data);
    }

    // forward

    // x , z1 , h1 , z2 , h2 : the vectors of the layers ( the backward needs them )
    // o : the 16 raw outputs of the net ( before the sigmoid ) , weights : 1.0f / 0.0f , is the neighbour valid , radiances : the colours of the neighbours
    // the weights of the neighbours are  w[i] = weights[i] * sigmoid( o[i] ) , wSum is their sum : both are rebuilt from o and weights
    float3 forwardNKPD(NKPDOptimal info ,
                    out Coop192 x , out Coop64 z1 , out Coop64 h1 , out Coop64 z2 , out Coop64 h2 ,
                    out Coop16 o , out float weights[PIXEL_COUNT] , out float3 radiances[PIXEL_COUNT] ,
                    out float history)
    {
        Coop192 input;
        fillFirstLayerVector(info , weights , input , radiances);
        x = input;
        const BufPtr pInputWeights = BufPtr(inputPtrBuffer.pWeightBuffer);
        const BufPtr pHiddenWeights = BufPtr(hiddenPtrBuffer.pWeightBuffer);
        const BufPtr pOutputWeights = BufPtr(outputPtrBuffer.pWeightBuffer);

        Coop64 inputResult = coop::matMulAdd<Coop64>(input , NP_F16 , pInputWeights.Get().data , 0 , NP_F16,
                                                    pInputWeights.Get().data , int(inputOffsetBuffer.BIAS_OFFSET) , NP_F16,
                                                    64 , 192 , NP_LAYOUT_INFERENCING_OPTIMAL , false , 192 * 2 , 0);


        z1 = inputResult;
        inputResult = coop::relu(inputResult);
        h1 = inputResult;

        Coop64 hiddenResult = coop::matMulAdd<Coop64>(inputResult , NP_F16 , pHiddenWeights.Get().data , 0 , NP_F16,
                                                    pHiddenWeights.Get().data , int(hiddenOffsetBuffer.BIAS_OFFSET) , NP_F16,
                                                    64 , 64 , NP_LAYOUT_INFERENCING_OPTIMAL , false , 64 * 2 , 0);

        z2 = hiddenResult;
        hiddenResult = coop::relu(hiddenResult);
        h2 = hiddenResult;

        Coop16 outputResult = coop::matMulAdd<Coop16>(hiddenResult , NP_F16 , pOutputWeights.Get().data , 0 , NP_F16,
                                                    pOutputWeights.Get().data , int(outputOffsetBuffer.BIAS_OFFSET) , NP_F16,
                                                    16 , 64 , NP_LAYOUT_INFERENCING_OPTIMAL , false , 64 * 2 , 0);

        o = outputResult;

        float w[PIXEL_COUNT];
        float wSum = 0.0f;
        [unroll]
        for(uint i = 0 ; i < PIXEL_COUNT ; i++)
        {
            const float raw = float(coop::extractDynamic(outputResult , i));
            w[i] = weights[i] / (1.0f + exp(-raw));
            wSum += w[i];
        }

        uint2 id = info.currentCoord;
        uint2 pid = info.prevCoord;
        const float3 center = readInput(id , info.useIndirect);
        history = readHistoryCount(id , info.useIndirect);
        // the signal of the LAST frame ( not the current one ) at the reprojected pixel
        const float3 prev = readPrevInput(pid , info.useIndirect);
        float alpha = 1.0f;
        if(info.validUV)
        {
            alpha = 1.0f / min(history + 1.0f , float(MAX_ACCUM_FRAMES));
        }
        else
        {
            history = 1.0f;
            alpha = 1.0f;
        }

        float3 sum = lerp(prev , center , alpha);

        [unroll]
        for(uint j = 0 ; j < PIXEL_COUNT ; j++)
        {
            sum += w[j] * radiances[j];
        }

        return sum / (1.0f + wSum);
    }

    // the info of a pixel : the pixel of the last frame ( by the motion vector ) and is it usable
    NKPDOptimal buildNKPDInfo(const uint2 pixel , const bool useIndirect)
    {
        uint width , height;
        depthImage.GetDimensions(width , height);
        const float2 size = float2(width , height);
        const float2 uv = (float2(pixel) + 0.5f) / size;
        const float2 prevUV = uv + motionVector[pixel];

        NKPDOptimal info;
        info.currentCoord = pixel;
        info.prevCoord = uint2(round(clamp(prevUV * size - 0.5f , float2(0.0f , 0.0f) , size - 1.0f)));
        info.useIndirect = useIndirect;
        info.validUV = !(any(prevUV > float2(1.0f , 1.0f)) || any(prevUV < float2(0.0f , 0.0f))) && checkDisocclusion(pixel , info.prevCoord);
        return info;
    }

    // the ground truth of the training : the result of ReBLUR
    float3 readIdeal(const uint2 p , const bool useIndirect)
    {
        if(useIndirect) return indirectImage[p].xyz;
        return diffuseImage[p].xyz;
    }

    // ask

    float3 askNKPD(NKPDOptimal info , out float history)
    {
        Coop192 x ; Coop64 z1 ; Coop64 h1 ; Coop64 z2 ; Coop64 h2 ;
        Coop16 o ;
        float weights[PIXEL_COUNT];
        float3 radiances[PIXEL_COUNT];
        const float3 result = forwardNKPD(info , x , z1 , h1 , z2 , h2 , o , weights , radiances , history);
        return result;
    }

    // backstep

    // ONLY NON SKYBOX
    void processBackStep(const float3 ideal , NKPDOptimal info , const bool valid)
    {
        Coop192 x ; Coop64 z1 ; Coop64 h1 ; Coop64 z2 ; Coop64 h2 ;
        Coop16 o ;
        float weights[PIXEL_COUNT];
        float3 radiances[PIXEL_COUNT];
        float history;
        const float3 result = forwardNKPD(info , x , z1 , h1 , z2 , h2 , o , weights , radiances , history);
        float dOut[PIXEL_COUNT];

        float wSum = 0.0f;
        [unroll]
        for(uint i = 0 ; i < PIXEL_COUNT ; i++)
        {
            // the same weight as in forwardNKPD : the mask times the SIGMOID of the raw output ( the raw output itself is not a weight )
            const float raw = float(coop::extractDynamic(o , i));
            wSum += weights[i] / (1.0f + exp(-raw));
        }
        const float3 g = 2.0f * (result - ideal) / (result * result + 1e-3f);
        [unroll]
        for(uint i = 0 ; i < PIXEL_COUNT ; i++)
        {
            const float raw = float(coop::extractDynamic(o , i));
            const float s = 1.0f / (1.0f + exp(-raw));
            const float dw_draw = weights[i] * s * (1.0f - s);                    
            const float dL_dw = dot(g , radiances[i] - result) / (1.0f + wSum);
            dOut[i] = dL_dw * dw_draw * (valid ? 1.0f / 16.0f : 0.0f) * weights[i];  
        }
        Coop16 inversedCoop = coop::construct16(float16_t(dOut[0]) , float16_t(dOut[1]) , float16_t(dOut[2]), float16_t(dOut[3])
                                            , float16_t(dOut[4]) , float16_t(dOut[5]) ,float16_t(dOut[6]) ,float16_t(dOut[7])
                                            , float16_t(dOut[8]) , float16_t(dOut[9]) ,float16_t(dOut[10]) ,float16_t(dOut[11])
                                            ,float16_t(dOut[12]) ,float16_t(dOut[13]) ,float16_t(dOut[14]) ,float16_t(dOut[15]));

        const BufPtr pInputWeights = BufPtr(inputPtrBuffer.pWeightBuffer);
        const BufPtr pHiddenWeights = BufPtr(hiddenPtrBuffer.pWeightBuffer);
        const BufPtr pOutputWeights = BufPtr(outputPtrBuffer.pWeightBuffer);

        const BufPtr pInputGrads = BufPtr(inputPtrBuffer.pGradBuffer);
        const BufPtr pHiddenGrads = BufPtr(hiddenPtrBuffer.pGradBuffer);
        const BufPtr pOutputGrads = BufPtr(outputPtrBuffer.pGradBuffer);

        coop::outerProduct(pOutputGrads.Get().data , 0 , inversedCoop , h2 , NP_LAYOUT_TRAINING_OPTIMAL , NP_F16 , 64 * 2);

        // Out matMul(In input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);
        // void outerProduct([[vk::ext_reference]] uint pointer[1 << 20] , int offset , A a , B b , int layout , int matrixInterp , uint stride);

        Coop64 layer2 = coop::matMul<Coop64>(inversedCoop , NP_F16 , pOutputWeights.Get().data , int(outputOffsetBuffer.TRAINING_W_OFFSET), NP_F16 , 64 , 16 ,  NP_LAYOUT_TRAINING_OPTIMAL , true , 64 * 2 , 0);
        Coop64 gate2 = coop::reluGate(z2);
        Coop64 dz2 = coop::mul(layer2 , gate2);
        coop::outerProduct(pHiddenGrads.Get().data , 0 , dz2 , h1 , NP_LAYOUT_TRAINING_OPTIMAL , NP_F16 , 64 * 2);

        Coop64 layer1 = coop::matMul<Coop64>(dz2 , NP_F16 , pHiddenWeights.Get().data , int(hiddenOffsetBuffer.TRAINING_W_OFFSET), NP_F16 , 64 , 64 ,  NP_LAYOUT_TRAINING_OPTIMAL , true , 64 * 2 , 0);
        Coop64 gate1 = coop::reluGate(z1);
        Coop64 dz1 = coop::mul(layer1 , gate1);
        coop::outerProduct(pInputGrads.Get().data , 0 , dz1 , x , NP_LAYOUT_TRAINING_OPTIMAL , NP_F16 , 192 * 2);

        coop::reduceSum(pOutputGrads.Get().data , outputOffsetBuffer.BIAS_GRAD_OFFSET , inversedCoop);
        coop::reduceSum(pHiddenGrads.Get().data , hiddenOffsetBuffer.BIAS_GRAD_OFFSET , dz2);
        coop::reduceSum(pInputGrads.Get().data , inputOffsetBuffer.BIAS_GRAD_OFFSET , dz1);
    }

    // apply

    static const float NP_LR_MLP = 1e-3f;
    static const float NP_GRAD_SCALE = 1.0f / 16.0f;

    void recordAdam_input(const uint3 id)
    {
        const uint validCount = NKPDGeneral[0].validCount;
        if(validCount == 0) return;
        const uint pos = id.x;
        static const uint matrixSize = 64 * 192;
        const uint totalSize = matrixSize + 64;
        if(pos >= totalSize) return;
        float g = 0.0f;

        const BufPtr ptrW = BufPtr(inputPtrBuffer.pWeightBuffer);
        const BufPtr ptrG = BufPtr(inputPtrBuffer.pGradBuffer);

        if(pos < matrixSize)
        {
            g = inputStagingGradBuffer[pos] / (float(validCount) * NP_GRAD_SCALE);
        }
        else
        {
            // the bias : its gradient is fp16 inside of the grads buffer after the TRAINING_OPTIMAL matrix
            const uint b = pos - matrixSize;
            g = readHalf(ptrG , inputOffsetBuffer.BIAS_GRAD_OFFSET / 2 + b) / (float(validCount) * NP_GRAD_SCALE);
        }

        AI::Parameter p = inputParamBuffer[pos];
        if(!isfinite(g)) return;
        adamStep(p , g , NP_LR_MLP);

        inputParamBuffer[pos] = p;

        if(pos < matrixSize)
        {
            inputStagingWeightBuffer[pos] = p.w;
        }
        else
        {
            writeHalf(ptrW , inputOffsetBuffer.BIAS_OFFSET / 2 + (pos - matrixSize) , p.w);
        }
    }

    void recordAdam_hidden(const uint3 id)
    {
        const uint validCount = NKPDGeneral[0].validCount;
        if(validCount == 0) return;
        const uint pos = id.x;
        static const uint matrixSize = 64 * 64;
        const uint totalSize = matrixSize + 64;
        if(pos >= totalSize) return;
        float g = 0.0f;

        const BufPtr ptrW = BufPtr(hiddenPtrBuffer.pWeightBuffer);
        const BufPtr ptrG = BufPtr(hiddenPtrBuffer.pGradBuffer);

        if(pos < matrixSize)
        {
            g = hiddenStagingGradBuffer[pos] / (float(validCount) * NP_GRAD_SCALE);
        }
        else
        {
            const uint b = pos - matrixSize;
            g = readHalf(ptrG , hiddenOffsetBuffer.BIAS_GRAD_OFFSET / 2 + b) / (float(validCount) * NP_GRAD_SCALE);
        }

        AI::Parameter p = hiddenParamBuffer[pos];
        if(!isfinite(g)) return;
        adamStep(p , g , NP_LR_MLP);

        hiddenParamBuffer[pos] = p;

        if(pos < matrixSize)
        {
            hiddenStagingWeightBuffer[pos] = p.w;
        }
        else
        {
            writeHalf(ptrW , hiddenOffsetBuffer.BIAS_OFFSET / 2 + (pos - matrixSize) , p.w);
        }
    }

    void recordAdam_output(const uint3 id)
    {
        const uint validCount = NKPDGeneral[0].validCount;
        if(validCount == 0) return;
        const uint pos = id.x;
        static const uint matrixSize = 16 * 64;
        const uint totalSize = matrixSize + 16;
        if(pos >= totalSize) return;
        float g = 0.0f;

        const BufPtr ptrW = BufPtr(outputPtrBuffer.pWeightBuffer);
        const BufPtr ptrG = BufPtr(outputPtrBuffer.pGradBuffer);

        if(pos < matrixSize)
        {
            g = outputStagingGradBuffer[pos] / (float(validCount) * NP_GRAD_SCALE);
        }
        else
        {
            const uint b = pos - matrixSize;
            g = readHalf(ptrG , outputOffsetBuffer.BIAS_GRAD_OFFSET / 2 + b) / (float(validCount) * NP_GRAD_SCALE);
        }

        AI::Parameter p = outputParamBuffer[pos];
        if(!isfinite(g)) return;
        adamStep(p , g , NP_LR_MLP);

        outputParamBuffer[pos] = p;

        if(pos < matrixSize)
        {
            outputStagingWeightBuffer[pos] = p.w;
        }
        else
        {
            writeHalf(ptrW , outputOffsetBuffer.BIAS_OFFSET / 2 + (pos - matrixSize) , p.w);
        }
    }

#endif

}
}
 #endif
#endif
