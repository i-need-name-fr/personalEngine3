#include "../../include/ShaderStructures/ShaderStructure.h"

#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"

const static float adaptationLight = 3.0f;
const static float adaptationNight = 1.5f;

static const float MIDDLE_GRAY_EV = -2.474f;   // log2(0.18)
static const float LOW_PERCENT  = 0.65f;
static const float HIGH_PERCENT = 0.90f;

static const float MAX_EV = 2.0f;
static const float MIN_EV = -6.0f;

// one thread , once per frame : 255 bins
[shader("compute")]
[numthreads(1 , 1 , 1)]
void main()
{
    uint total = 0;
    for(uint i = 1 ; i < 256 ; i++) total += histoGram[i];

    const float currentEV = EVBuffer[0];
    const bool firstFrame = (EVBuffer[1] == 0.0f);

    const float lowCount  = float(total) * LOW_PERCENT;
    const float highCount = float(total) * HIGH_PERCENT;

    float running = 0.0f;
    float weightSum = 0.0f;
    float countedPixel = 0.0f;

    for(uint b = 1 ; b < 256 ; b++)
    {
        const float t = (float(b) - 0.5f) / 254.0f;
        const float evOfBin = minLogRange + t * lumRange;
        const float nf = float(histoGram[b]);

        const float inWindow = max(0.0f , min(running + nf , highCount) - max(running , lowCount));

        weightSum += evOfBin * inWindow;
        countedPixel += inWindow;
        running += nf;

        histoGram[b] = 0;
    }
    histoGram[0] = 0;

    if(!(countedPixel > 0.0f)) return;

    const float averageEV = weightSum / countedPixel;
    const float targetEV = clamp(MIDDLE_GRAY_EV - averageEV , MIN_EV , MAX_EV);

    float newEV;
    if(firstFrame)
    {
        newEV = targetEV;            // первый кадр : прыгаем сразу
        EVBuffer[1] = 1.0f;
    }
    else
    {
        const float speed = (targetEV > currentEV) ? adaptationNight : adaptationLight;
        newEV = currentEV + (targetEV - currentEV) * (1.0f - exp(-generalBuffer.dt * speed));
    }

    EVBuffer[0] = newEV;
}