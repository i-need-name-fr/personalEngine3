#include "../../include/ShaderStructures/ShaderStructure.h"

#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"

const static float adaptationLight = 3.0f;
const static float adaptationNight = 1.5f;

static const float MIDDLE_GRAY_EV = -2.474f;   // log2(0.18)
static const float LOW_PERCENT  = 0.75f;
static const float HIGH_PERCENT = 0.853f;

// the limits of the exposure ( EV : the picture is multiplied by 2^EV ) . the night is dark : without the limit the eye adaptation lifts it up to a day picture
// MAX_EV : how much the exposure may brighten ( 1.0 = x2 , 2.0 = x4 ) , the smaller it is the darker the night stays . MIN_EV : how much it may darken ( the bright day )
static const float MAX_EV = 1.5f;
static const float MIN_EV = -6.0f;

// one thread , once per frame : 255 bins
[shader("compute")]
[numthreads(1 , 1 , 1)]
void main()
{
    // the bin 0 is the black one , it is not counted
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

        // the bin is read for the last time : the histogram of the next frame starts from zero
        histoGram[b] = 0;
    }
    histoGram[0] = 0;

    // the histogram is already cleared above , so this return does not leave old data
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