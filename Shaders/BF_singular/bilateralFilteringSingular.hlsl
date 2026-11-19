#include "../../include/ShaderStructures/ShaderStructure.h"
#define MOX_SCENE_BUFFERS_INCLUDED 0
#include "../../include/moxSourceCluster/SceneBuffers.hlsl"
#define MOX_SCENE_IMAGES_INCLUDED 1
#include "../../include/moxSourceCluster/SceneImages.hlsl"

[[vk::push_constant]] struct{
    int isAO;
} push;

// SPIR-V cannot select a resource ( image ) at runtime through a local variable ( OpPhi of OpTypeImage is invalid )
// so we select the PIXEL , not the image
float readInput(int2 p)
{
    if(push.isAO) return ambientImage[p];
    return shadowImage[p];
}

void writeOutput(int2 p , float value)
{
    if(push.isAO) fAmbientImage[p] = value;
    else fShadowImage[p] = value;
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID){
    uint width , height;
    // all the screen images have the same size
    ambientImage.GetDimensions(width , height);

    if(id.x >= width || id.y >= height) return;   // >=, the last row/column is width-1

    // signed from here on: `uint2 + int2` promotes to uint2, so pixel 0 with dx=-1
    // wraps to 0xFFFFFFFF and clamp() drags it to the opposite edge of the screen
    const int2 pixel = int2(id.xy);
    const int2 maxCoord = int2(width , height) - 1;  

    if(pixelCheck[id.xy] != COLOR_PIXEL){
        writeOutput(pixel , 1.0f);
        return;
    }

    if(push.isAO ){
        if(features.AO == AO_NOT_REQUIRED){
            writeOutput(pixel , 1.0f);
            return;
        }
    }

    int3 UV = int3(pixel, 0);
    const float3 centerNormal = normalImage[pixel].xyz;
    const float centerDepth = depthImage[pixel];

    float filtered = 0.0f;
    float weight = 0.0f;

    for(int dx = -2 ; dx <= 2 ; dx++){
        for(int dy = -2 ; dy <= 2 ; dy++){

            int2 nPixel = clamp(pixel + int2(dx , dy) , int2(0 , 0) , maxCoord);
            const int3 pUV = int3(nPixel , 0);
            const float pInput = readInput(nPixel);
            const float3 pNormal = normalImage[nPixel].xyz;
            const float pDepth = depthImage[nPixel];

            const float distSq = float(dx * dx + dy * dy) ;
            const float spatialW = exp(-distSq / 8.0f);
            const float normalW = pow(max(dot(centerNormal , pNormal) ,0.0f ) , 32.0f);
            // depth lives in [0,1], so neighbours differ by ~1e-4 - without the
            // sigma the exponent is ~0 and depth edge-stopping does nothing
            const float depthW = exp(-abs(centerDepth - pDepth) / 0.1f);

            const float w = (dx == 0 && dy == 0) ? spatialW : (spatialW * normalW * depthW);

            weight += w;
            filtered += w * pInput;
        }
    }

    writeOutput(pixel , filtered / max(weight , 0.0001f));
}