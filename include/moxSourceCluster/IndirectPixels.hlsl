#pragma once
#define MOX_INDIRECT_PIXELS_INCLUDED

#if defined (MOX_SCENE_IMAGES_INCLUDED)
uint2 directSize()
{
    uint width , height;
    positionImage.GetDimensions(width , height);
    return uint2(width , height);
}

uint2 indirectSize()
{
    uint width , height;
    indirectImage.GetDimensions(width , height);
    return uint2(width , height);
}

// the pixel of the G buffer that the indirect pixel looks at : the indirect pixel covers a block of the full resolution , jitter01 ( [0 , 1) ) picks the point of the block
uint2 indirectToDirectPixel(const uint2 indirectPixel , const float2 jitter01)
{
    const uint2 dSize = directSize();
    const float2 scale = float2(dSize) / float2(indirectSize());
    return min(uint2((float2(indirectPixel) + jitter01) * scale) , dSize - 1u);
}

// the indirect pixel whose block contains the direct pixel ( the nearest one : for the values that the direct pass reads from the indirect pass )
uint2 directToIndirectPixel(const uint2 directPixel)
{
    const uint2 iSize = indirectSize();
    const float2 uv = (float2(directPixel) + 0.5f) / float2(directSize());
    return min(uint2(uv * float2(iSize)) , iSize - 1u);
}

// the pixel of the indirect resolution / of the G buffer that a point of the screen ( uv ) falls into
uint2 uvToIndirectPixel(const float2 uv)
{
    const uint2 iSize = indirectSize();
    return min(uint2(uv * float2(iSize)) , iSize - 1u);
}

uint2 uvToDirectPixel(const float2 uv)
{
    const uint2 dSize = directSize();
    return min(uint2(uv * float2(dSize)) , dSize - 1u);
}

// the point of the block of the full resolution that the indirect pixel looks at in the frame : a function of ( the pixel , the frame ) only , so the last frame can be recomputed .
// the seed is salted : it is not the same stream as the seed that the passes use for the reservoirs
float2 indirectJitter(const uint2 indirectPixel , const uint frameIndex)
{
    uint jitterSeed = initSeed(indirectPixel , frameIndex ^ 0x68E31DA4u);
    const float x = rand(jitterSeed);
    const float y = rand(jitterSeed);
    return float2(x , y);
}

// the pixel of the G buffer ( of the last frame ) that the reservoir prevIndirectPixel was made for : the same jitter as the one of the last frame
uint2 prevDirectPixelOf(const uint2 prevIndirectPixel)
{
    return indirectToDirectPixel(prevIndirectPixel , indirectJitter(prevIndirectPixel , generalBuffer.frameIndex - 1u));
}

// the indirect light of a pixel of the full resolution : the 4 nearest pixels of indirectImage ( bilinear ) , every one weighted by the geometry of the G buffer : the point that THAT indirect pixel
// looked at in this frame ( the same jitter as the indirect pass ) has to lie on the plane of the surface of this pixel and to have about the same normal . the indirect pixel on another surface does not leak
// ( the edge of a pillar in front of a wall ) . no tap fits : the nearest indirect pixel . the same size as the G buffer : the pixel itself ( the old way , no cost )
// it reads the images of THIS frame : the jitter of the frame is the one that the indirect pass used
float3 upsampleIndirect(const uint2 directPixel)
{
    const uint2 iSize = indirectSize();
    const uint2 dSize = directSize();
    if(all(iSize == dSize)) return indirectImage[directPixel].xyz;

    const float3 P = positionImage[directPixel].xyz;
    const float3 N = normalImage[directPixel].xyz;
    const float tolerance = max(0.05f , length(P - generalBuffer.playerPosition) * 0.02f);

    // the centre of the indirect pixel i is at i + 0.5 in the grid of the indirect resolution
    const float2 gridPosition = (float2(directPixel) + 0.5f) / float2(dSize) * float2(iSize) - 0.5f;
    const float2 base = floor(gridPosition);
    const float2 fraction = gridPosition - base;

    float3 sum = 0.0f;
    float weightSum = 0.0f;
    [unroll]
    for(int y = 0 ; y < 2 ; y++)
    {
        [unroll]
        for(int x = 0 ; x < 2 ; x++)
        {
            const int2 tapSigned = clamp(int2(base) + int2(x , y) , int2(0 , 0) , int2(iSize) - 1);
            const uint2 tap = uint2(tapSigned);
            const float bilinear = ((x == 0) ? 1.0f - fraction.x : fraction.x) * ((y == 0) ? 1.0f - fraction.y : fraction.y);

            const uint2 tapDirect = indirectToDirectPixel(tap , indirectJitter(tap , generalBuffer.frameIndex));
            if(pixelCheck[tapDirect] != COLOR_PIXEL) continue;

            const float planeDistance = abs(dot(positionImage[tapDirect].xyz - P , N));
            const float geometry = exp(-2.0f * planeDistance / tolerance) * pow(saturate(dot(N , normalImage[tapDirect].xyz)) , 16.0f);
            const float weight = bilinear * geometry;
            sum += indirectImage[tap].xyz * weight;
            weightSum += weight;
        }
    }

    if(weightSum < 1e-5f) return indirectImage[directToIndirectPixel(directPixel)].xyz;
    return sum / weightSum;
}
#endif