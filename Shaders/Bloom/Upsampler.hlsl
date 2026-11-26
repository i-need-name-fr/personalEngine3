#include "../../include/ShaderStructures/ShaderStructure.h"
// this shader is a function of BloomSource : the only descriptor set is the one of the bloom ( bloomBuffer , bloomImages , bloomSampler , bloomTexture )
#define MOX_BLOOM_SOURCE_INCLUDED 0
#include "../../include/moxSourceCluster/BloomSource.hlsl"

[[vk::push_constant]] struct {
    uint lowerWidth;    // the size of the level i + 1 ( the one that is upsampled )
    uint lowerHeight;
    uint dstWidth;      // the size of the level i ( the destination )
    uint dstHeight;
    uint lowerLevel;    // i + 1
    uint dstLevel;      // i
} push;

// the 3x3 tent ( 1 2 1 / 2 4 2 / 1 2 1 ) / 16 , the taps are one texel of the LOWER level apart : the bilinear filter makes it a smooth quadratic surface , no diamonds of the plain bilinear
float3 UpsampleTent(const float2 uv , const float2 texel)
{
    const uint2 lowerSize = uint2(push.lowerWidth , push.lowerHeight);
    const uint level = push.lowerLevel;

    const float3 a = bloomSampleLevel(level , lowerSize , uv + float2(-texel.x , -texel.y));
    const float3 b = bloomSampleLevel(level , lowerSize , uv + float2( 0.0f    , -texel.y)) * 2.0f;
    const float3 c = bloomSampleLevel(level , lowerSize , uv + float2( texel.x , -texel.y));
    const float3 d = bloomSampleLevel(level , lowerSize , uv + float2(-texel.x ,  0.0f   )) * 2.0f;
    const float3 e = bloomSampleLevel(level , lowerSize , uv) * 4.0f;
    const float3 f = bloomSampleLevel(level , lowerSize , uv + float2( texel.x ,  0.0f   )) * 2.0f;
    const float3 g = bloomSampleLevel(level , lowerSize , uv + float2(-texel.x ,  texel.y));
    const float3 h = bloomSampleLevel(level , lowerSize , uv + float2( 0.0f    ,  texel.y)) * 2.0f;
    const float3 i = bloomSampleLevel(level , lowerSize , uv + float2( texel.x ,  texel.y));

    return (a + b + c + d + e + f + g + h + i) / 16.0f;
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if(id.x >= push.dstWidth || id.y >= push.dstHeight) return;

    // the same point of the screen in both levels : uv is in [0 , 1] of the level , the sizes differ ( 2x ) but uv does not
    const float2 uv = (float2(id.xy) + 0.5f) / float2(push.dstWidth , push.dstHeight);
    const float2 lowerTexel = 1.0f / float2(push.lowerWidth , push.lowerHeight);

    const float3 upsampled = UpsampleTent(uv , lowerTexel) * saturate(bloomBuffer.range);

    const uint3 texel = uint3(id.xy , push.dstLevel);
    const float3 current = bloomImages[texel].rgb;   // down[i] , the result of the downsample
    bloomImages[texel] = float4(current + upsampled , 1.0f);
}
