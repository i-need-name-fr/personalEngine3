#include "../../include/ShaderStructures/ShaderStructure.h"
// this shader is a function of BloomSource : the only descriptor set is the one of the bloom ( bloomBuffer , bloomImages , bloomSampler , bloomTexture )
#define MOX_BLOOM_SOURCE_INCLUDED 0
#include "../../include/moxSourceCluster/BloomSource.hlsl"

// one step of the pyramid : the level srcLevel ( the size src ) -> the level dstLevel ( the size dst , half of it ) , one thread = one texel of the destination
[[vk::push_constant]] struct {
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
    uint srcLevel;
    uint dstLevel;
    uint firstIteration;   // the first step ( the full resolution -> the first mip ) : Karis average
} push;

float karisLuma(const float3 c)
{
    return dot(c , float3(0.2126f , 0.7152f , 0.0722f));
}
struct Taps13
{
    float3 a , b , c , d , e , f , g , h , i , j , k , l , m;
};

Taps13 loadTaps(const float2 uv , const float2 texel)
{
    Taps13 t;
    const uint2 srcSize = uint2(push.srcWidth , push.srcHeight);
    t.a = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2(-2.0f , -2.0f));
    t.b = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2( 0.0f , -2.0f));
    t.c = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2( 2.0f , -2.0f));
    t.d = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2(-1.0f , -1.0f));
    t.e = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2( 1.0f , -1.0f));
    t.f = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2(-2.0f ,  0.0f));
    t.g = bloomSampleLevel(push.srcLevel , srcSize , uv);
    t.h = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2( 2.0f ,  0.0f));
    t.i = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2(-1.0f ,  1.0f));
    t.j = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2( 1.0f ,  1.0f));
    t.k = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2(-2.0f ,  2.0f));
    t.l = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2( 0.0f ,  2.0f));
    t.m = bloomSampleLevel(push.srcLevel , srcSize , uv + texel * float2( 2.0f ,  2.0f));
    return t;
}

// the plain step : 4 groups of the corners ( weight 0.125 each ) + the centre group ( 0.5 ) = 1 , the energy is kept
float3 DownsampleBox13(const float2 uv , const float2 texel)
{
    const Taps13 t = loadTaps(uv , texel);

    const float3 group1 = (t.a + t.b + t.f + t.g) * 0.25f;
    const float3 group2 = (t.b + t.c + t.g + t.h) * 0.25f;
    const float3 group3 = (t.f + t.g + t.k + t.l) * 0.25f;
    const float3 group4 = (t.g + t.h + t.l + t.m) * 0.25f;
    const float3 group5 = (t.d + t.e + t.i + t.j) * 0.25f;

    return (group1 + group2 + group3 + group4) * 0.125f + group5 * 0.5f;
}

float3 DownsampleKaris13(const float2 uv , const float2 texel)
{
    const Taps13 t = loadTaps(uv , texel);

    const float3 group1 = (t.a + t.b + t.f + t.g) * 0.25f;
    const float3 group2 = (t.b + t.c + t.g + t.h) * 0.25f;
    const float3 group3 = (t.f + t.g + t.k + t.l) * 0.25f;
    const float3 group4 = (t.g + t.h + t.l + t.m) * 0.25f;
    const float3 group5 = (t.d + t.e + t.i + t.j) * 0.25f;

    const float w1 = 0.125f / (1.0f + karisLuma(group1));
    const float w2 = 0.125f / (1.0f + karisLuma(group2));
    const float w3 = 0.125f / (1.0f + karisLuma(group3));
    const float w4 = 0.125f / (1.0f + karisLuma(group4));
    const float w5 = 0.5f   / (1.0f + karisLuma(group5));

    return (group1 * w1 + group2 * w2 + group3 * w3 + group4 * w4 + group5 * w5) / (w1 + w2 + w3 + w4 + w5);
}

[shader("compute")]
[numthreads(8 , 8 , 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    // the dispatch is rounded up to whole 8x8 groups , the tail threads have to bail
    if(id.x >= push.dstWidth || id.y >= push.dstHeight) return;

    // the centre of the destination texel in [0 , 1] of the level ; it is the same point in the source ( the source is 2x bigger ) , the offsets of the taps are in the texels of the source
    const float2 uv = (float2(id.xy) + 0.5f) / float2(push.dstWidth , push.dstHeight);
    const float2 srcTexel = 1.0f / float2(push.srcWidth , push.srcHeight);

    const float3 result = (push.firstIteration != 0u) ? DownsampleKaris13(uv , srcTexel) : DownsampleBox13(uv , srcTexel);
    bloomImages[uint3(id.xy , push.dstLevel)] = float4(result , 1.0f);
}
