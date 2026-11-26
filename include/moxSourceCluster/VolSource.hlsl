#pragma once
// #include "../../include/moxSourceCluster/VolSource.hlsl"
#if defined (MOX_VOL_SOURCE_INCLUDED) && MOX_VOL_SOURCE_INCLUDED >= 0ul && defined (MOX_SHADER_STRUCTURES)

[[vk::binding(0 , MOX_VOL_SOURCE_INCLUDED)]] ConstantBuffer<FogVolume> fogBuffer;
[[vk::binding(1 , MOX_VOL_SOURCE_INCLUDED)]] Texture2DArray txFogImageArray;
[[vk::image_format("rgba16f")]]
[[vk::binding(2 , MOX_VOL_SOURCE_INCLUDED)]] RWTexture2DArray<float4> RWFogImageArray;
// the volume of the LAST frame ( the temporal accumulation of Volumentric )
[[vk::binding(3 , MOX_VOL_SOURCE_INCLUDED)]] Texture2DArray txFogHistory;

// the density ( the extinction sigmaT ) of the fog in the world point : it is NOT stored in the volume ( the alpha of the volume is the count of the accumulated frames ) , both Volumentric and PostProcess take it from here
float fogDensityAt(const float3 position){
    return fogBuffer.density * exp(-fogBuffer.heightFallOff * max(position.y - fogBuffer.fogBaseHeight , 0.0f));
}

// ---------------------------------------------------------------- the DUST : the density of the air that is lit by the sun and the lamps ( the shafts ) . a soft noise that the wind carries ,
// no height falloff : the dust is everywhere , it is seen only where the light is
float dustHash(float3 p)
{
    p = frac(p * 0.3183099f + 0.1f);
    p *= 17.0f;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float dustValueNoise(const float3 x)
{
    const float3 i = floor(x);
    float3 f = frac(x);
    f = f * f * (3.0f - 2.0f * f);

    const float n000 = dustHash(i + float3(0.0f , 0.0f , 0.0f));
    const float n100 = dustHash(i + float3(1.0f , 0.0f , 0.0f));
    const float n010 = dustHash(i + float3(0.0f , 1.0f , 0.0f));
    const float n110 = dustHash(i + float3(1.0f , 1.0f , 0.0f));
    const float n001 = dustHash(i + float3(0.0f , 0.0f , 1.0f));
    const float n101 = dustHash(i + float3(1.0f , 0.0f , 1.0f));
    const float n011 = dustHash(i + float3(0.0f , 1.0f , 1.0f));
    const float n111 = dustHash(i + float3(1.0f , 1.0f , 1.0f));

    return lerp(lerp(lerp(n000 , n100 , f.x) , lerp(n010 , n110 , f.x) , f.y) ,
                lerp(lerp(n001 , n101 , f.x) , lerp(n011 , n111 , f.x) , f.y) , f.z);
}

// the extinction sigmaT of the dust in the world point : Volumentric and PostProcess take it from here when dustMode is on
float dustDensityAt(const float3 position)
{
    const float3 wind = float3(1.0f , 0.12f , 0.55f) * (fogBuffer.dustSpeed * fogBuffer.dustTime);
    const float3 q = (position + wind) * fogBuffer.dustScale;
    const float n = 0.6f * dustValueNoise(q) + 0.3f * dustValueNoise(q * 2.13f + 7.1f) + 0.1f * dustValueNoise(q * 4.71f + 3.3f);
    // the clumps : the density is never 0 ( the shafts are whole ) and never over the set value
    return fogBuffer.dustDensity * lerp(0.35f , 1.0f , saturate(n * 1.4f - 0.2f));
}

#endif
