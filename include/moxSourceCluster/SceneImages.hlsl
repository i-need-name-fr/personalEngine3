#pragma once
// #include "../../include/moxSourceCluster/SceneImages.hlsl"

#if defined(MOX_SCENE_IMAGES_INCLUDED) && MOX_SCENE_IMAGES_INCLUDED >= 0l

[[vk::image_format("rgba32f")]]
[[vk::binding(  0   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> positionImage;
[[vk::image_format("rgba32f")]]
[[vk::binding(  1   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> prevPositionImage;
[[vk::image_format("r32f")]]
[[vk::binding(  2   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> depthImage;
[[vk::image_format("r32f")]]
[[vk::binding(  3   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> prevDepthImage;
[[vk::image_format("rgba16f")]]
[[vk::binding(  4   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> normalImage;
[[vk::image_format("rgba16f")]]
[[vk::binding(  5   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> prevNormalImage;
[[vk::image_format("rgba16f")]]
[[vk::binding(  6   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> diffuseImage;
[[vk::image_format("rgba16f")]]
[[vk::binding(  7   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> prevDiffuseImage;
[[vk::image_format("r32f")]]
[[vk::binding(  8   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> diffuseDistance;
[[vk::image_format("r32f")]]
[[vk::binding(  9   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> prevDiffuseDistance;
[[vk::image_format("rgba16f")]]
[[vk::binding( 10   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> specularImage;
[[vk::image_format("rgba16f")]]
[[vk::binding( 11   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> prevSpecularImage;
[[vk::image_format("r32f")]]
[[vk::binding( 12   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> specularDistance;
[[vk::image_format("r32f")]]
[[vk::binding( 13   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> prevSpecularDistance;
[[vk::image_format("rgba16f")]]
[[vk::binding( 14   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> indirectImage;
[[vk::image_format("rgba16f")]]
[[vk::binding( 15   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> prevIndirectImage;
[[vk::image_format("r32f")]]
[[vk::binding( 16   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> indirectDistance;
[[vk::image_format("r32f")]]
[[vk::binding( 17   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> prevIndirectDistance;

[[vk::image_format("rgba16f")]]
[[vk::binding( 18   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> albedoImage;
[[vk::image_format("r32f")]]
[[vk::binding( 19   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> roughnessImage;
[[vk::image_format("r32f")]]
[[vk::binding( 20   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> metallicImage;
[[vk::image_format("r32f")]]
[[vk::binding( 21   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> objectAOImage;

[[vk::image_format("rg32f")]]
[[vk::binding( 22   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float2> shadowImage;
[[vk::image_format("rg32f")]]
[[vk::binding( 23   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float2> prevShadowImage;
[[vk::image_format("r32f")]]
[[vk::binding( 24   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> ambientImage;
[[vk::image_format("r32f")]]
[[vk::binding( 25   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> prevAmbientImage;
[[vk::image_format("r32f")]]
[[vk::binding( 26   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> fShadowImage;
[[vk::image_format("r32f")]]
[[vk::binding( 27   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> fAmbientImage;

[[vk::image_format("rgba16f")]]
[[vk::binding( 28   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> HDRImage;
[[vk::image_format("rgba16f")]]
[[vk::binding( 29   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> prevHDRImage;
[[vk::image_format("rgba8")]]
[[vk::binding( 30   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> finalImage;
[[vk::image_format("rgba8")]]
[[vk::binding( 31   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> guiImage;
[[vk::image_format("rgba8")]]
[[vk::binding( 32   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> guiCleanImage;

[[vk::image_format("r32f")]]
[[vk::binding( 33   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float> pixelCheck;
[[vk::image_format("rg16f")]]
[[vk::binding( 34   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float2> motionVector;
[[vk::image_format("rgba32f")]]
[[vk::binding( 35   ,   MOX_SCENE_IMAGES_INCLUDED   )]] RWTexture2D<float4> viewImage;
[[vk::binding( 36   ,   MOX_SCENE_IMAGES_INCLUDED   )]] Texture2D HiZImage;
#endif
