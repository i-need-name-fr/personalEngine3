#pragma once

// MOX_RESTIR_INCLUDED is the index of the set of SceneReSTIR
// ping pong : ReSTIRGIBuffer / ReSTIRPTBuffer are the reservoirs that THIS frame writes ( the "next" ones ) , prev... are the ones of the previous frame

#if defined(MOX_RESTIR_INCLUDED) && MOX_RESTIR_INCLUDED >= 0 && defined(MOX_SHADER_STRUCTURES)

[[vk::binding(0 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirGI> ReSTIRGIBuffer;
[[vk::binding(1 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirGI> prevReSTIRGIBuffer;
[[vk::binding(2 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirPT> ReSTIRPTBuffer;
[[vk::binding(3 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirPT> prevReSTIRPTBuffer;
// ReSTIR SSS ( the subsurface scattering ) : the same ping pong
[[vk::binding(4 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirSS> ReSTIRSSBuffer;
[[vk::binding(5 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirSS> prevReSTIRSSBuffer;

#endif
