#pragma once
// #include "../../include/AI/CoopVec.hlsl"
#define MOX_COOP_VECTOR

template<int N>
using Coop = vk::SpirvType<5288 , N * 2 , 2 , float16_t , vk::integral_constant<int , N> >;
typedef Coop<16> Coop16;
typedef Coop<32> Coop32;
typedef Coop<64> Coop64;

// ---------------------------------------------------------------- the atomic add of a float32 ( SPV_EXT_shader_atomic_float_add , OpAtomicFAddEXT ) : the scalar one , for a buffer of floats
// the device has to have shaderBufferFloat32AtomicAdd ( LogicalDevice.hpp turns it on ) . the pointer is an element of a buffer : atomicAddFloat( buffer[ i ] , ATOMIC_SCOPE_DEVICE , ATOMIC_SEMANTICS_RELAXED , value ) ,
// the old value is given back
static const uint ATOMIC_SCOPE_DEVICE = 1u;          // the memory scope : the whole device ( 2 : the workgroup , 3 : the subgroup )
static const uint ATOMIC_SCOPE_WORKGROUP = 2u;
static const uint ATOMIC_SEMANTICS_RELAXED = 0u;     // no ordering of the memory : an add does not need it

[[vk::ext_capability(6033)]]
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_instruction(6035)]]
float atomicAddFloat([[vk::ext_reference]] float pointer , uint scope , uint semantics , float value);

namespace coop
{

// the size of a vector as a COMPILE TIME constant : coop::Size<Coop64>::value . M and K of the matrix operations have to be constants ( SPIR-V : "M is not a constant instruction" ) , so they cannot come from a buffer
template<typename V> struct Size;
template<int N> struct Size< Coop<N> > { static const int value = N; };

// ---------------------------------------------------------------- matrix and vector
// W is ( M x K ) : M is the size of the result , K is the size of the input . the stride is the bytes of a row of the matrix AS IT LIES in memory ( with transpose = true the matrix keeps its own rows )
template<typename Out , typename In>
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5292)]]
Out matMulAdd(In input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , [[vk::ext_reference]] uint bias[1 << 20] , int biasOffset , int biasInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);

template<typename Out , typename In>
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5289)]]
Out matMul(In input , int inputInterp , [[vk::ext_reference]] uint matrix[1 << 20] , int matrixOffset , int matrixInterp , int M , int K , int layout , bool transpose , uint stride , [[vk::ext_literal]] int memoryOperand);

// ---------------------------------------------------------------- the gradients ( CooperativeVectorTrainingNV )
// every thread gives its own a and b , the extension adds the products of ALL the threads of the wave into the buffer itself : no atomic add of your own
template<typename A , typename B>
[[vk::ext_capability(5435)]]
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5290)]]
void outerProduct([[vk::ext_reference]] uint pointer[1 << 20] , int offset , A a , B b , int layout , int matrixInterp , uint stride);

template<typename V>
[[vk::ext_capability(5435)]]
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(5291)]]
void reduceSum([[vk::ext_reference]] uint pointer[1 << 20] , int offset , V v);

// ---------------------------------------------------------------- the vectors : to build , to read , to change
// OpCompositeConstruct needs a number for EVERY component : the count of the arguments is the size , so every size has its own function
[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop16 construct16(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 ,
                   float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15);

[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop32 construct32(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 ,
                   float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15 ,
                   float16_t a16 , float16_t a17 , float16_t a18 , float16_t a19 , float16_t a20 , float16_t a21 , float16_t a22 , float16_t a23 ,
                   float16_t a24 , float16_t a25 , float16_t a26 , float16_t a27 , float16_t a28 , float16_t a29 , float16_t a30 , float16_t a31);

[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop64 construct64(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 ,
                   float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15 ,
                   float16_t a16 , float16_t a17 , float16_t a18 , float16_t a19 , float16_t a20 , float16_t a21 , float16_t a22 , float16_t a23 ,
                   float16_t a24 , float16_t a25 , float16_t a26 , float16_t a27 , float16_t a28 , float16_t a29 , float16_t a30 , float16_t a31 ,
                   float16_t a32 , float16_t a33 , float16_t a34 , float16_t a35 , float16_t a36 , float16_t a37 , float16_t a38 , float16_t a39 ,
                   float16_t a40 , float16_t a41 , float16_t a42 , float16_t a43 , float16_t a44 , float16_t a45 , float16_t a46 , float16_t a47 ,
                   float16_t a48 , float16_t a49 , float16_t a50 , float16_t a51 , float16_t a52 , float16_t a53 , float16_t a54 , float16_t a55 ,
                   float16_t a56 , float16_t a57 , float16_t a58 , float16_t a59 , float16_t a60 , float16_t a61 , float16_t a62 , float16_t a63);

// one component with the number written in the code ( OpCompositeExtract ) : the index is a literal , it cannot be a variable
template<typename V>
[[vk::ext_instruction(81)]]
float16_t extract(V v , [[vk::ext_literal]] int index);

// one component with an index that is computed ( OpVectorExtractDynamic ) and to change one component ( OpVectorInsertDynamic )
template<typename V>
[[vk::ext_instruction(77)]]
float16_t extractDynamic(V v , int index);

template<typename V>
[[vk::ext_instruction(78)]]
V insertDynamic(V v , float16_t component , int index);

// ---------------------------------------------------------------- the arithmetic , component by component ( the vectors have the same type )
template<typename V>
[[vk::ext_instruction(129)]]   // OpFAdd
V add(V a , V b);

template<typename V>
[[vk::ext_instruction(131)]]   // OpFSub
V sub(V a , V b);

template<typename V>
[[vk::ext_instruction(133)]]   // OpFMul
V mul(V a , V b);

template<typename V>
[[vk::ext_instruction(136)]]   // OpFDiv
V div(V a , V b);

template<typename V>
[[vk::ext_instruction(127)]]   // OpFNegate
V neg(V a);

template<typename V>
[[vk::ext_instruction(142)]]   // OpVectorTimesScalar
V timesScalar(V a , float16_t s);

template<typename V>
[[vk::ext_instruction(4 , "GLSL.std.450")]]   // FAbs
V abs(V a);

template<typename V>
[[vk::ext_instruction(37 , "GLSL.std.450")]]   // FMin
V min(V a , V b);

template<typename V>
[[vk::ext_instruction(40 , "GLSL.std.450")]]   // FMax
V max(V a , V b);

template<typename V>
[[vk::ext_instruction(43 , "GLSL.std.450")]]   // FClamp : min( max( x , lo ) , hi )
V clamp(V x , V lo , V hi);

template<typename V>
[[vk::ext_instruction(48 , "GLSL.std.450")]]   // Step : 0 where x < edge , 1 where x >= edge
V step(V edge , V x);

template<typename V>
[[vk::ext_instruction(50 , "GLSL.std.450")]]   // Fma : a * b + c
V fma(V a , V b , V c);

// ---------------------------------------------------------------- ReLU
// ReLU of a vector : max( z , 0 ) . there is no way to write "0" for a whole vector , so it is z - z
template<typename V>
V relu(V z)
{
    const V zero = coop::sub<V>(z , z);
    return coop::max<V>(z , zero);
}

// the gate of ReLU for the signal back : 1 where z was above 0 , 0 where the neuron was shut ( dz = dh * gate )
template<typename V>
V reluGate(V z)
{
    const V zero = coop::sub<V>(z , z);
    return coop::step<V>(zero , z);
}

}
