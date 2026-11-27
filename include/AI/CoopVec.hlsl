#pragma once
// #include "../../include/AI/CoopVec.hlsl"
#define MOX_COOP_VECTOR

template<int N>
using Coop = vk::SpirvType<5288 , N * 2 , 2 , float16_t , vk::integral_constant<int , N> >;
typedef Coop<16> Coop16;
typedef Coop<32> Coop32;
typedef Coop<64> Coop64;
typedef Coop<272> Coop272;   // the input of NKPD : 16 neighbours x 16 features + 16 numbers of the centre ( the context )
typedef Coop<128> Coop128;
typedef Coop<256> Coop256;   // the input of NKPD : 16 neighbours x 16 features
typedef Coop<176> Coop176;   // the input of NKPD : 16 neighbours x 11 features
typedef Coop<192> Coop192;   // the input of NKPD : 16 neighbours x 12 features

static const uint ATOMIC_SCOPE_DEVICE = 1u;          // the memory scope : the whole device ( 2 : the workgroup , 3 : the subgroup )
static const uint ATOMIC_SCOPE_WORKGROUP = 2u;
static const uint ATOMIC_SEMANTICS_RELAXED = 0u;     // no ordering of the memory : an add does not need it

[[vk::ext_capability(6033)]]
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_instruction(6035)]]
float atomicAddFloat([[vk::ext_reference]] float pointer , uint scope , uint semantics , float value);

namespace coop
{

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

[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop128 construct128(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 , float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15 , float16_t a16 , float16_t a17 , float16_t a18 , float16_t a19 , float16_t a20 , float16_t a21 , float16_t a22 , float16_t a23 , float16_t a24 , float16_t a25 , float16_t a26 , float16_t a27 , float16_t a28 , float16_t a29 , float16_t a30 , float16_t a31 , float16_t a32 , float16_t a33 , float16_t a34 , float16_t a35 , float16_t a36 , float16_t a37 , float16_t a38 , float16_t a39 , float16_t a40 , float16_t a41 , float16_t a42 , float16_t a43 , float16_t a44 , float16_t a45 , float16_t a46 , float16_t a47 , float16_t a48 , float16_t a49 , float16_t a50 , float16_t a51 , float16_t a52 , float16_t a53 , float16_t a54 , float16_t a55 , float16_t a56 , float16_t a57 , float16_t a58 , float16_t a59 , float16_t a60 , float16_t a61 , float16_t a62 , float16_t a63 , float16_t a64 , float16_t a65 , float16_t a66 , float16_t a67 , float16_t a68 , float16_t a69 , float16_t a70 , float16_t a71 , float16_t a72 , float16_t a73 , float16_t a74 , float16_t a75 , float16_t a76 , float16_t a77 , float16_t a78 , float16_t a79 , float16_t a80 , float16_t a81 , float16_t a82 , float16_t a83 , float16_t a84 , float16_t a85 , float16_t a86 , float16_t a87 , float16_t a88 , float16_t a89 , float16_t a90 , float16_t a91 , float16_t a92 , float16_t a93 , float16_t a94 , float16_t a95 , float16_t a96 , float16_t a97 , float16_t a98 , float16_t a99 , float16_t a100 , float16_t a101 , float16_t a102 , float16_t a103 , float16_t a104 , float16_t a105 , float16_t a106 , float16_t a107 , float16_t a108 , float16_t a109 , float16_t a110 , float16_t a111 , float16_t a112 , float16_t a113 , float16_t a114 , float16_t a115 , float16_t a116 , float16_t a117 , float16_t a118 , float16_t a119 , float16_t a120 , float16_t a121 , float16_t a122 , float16_t a123 , float16_t a124 , float16_t a125 , float16_t a126 , float16_t a127);

[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop272 construct272(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 , float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15 , float16_t a16 , float16_t a17 , float16_t a18 , float16_t a19 , float16_t a20 , float16_t a21 , float16_t a22 , float16_t a23 , float16_t a24 , float16_t a25 , float16_t a26 , float16_t a27 , float16_t a28 , float16_t a29 , float16_t a30 , float16_t a31 , float16_t a32 , float16_t a33 , float16_t a34 , float16_t a35 , float16_t a36 , float16_t a37 , float16_t a38 , float16_t a39 , float16_t a40 , float16_t a41 , float16_t a42 , float16_t a43 , float16_t a44 , float16_t a45 , float16_t a46 , float16_t a47 , float16_t a48 , float16_t a49 , float16_t a50 , float16_t a51 , float16_t a52 , float16_t a53 , float16_t a54 , float16_t a55 , float16_t a56 , float16_t a57 , float16_t a58 , float16_t a59 , float16_t a60 , float16_t a61 , float16_t a62 , float16_t a63 , float16_t a64 , float16_t a65 , float16_t a66 , float16_t a67 , float16_t a68 , float16_t a69 , float16_t a70 , float16_t a71 , float16_t a72 , float16_t a73 , float16_t a74 , float16_t a75 , float16_t a76 , float16_t a77 , float16_t a78 , float16_t a79 , float16_t a80 , float16_t a81 , float16_t a82 , float16_t a83 , float16_t a84 , float16_t a85 , float16_t a86 , float16_t a87 , float16_t a88 , float16_t a89 , float16_t a90 , float16_t a91 , float16_t a92 , float16_t a93 , float16_t a94 , float16_t a95 , float16_t a96 , float16_t a97 , float16_t a98 , float16_t a99 , float16_t a100 , float16_t a101 , float16_t a102 , float16_t a103 , float16_t a104 , float16_t a105 , float16_t a106 , float16_t a107 , float16_t a108 , float16_t a109 , float16_t a110 , float16_t a111 , float16_t a112 , float16_t a113 , float16_t a114 , float16_t a115 , float16_t a116 , float16_t a117 , float16_t a118 , float16_t a119 , float16_t a120 , float16_t a121 , float16_t a122 , float16_t a123 , float16_t a124 , float16_t a125 , float16_t a126 , float16_t a127 , float16_t a128 , float16_t a129 , float16_t a130 , float16_t a131 , float16_t a132 , float16_t a133 , float16_t a134 , float16_t a135 , float16_t a136 , float16_t a137 , float16_t a138 , float16_t a139 , float16_t a140 , float16_t a141 , float16_t a142 , float16_t a143 , float16_t a144 , float16_t a145 , float16_t a146 , float16_t a147 , float16_t a148 , float16_t a149 , float16_t a150 , float16_t a151 , float16_t a152 , float16_t a153 , float16_t a154 , float16_t a155 , float16_t a156 , float16_t a157 , float16_t a158 , float16_t a159 , float16_t a160 , float16_t a161 , float16_t a162 , float16_t a163 , float16_t a164 , float16_t a165 , float16_t a166 , float16_t a167 , float16_t a168 , float16_t a169 , float16_t a170 , float16_t a171 , float16_t a172 , float16_t a173 , float16_t a174 , float16_t a175 , float16_t a176 , float16_t a177 , float16_t a178 , float16_t a179 , float16_t a180 , float16_t a181 , float16_t a182 , float16_t a183 , float16_t a184 , float16_t a185 , float16_t a186 , float16_t a187 , float16_t a188 , float16_t a189 , float16_t a190 , float16_t a191 , float16_t a192 , float16_t a193 , float16_t a194 , float16_t a195 , float16_t a196 , float16_t a197 , float16_t a198 , float16_t a199 , float16_t a200 , float16_t a201 , float16_t a202 , float16_t a203 , float16_t a204 , float16_t a205 , float16_t a206 , float16_t a207 , float16_t a208 , float16_t a209 , float16_t a210 , float16_t a211 , float16_t a212 , float16_t a213 , float16_t a214 , float16_t a215 , float16_t a216 , float16_t a217 , float16_t a218 , float16_t a219 , float16_t a220 , float16_t a221 , float16_t a222 , float16_t a223 , float16_t a224 , float16_t a225 , float16_t a226 , float16_t a227 , float16_t a228 , float16_t a229 , float16_t a230 , float16_t a231 , float16_t a232 , float16_t a233 , float16_t a234 , float16_t a235 , float16_t a236 , float16_t a237 , float16_t a238 , float16_t a239 , float16_t a240 , float16_t a241 , float16_t a242 , float16_t a243 , float16_t a244 , float16_t a245 , float16_t a246 , float16_t a247 , float16_t a248 , float16_t a249 , float16_t a250 , float16_t a251 , float16_t a252 , float16_t a253 , float16_t a254 , float16_t a255 , float16_t a256 , float16_t a257 , float16_t a258 , float16_t a259 , float16_t a260 , float16_t a261 , float16_t a262 , float16_t a263 , float16_t a264 , float16_t a265 , float16_t a266 , float16_t a267 , float16_t a268 , float16_t a269 , float16_t a270 , float16_t a271);

[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop256 construct256(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 , float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15 ,
                    float16_t a16 , float16_t a17 , float16_t a18 , float16_t a19 , float16_t a20 , float16_t a21 , float16_t a22 , float16_t a23 , float16_t a24 , float16_t a25 , float16_t a26 , float16_t a27 , float16_t a28 , float16_t a29 , float16_t a30 , float16_t a31 ,
                    float16_t a32 , float16_t a33 , float16_t a34 , float16_t a35 , float16_t a36 , float16_t a37 , float16_t a38 , float16_t a39 , float16_t a40 , float16_t a41 , float16_t a42 , float16_t a43 , float16_t a44 , float16_t a45 , float16_t a46 , float16_t a47 ,
                    float16_t a48 , float16_t a49 , float16_t a50 , float16_t a51 , float16_t a52 , float16_t a53 , float16_t a54 , float16_t a55 , float16_t a56 , float16_t a57 , float16_t a58 , float16_t a59 , float16_t a60 , float16_t a61 , float16_t a62 , float16_t a63 ,
                    float16_t a64 , float16_t a65 , float16_t a66 , float16_t a67 , float16_t a68 , float16_t a69 , float16_t a70 , float16_t a71 , float16_t a72 , float16_t a73 , float16_t a74 , float16_t a75 , float16_t a76 , float16_t a77 , float16_t a78 , float16_t a79 ,
                    float16_t a80 , float16_t a81 , float16_t a82 , float16_t a83 , float16_t a84 , float16_t a85 , float16_t a86 , float16_t a87 , float16_t a88 , float16_t a89 , float16_t a90 , float16_t a91 , float16_t a92 , float16_t a93 , float16_t a94 , float16_t a95 ,
                    float16_t a96 , float16_t a97 , float16_t a98 , float16_t a99 , float16_t a100 , float16_t a101 , float16_t a102 , float16_t a103 , float16_t a104 , float16_t a105 , float16_t a106 , float16_t a107 , float16_t a108 , float16_t a109 , float16_t a110 , float16_t a111 ,
                    float16_t a112 , float16_t a113 , float16_t a114 , float16_t a115 , float16_t a116 , float16_t a117 , float16_t a118 , float16_t a119 , float16_t a120 , float16_t a121 , float16_t a122 , float16_t a123 , float16_t a124 , float16_t a125 , float16_t a126 , float16_t a127 ,
                    float16_t a128 , float16_t a129 , float16_t a130 , float16_t a131 , float16_t a132 , float16_t a133 , float16_t a134 , float16_t a135 , float16_t a136 , float16_t a137 , float16_t a138 , float16_t a139 , float16_t a140 , float16_t a141 , float16_t a142 , float16_t a143 ,
                    float16_t a144 , float16_t a145 , float16_t a146 , float16_t a147 , float16_t a148 , float16_t a149 , float16_t a150 , float16_t a151 , float16_t a152 , float16_t a153 , float16_t a154 , float16_t a155 , float16_t a156 , float16_t a157 , float16_t a158 , float16_t a159 ,
                    float16_t a160 , float16_t a161 , float16_t a162 , float16_t a163 , float16_t a164 , float16_t a165 , float16_t a166 , float16_t a167 , float16_t a168 , float16_t a169 , float16_t a170 , float16_t a171 , float16_t a172 , float16_t a173 , float16_t a174 , float16_t a175 ,
                    float16_t a176 , float16_t a177 , float16_t a178 , float16_t a179 , float16_t a180 , float16_t a181 , float16_t a182 , float16_t a183 , float16_t a184 , float16_t a185 , float16_t a186 , float16_t a187 , float16_t a188 , float16_t a189 , float16_t a190 , float16_t a191 ,
                    float16_t a192 , float16_t a193 , float16_t a194 , float16_t a195 , float16_t a196 , float16_t a197 , float16_t a198 , float16_t a199 , float16_t a200 , float16_t a201 , float16_t a202 , float16_t a203 , float16_t a204 , float16_t a205 , float16_t a206 , float16_t a207 ,
                    float16_t a208 , float16_t a209 , float16_t a210 , float16_t a211 , float16_t a212 , float16_t a213 , float16_t a214 , float16_t a215 , float16_t a216 , float16_t a217 , float16_t a218 , float16_t a219 , float16_t a220 , float16_t a221 , float16_t a222 , float16_t a223 ,
                    float16_t a224 , float16_t a225 , float16_t a226 , float16_t a227 , float16_t a228 , float16_t a229 , float16_t a230 , float16_t a231 , float16_t a232 , float16_t a233 , float16_t a234 , float16_t a235 , float16_t a236 , float16_t a237 , float16_t a238 , float16_t a239 ,
                    float16_t a240 , float16_t a241 , float16_t a242 , float16_t a243 , float16_t a244 , float16_t a245 , float16_t a246 , float16_t a247 , float16_t a248 , float16_t a249 , float16_t a250 , float16_t a251 , float16_t a252 , float16_t a253 , float16_t a254 , float16_t a255);

[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop176 construct176(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 , float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15 ,
                    float16_t a16 , float16_t a17 , float16_t a18 , float16_t a19 , float16_t a20 , float16_t a21 , float16_t a22 , float16_t a23 , float16_t a24 , float16_t a25 , float16_t a26 , float16_t a27 , float16_t a28 , float16_t a29 , float16_t a30 , float16_t a31 ,
                    float16_t a32 , float16_t a33 , float16_t a34 , float16_t a35 , float16_t a36 , float16_t a37 , float16_t a38 , float16_t a39 , float16_t a40 , float16_t a41 , float16_t a42 , float16_t a43 , float16_t a44 , float16_t a45 , float16_t a46 , float16_t a47 ,
                    float16_t a48 , float16_t a49 , float16_t a50 , float16_t a51 , float16_t a52 , float16_t a53 , float16_t a54 , float16_t a55 , float16_t a56 , float16_t a57 , float16_t a58 , float16_t a59 , float16_t a60 , float16_t a61 , float16_t a62 , float16_t a63 ,
                    float16_t a64 , float16_t a65 , float16_t a66 , float16_t a67 , float16_t a68 , float16_t a69 , float16_t a70 , float16_t a71 , float16_t a72 , float16_t a73 , float16_t a74 , float16_t a75 , float16_t a76 , float16_t a77 , float16_t a78 , float16_t a79 ,
                    float16_t a80 , float16_t a81 , float16_t a82 , float16_t a83 , float16_t a84 , float16_t a85 , float16_t a86 , float16_t a87 , float16_t a88 , float16_t a89 , float16_t a90 , float16_t a91 , float16_t a92 , float16_t a93 , float16_t a94 , float16_t a95 ,
                    float16_t a96 , float16_t a97 , float16_t a98 , float16_t a99 , float16_t a100 , float16_t a101 , float16_t a102 , float16_t a103 , float16_t a104 , float16_t a105 , float16_t a106 , float16_t a107 , float16_t a108 , float16_t a109 , float16_t a110 , float16_t a111 ,
                    float16_t a112 , float16_t a113 , float16_t a114 , float16_t a115 , float16_t a116 , float16_t a117 , float16_t a118 , float16_t a119 , float16_t a120 , float16_t a121 , float16_t a122 , float16_t a123 , float16_t a124 , float16_t a125 , float16_t a126 , float16_t a127 ,
                    float16_t a128 , float16_t a129 , float16_t a130 , float16_t a131 , float16_t a132 , float16_t a133 , float16_t a134 , float16_t a135 , float16_t a136 , float16_t a137 , float16_t a138 , float16_t a139 , float16_t a140 , float16_t a141 , float16_t a142 , float16_t a143 ,
                    float16_t a144 , float16_t a145 , float16_t a146 , float16_t a147 , float16_t a148 , float16_t a149 , float16_t a150 , float16_t a151 , float16_t a152 , float16_t a153 , float16_t a154 , float16_t a155 , float16_t a156 , float16_t a157 , float16_t a158 , float16_t a159 ,
                    float16_t a160 , float16_t a161 , float16_t a162 , float16_t a163 , float16_t a164 , float16_t a165 , float16_t a166 , float16_t a167 , float16_t a168 , float16_t a169 , float16_t a170 , float16_t a171 , float16_t a172 , float16_t a173 , float16_t a174 , float16_t a175);

[[vk::ext_capability(5394)]]
[[vk::ext_extension("SPV_NV_cooperative_vector")]]
[[vk::ext_instruction(80)]]
Coop192 construct192(float16_t a0 , float16_t a1 , float16_t a2 , float16_t a3 , float16_t a4 , float16_t a5 , float16_t a6 , float16_t a7 , float16_t a8 , float16_t a9 , float16_t a10 , float16_t a11 , float16_t a12 , float16_t a13 , float16_t a14 , float16_t a15 ,
                    float16_t a16 , float16_t a17 , float16_t a18 , float16_t a19 , float16_t a20 , float16_t a21 , float16_t a22 , float16_t a23 , float16_t a24 , float16_t a25 , float16_t a26 , float16_t a27 , float16_t a28 , float16_t a29 , float16_t a30 , float16_t a31 ,
                    float16_t a32 , float16_t a33 , float16_t a34 , float16_t a35 , float16_t a36 , float16_t a37 , float16_t a38 , float16_t a39 , float16_t a40 , float16_t a41 , float16_t a42 , float16_t a43 , float16_t a44 , float16_t a45 , float16_t a46 , float16_t a47 ,
                    float16_t a48 , float16_t a49 , float16_t a50 , float16_t a51 , float16_t a52 , float16_t a53 , float16_t a54 , float16_t a55 , float16_t a56 , float16_t a57 , float16_t a58 , float16_t a59 , float16_t a60 , float16_t a61 , float16_t a62 , float16_t a63 ,
                    float16_t a64 , float16_t a65 , float16_t a66 , float16_t a67 , float16_t a68 , float16_t a69 , float16_t a70 , float16_t a71 , float16_t a72 , float16_t a73 , float16_t a74 , float16_t a75 , float16_t a76 , float16_t a77 , float16_t a78 , float16_t a79 ,
                    float16_t a80 , float16_t a81 , float16_t a82 , float16_t a83 , float16_t a84 , float16_t a85 , float16_t a86 , float16_t a87 , float16_t a88 , float16_t a89 , float16_t a90 , float16_t a91 , float16_t a92 , float16_t a93 , float16_t a94 , float16_t a95 ,
                    float16_t a96 , float16_t a97 , float16_t a98 , float16_t a99 , float16_t a100 , float16_t a101 , float16_t a102 , float16_t a103 , float16_t a104 , float16_t a105 , float16_t a106 , float16_t a107 , float16_t a108 , float16_t a109 , float16_t a110 , float16_t a111 ,
                    float16_t a112 , float16_t a113 , float16_t a114 , float16_t a115 , float16_t a116 , float16_t a117 , float16_t a118 , float16_t a119 , float16_t a120 , float16_t a121 , float16_t a122 , float16_t a123 , float16_t a124 , float16_t a125 , float16_t a126 , float16_t a127 ,
                    float16_t a128 , float16_t a129 , float16_t a130 , float16_t a131 , float16_t a132 , float16_t a133 , float16_t a134 , float16_t a135 , float16_t a136 , float16_t a137 , float16_t a138 , float16_t a139 , float16_t a140 , float16_t a141 , float16_t a142 , float16_t a143 ,
                    float16_t a144 , float16_t a145 , float16_t a146 , float16_t a147 , float16_t a148 , float16_t a149 , float16_t a150 , float16_t a151 , float16_t a152 , float16_t a153 , float16_t a154 , float16_t a155 , float16_t a156 , float16_t a157 , float16_t a158 , float16_t a159 ,
                    float16_t a160 , float16_t a161 , float16_t a162 , float16_t a163 , float16_t a164 , float16_t a165 , float16_t a166 , float16_t a167 , float16_t a168 , float16_t a169 , float16_t a170 , float16_t a171 , float16_t a172 , float16_t a173 , float16_t a174 , float16_t a175 ,
                    float16_t a176 , float16_t a177 , float16_t a178 , float16_t a179 , float16_t a180 , float16_t a181 , float16_t a182 , float16_t a183 , float16_t a184 , float16_t a185 , float16_t a186 , float16_t a187 , float16_t a188 , float16_t a189 , float16_t a190 , float16_t a191);

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

static const float16_t LEAKY_SLOPE = float16_t(0.01f);

// max( z , 0.01 * z ) : z for z > 0 , 0.01 * z for z < 0
template<typename V>
V leakyRelu(V z)
{
    return coop::max<V>(z , coop::timesScalar<V>(z , LEAKY_SLOPE));
}

template<typename V>
V leakyReluGate(V z)
{
    const V zero = coop::sub<V>(z , z);
    const V one = coop::step<V>(zero , zero);
    const V open = coop::step<V>(zero , z);
    return coop::max<V>(open , coop::timesScalar<V>(one , LEAKY_SLOPE));
}

// ---------------------------------------------------------------- the guard of fp16
// the same value in all the components ( a constant vector that does not depend on the data : z - z is NaN for an inf )
Coop16 splat16(const float16_t v)
{
    return construct16(v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v);
}

Coop64 splat64(const float16_t v)
{
    return construct64(v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v);
}

Coop256 splat256(const float16_t v)
{
    return construct256(v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v);
}

Coop128 splat128(const float16_t v)
{
    return construct128(v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v);
}

Coop192 splat192(const float16_t v)
{
    return construct192(v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v , v);
}

static const float16_t FP16_SAFE_LIMIT = float16_t(30000.0f);

Coop64 clampFinite(Coop64 z)
{
    return clamp<Coop64>(z , splat64(-FP16_SAFE_LIMIT) , splat64(FP16_SAFE_LIMIT));
}

Coop256 clampFinite(Coop256 z)
{
    return clamp<Coop256>(z , splat256(-FP16_SAFE_LIMIT) , splat256(FP16_SAFE_LIMIT));
}

Coop128 clampFinite(Coop128 z)
{
    return clamp<Coop128>(z , splat128(-FP16_SAFE_LIMIT) , splat128(FP16_SAFE_LIMIT));
}

Coop192 clampFinite(Coop192 z)
{
    return clamp<Coop192>(z , splat192(-FP16_SAFE_LIMIT) , splat192(FP16_SAFE_LIMIT));
}

Coop16 clampFinite(Coop16 z)
{
    return clamp<Coop16>(z , splat16(-FP16_SAFE_LIMIT) , splat16(FP16_SAFE_LIMIT));
}

}
