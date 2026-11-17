#pragma once

#include "VertexHeap.hpp"
#include <meshoptimizer.h>

namespace mox
{
    class Entity final
    {
        // the heap is not a base : only its static recording functions are used ( Entity is a friend of it )
        using Heap = Source::VertexHeap;

    public:

        Entity() = delete;
        Entity(const Entity& input) = delete;
        Entity(Entity& input) = delete;
        Entity& operator=(Entity& input) = delete;
        Entity& operator=(const Entity& input) = delete;


        struct CreateInfo
        {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;

            glm::vec3 color{1.0f};
            glm::vec3 emission{0.0f};

            std::filesystem::path diffusePath{moxUndefinedBehavior};
            std::filesystem::path normalPath{moxUndefinedBehavior};
            std::filesystem::path depthPath{moxUndefinedBehavior};
            std::filesystem::path roughnessPath{moxUndefinedBehavior};
            std::filesystem::path metallicPath{moxUndefinedBehavior};

            std::filesystem::path ao;
            std::filesystem::path arm;

            float alphaCut{0.5f};
            float density{0.0f};
            float c{0.0f};
            float sigma{0.0f};
            float roughness{1.0f};
            float metallic{0.0f};
            
            bool needsTBN{false};
            bool needsNormal{false};
            bool alphaRequired{false};
            bool isLightObject{false};
            bool requiredDensity{false};
        };

        explicit Entity(CreateInfo& info)
        {
            if(Heap::getStatus() != VertexHeapStatus::Recording) 
            {
                engineLogger(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create Entity , since the Begin() hasnt started"});
                THROW_MESSAGE;
            }

            const auto result = _implProcessInputVertices(info);
            if(!result.has_value())
            {
                engineLogger(result);
                THROW_MESSAGE;
            }
        }

        MOX_NO_DISCARD ErrorDataOutput<void> changeLightStatus(const bool status) noexcept
        {
            auto ourLUT = Heap::lutsAt(LUT_index);
            ourLUT.isLightSource = status;
            return Heap::changeLUT(LUT_index , ourLUT);
        }

        void setMatrix(const uint32_t index , const glm::mat4 matrix) noexcept
        {
            if(index >= instanceIndices.size())
            {
                engineLogger(std::format("failed to change matrix for index {} , since the index refers our of the array" , index));
                return;
            }
            auto aabb = _implGetAABBCorners(localBoxMin , localBoxMax);
            glm::vec3 boxMin{BOX_MIN} , boxMax{BOX_MAX};
            for(auto& corner : aabb){
                glm::vec3 worldCorner = glm::vec3(matrix * glm::vec4(corner , 1.0f));
                boxMin.x = std::min(boxMin.x , worldCorner.x);
                boxMin.y = std::min(boxMin.y , worldCorner.y);
                boxMin.z = std::min(boxMin.z , worldCorner.z);

                boxMax.x = std::max(boxMax.x , worldCorner.x);
                boxMax.y = std::max(boxMax.y , worldCorner.y);
                boxMax.z = std::max(boxMax.z , worldCorner.z);
            }

            const glm::vec3 center = (boxMin + boxMax) * 0.5f; 
            const float radius = glm::length(boxMax - center);
            auto mesh = Heap::meshsAt(Heap::instanceLayoutAt(instanceIndices[index]).meshDataIndex);

            mesh.boxCenter = center;
            mesh.boxRadius = radius;
            mesh.boxMax = boxMax;   
            mesh.boxMin = boxMin;

            {
                const auto result = Heap::changeMesh(Heap::instanceLayoutAt(instanceIndices[index]).meshDataIndex , mesh);
                if(!result.has_value())
                {
                    engineLogger(result);
                }
            }

            const auto result = Heap::changeInstance(mesh.instanceIndex , matrix);
            if(!result.has_value())
            {
                engineLogger(result);
            }
        }

        MOX_NO_DISCARD ErrorDataOutput<void> addMatrix(const glm::mat4 matrix) noexcept
        {

            auto aabb = _implGetAABBCorners(localBoxMin , localBoxMax);
            glm::vec3 boxMin{BOX_MIN} , boxMax{BOX_MAX};
            for(auto& corner : aabb){
                glm::vec3 worldCorner = glm::vec3(matrix * glm::vec4(corner , 1.0f));
                boxMin.x = std::min(boxMin.x , worldCorner.x);
                boxMin.y = std::min(boxMin.y , worldCorner.y);
                boxMin.z = std::min(boxMin.z , worldCorner.z);

                boxMax.x = std::max(boxMax.x , worldCorner.x);
                boxMax.y = std::max(boxMax.y , worldCorner.y);
                boxMax.z = std::max(boxMax.z , worldCorner.z);
            }

            const glm::vec3 center = (boxMin + boxMax) * 0.5f; 
            const float radius = glm::length(boxMax - center);

            const auto iID = Heap::instanceLOffset();

            instanceIndices.push_back(iID);

            const auto meshID = Heap::meshOffset();
            const auto instanceID = Heap::instanceOffset();


            {
                InstanceData data{};
                data.meshDataIndex = meshID;
                data.LUT_index = LUT_index;
                data.BLAS_index = BLAS_index;
                data.needsToBeDraw = true;
                const auto result = Heap::appendInstanceLayout(data);
                MOX_CHECK_RESULT(result)
            }

            {
                MeshData data{};
                data = meshBase;
                data.instanceIndex = instanceID;
                data.boxCenter = center;
                data.boxRadius = radius;
                data.boxMax = boxMax;
                data.boxMin = boxMin;

                const auto result = Heap::appendMesh(data);
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = Heap::appendInstance(matrix);
                MOX_CHECK_RESULT(result)
            }

            return {};
        }

    private:

        // change LUT status for light
        // add and change matrix
        // process the textures

        const glm::vec3 BOX_MIN = glm::vec3(FLT_MAX);
        const glm::vec3 BOX_MAX = glm::vec3(-FLT_MAX);
        glm::vec3 localBoxMin = BOX_MIN;
        glm::vec3 localBoxMax = BOX_MAX;

        uint32_t LUT_index = 0;
        uint32_t BLAS_index = 0;
        std::vector<uint32_t> instanceIndices{};
        MeshData meshBase{};


        inline std::vector<glm::vec3> _implGetAABBCorners(glm::vec3 boxMin, glm::vec3 boxMax) {
            std::vector<glm::vec3> corners(8);

            corners[0] = glm::vec3(boxMin.x, boxMin.y, boxMin.z);
            corners[1] = glm::vec3(boxMax.x, boxMin.y, boxMin.z);
            corners[2] = glm::vec3(boxMin.x, boxMax.y, boxMin.z);
            corners[3] = glm::vec3(boxMax.x, boxMax.y, boxMin.z);
            corners[4] = glm::vec3(boxMin.x, boxMin.y, boxMax.z);
            corners[5] = glm::vec3(boxMax.x, boxMin.y, boxMax.z);
            corners[6] = glm::vec3(boxMin.x, boxMax.y, boxMax.z);
            corners[7] = glm::vec3(boxMax.x, boxMax.y, boxMax.z);

            return corners;
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessTexturePath(const std::filesystem::path &path , VkFormat format , int32_t& index ) noexcept
        {

            namespace fs = std::filesystem;

            if(!fs::exists(path)) 
            {
                engineLogger(std::format("failed to create a texture for mesh , however it wont be considered as a error , path -> {}" , path.string()));
                index = -1;
                return {};
            }

            Image::Texture2::CreateInfo info{};
            info.context = Heap::context;
            info.format = format;
            info.path = path.string();
            index = NON_VALID_SLOT;
            const auto result = Heap::txHub->addTexture(info);
            MOX_CHECK_RESULT(result)

            index = result.value();

            return {};
        }
        
        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessInputVertices(CreateInfo& input)
        {
            // nothing is written to the heap before these checks : the meshlets and the box can not be built from an empty or a broken mesh
            if(input.vertices.empty())
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , "failed to create Entity , since the input has no vertices"});
            }

            if(input.indices.size() < 3 || input.indices.size() % 3 != 0)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to create Entity , the number of indices must be a multiple of 3 and at least 3 , indices -> {}" , input.indices.size())});
            }

            for(const uint32_t index : input.indices)
            {
                if(index >= input.vertices.size())
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to create Entity , an index refers out of the vertices , index -> {} , vertices -> {}" , index , input.vertices.size())});
                }
            }

            if(input.needsNormal)
            {
                for(Vertex& vertex : input.vertices){
                    vertex.normal = glm::vec3(0.0f);
                }

                for(uint32_t i = 0 ; i + 2 < input.indices.size() ; i += 3)
                {
                    const uint32_t ia = input.indices[i];
                    const uint32_t ib = input.indices[i + 1];
                    const uint32_t ic = input.indices[i + 2];
                    if(ia >= input.vertices.size() || ib >= input.vertices.size() || ic >= input.vertices.size()) continue;

                    const glm::vec3& a = input.vertices[ia].position;
                    const glm::vec3& b = input.vertices[ib].position;
                    const glm::vec3& c = input.vertices[ic].position;

                    // counter clockwise ( glTF ) : the normal looks out of the surface . its length is twice the area of the triangle
                    const glm::vec3 areaNormal = glm::cross(b - a , c - a);

                    input.vertices[ia].normal += areaNormal;
                    input.vertices[ib].normal += areaNormal;
                    input.vertices[ic].normal += areaNormal;
                }

                for(Vertex& vertex : input.vertices){
                    const float length = glm::length(vertex.normal);
                    vertex.normal = (length > 1e-12f) ? (vertex.normal / length) : glm::vec3(0.0f , 1.0f , 0.0f);
                }
            }

            if(input.needsTBN){
                std::vector<glm::vec3> tangentSum(input.vertices.size() , glm::vec3(0.0f));
                std::vector<glm::vec3> dPdvSum(input.vertices.size() , glm::vec3(0.0f));

                for(uint32_t i = 0 ; i < input.indices.size() ; i +=3){
                    const uint32_t ia = input.indices[i];
                    const uint32_t ib = input.indices[i + 1];
                    const uint32_t ic = input.indices[i + 2];
                    if(ia >= input.vertices.size() || ib >= input.vertices.size() || ic >= input.vertices.size()) continue;

                    Vertex& a = input.vertices[ia];
                    Vertex& b = input.vertices[ib];
                    Vertex& c = input.vertices[ic];

                    const glm::vec3 edge1 = b.position - a.position;
                    const glm::vec3 edge2 = c.position - a.position;

                    const glm::vec2 uv1 = b.UV - a.UV;
                    const glm::vec2 uv2 = c.UV - a.UV;

                    const float det = uv1.x * uv2.y - uv1.y * uv2.x;
                    if(std::abs(det) < 1e-8f) continue;
                    const float f = 1.0f / det;

                    const glm::vec3 dPdu = f * (edge1 * uv2.y - edge2 * uv1.y);
                    const glm::vec3 dPdv = f * (edge2 * uv1.x - edge1 * uv2.x);

                    const float weight = glm::length(glm::cross(edge1 , edge2));
                    for(const uint32_t index : {ia , ib , ic}){
                        if(glm::length(dPdu) > 1e-12f) tangentSum[index] += glm::normalize(dPdu) * weight;
                        if(glm::length(dPdv) > 1e-12f) dPdvSum[index] += glm::normalize(dPdv) * weight;
                    }
                }

                for(uint32_t v = 0 ; v < input.vertices.size() ; v++){
                    Vertex& vertex = input.vertices[v];
                    const glm::vec3 N = glm::normalize(vertex.normal);

                    glm::vec3 T = tangentSum[v] - glm::dot(N , tangentSum[v]) * N;
                    if(glm::length(T) < 1e-6f){
                        const glm::vec3 helper = (std::abs(N.y) < 0.99f) ? glm::vec3(0.0f , 1.0f , 0.0f) : glm::vec3(1.0f , 0.0f , 0.0f);
                        T = glm::cross(helper , N);
                    }
                    T = glm::normalize(T);

                    const glm::vec3 B = glm::cross(N , T);
                    const float handedness = (glm::dot(B , dPdvSum[v]) > 0.0f) ? -1.0f : 1.0f;

                    vertex.tangent = T;
                    vertex.bitangent = B * handedness;
                }
            }

            localBoxMax = BOX_MAX;
            localBoxMin =BOX_MIN;
            for(auto& vertex : input.vertices){

                localBoxMax.x = glm::max(localBoxMax.x , vertex.position.x);
                localBoxMax.y = glm::max(localBoxMax.y , vertex.position.y);
                localBoxMax.z = glm::max(localBoxMax.z , vertex.position.z);

                localBoxMin.x = glm::min(localBoxMin.x , vertex.position.x);
                localBoxMin.y = glm::min(localBoxMin.y , vertex.position.y);
                localBoxMin.z = glm::min(localBoxMin.z , vertex.position.z);
            }

            LUT lut{};
            lut.alphaCutoff = input.alphaCut;
            lut.alphaMode = input.alphaRequired;
            lut.isLightSource = input.isLightObject ? 1 : 0;
            lut.requiredDensity = input.requiredDensity;
            lut.density = input.density;
            lut.c = input.c;
            lut.sigma = input.sigma;
            lut.emission = input.emission;
            lut.color =input.color;
            lut.roughness = input.roughness;
            lut.metallic = input.metallic;

            meshBase.firstIndex = Heap::indicesOffset();
            meshBase.indexCount = input.indices.size();
            meshBase.firstVertex = Heap::vertexOffset();
            meshBase.geometryIndex = Heap::geometryOffset();

            // creating the meshlet
            {
                const auto bound = meshopt_buildMeshletsBound(input.indices.size() , ms::MAX_BOUND_VERTEX_COUNT , ms::MAX_BOUND_TRIANGLE_COUNT);
                ms::GeometryData geometry{};
                geometry.firstMeshletVertex = Heap::mVertexOffset();
                geometry.meshletOffset = Heap::meshletOffset();
                geometry.firstMeshletIndex = Heap::mIndicesOffset();

                std::vector<meshopt_Meshlet> meshlets(bound);
                std::vector<uint32_t> mVertices(bound * ms::MAX_BOUND_VERTEX_COUNT);
                std::vector<uint8_t> mIndices(bound * ms::MAX_BOUND_TRIANGLE_COUNT * 3);
                const auto realSize = meshopt_buildMeshlets(meshlets.data() , mVertices.data() , mIndices.data()
                                    , input.indices.data() , input.indices.size() , 
                                    &input.vertices[0].position.x , input.vertices.size() , sizeof(Vertex) ,
                                    ms::MAX_BOUND_VERTEX_COUNT , ms::MAX_BOUND_TRIANGLE_COUNT , 0.5f) ;
                if(realSize == 0)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , "failed to create Entity , meshopt built no meshlets"});
                }
                meshlets.resize(realSize);
                // the real number of the meshlets is known only after meshopt_buildMeshlets
                geometry.meshletCount = realSize;

                const auto last = meshlets[realSize - 1];
                mVertices.resize(last.vertex_count + last.vertex_offset);
                mIndices.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
                std::vector<ms::Meshlet> m(realSize);
                for(uint32_t i = 0; i < realSize ; i++)
                {
                    m[i].firstIndex = meshlets[i].triangle_offset + geometry.firstMeshletIndex;
                    m[i].firstVertex = meshlets[i].vertex_offset + geometry.firstMeshletVertex;
                    m[i].vertexCount = meshlets[i].vertex_count;
                    m[i].indexCount = meshlets[i].triangle_count;
                }

                std::vector<ms::MeshletBound> mBound(realSize);
                for(uint32_t i = 0 ; i < realSize ; i++)
                {
                    auto& mesh = meshlets[i];

                    const auto b = meshopt_computeMeshletBounds(&mVertices[mesh.vertex_offset] , &mIndices[mesh.triangle_offset] , mesh.triangle_count , &input.vertices[0].position.x ,input.vertices.size() , sizeof(Vertex) );
                    ms::MeshletBound mB{};
                    mB.sphere = glm::vec4(b.center[0] , b.center[1] , b.center[2] , b.radius);
                    mB.cone = glm::vec4(b.cone_axis[0] , b.cone_axis[1] , b.cone_axis[2] , b.cone_cutoff);
                    const auto result = Heap::appendBounds({mB});
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = Heap::appendGeometry(geometry);
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = Heap::appendMeshlets(m);
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = Heap::appendMeshletVertices(mVertices);
                    MOX_CHECK_RESULT(result);
                }

                {
                    std::vector<uint32_t> indices(mIndices.size());
                    for(uint32_t i = 0 ; i < indices.size() ; i++)
                    {   
                        indices[i] = mIndices[i];
                    }

                    const auto result = Heap::appendMeshletIndices(indices);
                    MOX_CHECK_RESULT(result);
                }
            
            }

            LUT_index = Heap::lutsOffset();
            BLAS_index = Heap::asLayoutsOffset();

            {
                BLAS2_layout data{};
                data.firstVertex = meshBase.firstVertex;
                data.vertexCount = input.vertices.size();
                data.indexCount = input.indices.size();
                data.indexOffset = meshBase.firstIndex;
                data.mask = (input.isLightObject ? moxInstanceMaskLight : moxInstanceMaskNonLight);

                const auto result  = Heap::appendASLayout(data);
                MOX_CHECK_RESULT(result)
            }

            // diffuse
            {
                const auto result = _implProcessTexturePath(input.diffusePath , VK_FORMAT_R8G8B8A8_UNORM ,lut.diffuseTextureID);
                MOX_CHECK_RESULT(result)
            }

            // normal
            {
                const auto result = _implProcessTexturePath(input.normalPath , VK_FORMAT_R8G8_UNORM ,lut.normalTextureID);
                MOX_CHECK_RESULT(result)
            }

            // depth
            {
                const auto result = _implProcessTexturePath(input.depthPath , VK_FORMAT_R32_SFLOAT ,lut.depthTextureID);
                MOX_CHECK_RESULT(result)
            }

            // roughness
            {
                const auto result = _implProcessTexturePath(input.roughnessPath , VK_FORMAT_R32_SFLOAT ,lut.roughnessTextureID);
                MOX_CHECK_RESULT(result)
            }

            // metallic
            {
                const auto result = _implProcessTexturePath(input.metallicPath , VK_FORMAT_R32_SFLOAT ,lut.metallicTextureID);
                MOX_CHECK_RESULT(result)
            }

            // ARM
            {
                const auto result = _implProcessTexturePath(input.arm , VK_FORMAT_R8G8B8A8_UNORM ,lut.armTextureID);
                MOX_CHECK_RESULT(result)
            }

            // ao
            {
                const auto result = _implProcessTexturePath(input.ao , VK_FORMAT_R32_SFLOAT ,lut.aoTextureID);
                MOX_CHECK_RESULT(result)
            }
        
            {
                const auto result = Heap::appendLUT(lut);
                MOX_CHECK_RESULT(result)
            }            


            {
                const auto result = Heap::appendVertices(input.vertices);
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = Heap::appendIndices(input.indices);
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = addMatrix(glm::mat4(1.0f));
                MOX_CHECK_RESULT(result)
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(Entity)
}