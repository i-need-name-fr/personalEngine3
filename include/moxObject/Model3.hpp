#pragma once

#include "Entity.hpp"
#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <shared_mutex>
#include <random>
#include <unordered_map>

namespace mox
{
    
    class Model3 final
    {
    public:

        SetTypicalOperators(Model3)
        SetTypicalTransferOperators(Model3)

        struct CreateInfo
        {
            std::filesystem::path modelPath{};
            std::filesystem::path armPath{}; // personal
            std::filesystem::path depthPath{};
            glm::vec3 color{0.0f};
            glm::vec3 emission{0.0f};

            float density{0.0f};
            float c{0.0f};
            float sigma{0.0f};
            float roughness{1.0f};
            float metallic{0.0f};
            bool isLight{false};
            bool skipBlend{false};
            // the nodes of the file that are not made at all ( by the NAME of the node ) : Intel Sponza has the node "exterior" , a simplified shell of the building ( for the light ) with a flat roof
            // that lies over the tiles : only the ridges of the tiles were visible through it
            std::vector<std::string> skipNodes{};

            // the test of ReSTIR SSS : EVERY mesh of the model gets its own random sigma ( the width of the Gaussian profile , meters ) and its own random M ( the first Fresnel moment ) , c = 1 - 2 M .
            // the values are in the LUT of the mesh ( lut.sigma , lut.c ) . the seed is fixed : the same picture in every run . false : c and sigma of the model are used as they are
            bool randomSSS{false};
            float sigmaMin{0.05f};
            float sigmaMax{0.5f};
            // M of the Schlick Fresnel with F0 = 0.04 is about 0.04 , so 0.02 .. 0.1 gives c = 0.8 .. 0.96
            float momentMin{0.02f};
            float momentMax{0.10f};
            uint32_t sssSeed{12345u};
        };

        explicit Model3(const CreateInfo &info) 
        {
            const auto result = _implProcessModel(info);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        // add matrix , change matrix

        MOX_NO_DISCARD ErrorDataOutput<void> addMatrix(const glm::mat4 matrix) noexcept
        {
            for(auto& m : entities)
            {
                const auto result = m->addMatrix(matrix);
                MOX_CHECK_RESULT(result)
            }

            return {};
        }

        void setMatrix(const glm::mat4 matrix , const std::uint32_t index = 0) noexcept
        {
            for(auto& m : entities)
            {
                m->setMatrix(index , matrix);
            }
        }

        // change status

        MOX_NO_DISCARD ErrorDataOutput<void> changeStatus(const bool status) noexcept
        {
            for(auto& m : entities)
            {
                const auto result = m->changeLightStatus(status);
                MOX_CHECK_RESULT(result)
            }
            return {};
        }

        ~Model3() noexcept
        {
            processDelete();
        }

    private:

        void processCopy(Model3&& input) noexcept
        {
            entities = std::move(input.entities);
        }

        void processDelete() noexcept
        {
            entities.clear();
        }

        std::vector<pEntity> entities{};
        inline static std::optional<fastgltf::Parser> parser_{};

        // the random numbers of the SSS parameters of the meshes ( one generator for the whole model : every mesh takes the next numbers )
        std::mt19937 sssRng{};

        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessModel(const CreateInfo& info)
        {
            sssRng.seed(info.sssSeed);

            if(!parser_.has_value())
            {
                parser_ = std::move(fastgltf::Parser());
            }

            const std::filesystem::path directory = info.modelPath.parent_path();
            auto data = fastgltf::GltfDataBuffer::FromPath(info.modelPath);
            if(data.error()  != fastgltf::Error::None)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to create model -> {} , Error: {}" , info.modelPath.string() , fastgltf::getErrorName(data.error()) ));
            }

            const auto options = fastgltf::Options::GenerateMeshIndices | fastgltf::Options::LoadExternalBuffers;
            auto asset_check = parser_.value().loadGltf(data.get() , directory , options);
            if(asset_check.error() != fastgltf::Error::None) 
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to create model -> {} , Error: {}" , info.modelPath.string() , fastgltf::getErrorName(asset_check.error()) ));
            }

            auto asset = std::move(asset_check.get());

            std::size_t sceneIndex = asset.defaultScene.has_value() ? asset.defaultScene.value() : 0;
            if(!asset.scenes.empty() && sceneIndex < asset.scenes.size())
            {
                ErrorDataOutput<void> status{};
                fastgltf::iterateSceneNodes(asset , sceneIndex , fastgltf::math::fmat4x4() , [&](fastgltf::Node &node , const fastgltf::math::fmat4x4 &matrix) -> void {
                    if(!status.has_value() || !node.meshIndex.has_value()) return;
                    if(std::find(info.skipNodes.begin() , info.skipNodes.end() , std::string(node.name)) != info.skipNodes.end()) return;
                    const glm::mat4 world = glm::make_mat4(matrix.data());
                    status = _implProcessMesh(asset.meshes[node.meshIndex.value()] , asset , info ,directory, world);

                });
                MOX_CHECK_RESULT(status)
            }
            else
            {
                for(const auto& mesh : asset.meshes)
                {
                    const auto result = _implProcessMesh(mesh , asset ,  info ,directory, glm::mat4(1.0f));
                    MOX_CHECK_RESULT(result)
                }
            }
            
            return {};
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessMesh(fastgltf::Mesh const &inputMesh , const fastgltf::Asset& asset , const CreateInfo& input, const std::filesystem::path& directory , const glm::mat4 world) noexcept
        {
            try
            {

                for(auto& prim : inputMesh.primitives)
                {
                    if(prim.type != fastgltf::PrimitiveType::Triangles)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process the mesh for the model , since the model topology isnt triangles"});
                    }

                    std::vector<Vertex> vertices{};
                    std::vector<uint32_t> indices{};
                    Entity::CreateInfo info{};
                    if(input.skipBlend && prim.materialIndex.has_value() && asset.materials[prim.materialIndex.value()].alphaMode == fastgltf::AlphaMode::Blend) continue;

                    info.isLightObject = input.isLight;
                    info.needsTBN = true;
                    info.metallic = input.metallic;
                    info.roughness = input.roughness;

                    // the subsurface scattering of the mesh ( the LUT ) : of the model , or a random one for every mesh ( the test of ReSTIR SSS )
                    info.density = input.density;
                    info.c = input.c;
                    info.sigma = input.sigma;
                    if(input.randomSSS)
                    {
                        const float sigma = std::uniform_real_distribution<float>(input.sigmaMin , input.sigmaMax)(sssRng);
                        const float moment = std::uniform_real_distribution<float>(input.momentMin , input.momentMax)(sssRng);
                        info.sigma = sigma;
                        // the shader makes c = 1 - 2 M itself ( LUT.c is the moment M ) : c = 1 - 2 M here was applied twice ( 1 - 2 ( 1 - 2 M ) < 0 : a negative light )
                        info.c = moment;
                    }
                    if(!prim.indicesAccessor.has_value())
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process mesh for the model , since the model doesnt have indices"});
                    }

                    {
                        const auto& acc =  asset.accessors[prim.indicesAccessor.value()];
                        indices.resize(acc.count);
                        fastgltf::iterateAccessorWithIndex<std::uint32_t>(asset , acc , [&](std::uint32_t value, std::size_t id) ->void {indices[id] = value;});
                    }


                    {
                        auto* it = prim.findAttribute("POSITION");
                        if(it == prim.attributes.end()) 
                        {
                            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to process mesh model , since the mesh doesnt have a POSITION attribute that is required");
                        }

                        const auto& acc = asset.accessors[it->accessorIndex];
                        vertices.resize(acc.count);
                        fastgltf::iterateAccessorWithIndex<glm::vec3>(asset , acc , [&](glm::vec3 pos , std::size_t id)-> void{vertices[id].position = pos;});
                    }


                    {
                        auto* it = prim.findAttribute("NORMAL");
                        if(it == prim.attributes.end()) 
                        {
                            info.needsNormal = true;
                        }
                        else
                        {
                            info.needsNormal = false;
                            const auto& acc = asset.accessors[it->accessorIndex];
                            fastgltf::iterateAccessorWithIndex<glm::vec3>(asset , acc , [&](glm::vec3 n , std::size_t id)-> void{vertices[id].normal = n;});
                        }

                    }

                    {
                        auto* it = prim.findAttribute("TEXCOORD_0");
                        if(it == prim.attributes.end())
                        {
                            const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create one of the meshes for {}, however will set to 0 , since the UV wasnt defined" , inputMesh.name)};
                            engineLogger(error);
                        }
                        else
                        {
                            const auto& acc = asset.accessors[it->accessorIndex];
                            fastgltf::iterateAccessorWithIndex<glm::vec2>(asset , acc , [&](glm::vec2 data , std::size_t idx) ->void {vertices[idx].UV = data;});
                        }
                    }

                    {
                        const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(world)));
                        for(Vertex& vertex : vertices){
                            vertex.position = glm::vec3(world * glm::vec4(vertex.position , 1.0f));
                            const glm::vec3 transformed = normalMatrix * vertex.normal;
                            vertex.normal = (glm::length(transformed) > 1e-12f) ? glm::normalize(transformed) : transformed;
                        }

                        if(glm::determinant(glm::mat3(world)) < 0.0f){
                            for(std::size_t i = 0 ; i + 2 < indices.size() ; i += 3){
                                std::swap(indices[i + 1] , indices[i + 2]);
                            }
                        }
                    }

                    _implAnalyzeTexture(info ,input , prim , asset , directory );
                    info.vertices = vertices;
                    info.indices = indices;
                    entities.push_back(std::make_shared<Entity>(info));
                }   

            }
            catch(std::exception &e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process Mesh for model , please check above"});
            }
            return {};
        }

        // process textures
        void _implAnalyzeTexture(Entity::CreateInfo &info ,const CreateInfo& input, const fastgltf::Primitive &prim , const fastgltf::Asset &asset , const std::filesystem::path &directory ) noexcept
        {   

            info.diffusePath = moxUndefinedBehavior;
            info.normalPath = moxUndefinedBehavior;
            info.depthPath = moxUndefinedBehavior;
            info.roughnessPath = moxUndefinedBehavior;
            info.metallicPath = moxUndefinedBehavior;
            info.ao = moxUndefinedBehavior;
            info.arm = moxUndefinedBehavior;

            info.density = input.density;
            info.c = input.c;
            info.sigma = input.sigma;
            info.emission = input.emission;
            info.color = input.color;

            const bool armRequired = (input.armPath == moxUndefinedBehavior ? false : std::filesystem::exists(input.armPath));
            const bool depthPath = (input.depthPath == moxUndefinedBehavior ? false : std::filesystem::exists(input.depthPath));

            if(prim.materialIndex.has_value())
            {
                namespace gl = fastgltf;
                const auto& mat = asset.materials[prim.materialIndex.value()];
                if(mat.pbrData.baseColorTexture.has_value()) // diffuse
                {
                    const gl::TextureInfo& txInfo = mat.pbrData.baseColorTexture.value();
                    const auto& tx = asset.textures[txInfo.textureIndex];
                    if(tx.imageIndex.has_value())
                    {
                        const auto& imageID = tx.imageIndex.value();
                        const auto& image = asset.images[imageID];
                        if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                        {
                            const auto path = uriSource->uri.fspath();
                            info.diffusePath = _implFullPath(directory ,path);
                        }
                    }
                }

                if(mat.normalTexture.has_value())
                {
                    const gl::TextureInfo& txInfo = mat.normalTexture.value();
                    const auto& tx = asset.textures[txInfo.textureIndex];
                    if(tx.imageIndex.has_value())
                    {
                        const auto& imageID = tx.imageIndex.value();
                        const auto& image = asset.images[imageID];
                        if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                        {
                            const auto path = uriSource->uri.fspath();
                            info.normalPath = _implFullPath(directory ,path);
                        }
                    }
                }

                if(mat.occlusionTexture.has_value())
                {
                    const gl::TextureInfo& texInfo = mat.occlusionTexture.value();
                    const auto& tex = asset.textures[texInfo.textureIndex];
                    if(tex.imageIndex.has_value())
                    {
                        const auto& image = asset.images[tex.imageIndex.value()];
                        if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                        {
                            const auto path = uriSource->uri.fspath();
                            info.ao = _implFullPath(directory , path);
                        }
                    }
                }

                // RM
                if(!armRequired)
                {
                    if(mat.pbrData.metallicRoughnessTexture.has_value())
                    {
                        const gl::TextureInfo& texInfo = mat.pbrData.metallicRoughnessTexture.value();
                        const auto& tex = asset.textures[texInfo.textureIndex];
                        if(tex.imageIndex.has_value())
                        {
                            const auto& image = asset.images[tex.imageIndex.value()];
                            if(auto* uriSource = std::get_if<gl::sources::URI>(&image.data))
                            {
                                const auto path = uriSource->uri.fspath();
                                info.roughnessPath = _implFullPath(directory , path);
                                info.metallicPath = _implFullPath(directory , path);
                            }
                        }
                    }
                }
                else
                {
                    info.arm = input.armPath;
                }

                if(depthPath)
                {
                    info.depthPath = input.depthPath;
                }

                if(mat.alphaMode == gl::AlphaMode::Mask)
                {
                    info.alphaRequired = true;
                    info.alphaCut = static_cast<float>(mat.alphaCutoff);
                }

                if(mat.pbrData.metallicFactor == 0.0f)
                {
                    info.metallicPath = moxUndefinedBehavior;
                    info.metallic = 0.0f;
                }
            }
            
        }

        const std::string _implFullPath(const std::filesystem::path directory , const std::filesystem::path file) const noexcept
        {
            return (directory /file).string();
        }

    };


    MOX_DECLARE_PTR_STRUCTURE(Model3)
}