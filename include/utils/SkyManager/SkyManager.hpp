#pragma once

#include "../../moxSourceCluster/AtmoScat.hpp"
#include "../../moxSourceCluster/SceneBuffers.hpp"
#include "../../ShaderPipe2/ShaderPipe2.hpp"

namespace mox
{
namespace utils
{
    class SkyManager final
    {
    public:

        struct CreateInfo
        {
            moxContext2* _Nonnull context;
            Source::SceneBuffers* _Nonnull sceneBuffers;
            std::filesystem::path IBL;
            std::filesystem::path MS;
            std::filesystem::path transition;
            std::filesystem::path transmission;
            std::filesystem::path skyview;
        };

        SetTypicalOperators(SkyManager)
        SetTypicalTransferOperators(SkyManager)

        explicit SkyManager(const CreateInfo &info)
        {
            const auto result = _implInit(info);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        Source::pAtmoScat source{nullptr};

        void update(VkCommandBuffer& cmd) noexcept __attribute__((nonnull))
        {
            _implUpdateSource(cmd);
        }

        ~SkyManager() noexcept
        {
            processDelete();
        }

    private:

        
        pShaderPipe2 __IBL{}; // multi execution , requires Update()
        pShaderPipe2 __MS{}; // single execution
        pShaderPipe2 __transmission{}; // single execution
        pShaderPipe2 __transition{}; // multi execution , requires Update()
        pShaderPipe2 __skyView{}; // multi execution , requires Update()

        std::array<DescriptorBuffer2* , 2> __dbs{};

        void processCopy(SkyManager&& input) noexcept
        {
            source = std::move(input.source);
            __IBL = std::move(input.__IBL);
            __MS = std::move(input.__MS);
            __transmission = std::move(input.__transmission);
            __transition = std::move(input.__transition);
            __skyView = std::move(input.__skyView);

            // raw pointers , std::move does not clear them , so we do it by hand
            __dbs = std::exchange(input.__dbs , {});
        }

        void processDelete() noexcept
        {
            source = nullptr;
            __IBL = nullptr;
            __dbs = {};
            __MS = nullptr;
            __skyView = nullptr;
            __transmission = nullptr;
            __transition = nullptr;

        }


        void _implUpdateSource(VkCommandBuffer& cmd) noexcept __attribute__((nonnull)) 
        {
            std::array<VkDescriptorBufferBindingInfoEXT , 2> infos{};
            infos[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
            infos[0].pNext = nullptr;
            infos[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
            infos[0].address = __dbs[0]->pBuffer;

            infos[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
            infos[1].pNext = nullptr;
            infos[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
            infos[1].address = __dbs[1]->pBuffer;

            cmdBindDescriptorBuffersEXT(cmd , infos.size() , infos.data());
            
            const uint32_t sets[2] = {0 , 1};
            const uint64_t offsets[2]= { 0 , 0};
            cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , __skyView->layout->layout , 0  , 2 , sets , offsets);

            VkShaderEXT shaders[1] = {nullptr};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            // SkyView first
            {
                shaders[0] = __skyView->shaders[0]->shader;
                cmdBindShadersEXT(cmd , 1 , stage , shaders);

                // in Skyview we are reading both multi scatter and LUT , however we dont need to set barriers for them , since in ExecuteSingle 
                // both of them have the same READ for ALL_COMMANDS
                // so here we only have to translate the skyview image itself
                auto barrier = source->skyViewLUT->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                const auto dep = TypicalDependency<1>(&barrier);
                vkCmdPipelineBarrier2(cmd , &dep);

                const std::uint32_t x = (source->skyViewLUT->extent.width + 7) / 8;
                const std::uint32_t y = (source->skyViewLUT->extent.height + 7) / 8;

                vkCmdDispatch(cmd , x,  y, 1);
            }

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            // transition to cube
            {
                // now we are reading the sky view ALL COMMANDS
                // and writing into the skybox
                shaders[0] = __transition->shaders[0]->shader;
                cmdBindShadersEXT(cmd , 1 , stage , shaders);

                std::array<VkImageMemoryBarrier2 , 2> barriers{};
                barriers[0] = source->skyViewLUT->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                barriers[1] = source->skybox->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT ,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT );
                const auto dep = TypicalDependency<2>(barriers.data());
                vkCmdPipelineBarrier2(cmd , &dep);

                const std::uint32_t x = (source->skybox->extent.width + 7) / 8;
                const std::uint32_t y = (source->skybox->extent.height + 7) / 8;

                vkCmdDispatch(cmd , x,  y, 6);   
            }

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            // IBL
            {
                shaders[0] = __IBL->shaders[0]->shader;
                cmdBindShadersEXT(cmd , 1 , stage , shaders);

                std::array<VkImageMemoryBarrier2 , 2> barriers{};
                barriers[0] = source->skybox->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                barriers[1] = source->IBL->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT ,VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT );
                const auto dep = TypicalDependency<2>(barriers.data());
                vkCmdPipelineBarrier2(cmd , &dep);

                const std::uint32_t x = (source->IBL->extent.width + 7) / 8;
                const std::uint32_t y = (source->IBL->extent.height + 7) / 8;

                vkCmdDispatch(cmd , x,  y, 6);   
            }

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            // last 
            { 
                auto barrier = source->IBL->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                const auto dep = TypicalDependency<1>(&barrier);
                vkCmdPipelineBarrier2(cmd , &dep);
            }
        }

        // this function will go as a lambda for Queue Manager
        void _implExecuteSingle(VkCommandBuffer& cmd) noexcept __attribute__((nonnull))
        {
            // since they all use the same attachments , we can bind it only once :3
            std::array<VkDescriptorBufferBindingInfoEXT , 2> infos{};
            infos[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
            infos[0].pNext = nullptr;
            infos[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
            infos[0].address = __dbs[0]->pBuffer;

            infos[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
            infos[1].pNext = nullptr;
            infos[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
            infos[1].address = __dbs[1]->pBuffer;

            cmdBindDescriptorBuffersEXT(cmd , infos.size() , infos.data());
            
            const uint32_t sets[2] = {0 , 1};
            const uint64_t offsets[2]= { 0 , 0};
            cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , __transmission->layout->layout , 0  , 2 , sets , offsets);

            VkShaderEXT shaders[1] = {__transmission->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
            
            { // transmission first
                auto barrier = source->transmissionLUT->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                const auto dep = TypicalDependency<1>(&barrier);
                vkCmdPipelineBarrier2(cmd , &dep);


                cmdBindShadersEXT(cmd , 1 , stage , shaders);

                const std::uint32_t x = (source->transmissionLUT->extent.width + 7) / 8;
                const std::uint32_t y = (source->transmissionLUT->extent.height + 7) / 8;

                vkCmdDispatch(cmd , x,  y, 1);
                
            }

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            // now MS
            {
                shaders[0] = __MS->shaders[0]->shader;
                std::array<VkImageMemoryBarrier2 ,2> barriers{};
                barriers[0] = source->transmissionLUT->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                barriers[1] = source->multiscatterLUT->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

                const auto dep = TypicalDependency<2>(barriers.data());
                vkCmdPipelineBarrier2(cmd , &dep);

                cmdBindShadersEXT(cmd , 1 , stage , shaders);

                const std::uint32_t x = (source->multiscatterLUT->extent.width + 7) / 8;
                const std::uint32_t y = (source->multiscatterLUT->extent.height + 7) / 8;

                vkCmdDispatch(cmd , x,  y, 1);
            }   

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            { // for everyone later to read this texture
                auto barrier = source->multiscatterLUT->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                const auto dep = TypicalDependency<1>(&barrier);
                vkCmdPipelineBarrier2(cmd , &dep);
            }

            _implUpdateSource(cmd); // we want to make them from start , so 
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implCreateSingleUsageImages(moxContext2* context) noexcept __attribute__((nonnull))
        {
            cmdFunctionType function = [&](VkCommandBuffer& cmd) -> void {_implExecuteSingle(cmd);};
            return context->executeCommand(function);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit(const CreateInfo &input) noexcept
        {

            try
            {
                // since they all share the same buffers , there is no need to create dbuffers, and dlayouts , only pipeline layout
                // __IBL first

                source = std::make_shared<Source::AtmoScat>(input.context);

                std::vector<DescriptorLayout*> layouts(2);
                layouts[0] = source->requestLayout();
                layouts[1] = input.sceneBuffers->requestLayout();

                __dbs[0] = source->requestBuffer();
                __dbs[1] = input.sceneBuffers->requestBuffer();

                PipelineLayout::CreateInfo pInfo{};
                pInfo.layouts = layouts;
                pInfo.ranges = {};
                pInfo.context = input.context;

                ShaderObject2::CreateInfo shader{};
                shader.context = input.context;
                shader.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                shader.nextStage = 0;
                shader.layouts = layouts;
                shader.pushes = {};

                {
                    __IBL = std::make_shared<ShaderPipe2>(input.context);
                    __IBL->layout = std::make_shared<PipelineLayout>(pInfo);
                    shader.filePath = input.IBL.string();
                    __IBL->addShader(shader);
                }

                // MS
                {
                    __MS = std::make_shared<ShaderPipe2>(input.context);
                    __MS->layout = std::make_shared<PipelineLayout>(pInfo);
                    shader.filePath = input.MS.string();
                    __MS->addShader(shader);
                }

                // Skyview

                {
                    __skyView = std::make_shared<ShaderPipe2>(input.context);
                    __skyView->layout = std::make_shared<PipelineLayout>(pInfo);
                    shader.filePath = input.skyview.string();
                    __skyView->addShader(shader);
                }

                // transmission

                {
                    __transmission = std::make_shared<ShaderPipe2>(input.context);
                    __transmission->layout = std::make_shared<PipelineLayout>(pInfo);
                    shader.filePath = input.transmission.string();
                    __transmission->addShader(shader);
                }

                // transition
                {
                    __transition = std::make_shared<ShaderPipe2>(input.context);
                    __transition->layout = std::make_shared<PipelineLayout>(pInfo);
                    shader.filePath = input.transition.string();
                    __transition->addShader(shader);
                }

            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to impl init for Sky manager");
            }

            const auto result = _implCreateSingleUsageImages(input.context);
            MOX_CHECK_RESULT(result)

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(SkyManager)
}
}