#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxObject/VertexHeap.hpp"
#include "../moxSourceCluster/SceneImages.hpp"

namespace mox
{
namespace Passes
{
    class mDeferredPass : public NodePass
    {
    public:

        SetTypicalOperators(mDeferredPass)

        struct CreateInfo
        {
            moxContext2* context{};
            std::filesystem::path task_F;
            std::filesystem::path mesh_F;
            std::filesystem::path fragment_F;
            std::filesystem::path general_F; // general is the mox file
        };

        explicit mDeferredPass(const CreateInfo& info) : NodePass(info.context , "mDeferredRender")
        {
            const auto result = this->initFile(info.general_F);
            if(!result.has_value()) 
            {
                engineLogger(result);
                THROW_MESSAGE;
            }

            inputData = info;
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> initPass() noexcept override
        {
            return _implCreatePipe(inputData);
        }

        virtual bool requestExecution() const noexcept override
        {
            return true;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            VkShaderEXT shaders[4] = {VK_NULL_HANDLE , pipe->shaders[0]->shader , pipe->shaders[1]->shader , pipe->shaders[2]->shader};
            VkShaderStageFlagBits stages[4] = {VK_SHADER_STAGE_VERTEX_BIT , VK_SHADER_STAGE_TASK_BIT_EXT , VK_SHADER_STAGE_MESH_BIT_EXT , VK_SHADER_STAGE_FRAGMENT_BIT};
            const std::uint32_t id = context->sc->frameIndex %2;
            cmdBindShadersEXT(cmd , 4 , stages , shaders);

            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_GRAPHICS , pipe->layout.get());

            // main setup for pipeline
            VkViewport viewport{};
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            viewport.x = 0.0f;
            viewport.width = static_cast<float>(inputData.context->getResolutionExtent().width);
            viewport.y = static_cast<float>(inputData.context->getResolutionExtent().height);
            viewport.height = -static_cast<float>(inputData.context->getResolutionExtent().height);

            VkRect2D scissor{};
            scissor.extent = {inputData.context->getResolutionExtent().width , inputData.context->getResolutionExtent().height};
            scissor.offset = {0 , 0};

            vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
            vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
            vkCmdSetDepthTestEnable(cmd, VK_TRUE);
            vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
            vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_LESS);
            vkCmdSetViewportWithCount(cmd, 1, &viewport);
            vkCmdSetScissorWithCount(cmd, 1, &scissor);


            vkCmdSetRasterizerDiscardEnable(cmd , VK_FALSE);
            vkCmdSetDepthBiasEnable(cmd , VK_TRUE);
            // a clamp that is not 0 needs the depthBiasClamp feature of the device ( LogicalDevice )
            vkCmdSetDepthBias(cmd , DEPTH_BIAS_CONSTANT , DEPTH_BIAS_CLAMP , DEPTH_BIAS_SLOPE);
            vkCmdSetStencilTestEnable(cmd , VK_FALSE);
            cmdSetPolygonModeEXT(cmd , VK_POLYGON_MODE_FILL);
            cmdSetRasterizationSamplesEXT(cmd , VK_SAMPLE_COUNT_1_BIT);
            const VkSampleMask sampleMask = 0xFFFFFFFF;
            cmdSetSampleMaskEXT(cmd , VK_SAMPLE_COUNT_1_BIT , &sampleMask);
            cmdSetAlphaToCoverageEnableEXT(cmd , VK_FALSE);

            {
                const VkBool32 blend[8] = {VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE , VK_FALSE};
                const VkColorComponentFlags all = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
                const VkColorComponentFlags mask[8] = {all , all , all , all , all , all , all , all};

                cmdSetColorBlendEnableEXT(cmd , 0 , 8 , blend);
                cmdSetColorWriteMaskEXT(cmd , 0 , 8 , mask);
            }

            const VkExtent2D shadingRate = {1 , 1};
            const VkFragmentShadingRateCombinerOpKHR combiners[2] = {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR , VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR};
            cmdSetFragmentShadingRateKHR(cmd , &shadingRate , combiners);

            {
                std::array<VkImageMemoryBarrier2 , 8> gBufferBarriers{};
                for(std::size_t i = 0 ; i < gBufferBarriers.size() ; i++)
                {
                    gBufferBarriers[i] = targets[id][i]->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
                }

                const auto dep = TypicalDependency<8>(gBufferBarriers.data());
                vkCmdPipelineBarrier2(cmd , &dep);
            }

            // albedo  ,position , normal  , depth , roughness , metallic , pixelCheck , objectAmbient

            depthBuffer->cleanAndTranslate(cmd , VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT , VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);

            // now the Dynamic Render

            shaders[0] = nullptr;
            shaders[1] = nullptr;
            shaders[2] = nullptr;
            shaders[3] = nullptr;

            {
                // depth Buffer is the last

                VkRenderingInfo info{};
                info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                info.colorAttachmentCount = 8;
                info.pColorAttachments = attachments[id].data();
                info.layerCount = 1;
                info.pDepthAttachment = &depthAtt;
                info.pNext = nullptr;
                info.renderArea.extent = {inputData.context->getResolutionExtent().width , inputData.context->getResolutionExtent().height};
                info.renderArea.offset = {0 ,0};

                vkCmdBeginRendering(cmd , &info);
            }

            const std::uint32_t x = (Source::VertexHeap::getMaxGeometry() + TASK_GROUP_SIZE - 1) / TASK_GROUP_SIZE;
            const std::uint32_t y = (Source::VertexHeap::getInstanceLayoutSize());

            if(x > 0 && y > 0)
            {
                cmdDrawMeshTasksEXT(cmd , x , y , 1);
            }

            vkCmdEndRendering(cmd);

            cmdBindShadersEXT(cmd , 4 , stages , shaders);

            if(!context->user->fixedPosition)
            {
                const auto result =  recordHiZ(cmd , depthBuffer.get());
                MOX_CHECK_RESULT(result)
            }

            return {};
        }

        ~mDeferredPass() noexcept
        {
            processDelete();
        }

        Image::pRenderImage2 depthBuffer{nullptr};

    private:

        inline static constexpr float DEPTH_BIAS_CONSTANT = 1.0f;
        inline static constexpr float DEPTH_BIAS_CLAMP = 1.0e-5f;
        inline static constexpr float DEPTH_BIAS_SLOPE = 1.0f;
        inline static constexpr uint32_t TASK_GROUP_SIZE = 32;

        pShaderPipe2 pipe{};
        CreateInfo inputData;
        Source::SceneImages::RecordHiZFunctionType recordHiZ{};

        std::array<std::array<VkRenderingAttachmentInfo , 8> , 2> attachments{};
        VkRenderingAttachmentInfo depthAtt{};
        std::array<std::array<Image::RenderImage2* , 8> , 2> targets{};

        void processDelete() noexcept override
        {
            pipe = nullptr;
            inputData = {};
            depthBuffer= nullptr;
            NodePass::processDelete();
        }

        void processCopy(mDeferredPass&& input) noexcept
        {
            pipe = std::move(input.pipe);
            inputData = input.inputData;
            depthBuffer = std::move(input.depthBuffer);
            attachments = input.attachments;
            depthAtt = input.depthAtt;
            targets = input.targets;
            NodePass::processCopy(std::move(input));
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implCreatePipe(const CreateInfo& input) noexcept
        {
            try
            {
                pipe = std::make_shared<ShaderPipe2>(context);
                const auto result = processPipe(pipe.get() , VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT);
                MOX_CHECK_RESULT(result)

                { // task
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_TASK_BIT_EXT;
                    info.nextStage = VK_SHADER_STAGE_MESH_BIT_EXT;
                    info.filePath = input.task_F.string();
                    info.pushes = {};
                    info.layouts = pipe->getLayouts();
                    pipe->addShader(info);
                }

                { // mesh
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_MESH_BIT_EXT;
                    info.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    info.filePath = input.mesh_F.string();
                    info.pushes = {};
                    info.layouts = pipe->getLayouts();
                    pipe->addShader(info);
                }

                { // fragment
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_FRAGMENT_BIT;
                    info.nextStage = 0;
                    info.filePath = input.fragment_F.string();
                    info.pushes = {};
                    info.layouts = pipe->getLayouts();
                    pipe->addShader(info);
                }

                {
                    Image::RenderImage2::CreateInfo info{};
                    info.context = context;
                    info.imageFormat = VK_FORMAT_D32_SFLOAT;
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                    info.extent = context->getResolutionExtent();
                    info.layerCount = 1;
                    info.mipMapCount = 1;
                    info.imageUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                    info.clear.float32[0] = 1.0f;
                    info.clear.float32[1] = 1.0f;
                    info.clear.float32[2] = 1.0f;
                    info.clear.float32[3] = 1.0f;
                    depthBuffer = std::make_shared<Image::RenderImage2>(info);
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe for mDeferredPass");
            }

            const auto result = cSources[Source::SceneImages::Name].source->requestFunction<Source::SceneImages::RecordHiZFunctionType>(Source::SceneImages::recordHiZFunctionName);
            MOX_CHECK_RESULT(result)

            recordHiZ = result.value();

            // the depth buffer of the pass
            depthAtt = {};
            depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAtt.pNext = nullptr;
            depthAtt.imageView = depthBuffer->view;
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.clearValue.depthStencil = {1.0f , 0};
            depthAtt.resolveMode = VK_RESOLVE_MODE_NONE;
            depthAtt.resolveImageView = nullptr;
            depthAtt.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            struct GBufferTarget
            {
                const char* name;
                bool pingPong;
                float clear;
            };

            static constexpr std::array<GBufferTarget , 8> gBuffer = {{
                {"albedoImage" , false , 0.0f} ,
                {"positionImage" , true , 0.0f} ,
                {"normalImage" , true , 0.0f} ,
                {"depthImage" , true , 0.0f} ,
                {"roughnessImage" , false , 0.0f} ,
                {"metallicImage" , false , 0.0f} ,
                {"pixelCheck" , false , 0.0f} ,
                {"objectAOImage" , false , 1.0f}
            }};

            const auto sceneIt = cSources.find(Source::SceneImages::Name);
            if(sceneIt == cSources.end() || !sceneIt->second.source)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create mDeferredPass , since the SourceCluster SceneImages is not given");
            }
            auto* scene = sceneIt->second.source;

            for(std::uint8_t parity = 0 ; parity < 2 ; parity++)
            {
                for(std::size_t i = 0 ; i < gBuffer.size() ; i++)
                {
                    const std::string name = gBuffer[i].pingPong ? std::string(gBuffer[i].name) + "_" + std::to_string(parity) : std::string(gBuffer[i].name);

                    const auto result = scene->requestStructure<Image::RenderImage2*>(name);
                    MOX_CHECK_RESULT(result)
                    Image::RenderImage2* image = result.value();
                    targets[parity][i] = image;

                    VkRenderingAttachmentInfo& attachment = attachments[parity][i];
                    attachment = {};
                    attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    attachment.pNext = nullptr;
                    attachment.imageView = image->view;
                    attachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    attachment.resolveMode = VK_RESOLVE_MODE_NONE;
                    attachment.resolveImageView = nullptr;
                    attachment.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    attachment.clearValue.color.float32[0] = gBuffer[i].clear;
                    attachment.clearValue.color.float32[1] = gBuffer[i].clear;
                    attachment.clearValue.color.float32[2] = gBuffer[i].clear;
                    attachment.clearValue.color.float32[3] = gBuffer[i].clear;
                }
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(mDeferredPass)
}
}