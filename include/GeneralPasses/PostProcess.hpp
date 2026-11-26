#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../moxSourceCluster/BloomSource.hpp"

namespace mox
{
namespace Passes
{
    class PostProcess : public NodePass
    {
    public:

        struct CreateInfo       
        {
            moxContext2* _Nonnull context;
            std::string path_S;
        };

        SetTypicalOperators(PostProcess)

        explicit PostProcess(const CreateInfo& info) : NodePass(info.context , "PostProcess")
        {
            const std::filesystem::path file = info.path_S + ".mox";
            const auto result = initFile(file);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
            input = info;
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> initPass() noexcept override
        {
            return _implCreatePipe();
        }

        virtual bool requestExecution() const noexcept override
        {
            return true;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            static const std::uint32_t x = (context->getResolutionExtent().width + 7) / 8;
            static const std::uint32_t y = (context->getResolutionExtent().height + 7) / 8;

            // now the shaders
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);
            const auto features = requestFeatures();
            if(features.bloomRequired)
            {
                translateBloom(cmd);
            }
            return {};
        }

        ~PostProcess() noexcept
        {
            processDelete();
        }

    private:

        pShaderPipe2 pipe{nullptr};
        CreateInfo input{};
        Source::BloomSource::FunctionType translateBloom{};
        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};

        void processDelete() noexcept override
        {
            pipe = nullptr;
            input = {};
            NodePass::processDelete();
        }

        void processCopy(PostProcess&& other) noexcept
        {
            pipe = std::move(other.pipe);
            input = other.input;
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implCreatePipe() noexcept
        {
            try
            {
                pipe = std::make_shared<ShaderPipe2>(context);
                const auto result = processPipe(pipe.get() , VK_SHADER_STAGE_COMPUTE_BIT);
                MOX_CHECK_RESULT(result);
                {
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = input.path_S + ".hlsl";
                    info.layouts = pipe->getLayouts();
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception& e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create pipe for PostProcess");
            }

            {
                const auto result = cSources[Source::BloomSource::Name].source->requestFunction<Source::BloomSource::FunctionType>(Source::BloomSource::FunctionName);
                MOX_CHECK_RESULT(result)
                translateBloom = result.value();
            }

            {
                const auto result = cSources[Source::SceneBuffers::Name].source->requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
                MOX_CHECK_RESULT(result)

                requestFeatures = result.value();
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(PostProcess)
}
}