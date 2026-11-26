#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"

namespace mox
{
namespace Passes
{
    class Volumentric : public NodePass
    {
    public:

        struct CreateInfo       
        {
            moxContext2* _Nonnull context;
            std::string path_S;
            // the size of the froxel volume ( VolSource::extent , VolSource::layerCount ) : one thread = one froxel
            VkExtent2D extent{};
            std::uint32_t layerCount = 1;
        };

        SetTypicalOperators(Volumentric)

        explicit Volumentric(const CreateInfo& info) : NodePass(info.context , "Volumentric")
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

        // the volume is read by PostProcess only when the fog is on
        virtual bool requestExecution() const noexcept override
        {
            const auto features = requestFeatures();
            return features.volumentricFog != 0 || features.dustMode != 0;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            const std::uint32_t x = (input.extent.width + 7) / 8;
            const std::uint32_t y = (input.extent.height + 7) / 8;

            // now the shaders
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            vkCmdDispatch(cmd , x , y , input.layerCount);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            return {};
        }

        ~Volumentric() noexcept
        {
            processDelete();
        }

    private:

        pShaderPipe2 pipe{nullptr};
        CreateInfo input{};
        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};

        void processDelete() noexcept override
        {
            pipe = nullptr;
            input = {};
            requestFeatures = {};
            NodePass::processDelete();
        }

        void processCopy(Volumentric&& other) noexcept
        {
            pipe = std::move(other.pipe);
            input = other.input;
            requestFeatures = std::move(other.requestFeatures);
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

                const auto fResult = cSources[Source::SceneBuffers::Name].source->requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
                MOX_CHECK_RESULT(fResult)
                requestFeatures = fResult.value();
            }
            catch(std::exception& e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create pipe for Volumentric");
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(Volumentric)
}
}