#pragma once

#include <array>
#include <cstdint>

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/IndirectExtent.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../AI/NKPD/NKPD.hpp"
#include "NKPDPassCommon.hpp"

namespace mox
{
namespace Passes
{
    class NKPDInference : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        SetTypicalOperators(NKPDInference)

        explicit NKPDInference(const CreateInfo& info) : NodePass(info.context , "NKPDInference_Indirect")
        {
            const std::filesystem::path fileMOX = info.path_S + ".mox";
            const auto result = initFile(fileMOX);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }

            inputData = info;
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> initPass() noexcept override
        {
            return _implCreatePipe();
        }

        // only if the neural denoiser works ( it works INSTEAD of ReBLUR )
        virtual bool requestExecution() const noexcept override
        {
            const auto features = requestFeatures();
            return features.NKPD;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            const VkExtent2D extent = Source::indirectExtent(inputData.context->getResolutionExtent());
            for(std::size_t i = 0 ; i < NKPD_PASS_STEPS.size() ; i++)
            {
                const NKPDPassPush push{NKPD_PASS_STEPS[i] , (i + 1 == NKPD_PASS_STEPS.size()) ? 1u : 0u};
                vkCmdPushConstants(cmd , pipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(NKPDPassPush) , &push);
                vkCmdDispatch(cmd , (extent.width + 7) / 8 , (extent.height + 7) / 8 , 1);
                if(i + 1 < NKPD_PASS_STEPS.size()) nkpdPassBarrier(cmd);
            }

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            return {};
        }

        ~NKPDInference() noexcept
        {
            processDelete();
        }

    private:

        pShaderPipe2 pipe{nullptr};
        CreateInfo inputData{};
        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};

        void processDelete() noexcept override
        {
            pipe = nullptr;
            inputData = {};
            requestFeatures = {};
            NodePass::processDelete();
        }

        void processCopy(NKPDInference&& input) noexcept
        {
            pipe = std::move(input.pipe);
            inputData = input.inputData;
            requestFeatures = std::move(input.requestFeatures);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implCreatePipe() noexcept
        {
            moxContext2* context = inputData.context;

            try
            {
                VkPushConstantRange push{};
                push.offset = 0;
                push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                push.size = sizeof(NKPDPassPush);
                pipe = std::make_shared<ShaderPipe2>(inputData.context);
                const auto result = processPipe(pipe.get() , VK_SHADER_STAGE_COMPUTE_BIT , {push});
                MOX_CHECK_RESULT(result)

                {
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = inputData.path_S + ".hlsl";
                    info.layouts = pipe->getLayouts();
                    info.pushes = {push};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe or shader for NKPDInference pass , please check the logs above");
            }

            {
                const auto result = cSources[Source::SceneBuffers::Name].source->template requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
                MOX_CHECK_RESULT(result)

                requestFeatures = result.value();
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(NKPDInference)
}
}
