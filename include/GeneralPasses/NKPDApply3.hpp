#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../AI/NKPD/NKPD.hpp"

namespace mox
{
namespace Passes
{
    class NKPDApply3 : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        SetTypicalOperators(NKPDApply3)

        explicit NKPDApply3(const CreateInfo& info) : NodePass(info.context , "NKPDApply3")
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

        virtual bool requestExecution() const noexcept override
        {
            const auto features = requestFeatures();
            return features.NKPDTrain;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            // now the shaders
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);
            const uint32_t x = (16 * 128  + 16 + 127) / 128;
            vkCmdDispatch(cmd , x , 1 , 1);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);
            translate(cmd , Source::AI::NKPD::TranslateType::ToOptimal);
            annulment(cmd);
            return {};
        }

        ~NKPDApply3() noexcept
        {
            processDelete();
        }

    private:

        pShaderPipe2 pipe{nullptr};
        CreateInfo inputData{};
        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};
        Source::AI::NKPD::requestTranslate translate{};
        Source::AI::NKPD::requestAnnulment annulment{};
        
        void processDelete() noexcept override
        {
            pipe = nullptr;
            inputData = {};
            requestFeatures = {};
            NodePass::processDelete();
        }

        void processCopy(NKPDApply3&& input) noexcept
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
                pipe = std::make_shared<ShaderPipe2>(inputData.context);
                const auto result = processPipe(pipe.get() , VK_SHADER_STAGE_COMPUTE_BIT);
                MOX_CHECK_RESULT(result)

                {
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = inputData.path_S + ".hlsl";
                    info.layouts = pipe->getLayouts();
                    info.pushes=  {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe or shader for NKPDApply3 pass , please check the ,logs above");
            } 

            {
                const auto result = cSources[Source::SceneBuffers::Name].source->requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
                MOX_CHECK_RESULT(result)

                requestFeatures = result.value();
            }

            {
                const auto result = cSources[Source::AI::NKPD::Name].source->requestFunction<Source::AI::NKPD::requestTranslate>(Source::AI::NKPD::requestTranslateName);
                MOX_CHECK_RESULT(result)

                translate = result.value();
            }

            {
                const auto result = cSources[Source::AI::NKPD::Name].source->requestFunction<Source::AI::NKPD::requestAnnulment>(Source::AI::NKPD::requestAnnulmentName);
                MOX_CHECK_RESULT(result)

                annulment = result.value();
            }

            return {};
        }        
    };

    MOX_DECLARE_PTR_STRUCTURE(NKPDApply3)
}
}