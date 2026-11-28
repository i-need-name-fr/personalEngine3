#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../AI/NKPD/NKPD.hpp"

namespace mox
{
namespace Passes
{
    class NKPDApply2 : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        SetTypicalOperators(NKPDApply2)

        explicit NKPDApply2(const CreateInfo& info) : NodePass(info.context , "NKPDApply2")
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

            // Adam of the layer 1 ( NeuralBuffer::adamStep , processAdam.hlsl ) : it makes its own barriers
            adam(cmd);
            return {};
        }

        ~NKPDApply2() noexcept
        {
            processDelete();
        }

    private:

        pShaderPipe2 pipe{nullptr};
        CreateInfo inputData{};
        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};
        Source::AI::NKPD::requestTranslate translate{};
        Source::AI::NKPD::requestAnnulment annulment{};
        Source::AI::NKPD::requestAdam adam{};
        
        void processDelete() noexcept override
        {
            pipe = nullptr;
            inputData = {};
            requestFeatures = {};
            NodePass::processDelete();
        }

        void processCopy(NKPDApply2&& input) noexcept
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
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe or shader for NKPDApply2 pass , please check the ,logs above");
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

            {
                const auto result = cSources[Source::AI::NKPD::Name].source->requestFunction<Source::AI::NKPD::requestAdam>(Source::AI::NKPD::requestAdamHiddenName);
                MOX_CHECK_RESULT(result)

                adam = result.value();
            }

            return {};
        }        
    };

    MOX_DECLARE_PTR_STRUCTURE(NKPDApply2)
}
}