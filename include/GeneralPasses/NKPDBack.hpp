#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../AI/NKPD/NKPD.hpp"

namespace mox
{
namespace Passes
{   
    template<uint8_t forIndirect>
    class NKPDBack : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        SetTypicalOperators(NKPDBack)

        // the name has to be unique in a path of the graph ( the diffuse and the indirect pass are both in it ) , the same as the names of ReBLUR
        explicit NKPDBack(const CreateInfo& info) : NodePass(info.context , (forIndirect ? "NKPDBack_Indirect" : "NKPDBack_Diffuse"))
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

            static const std::uint32_t x = ( inputData.context->sc->extent.width / MOX_NKPD_RESOLUTION_DIVISOR + 7) / 8;
            static const std::uint32_t y = ( inputData.context->sc->extent.height / MOX_NKPD_RESOLUTION_DIVISOR + 7) / 8;
            const uint32_t push = forIndirect ? 1 : 0 ;
            vkCmdPushConstants(cmd , pipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(uint32_t), &push);
            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);
            if constexpr (forIndirect) translate(cmd , Source::AI::NKPD::TranslateType::ToRowMajor);
            return {};
        }

        ~NKPDBack() noexcept
        {
            processDelete();
        }

    private:

        pShaderPipe2 pipe{nullptr};
        CreateInfo inputData{};
        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};
        Source::AI::NKPD::requestTranslate translate{};
        
        void processDelete() noexcept override
        {
            pipe = nullptr;
            inputData = {};
            requestFeatures = {};
            NodePass::processDelete();
        }

        void processCopy(NKPDBack&& input) noexcept
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
                push.size = sizeof(uint32_t);
                pipe = std::make_shared<ShaderPipe2>(inputData.context);
                const auto result = processPipe(pipe.get() , VK_SHADER_STAGE_COMPUTE_BIT, {push});
                MOX_CHECK_RESULT(result)

                {
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = inputData.path_S + ".hlsl";
                    info.layouts = pipe->getLayouts();
                    info.pushes=  {push};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe or shader for NKPDBack pass , please check the ,logs above");
            } 

            {
                const auto result = cSources[Source::SceneBuffers::Name].source->template requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
                MOX_CHECK_RESULT(result)

                requestFeatures = result.value();
            }

            {
                const auto result = cSources[Source::AI::NKPD::Name].source->template requestFunction<Source::AI::NKPD::requestTranslate>(Source::AI::NKPD::requestTranslateName);
                MOX_CHECK_RESULT(result)

                translate = result.value();
            }

            return {};
        }        
    };

    template<uint8_t indirect>
    using pNKPDBack = std::shared_ptr<NKPDBack<indirect>>;
}
}