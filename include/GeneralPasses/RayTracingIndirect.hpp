#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../moxSourceCluster/IndirectExtent.hpp"


namespace mox
{
namespace Passes
{
    class RayTracingIndirect : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        SetTypicalOperators(RayTracingIndirect)

        explicit RayTracingIndirect(const CreateInfo& info) : NodePass(info.context , "RayTracingIndirect")
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

        // this pass is only in the graph of the ray tracing mode : the shader itself chooses the source of the indirect light ( ReSTIR GI , SHaRC , NRC ) by the features , so it runs every frame
        virtual bool requestExecution() const noexcept override
        {
            return true;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            // the indirect light has its own resolution : the same extent as indirectImage , read every frame ( the size of the window can change )
            const VkExtent2D extent = Source::indirectExtent(inputData.context->sc->extent);
            const std::uint32_t x = (extent.width + 7) / 8;
            const std::uint32_t y = (extent.height + 7) / 8;

            // now the shaders
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            return {};
        }

        ~RayTracingIndirect() noexcept
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

        void processCopy(RayTracingIndirect&& input) noexcept
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
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe or shader for RayTracingIndirect pass , please check the ,logs above");
            } 

            const auto result = cSources[Source::SceneBuffers::Name].source->requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
            MOX_CHECK_RESULT(result)

            requestFeatures = result.value();

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(RayTracingIndirect)
}
}