#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"

// =============================== INFO ===============================
// mRTPass.hpp - the G BUFFER by ray tracing ( Shaders/mRT/mRT.hlsl ) : one primary ray per pixel instead of the mesh shaders . it does not need Mesh Shading , it is a compute pass with RayQuery .
// it runs right after the Deferred render and overwrites the images of the G buffer ( the depth attachment of the raster pass stays for DLSS and HiZ ) . OFF by default : mRTPass::enabled ( the button in ImGui )
// the test it was made for : if a picture bug is gone with mRT ON , the bug was in the raster G buffer ( culling , missing meshlets )
// =============================== END ===============================

namespace mox
{
namespace Passes
{
    class mRTPass : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        // the switch of the pass ( the GUI flips it )
        inline static bool enabled = false;

        SetTypicalOperators(mRTPass)

        explicit mRTPass(const CreateInfo& info) : NodePass(info.context , "mRTPass")
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
            return enabled;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            const std::uint32_t x = (inputData.context->getResolutionExtent().width + 7) / 8;
            const std::uint32_t y = (inputData.context->getResolutionExtent().height + 7) / 8;

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            return {};
        }

        ~mRTPass() noexcept
        {
            processDelete();
        }

    private:

        pShaderPipe2 pipe{nullptr};
        CreateInfo inputData{};

        void processDelete() noexcept override
        {
            pipe = nullptr;
            inputData = {};
            NodePass::processDelete();
        }

        void processCopy(mRTPass&& input) noexcept
        {
            pipe = std::move(input.pipe);
            inputData = input.inputData;
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
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe or shader for mRT pass , please check the ,logs above");
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(mRTPass)
}
}
