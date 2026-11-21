#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../moxSourceCluster/LightHub.hpp"


namespace mox
{
namespace Passes
{
    class ForwardPlus : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        SetTypicalOperators(ForwardPlus)

        explicit ForwardPlus(const CreateInfo& info) : NodePass(info.context , "ForwardPlus")
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
            return true;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            {
                std::array<VkBufferMemoryBarrier2 , 2> clearBarriers{};
                clearBarriers[0] = tileLightIndices->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
                clearBarriers[1] = lightCountPerTile->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.bufferMemoryBarrierCount = static_cast<std::uint32_t>(clearBarriers.size());
                dep.pBufferMemoryBarriers = clearBarriers.data();
                vkCmdPipelineBarrier2(cmd , &dep);

                vkCmdFillBuffer(cmd , tileLightIndices->buffer , 0 , tileLightIndices->getBufferSize() , 0);
                vkCmdFillBuffer(cmd , lightCountPerTile->buffer , 0 , lightCountPerTile->getBufferSize() , 0);
            }

            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            // forward.hlsl : groupId.xy is the tile , so the grid is TILE_PER_ROW x rows . TILE_SIZE is the same as in LightHub ( ForwardPlusInfo::TILE_SIZE )
            const std::uint32_t x = (inputData.context->sc->extent.width + TILE_SIZE - 1) / TILE_SIZE;
            const std::uint32_t y = (inputData.context->sc->extent.height + TILE_SIZE - 1) / TILE_SIZE;

            // now the shaders
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            return {};
        }

        ~ForwardPlus() noexcept
        {
            processDelete();
        }

    private:

        inline static constexpr std::uint32_t TILE_SIZE = 16;

        pShaderPipe2 pipe{nullptr};
        CreateInfo inputData{};
        // the buffers of LightHub that must be cleared , the pointers are owned by LightHub
        Buffer::BufferBase2* tileLightIndices{nullptr};
        Buffer::BufferBase2* lightCountPerTile{nullptr};

        void processDelete() noexcept override
        {
            pipe = nullptr;
            inputData = {};
            tileLightIndices = nullptr;
            lightCountPerTile = nullptr;
            NodePass::processDelete();
        }

        void processCopy(ForwardPlus&& input) noexcept
        {
            pipe = std::move(input.pipe);
            inputData = input.inputData;
            tileLightIndices = input.tileLightIndices;
            lightCountPerTile = input.lightCountPerTile;
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
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create Pipe or shader for ForwardPlus pass , please check the logs above");
            }

            const auto hubIt = cSources.find(Source::LightHub::Name);
            if(hubIt == cSources.end() || !hubIt->second.source)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create ForwardPlus pass , since the SourceCluster LightHub is not given");
            }

            {
                const auto result = hubIt->second.source->requestStructure<Buffer::BufferBase2*>("tileLightIndices");
                MOX_CHECK_RESULT(result)
                tileLightIndices = result.value();
            }

            {
                const auto result = hubIt->second.source->requestStructure<Buffer::BufferBase2*>("lightCountPerTile");
                MOX_CHECK_RESULT(result)
                lightCountPerTile = result.value();
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(ForwardPlus)
}
}
