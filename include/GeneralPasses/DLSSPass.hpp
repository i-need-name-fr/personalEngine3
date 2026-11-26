#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../Vulkan/vulkanContext/StreamlineContext.hpp"

namespace mox
{
namespace Passes
{
    class DLSSPass : public NodePass
    {
    public:

        struct CreateInfo
        {
            moxContext2* _Nonnull context;
            std::string path_S;
            // the ADDRESS of the pointer of the depth ( mDeferredPass::depthBuffer ) : the depth image is created later , in initPass of mDeferredPass , so it is read at the execution , not at the creation of this pass
            const Image::pRenderImage2* _Nonnull depth;
        };

        SetTypicalOperators(DLSSPass)

        explicit DLSSPass(const CreateInfo& info) : NodePass(info.context , "DLSSPass")
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
            auto& scene = cSources[Source::SceneImages::Name].source;

            for(uint32_t i = 0 ; i < 2 ; i++)
            {
                const auto result = scene->template requestStructure<Image::RenderImage2*>(moxPingPongName("HDRImage" , i));
                MOX_CHECK_RESULT(result)
                hdr[i] = result.value();
            }

            {
                const auto result = scene->template requestStructure<Image::RenderImage2*>("motionVector");
                MOX_CHECK_RESULT(result)
                motion = result.value();
            }

            {
                const auto result = scene->template requestStructure<Image::RenderImage2*>("DLSSImage");
                MOX_CHECK_RESULT(result)
                output = result.value();
            }

            return {};
        }

        // always : DLSS or the copy , the picture has to get into DLSSImage in any case
        virtual bool requestExecution() const noexcept override
        {
            return true;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            Image::RenderImage2* color = hdr[moxPingPongIndex(static_cast<uint32_t>(context->sc->frameIndex % 2) , 1u)];

            Image::RenderImage2* depthImage = input.depth ? input.depth->get() : nullptr;
            const bool useDLSS = depthImage && context->stream && context->stream->isDLSSReady() && context->stream->currentFrame;
            if(useDLSS) return _implRunDLSS(cmd , color , depthImage);

            _implCopy(cmd , color);
            return {};
        }

        ~DLSSPass() noexcept
        {
            processDelete();
        }

    private:

        CreateInfo input{};
        Image::RenderImage2* hdr[2]{nullptr , nullptr};
        Image::RenderImage2* motion{nullptr};
        Image::RenderImage2* output{nullptr};

        void processDelete() noexcept override
        {
            input = {};
            hdr[0] = nullptr;
            hdr[1] = nullptr;
            motion = nullptr;
            output = nullptr;
            NodePass::processDelete();
        }

        void processCopy(DLSSPass&& other) noexcept
        {
            input = other.input;
            hdr[0] = other.hdr[0];
            hdr[1] = other.hdr[1];
            motion = other.motion;
            output = other.output;
        }

        static void _implBarriers(VkCommandBuffer& cmd , VkImageMemoryBarrier2* barriers , const uint32_t count) noexcept
        {
            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.pNext = nullptr;
            dependency.dependencyFlags = 0;
            dependency.memoryBarrierCount = 0;
            dependency.pMemoryBarriers = nullptr;
            dependency.bufferMemoryBarrierCount = 0;
            dependency.pBufferMemoryBarriers = nullptr;
            dependency.imageMemoryBarrierCount = count;
            dependency.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd , &dependency);
        }

        // DLSS off : the picture of PostProcess goes to DLSSImage as it is
        void _implCopy(VkCommandBuffer& cmd , Image::RenderImage2* color) noexcept
        {
            VkImageMemoryBarrier2 barriers[2] = {
                color->translateImage(VK_ACCESS_2_TRANSFER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT),
                output->translateImage(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT)
            };
            _implBarriers(cmd , barriers , 2);

            VkImageCopy region{};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.mipLevel = 0;
            region.srcSubresource.baseArrayLayer = 0;
            region.srcSubresource.layerCount = 1;
            region.dstSubresource = region.srcSubresource;
            region.srcOffset = {0 , 0 , 0};
            region.dstOffset = {0 , 0 , 0};
            // the render and the output can differ in size : the part that both have
            region.extent = {std::min(color->extent.width , output->extent.width) , std::min(color->extent.height , output->extent.height) , 1};

            vkCmdCopyImage(cmd , color->image , VK_IMAGE_LAYOUT_GENERAL , output->image , VK_IMAGE_LAYOUT_GENERAL , 1 , &region);
        }

        // DLSS on
        ErrorDataOutput<void> _implRunDLSS(VkCommandBuffer& cmd , Image::RenderImage2* color , Image::RenderImage2* depth) noexcept
        {
            auto& stream = context->stream;

            // everything that DLSS reads is ready , the output is free to be written . the stage is ALL_COMMANDS : SL records its own work into this command buffer
            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkImageMemoryBarrier2 barriers[4] = {
                color->translateImage(VK_ACCESS_2_MEMORY_READ_BIT , stage),
                depth->translateImage(VK_ACCESS_2_MEMORY_READ_BIT , stage),
                motion->translateImage(VK_ACCESS_2_MEMORY_READ_BIT , stage),
                output->translateImage(VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT , stage)
            };
            _implBarriers(cmd , barriers , 4);

            // the tags of the frame ( the lifecycle is the same as in the tags of the frame generation : the resources live as long as the engine )
            sl::Resource colorRes = color->translateToResource();
            sl::Resource depthRes = depth->translateToResource();
            sl::Resource motionRes = motion->translateToResource();
            sl::Resource outputRes = output->translateToResource();

            const sl::ResourceTag tags[4] = {
                sl::ResourceTag(&colorRes , sl::kBufferTypeScalingInputColor , sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&outputRes , sl::kBufferTypeScalingOutputColor , sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&depthRes , sl::kBufferTypeDepth , sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&motionRes , sl::kBufferTypeMotionVectors , sl::ResourceLifecycle::eValidUntilPresent)
            };

            const sl::Result tagResult = stream->pSetTagForFrame(*stream->currentFrame , sl::ViewportHandle(0) , tags , 4 , nullptr);
            if(tagResult != sl::Result::eOk)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to set the tags of DLSS , error -> {} ( {} )" , sl::getResultAsStr(tagResult) , static_cast<int32_t>(tagResult))});
            }

            return stream->evaluateDLSS(cmd);
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(DLSSPass)
}
}
