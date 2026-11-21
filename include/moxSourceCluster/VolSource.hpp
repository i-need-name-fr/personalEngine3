#pragma once

#include "../shaderStructures/shaderStructure.h"
#include "SourceCluster.hpp"

namespace mox
{
namespace Source
{
    /*

    */

    class VolSource final : public SourceCluster
    {
    public:

        VolSource() = delete;

        static constexpr uint32_t screenDivisor = 4;
        static constexpr uint32_t layerCount = 96;

        using FogImage = Image::MultiViewImage2<layerCount , Image::ViewPerType::Layer>;

        Buffer::pDeviceReadableBuffer fogBuffer{nullptr};
        std::shared_ptr<FogImage> fogImageArray{nullptr};

        MOX_NO_DISCARD VkImageMemoryBarrier2 requestFogBarrier(const VkPipelineStageFlags2 stage , const VkAccessFlags2 access , const uint32_t count , const uint32_t layerIndex) noexcept
        {
            return fogImageArray->translateImage(access , stage , count , layerIndex);
        }

        VkExtent2D extent{};

        inline static const std::string Name = "VolSource";

        explicit VolSource(moxContext2* context) : SourceCluster(context , Name)
        {
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager | VkReqTypeSwapchain , context);
                if(!result.has_value())
                {
                    engineLogger(&result.error());
                    THROW_MESSAGE;
                }
            }

            const auto result = _implInit();
            if(!result.has_value())
            {
                engineLogger(&result.error());
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~VolSource() noexcept
        {
            processDelete();
        }

    private:

        // requestBarrier( name , stage , access ) of the base class does not know MultiViewImage2 : the volume is the whole image ( all the layers )
        MOX_NO_DISCARD ErrorDataOutput<std::variant<VkImageMemoryBarrier2 , VkBufferMemoryBarrier2>> _implAskForBarrier(const std::string& name , VkPipelineStageFlags2 const stage , VkAccessFlags2 const access) noexcept override
        {
            if(name == "fogImageArray")
            {
                if(!fogImageArray) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create barrier for fogImageArray in VolSource , since the image is nullptr"});
                return fogImageArray->translateWholeImage(access , stage);
            }

            return SourceCluster::_implAskForBarrier(name , stage , access);
        }

        void processDelete() noexcept override
        {
            fogBuffer = nullptr;
            fogImageArray = nullptr;
            SourceCluster::processDelete();
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit() noexcept
        {
            try
            {
                extent = {std::max(1u , context->sc->extent.width / screenDivisor) , std::max(1u , context->sc->extent.height / screenDivisor)};

                // the fog data : a uniform buffer , the CPU changes it ( DeviceReadableBuffer )
                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = sizeof(FogVolume);

                    FogVolume fog{};
                    fog.gridSize = static_cast<float>(layerCount);
                    fogBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , &fog);
                }

                // the froxel volume : one image , a 2D array
                {
                    FogImage::CreateInfo info{};
                    info.context = context;
                    info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                    info.extent = extent;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    // the view of ONE layer is a 2D view , fullView ( all the layers ) is always a 2D array
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;

                    fogImageArray = std::make_shared<FogImage>(info);
                }

                // the layout : 0 uniform buffer , 1 sampled image ( Texture2DArray ) , 2 storage image ( RWTexture2DArray )
                {
                    std::vector<DescriptorLayoutData> data(3);
                    data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.resize(1);
                    dlayouts[0] = std::make_shared<DescriptorLayout>(info);
                }

                {
                    const auto result = _implAddSource(fogBuffer.get() , "fogBuffer");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(fogImageArray.get() , "fogImageArray");
                    MOX_CHECK_RESULT(result)
                }

                // the descriptor buffer : nothing changes , so it is made once . the SAME view ( fullView : all the layers , a 2D array ) goes to the bindings 1 and 2 , only the descriptor type is different
                {
                    std::vector<DescriptorData2> data{};
                    data.reserve(3);
                    data.push_back(DescriptorData2::createData(fogBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0));
                    data.push_back(DescriptorData2::createData(fogImageArray->fullView , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1 , 0));
                    data.push_back(DescriptorData2::createData(fogImageArray->fullView , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0));

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = dlayouts[0].get();

                    dbuffers.resize(1);
                    dbuffers[0] = std::make_shared<DescriptorBuffer2>(info);
                }

                containsPingPong = false;
            }
            catch(std::exception& e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create VolSource ( the fog buffer , the froxel volume or the descriptor set )"});
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(VolSource)
}
}
