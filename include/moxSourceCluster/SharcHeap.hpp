#pragma once

#include "../Vulkan/Sampler/Sampler.hpp"
#include "../Vulkan/Image2/RenderImage2.hpp"
#include "../Vulkan/Image2/Skybox2.hpp"
#include "SourceCluster.hpp"

namespace mox
{
namespace Source
{
    class SharcHeap : public SourceCluster
    {
    public:

        SetTypicalOperators(SharcHeap)

        inline static const std::string Name = "SharcHeap";

        struct CreateInfo
        {
            moxContext2* context;
        };

        explicit SharcHeap(const CreateInfo& info) : SourceCluster(info.context , Name)
        {
            const auto result = _implInit(info.context);
            if(!result.has_value())
            {
                processDelete();
                engineLogger(result);
                THROW_MESSAGE;
            }
        }

        ~SharcHeap() noexcept
        {
            processDelete();
        }

        Buffer::pDeviceReadableBuffer tableBuffer{};
        Buffer::pDeviceReadableBuffer tableInfoBuffer{};

    private:
    
        void processDelete() noexcept
        {
            tableBuffer = nullptr;
            tableInfoBuffer = nullptr;
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit(moxContext2* context) noexcept
        __attribute__((nonnull))
        {

            const uint32_t baseT = 64;
            const float baseCellSize = 1.0f;

            try
            {
                std::vector<Sharc::TableInfo> infos{};
                std::vector<glm::vec4> radiance{};
                for(uint32_t layer = 1 ; layer <= MOX_SHARC_LAYER_COUNT ; layer++)
                {
                    Sharc::TableInfo info{};
                    info.cellSize = baseCellSize * (float)(1u << (layer - 1));
                    info.T = baseT;
                    info.minDistance = (layer == 1) ? 0.0f : 0.5f * (float)baseT * baseCellSize * (float)(1u << (layer - 2));
                    info.offset = radiance.size();
                    infos.push_back(info);
                    // 000 , 001 , 010 , 011 , 100 , 101 , 110 , 111

                    for(uint32_t z = 0 ; z < info.T ; z++)
                    for(uint32_t y = 0 ; y < info.T ; y++)
                    for(uint32_t x = 0 ; x < info.T ; x++)
                    for(uint32_t r = 0 ; r < 8 ; r++)
                    {
                        radiance.push_back(glm::vec4(0.0f));
                    }
                }
                
                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                    info.size = infos.size() * sizeof(Sharc::TableInfo);
                    
                    tableInfoBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , Buffer::translateToSpan(infos));
                }

                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                    info.size = radiance.size() * sizeof(glm::vec4);
                    
                    tableBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , Buffer::translateToSpan(radiance));
                }

                {
                    std::vector<DescriptorLayoutData> data(2);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.push_back(std::make_shared<DescriptorLayout>(info));
                }

                {
                    std::vector<DescriptorData2> data(2);
                    data[0] = mox::DescriptorData2::createData(tableBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 ,0 );
                    data[1] = mox::DescriptorData2::createData(tableInfoBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 ,0 );

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = dlayouts[0].get();

                    dbuffers.push_back(std::make_shared<DescriptorBuffer2>(info));
                }

                {
                    const auto result = _implAddSource(tableBuffer.get() , "tableBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(tableInfoBuffer.get() , "tableInfoBuffer");
                    MOX_CHECK_RESULT(result);
                }

                containsPingPong = false;
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create buffers for ShaRC");
            }
            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(SharcHeap)
}
}