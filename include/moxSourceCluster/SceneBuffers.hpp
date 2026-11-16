#pragma once

#include "../shaderStructures/shaderStructure.h"
#include "SourceCluster.hpp"

namespace mox
{
namespace Source
{
    /*

    SceneBuffers.hlsl :

    [[vk::binding(0 , MOX_SCENE_BUFFERS_INCLUDED)]] ConstantBuffer<GeneralInfo> generalBuffer;
    [[vk::binding(1 , MOX_SCENE_BUFFERS_INCLUDED)]] ConstantBuffer<GeneralFeatures> features;

    the names of the sources are the names of the HLSL variables ( the NAME in the .mox file ) , the order of the slots is the number of the binding

    */

    class SceneBuffers final : public SourceCluster
    {
    public:

        SceneBuffers() = delete;

        Buffer::pDeviceReadableBuffer generalBuffer{nullptr};
        Buffer::pDeviceReadableBuffer features{nullptr};

        explicit SceneBuffers(moxContext2* context) : SourceCluster(context , "SceneBuffers")
        {
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , context);
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

        ~SceneBuffers() noexcept
        {
            processDelete();
        }

    private:

        void processDelete() noexcept override
        {
            generalBuffer = nullptr;
            features = nullptr;
            SourceCluster::processDelete();
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit() noexcept
        {
            try
            {
                // the buffers
                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    GeneralInfo general{};
                    info.size = sizeof(GeneralInfo);
                    generalBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , &general);

                    GeneralFeatures defaultFeatures{};
                    info.size = sizeof(GeneralFeatures);
                    features = std::make_shared<Buffer::DeviceReadableBuffer>(info , &defaultFeatures);
                }

                // the layout : both are ConstantBuffer in the shader , so both are uniform buffers
                {
                    std::vector<DescriptorLayoutData> data(2 , DescriptorLayoutData{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , .count = 1});

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.resize(1);
                    dlayouts[0] = std::make_shared<DescriptorLayout>(info);
                }

                // the sources : the names are the names of the variables in SceneBuffers.hlsl
                {
                    const auto result = _implAddSource(generalBuffer.get() , "generalBuffer");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(features.get() , "features");
                    MOX_CHECK_RESULT(result)
                }

                // the descriptor buffer : the buffers never change , so it is made once ( there is no ping pong , one descriptor buffer )
                {
                    std::vector<DescriptorData2> data{};
                    data.reserve(2);
                    data.push_back(DescriptorData2::createData(generalBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0));
                    data.push_back(DescriptorData2::createData(features.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0));

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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SceneBuffers ( the buffers or the descriptor set )"});
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(SceneBuffers)
}
}
