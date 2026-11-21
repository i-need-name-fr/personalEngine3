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
    [[vk::binding(2 , MOX_SCENE_BUFFERS_INCLUDED)]] ConstantBuffer<SunSource> sunBuffer;

    the names of the sources are the names of the HLSL variables ( the NAME in the .mox file ) , the order of the slots is the number of the binding

    */

    class SceneBuffers final : public SourceCluster
    {
    public:

        SceneBuffers() = delete;

        using RequestFeaturesFunctionType =std::function<GeneralFeatures(void)>;
        inline static const std::string RequestFeaturesFunctionName = "__RequestFeatures__";

        Buffer::pDeviceReadableBuffer generalBuffer{nullptr};
        Buffer::pDeviceReadableBuffer features{nullptr};
        Buffer::pDeviceReadableBuffer sunBuffer{nullptr};
        Buffer::pDeviceReadableBuffer histoGram{nullptr};
        // the auto exposure : [ 0 ] = the EV that is applied now ( it lives between the frames ) , [ 1 ] = 0 until the first frame sets it ( then the EV jumps to the target , not crawls from 0 )
        Buffer::pDeviceReadableBuffer EVBuffer{nullptr};
        Buffer::pDeviceStaticBuffer SSAO_kernel{nullptr};

        inline static const std::string Name = "SceneBuffers";

        explicit SceneBuffers(moxContext2* context) : SourceCluster(context , Name)
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
            sunBuffer = nullptr;
            SSAO_kernel = nullptr;
            histoGram = nullptr;
            EVBuffer = nullptr;
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

                    SunSource defaultSun{};
                    info.size = sizeof(SunSource);
                    sunBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , &defaultSun);
                    std::vector<uint32_t> zeros(256 , 0);
                    info.size = sizeof(float) * 256;
                    histoGram = std::make_shared<Buffer::DeviceReadableBuffer>(info , Buffer::translateToSpan(zeros));

                    std::vector<float> evStart(2 , 0.0f);
                    info.size = sizeof(float) * evStart.size();
                    EVBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , Buffer::translateToSpan(evStart));

                    {
                        Buffer::DeviceStaticBuffer::CreateInfo info{};
                        info.context = context;
                        info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                        info.size = sizeof(glm::vec4) * 64;

                        std::vector<glm::vec4> kernels(64,  glm::vec4(0.0f));

                        static std::random_device rndDevice;
                        static std::mt19937 engine(rndDevice());
                        
                        std::uniform_real_distribution<float> rndFloat(0.0f , 1.0f);

                        for(auto& vec : kernels)
                        {
                            glm::vec3 sample = glm::vec3(rndFloat(engine) * 2.0f - 1.0f , rndFloat(engine) * 2.0f - 1.0f , rndFloat(engine));
                            sample = glm::normalize(sample) * rndFloat(engine);
                            vec = glm::vec4(sample.x , sample.y , sample.z , 0.0f);
                        }

                        SSAO_kernel = std::make_shared<Buffer::DeviceStaticBuffer>(info , Buffer::translateToSpan(kernels));
                    }
                }

                // the layout : 0 , 1 , 2 are ConstantBuffer in the shader ( uniform ) , 3 , 4 , 5 are RWStructuredBuffer ( storage )
                {
                    std::vector<DescriptorLayoutData> data(6 , DescriptorLayoutData{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , .count = 1});
                    data[3] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                    data[4] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};
                    data[5] = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1};

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

                {
                    const auto result = _implAddSource(sunBuffer.get() , "sunBuffer");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(SSAO_kernel.get() , "SSAO_kernel");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(histoGram.get() , "histoGram");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(EVBuffer.get() , "EVBuffer");
                    MOX_CHECK_RESULT(result)
                }

                // the descriptor buffer : the buffers never change , so it is made once ( there is no ping pong , one descriptor buffer )
                {
                    std::vector<DescriptorData2> data{};
                    data.reserve(6);
                    data.push_back(DescriptorData2::createData(generalBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0));
                    data.push_back(DescriptorData2::createData(features.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1 , 0));
                    data.push_back(DescriptorData2::createData(sunBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 2 , 0));
                    data.push_back(DescriptorData2::createData(SSAO_kernel.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0));
                    data.push_back(DescriptorData2::createData( histoGram.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0));
                    data.push_back(DescriptorData2::createData(EVBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 5 , 0));

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

            RequestFeaturesFunctionType function = [&]()-> GeneralFeatures
            {
                return features->getBuffetDataSingular<GeneralFeatures>();
            };

            const auto result = _implAddFunction(function , RequestFeaturesFunctionName);
            MOX_CHECK_RESULT(result)

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(SceneBuffers)
}
}
