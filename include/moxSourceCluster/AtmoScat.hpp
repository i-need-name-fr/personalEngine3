#pragma once

#include "../shaderStructures/shaderStructure.h"
#include "../Vulkan/Sampler/Sampler.hpp"
#include "../Vulkan/Image2/RenderImage2.hpp"
#include "../Vulkan/Image2/Skybox2.hpp"
#include "SourceCluster.hpp"

namespace mox
{
namespace Source
{

    class AtmoScat final : public SourceCluster
    {
    public:

        AtmoScat() = delete;

        static constexpr uint32_t transmissionWidth  = 512;
        static constexpr uint32_t transmissionHeight = 256;

        static constexpr uint32_t multiScatterWidth  = 256;
        static constexpr uint32_t multiScatterHeight = 128;

        static constexpr uint32_t skyViewWidth  = 768;
        static constexpr uint32_t skyViewHeight = 432;

        static constexpr uint32_t skyboxResolution = 1024;

        static constexpr uint32_t iblResolution = 32;

        Image::pRenderImage2 transmissionLUT{nullptr};
        Image::pRenderImage2 multiscatterLUT{nullptr};
        Image::pRenderImage2 skyViewLUT{nullptr};

        Image::pSkybox2 skybox{nullptr};
        Image::pSkybox2 IBL{nullptr};

        Buffer::pDeviceReadableBuffer skyBuffer{nullptr};

        inline static const std::string Name = "AtmoScat";

        explicit AtmoScat(moxContext2* context) __attribute__((nonnull)) : SourceCluster(context , Name) 
        {
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager , context);
                [[unlikely]] if(!result.has_value())
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

        ~AtmoScat() noexcept
        {
            processDelete();
        }

    private:

        // the barriers are made by SourceCluster::requestBarrier : RenderImage2 , Skybox2 ( all 6 faces ) and the buffers are all known to the base class

        pSampler clampSampler{nullptr};

        // NOT [[noreturn]] : the function returns , and for a function that is marked noreturn the compiler puts a trap ( int3 ) after the last line : the crash 0x80000003 on the shutdown
        void processDelete() noexcept override
        {
            // the descriptor buffer is destroyed first ( the base class ) , then the resources it pointed to
            SourceCluster::processDelete();

            transmissionLUT = nullptr;
            multiscatterLUT = nullptr;
            skyViewLUT = nullptr;
            skybox = nullptr;
            IBL = nullptr;
            skyBuffer = nullptr;
            clampSampler = nullptr;
        }

        MOX_NO_DISCARD Image::pRenderImage2 _implCreateLUT(const uint32_t width , const uint32_t height) const
        {
            Image::RenderImage2::CreateInfo info{};
            info.context = context;
            info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
            // written by a compute shader ( storage ) and read by the next ones through a sampler ( sampled )
            info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            info.extent = {width , height};
            info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.layerCount = 1;
            info.mipMapCount = 1;
            info.clear.float32[0] = 0.0f;
            info.clear.float32[1] = 0.0f;
            info.clear.float32[2] = 0.0f;
            info.clear.float32[3] = 0.0f;

            return std::make_shared<Image::RenderImage2>(info);
        }

        MOX_NO_DISCARD Image::pSkybox2 _implCreateCube(const uint32_t resolution) const
        {
            Image::Skybox2::CreateInfo info{};
            info.context = context;
            // not a texture from a file : the cube is written by the shaders ( the format is R16G16B16A16_SFLOAT ) . STORAGE_BIT makes the storageView ( the 2D array of the 6 faces )
            info.isTexture = false;
            info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            info.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
            info.extent = {resolution , resolution};
            info.resolution = resolution;

            return std::make_shared<Image::Skybox2>(info);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit() noexcept
        {
            try
            {
                // the sky settings : a high sun by default , so the first frame is not black before the CPU writes the real values
                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = sizeof(SkyViewData);

                    SkyViewData sky{};
                    sky.cameraHeight = 1.0f;
                    sky.sunCosZenith = 0.8f;
                    sky.sunCosAzimuth = 1.0f;
                    sky.sunSinAzimuth = 0.0f;
                    skyBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , &sky);
                }

                transmissionLUT = _implCreateLUT(transmissionWidth , transmissionHeight);
                multiscatterLUT = _implCreateLUT(multiScatterWidth , multiScatterHeight);
                skyViewLUT = _implCreateLUT(skyViewWidth , skyViewHeight);

                skybox = _implCreateCube(skyboxResolution);
                IBL = _implCreateCube(iblResolution);

                // the lookups of the LUTs and of the cube : linear , clamp to edge
                {
                    Sampler::CreateInfo info{};
                    info.context = context;
                    info.linearClampSampler = true;
                    clampSampler = std::make_shared<Sampler>(info);
                }

                if(!skybox->storageView || !IBL->storageView)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create AtmoScat , since a cube has no storageView ( it needs VK_IMAGE_USAGE_STORAGE_BIT )"});
                }

                // the layout : see the table at the top of the file
                {
                    std::vector<DescriptorLayoutData> data(12);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[5] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[6] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[7] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[8] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[9] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    data[10] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[11] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.resize(1);
                    dlayouts[0] = std::make_shared<DescriptorLayout>(info);
                }

                // the sources : the names of the HLSL variables ( the barrier is made for the whole image )
                {
                    const auto result = _implAddSource(transmissionLUT.get() , "transmissionLUT");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(multiscatterLUT.get() , "multiscatterLUT");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(skyViewLUT.get() , "skyViewLUT");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(skybox.get() , "skybox");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(IBL.get() , "IBL");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(skyBuffer.get() , "skyBuffer");
                    MOX_CHECK_RESULT(result)
                }

                // the descriptor buffer : nothing changes , it is made once . the same image is written through the storage binding and read through the sampled one ,
                // the cubes : the storage binding takes the 2D array view of the faces , the sampled one takes the cube view
                {
                    std::vector<DescriptorData2> data{};
                    data.reserve(12);
                    data.push_back(DescriptorData2::createData(transmissionLUT.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0 , 0));
                    data.push_back(DescriptorData2::createData(multiscatterLUT.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0));
                    data.push_back(DescriptorData2::createData(skyViewLUT.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0));
                    data.push_back(DescriptorData2::createData(skybox->storageView , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0));
                    data.push_back(DescriptorData2::createData(skybox->view , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 4 , 0));
                    data.push_back(DescriptorData2::createData(IBL->storageView , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 5 , 0));
                    data.push_back(DescriptorData2::createData(IBL->view , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 6 , 0));
                    data.push_back(DescriptorData2::createData(transmissionLUT.get() , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 7 , 0));
                    data.push_back(DescriptorData2::createData(multiscatterLUT.get() , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 8 , 0));
                    data.push_back(DescriptorData2::createData(skyViewLUT.get() , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 9 , 0));
                    data.push_back(DescriptorData2::createData(clampSampler.get() , 10 , 0));
                    data.push_back(DescriptorData2::createData(skyBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 11 , 0));

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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create AtmoScat ( the LUTs , the cubes , the sampler or the descriptor set )"});
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(AtmoScat)
}
}
