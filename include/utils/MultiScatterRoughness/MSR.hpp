#pragma once


#include "../../Vulkan/Image2/RenderImage2.hpp"
#include <filesystem>
#include "../../Vulkan/ShaderPipe2/ShaderPipe2.hpp"
namespace mox
{
namespace utils
{
    class MSR final
    {
    public:

        MOX_NO_DISCARD inline static ErrorDataOutput<void> Execute(moxContext2* context , Image::RenderImage2* image , const std::filesystem::path& path) noexcept 
        __attribute__((nonnull))
        {
            return _implCreatePipe(context , image , path);
        }

    private:


        MOX_NO_DISCARD inline static ErrorDataOutput<void> _implCreatePipe(moxContext2* context ,Image::RenderImage2* image , const std::filesystem::path path) noexcept
        __attribute__((nonnull))
        {
            pShaderPipe2 pipe = nullptr;
            try
            {
                pipe = std::make_shared<ShaderPipe2>(context);

                {
                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.data = {DescriptorLayoutData{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , .count = 1}};

                    pipe->addDescriptorLayout(info);
                }

                {
                    PipelineLayout::CreateInfo info{};
                    info.layouts = {pipe->descriptorLayouts[0].get()};
                    info.context = context;
                    info.ranges = {};

                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.filePath = path.string();
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.pushes = {};
                    info.layouts = {pipe->descriptorLayouts[0].get()};

                    pipe->addShader(info);
                }

                // no need for descriptor buffer , instead , we are going to use push descriptor set
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create pipe for MSR , please check the logs above");
            }

            cmdFunctionType function = [&](VkCommandBuffer& cmd) -> void 
            {
                _implExecutePass(cmd , image , pipe.get());
            };

            const auto result = context->executeCommand(function);
            MOX_CHECK_RESULT(result)

            deviceWaitIdle(context->logicalDevice->device);

            return {};
        }

        inline static void _implExecutePass(VkCommandBuffer& cmd , Image::RenderImage2* image ,ShaderPipe2* pipe ) noexcept
        __attribute__((nonnull))
        {
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            const uint32_t x = ( image->extent.width + 63) / 64;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            {
                auto bar = image->translateImage(VK_ACCESS_2_SHADER_WRITE_BIT , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
                const auto dep = TypicalDependency<1>(&bar);
                vkCmdPipelineBarrier2(cmd , &dep);
            }

            VkDescriptorImageInfo dimage{};
            dimage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            dimage.imageView = image->view;
            dimage.sampler = nullptr;
            
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.pNext = nullptr;
            write.pBufferInfo = nullptr;
            write.pTexelBufferView = nullptr;
            write.pImageInfo = &dimage;
            write.descriptorCount = 1l;
            write.dstArrayElement = 0;
            write.dstSet = nullptr;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.dstBinding = 0;

            cmdPushDescriptorSetKHR(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , &write);

            vkCmdDispatch(cmd , x , 1 , 1);

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            {
                auto bar = image->translateImage(VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
                const auto dep = TypicalDependency<1>(&bar);
                vkCmdPipelineBarrier2(cmd , &dep);
            }
        }
    };
}
}