#pragma once
#include "../shaderStructures/shaderStructure.h"
#include "../Vulkan/Sampler/Sampler.hpp"
#include "../Vulkan/Image2/RenderImage2.hpp"
#include "../Vulkan/Image2/MultiViewImage2.hpp"
#include "../Vulkan/Image2/Skybox2.hpp"
#include "SourceCluster.hpp"
#include "../Vulkan/ShaderPipe2/ShaderPipe2.hpp"

namespace mox
{
namespace Source
{
    #define MOX_BLOOM_IMAGE_COUNT 5
    class BloomSource : public SourceCluster
    {
    public: 

        using FunctionType = std::function<void(VkCommandBuffer&)>;
        inline static const std::string Name = "BloomSource";

        inline static const std::string FunctionName = "__MOX_BLOOM_SOURCE_FUNCTION_ANALYZE__";

        struct CreateInfo
        {
            moxContext2* context;
            std::filesystem::path upscal;
            std::filesystem::path downscale;
        };

        explicit BloomSource(const CreateInfo info) : SourceCluster(info.context,  Name)
        {
            const auto result = _implInit(info);
            if(!result.has_value())
            {
                engineLogger(result);
                bloomBuffer = nullptr;
                bloomImages = nullptr;
                upPipe = nullptr;
                downPipe = nullptr;
                sampler = nullptr;
                THROW_MESSAGE;
            }
        }

        Image::pMultiViewImage2<MOX_BLOOM_IMAGE_COUNT , Image::ViewPerType::Layer> bloomImages{};
        Buffer::pDeviceReadableBuffer bloomBuffer{};

        ~BloomSource()
        {
            deviceWaitIdle(context->logicalDevice->device);
            downPipe = nullptr;
            upPipe = nullptr;
            bloomBuffer = nullptr;
            bloomImages = nullptr;
            sampler = nullptr;
            SourceCluster::processDelete();
        }

    protected:

        MOX_NO_DISCARD ErrorDataOutput<std::variant<VkImageMemoryBarrier2 , VkBufferMemoryBarrier2>> _implAskForBarrier(const std::string& name , VkPipelineStageFlags2 const stage , VkAccessFlags2 const access) noexcept override
        {
            if(name == "bloomImages")
            {
                if(!bloomImages) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create the barrier of bloomImages , since the image is nullptr"});
                const VkImageMemoryBarrier2 barrier = bloomImages->translateWholeImage(access , stage);
                return barrier;
            }
            return SourceCluster::_implAskForBarrier(name , stage , access);
        }

    private:

        pSampler sampler{};
        pShaderPipe2 downPipe{};
        pShaderPipe2 upPipe{};

        struct PushDown {
            uint32_t srcWidth;
            uint32_t srcHeight;
            uint32_t dstWidth;
            uint32_t dstHeight;
            uint32_t srcLevel;
            uint32_t dstLevel;
            uint32_t firstIteration;   // the first step ( the full resolution -> the first mip ) : Karis average
        };
        static_assert(sizeof(PushDown) == 28);

        struct PushUp {
            uint32_t lowerWidth;    // the size of the level i + 1 ( the one that is upsampled )
            uint32_t lowerHeight;
            uint32_t dstWidth;      // the size of the level i ( the destination )
            uint32_t dstHeight;
            uint32_t lowerLevel;    // i + 1
            uint32_t dstLevel;      // i
        };
        static_assert(sizeof(PushUp) == 24);


        static VkExtent2D _implLevelExtent(const VkExtent2D full , const uint32_t level) noexcept
        {
            return {std::max(1u , full.width >> level) , std::max(1u , full.height >> level)};
        }

        void _implLayerBarriers(VkCommandBuffer& cmd , const std::initializer_list<std::pair<uint32_t , VkAccessFlags2>> layers , const VkPipelineStageFlags2 stage) noexcept
        {
            std::vector<VkImageMemoryBarrier2> barriers{};
            barriers.reserve(layers.size());
            for(const auto& [layer , access] : layers)
            {
                barriers.push_back(bloomImages->translateImage(access , stage , 1 , layer));
            }

            VkDependencyInfo dependency{};
            dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependency.pNext = nullptr;
            dependency.dependencyFlags = 0;
            dependency.memoryBarrierCount = 0;
            dependency.pMemoryBarriers = nullptr;
            dependency.bufferMemoryBarrierCount = 0;
            dependency.pBufferMemoryBarriers = nullptr;
            dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            dependency.pImageMemoryBarriers = barriers.data();

            vkCmdPipelineBarrier2(cmd , &dependency);
        }

        void _implBindSet(VkCommandBuffer& cmd , const ShaderPipe2* pipe) noexcept
        {
            const VkDeviceSize offset[] = {0};
            const uint32_t indices[] = {0};
            cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , indices , offset);
        }

        void _implTranslateBloom(VkCommandBuffer& cmd)
        {
            constexpr uint32_t levels = MOX_BLOOM_IMAGE_COUNT;
            const VkExtent2D full = context->sc->extent;

            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.address = dbuffers[0]->pBuffer;
                info.pNext = nullptr;
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;

                cmdBindDescriptorBuffersEXT(cmd , 1,  &info);
            }

            constexpr VkPipelineStageFlags2 computeStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            constexpr VkAccessFlags2 readAccess = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkAccessFlags2 writeAccess = VK_ACCESS_2_SHADER_WRITE_BIT;

            VkShaderEXT shaders[1] = {downPipe->shaders[0]->shader};
            VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            _implBindSet(cmd , downPipe.get());

            for(uint32_t level = 1 ; level < levels ; level++)
            {
                _implLayerBarriers(cmd , {{level - 1 , readAccess} , {level , writeAccess}} , computeStage);

                const VkExtent2D src = _implLevelExtent(full , level - 1);
                const VkExtent2D dst = _implLevelExtent(full , level);

                PushDown push{};
                push.srcWidth = src.width;
                push.srcHeight = src.height;
                push.dstWidth = dst.width;
                push.dstHeight = dst.height;
                push.srcLevel = level - 1;
                push.dstLevel = level;
                push.firstIteration = (level == 1) ? 1u : 0u;

                vkCmdPushConstants(cmd , downPipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(PushDown) , &push);
                vkCmdDispatch(cmd , (dst.width + 7) / 8 , (dst.height + 7) / 8 , 1);
            }

            shaders[0] = upPipe->shaders[0]->shader;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);
            _implBindSet(cmd , upPipe.get());

            for(uint32_t level = levels - 1 ; level > 0 ; level--)
            {
                const uint32_t dstLevel = level - 1;

                _implLayerBarriers(cmd , {{level , readAccess} , {dstLevel , readAccess | writeAccess}} , computeStage);

                const VkExtent2D lower = _implLevelExtent(full , level);
                const VkExtent2D dst = _implLevelExtent(full , dstLevel);

                PushUp push{};
                push.lowerWidth = lower.width;
                push.lowerHeight = lower.height;
                push.dstWidth = dst.width;
                push.dstHeight = dst.height;
                push.lowerLevel = level;
                push.dstLevel = dstLevel;

                vkCmdPushConstants(cmd , upPipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(PushUp) , &push);
                vkCmdDispatch(cmd , (dst.width + 7) / 8 , (dst.height + 7) / 8 , 1);
            }

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stages , shaders);

            _implLayerBarriers(cmd , {{0 , readAccess | writeAccess}} , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }


        MOX_NO_DISCARD pShaderPipe2 _implCreatePipe(const std::filesystem::path& path , const uint32_t pushSize)
        {
            pShaderPipe2 result = std::make_shared<ShaderPipe2>(context);
            result->setLayouts.resize(1);
            result->setLayouts[0] = dlayouts[0].get();

            VkPushConstantRange range{};
            range.offset = 0;
            range.size = pushSize;
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            {
                PipelineLayout::CreateInfo info{};
                info.context = context;
                info.layouts = {dlayouts[0].get()};
                info.ranges = {range};

                result->layout = std::make_shared<PipelineLayout>(info);
            }

            {
                ShaderObject2::CreateInfo info{};
                info.context = context;
                info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                info.nextStage = 0;
                info.filePath = path.string();
                info.pushes = {range};
                info.layouts = {dlayouts[0].get()};
                result->addShader(info);
            }
            return result;
        }

        // init

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit(const CreateInfo& input)
        {

            try
            {
                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(BloomInfo);
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

                    bloomBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info);
                }

                {
                    Image::MultiViewImage2<MOX_BLOOM_IMAGE_COUNT , Image::ViewPerType::Layer>::CreateInfo info{};
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.context = context;
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    info.extent = context->sc->extent;
                    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;\
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;
                    // info.lowerExtent = true;
                    
                    bloomImages = std::make_shared<Image::MultiViewImage2<MOX_BLOOM_IMAGE_COUNT , Image::ViewPerType::Layer>>(info);
                }

                {
                    Sampler::CreateInfo info{};
                    info.linearClampSampler = true;
                    info.context = context;
                    info.requiredComparingSampler = false;
                    info.textureSampler = false;
                    info.requiredTypicalSampler = true;
                    sampler = std::make_shared<Sampler>(info);
                }

                {
                    std::vector<DescriptorLayoutData> data(4);
                    data[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_SAMPLER , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                    
                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.push_back(std::make_shared<DescriptorLayout>(info));
                    containsPingPong = false;
                }

                {
                    std::vector<DescriptorData2> data(4);
                    data[0] = DescriptorData2::createData(bloomBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 0 , 0);
                    data[1] = DescriptorData2::createData(bloomImages->fullView , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0);
                    data[2] = DescriptorData2::createData(sampler.get(), 2 , 0);
                    data[3] = DescriptorData2::createData(bloomImages->fullView , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 3 , 0);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = dlayouts[0].get();

                    dbuffers.push_back(std::make_shared<DescriptorBuffer2>(info));
                }

                downPipe = _implCreatePipe(input.downscale , sizeof(PushDown));
                upPipe = _implCreatePipe(input.upscal , sizeof(PushUp));

            }   
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create sources for Bloom");
            }

            const FunctionType function = [&](VkCommandBuffer& cmd) -> void{_implTranslateBloom(cmd);};
            {
                const auto result = _implAddFunction(function , FunctionName);
                MOX_CHECK_RESULT(result)
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(BloomSource)
}
}