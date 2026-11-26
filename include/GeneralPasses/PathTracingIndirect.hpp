#pragma once

#include "../moxNode/NodePass.hpp"
#include "../Vulkan/Buffer2/SBT2.hpp"
#include "../Vulkan/ShaderModule2/ShaderModule2.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../moxSourceCluster/SceneReSTIR.hpp"
#include "../moxSourceCluster/IndirectExtent.hpp"

namespace mox
{
namespace Passes
{
    class PathTracingIndirect : public NodePass
    {
    public:

        struct CreateInfo
        {
            moxContext2* _Nonnull context;
            std::string directory = "Shaders/PathTracingIndirect/";
            std::string moxName = "pathTracingIndirect";
        };

        SetTypicalOperators(PathTracingIndirect)

        explicit PathTracingIndirect(const CreateInfo& info) : NodePass(info.context , "PathTracingIndirect")
        {
            const std::filesystem::path fileMOX = info.directory + info.moxName + ".mox";
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

        // only if ReSTIR PT is the source of the indirect light
        virtual bool requestExecution() const noexcept override
        {
            const auto features = requestFeatures();
            return features.PT != PT_NOT_REQUIRED;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            if(!pipe || !pipe->pipeline || !sbt) MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to execute PathTracingIndirect pass , since the pipeline or the SBT does not exist");

            // the sets first , then the barriers of the images that the pass reads and writes
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
            MOX_CHECK_RESULT(result)

            vkCmdBindPipeline(cmd , VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR , pipe->pipeline);

            // the indirect light has its own resolution ( the extent of the screen / MOX_INDIRECT_DIVISOR_RESOLUTION ) . the extent is read every frame : after a resize of the swapchain the size of the dispatch has to follow it
            const VkExtent2D extent = Source::indirectExtent(inputData.context->getResolutionExtent());
            const std::uint32_t width = extent.width;
            const std::uint32_t height = extent.height;

            sbt->traceRays(cmd , width , height , 1);

            return {};
        }

        ~PathTracingIndirect() noexcept
        {
            processDelete();
        }

    private:

        pPipeline2 pipe{nullptr};
        Buffer::pSBT2 sbt{nullptr};
        CreateInfo inputData{};
        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};

        void processDelete() noexcept override
        {
            requestFeatures = {};
            sbt = nullptr;
            pipe = nullptr;
            inputData = {};
            NodePass::processDelete();
        }

        void processCopy(PathTracingIndirect&& input) noexcept
        {
            pipe = std::move(input.pipe);
            sbt = std::move(input.sbt);
            inputData = input.inputData;
            requestFeatures = std::move(input.requestFeatures);
        }

        // a shader of the pipeline : the module is alive until the pipeline is created , and then it is not needed
        struct StageFile
        {
            std::string name;
            VkShaderStageFlagBits stage;
        };

        MOX_NO_DISCARD ErrorDataOutput<void> _implCreatePipe() noexcept
        {
            moxContext2* context = inputData.context;

            if(!createRayTracingPipelinesKHR) MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create PathTracingIndirect pass , since vkCreateRayTracingPipelinesKHR is not loaded");

            try
            {
                pipe = std::make_shared<Pipeline2>(Pipeline2::CreateInfo{.context = context , .cacheFile = {}});

                constexpr VkShaderStageFlags allStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
                const auto result = processPipe(pipe.get() , allStages);
                MOX_CHECK_RESULT(result)
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create the Pipeline or the layout for PathTracingIndirect pass , please check the logs above");
            }

            {
                const auto result = cSources[Source::SceneBuffers::Name].source->requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
                MOX_CHECK_RESULT(result)
                requestFeatures = result.value();
            }

            // ----------------------------------------------------------------- the shaders
            // the order of the stages in this list is the order of the indices in the groups below
            const std::string dir = inputData.directory;
            const std::array<StageFile , 13> files = {{
                {dir + "RayGen.hlsl"             , VK_SHADER_STAGE_RAYGEN_BIT_KHR},      // 0
                {dir + "shadowMiss.hlsl"         , VK_SHADER_STAGE_MISS_BIT_KHR},        // 1
                {dir + "GI_miss.hlsl"            , VK_SHADER_STAGE_MISS_BIT_KHR},        // 2
                {dir + "ReflectionMiss.hlsl"     , VK_SHADER_STAGE_MISS_BIT_KHR},        // 3
                {dir + "shadowHit.hlsl"          , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR}, // 4
                {dir + "shadow_any_hit.hlsl"     , VK_SHADER_STAGE_ANY_HIT_BIT_KHR},     // 5
                {dir + "GI_hit.hlsl"             , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR}, // 6
                {dir + "GI_any_hit.hlsl"         , VK_SHADER_STAGE_ANY_HIT_BIT_KHR},     // 7
                {dir + "ReflectionHit.hlsl"      , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR}, // 8
                {dir + "ReflectionAnyHit.hlsl"   , VK_SHADER_STAGE_ANY_HIT_BIT_KHR},     // 9
                {dir + "PTSecondaryMiss.hlsl"         , VK_SHADER_STAGE_MISS_BIT_KHR},        // 10
                {dir + "PTSecondaryHit.hlsl"          , VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR}, // 11
                {dir + "PTSecondaryAlphaAnyHit.hlsl"  , VK_SHADER_STAGE_ANY_HIT_BIT_KHR},     // 12
            }};

            std::vector<pShaderModule2> modules{};
            std::vector<VkPipelineShaderStageCreateInfo> stages{};
            modules.reserve(files.size());
            stages.reserve(files.size());

            try
            {
                for(const auto& file : files)
                {
                    modules.push_back(std::make_shared<ShaderModule2>(context , file.name));

                    VkPipelineShaderStageCreateInfo stage{};
                    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    stage.stage = file.stage;
                    stage.module = modules.back()->module;
                    stage.pName = "main";
                    stages.push_back(stage);
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to compile a shader of PathTracingIndirect pass , please check the logs above");
            }

            // ----------------------------------------------------------------- the groups ( the order = the order of groupTypes of the SBT )
            std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups{};
            std::vector<Buffer::SBTGroupType> groupTypes{};

            const auto general = [&](const uint32_t stageIndex , const Buffer::SBTGroupType type)
            {
                VkRayTracingShaderGroupCreateInfoKHR group{};
                group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                group.generalShader = stageIndex;
                group.closestHitShader = VK_SHADER_UNUSED_KHR;
                group.anyHitShader = VK_SHADER_UNUSED_KHR;
                group.intersectionShader = VK_SHADER_UNUSED_KHR;
                groups.push_back(group);
                groupTypes.push_back(type);
            };

            const auto hitGroup = [&](const uint32_t closestHit , const uint32_t anyHit)
            {
                VkRayTracingShaderGroupCreateInfoKHR group{};
                group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                group.generalShader = VK_SHADER_UNUSED_KHR;
                group.closestHitShader = closestHit;
                group.anyHitShader = anyHit;
                group.intersectionShader = VK_SHADER_UNUSED_KHR;
                groups.push_back(group);
                groupTypes.push_back(Buffer::SBTGroupType::Hit);
            };

            general(0 , Buffer::SBTGroupType::RayGeneration);
            // miss : 0 shadow , 1 GI , 2 reflection
            general(1 , Buffer::SBTGroupType::Miss);
            general(2 , Buffer::SBTGroupType::Miss);
            general(3 , Buffer::SBTGroupType::Miss);
            // 3 secondary bounce ( ReSTIR PT )
            general(10 , Buffer::SBTGroupType::Miss);
            // hit : 0 shadow , 1 GI , 2 reflection ( closest hit , any hit )
            hitGroup(4 , 5);
            hitGroup(6 , 7);
            hitGroup(8 , 9);
            hitGroup(11 , 12);

            VkRayTracingPipelineCreateInfoKHR info{};
            info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
            info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
            info.stageCount = static_cast<uint32_t>(stages.size());
            info.pStages = stages.data();
            info.groupCount = static_cast<uint32_t>(groups.size());
            info.pGroups = groups.data();
            // the recursion that the device can do : the pipeline can not ask for more than this
            uint32_t deviceRecursionLimit = 1u;
            {
                VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{};
                rtProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
                VkPhysicalDeviceProperties2 properties2{};
                properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                properties2.pNext = &rtProperties;
                vkGetPhysicalDeviceProperties2(context->physicalDevice->device , &properties2);
                deviceRecursionLimit = rtProperties.maxRayRecursionDepth;
            }
            // the secondary hit shader traces the next bounce itself ( MAX_PT_RAY_CAST bounces ) , and the shadow ray of the last one is one more level
            info.maxPipelineRayRecursionDepth = std::min(static_cast<uint32_t>(MAX_PT_RAY_CAST) + 2u , deviceRecursionLimit);
            info.layout = pipe->layout->layout;

            {
                const VkResult result = createRayTracingPipelinesKHR(context->logicalDevice->device , VK_NULL_HANDLE , pipe->loadPipelineCache() , 1 , &info , nullptr , &pipe->pipeline);
                if(result != VK_SUCCESS)
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the ray tracing pipeline of PathTracingIndirect pass , error ID -> {}" , static_cast<int32_t>(result)));
                }
            }

            // the modules are not needed any more
            modules.clear();

            try
            {
                Buffer::SBT2::CreateInfo sbtInfo{};
                sbtInfo.context = context;
                sbtInfo.pipeline = pipe->pipeline;
                sbtInfo.groupTypes = groupTypes;
                sbt = std::make_shared<Buffer::SBT2>(sbtInfo);
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the SBT of PathTracingIndirect pass , please check the logs above");
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(PathTracingIndirect)
}
}
