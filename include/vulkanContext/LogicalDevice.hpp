#pragma once

#include "PhysicalDevice.hpp"
#include <map>
#include <algorithm>

namespace mox{

    class LogicalDevice final{
    private:
        
        [[nodiscard]] ErrorDataOutput<void> createLogicalDevice(VkPhysicalDevice pDevice , VkSurfaceKHR surface, const uint32_t queueCount , const uint32_t extraGraphics , const uint32_t extraCompute){
            DetailsCreateInfo det{};
            det.device = pDevice;
            det.surface = surface;
            auto details = queueFamilyDetails::queryQueueFamilyDetails(det);

            // family -> how many queues of it are created . the engine takes queueCount queues from the graphics family and from the present family ( the same family in most cases ) ,
            // Streamline asks for extra graphics queues ( from the graphics family , AFTER the ones of the engine ) and extra compute queues ( from the compute only family )
            std::map<uint32_t , uint32_t> counts;
            counts[details.graphics.value()] = queueCount + extraGraphics;
            counts[details.present.value()] = std::max(counts[details.present.value()] , queueCount);

            // where the queues of SL start in their families : slSetVulkanInfo needs it
            slGraphicsFamily = details.graphics.value();
            slGraphicsIndex = queueCount;
            if(extraCompute > 0)
            {
                const uint32_t family = details.compute.value();
                const auto found = counts.find(family);
                slComputeFamily = family;
                slComputeIndex = (found != counts.end()) ? found->second : 0;
                counts[family] = slComputeIndex + extraCompute;
            }

            // a family can not give more queues than it has
            {
                uint32_t familyCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(pDevice , &familyCount , nullptr);
                std::vector<VkQueueFamilyProperties> properties(familyCount);
                vkGetPhysicalDeviceQueueFamilyProperties(pDevice , &familyCount , properties.data());
                for(const auto& [family , count] : counts)
                {
                    if(family >= familyCount || count > properties[family].queueCount)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("the queue family {} has {} queues , {} are required ( the engine and Streamline together )" , family , (family < familyCount ? properties[family].queueCount : 0u) , count)});
                    }
                }
            }

            uint32_t maxCount = 1;
            for(const auto& [family , count] : counts) maxCount = std::max(maxCount , count);
            std::vector<float> priority(maxCount , 1.0f);

            std::vector<VkDeviceQueueCreateInfo> queues;
            for(const auto& [family , count] : counts){
                VkDeviceQueueCreateInfo info{};
                info.queueFamilyIndex = family;
                info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                info.pQueuePriorities = priority.data();
                info.queueCount = count;
                info.flags = 0;
                info.pNext = nullptr;
                queues.push_back(info);
            }

            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
            atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
            atomicFloat.shaderBufferFloat32Atomics = VK_TRUE;
            atomicFloat.shaderBufferFloat32AtomicAdd = VK_TRUE;
            atomicFloat.shaderSharedFloat32Atomics = VK_TRUE;
            atomicFloat.shaderSharedFloat32AtomicAdd = VK_TRUE;

            VkPhysicalDeviceCooperativeVectorFeaturesNV vector{};
            vector.cooperativeVector = VK_TRUE;
            vector.cooperativeVectorTraining = VK_TRUE;
            vector.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
            vector.pNext = &atomicFloat;

            VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT helperIn{};
            helperIn.shaderDemoteToHelperInvocation = VK_TRUE;
            helperIn.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT;
            helperIn.pNext = &vector;

            VkPhysicalDeviceMeshShaderFeaturesEXT meshShading{};
            meshShading.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
            meshShading.meshShader= VK_TRUE;
            meshShading.taskShader=  VK_TRUE;
            meshShading.multiviewMeshShader = VK_FALSE;
            meshShading.primitiveFragmentShadingRateMeshShader = VK_TRUE;
            meshShading.pNext = &helperIn;

            VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional{};
            conditional.conditionalRendering = VK_TRUE;
            conditional.pNext = &meshShading;
            conditional.sType= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT;

            VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR images{};
            images.pNext = &conditional;
            images.unifiedImageLayouts = VK_TRUE;
            images.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR;

            VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRate{};
            shadingRate.primitiveFragmentShadingRate = VK_TRUE;
            shadingRate.attachmentFragmentShadingRate = VK_TRUE;
            shadingRate.pipelineFragmentShadingRate = VK_TRUE;
            shadingRate.pNext = &images;
            shadingRate.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;

            VkPhysicalDeviceSynchronization2Features sync2Features{};
            sync2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
            sync2Features.pNext = &shadingRate;
            sync2Features.synchronization2 = VK_TRUE;

            VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBuffer{};
            descriptorBuffer.descriptorBuffer = VK_TRUE;
            descriptorBuffer.descriptorBufferCaptureReplay = VK_TRUE;
            descriptorBuffer.pNext = &sync2Features;
            descriptorBuffer.descriptorBufferPushDescriptors = VK_TRUE;
            descriptorBuffer.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;

            VkPhysicalDeviceVulkan12Features vulkan12{};
            vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

            // vkCmdDrawIndexedIndirectCount
            vulkan12.drawIndirectCount = VK_TRUE;

            vulkan12.timelineSemaphore = VK_TRUE;
            vulkan12.bufferDeviceAddress = VK_TRUE;
            // half in the shader : the NRC weights and activations are fp16 ( tensor cores )
            vulkan12.shaderFloat16 = VK_TRUE;

            // the extension is enabled , so this has to be on as well
            vulkan12.descriptorIndexing = VK_TRUE;
            // Texture2D tex[] in the shader - an array with no compile time size
            vulkan12.runtimeDescriptorArray = VK_TRUE;
            // indexing that array with a value that differs between lanes
            vulkan12.shaderSampledImageArrayNonUniformIndexing  = VK_TRUE;
            vulkan12.shaderStorageImageArrayNonUniformIndexing  = VK_TRUE;
            vulkan12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
            vulkan12.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
            // slots may stay unwritten as long as the shader does not read them
            vulkan12.descriptorBindingPartiallyBound = VK_TRUE;
            // the real length is given at allocation time
            vulkan12.descriptorBindingVariableDescriptorCount = VK_TRUE;
            // rewrite a descriptor while the set is still bound
            vulkan12.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
            vulkan12.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
            vulkan12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
            vulkan12.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
            vulkan12.descriptorBindingUpdateUnusedWhilePending     = VK_TRUE;
            vulkan12.pNext = &descriptorBuffer;

            VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
            shaderObjectFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
            shaderObjectFeatures.shaderObject = VK_TRUE;
            shaderObjectFeatures.pNext = &vulkan12;

            VkPhysicalDeviceDynamicRenderingFeatures dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
            dynamic.dynamicRendering = VK_TRUE;
            dynamic.pNext = &shaderObjectFeatures;

            VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
            asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            asFeatures.accelerationStructure = VK_TRUE;
            asFeatures.pNext = &dynamic;

            VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeature{};
            rayTracingFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
            rayTracingFeature.pNext = &asFeatures;
            rayTracingFeature.rayTracingPipeline = VK_TRUE;

            VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeature{};
            rayQueryFeature.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            rayQueryFeature.rayQuery = VK_TRUE;
            rayQueryFeature.pNext    = &rayTracingFeature;

            VkPhysicalDeviceFeatures features{};
            features.samplerAnisotropy = VK_TRUE;
            features.fragmentStoresAndAtomics = VK_TRUE;
            features.vertexPipelineStoresAndAtomics = VK_TRUE;
            features.drawIndirectFirstInstance = VK_TRUE;
            features.multiDrawIndirect = VK_TRUE;
            features.depthBiasClamp = VK_TRUE;
            features.shaderInt64 = VK_TRUE;

            VkDeviceCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            info.flags = 0;
            info.pEnabledFeatures= &features;
            info.pNext = &rayQueryFeature;
            info.enabledExtensionCount = static_cast<uint32_t>(requiredExt.size());
            info.ppEnabledExtensionNames = requiredExt.data();
            info.pQueueCreateInfos = queues.data();
            info.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
            info.enabledLayerCount = (enableValidations ? static_cast<uint32_t>(validationLayers.size()) : 0);
            info.ppEnabledLayerNames = (enableValidations ? validationLayers.data() : nullptr);

            const auto result = createVkDevice(pDevice , &info , nullptr , &device);

            if(result != VK_SUCCESS){
                ErrorDataType error{MOX_ERROR_TYPE_CRITICAL , std::format("couldnt create the logical device , error ID -> {}" , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }
            return {};
        }

    public: 

        LogicalDevice() = delete;

        struct logicalDeviceCreateInfo {
            VkPhysicalDevice device;
            VkSurfaceKHR surface;
            uint32_t requiredForGraphics;
            uint32_t requiredForMultiThread;
            // the extra queues that Streamline asked for ( StreamlineContext::extraGraphicsQueues / extraComputeQueues ) , 0 = no Streamline
            uint32_t extraGraphicsQueues = 0;
            uint32_t extraComputeQueues = 0;
        };

        LogicalDevice(const logicalDeviceCreateInfo& info){
            if(!info.device || !info.surface) throw std::runtime_error("couldnt create the logicalDevice , since the info is nullptr");
            if(info.requiredForGraphics + info.requiredForMultiThread + info.extraGraphicsQueues > 16) throw std::runtime_error("couldnt create the logical device since the required queues is more than 16");
            auto result = createLogicalDevice(info.device , info.surface,  info.requiredForGraphics + info.requiredForMultiThread , info.extraGraphicsQueues , info.extraComputeQueues);
            if(result.has_value()){
                ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , "logical device has been created "};
                engineLogger(&res);
            }else{
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }


        LogicalDevice(LogicalDevice& input) = delete;
        LogicalDevice(LogicalDevice&& input){
            device = std::exchange(input.device , nullptr);
        }

        LogicalDevice& operator=(LogicalDevice& input) = delete;
        LogicalDevice& operator=(LogicalDevice&& input){
            device = std::exchange(input.device , nullptr);
            return *this;
        }

        VkDevice device = nullptr;

        // where the queues of Streamline are : the family and the index of the FIRST queue of SL in it ( slSetVulkanInfo ) . the queues of the engine are before it
        uint32_t slGraphicsFamily = 0;
        uint32_t slGraphicsIndex = 0;
        uint32_t slComputeFamily = 0;
        uint32_t slComputeIndex = 0;

        ~LogicalDevice(){
            if(device) vkDestroyDevice(device , nullptr);
            ErrorDataType res{MOX_ERROR_TYPE_NONE , "deleting the logical device"};
            engineLogger(&res);
        }
    };
}