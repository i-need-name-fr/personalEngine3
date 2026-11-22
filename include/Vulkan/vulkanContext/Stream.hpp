#pragma once

#include "PhysicalDevice.hpp"
#include "FamilyDetails.hpp"
#include "Instance.hpp"
#include "../../logger/logger.hpp"
namespace mox
{   

    enum class StreamType
    {
        Compute,
        Graphics
    };

    // class stream is a brand new type of queue
    template<StreamType type , std::uint32_t allocCount>
    requires((allocCount > 0))
    class Stream final
    {
    public:

        SetTypicalOperators(Stream)
        SetTypicalTransferOperators(Stream)

        struct CreateInfo
        {
            VkDevice logicalDevice{};
            VkPhysicalDevice physicalDevice{};
            VkSurfaceKHR surface{};
            VkInstance instance{};
            uint32_t index{};
        };  

        explicit Stream(const CreateInfo &info)
        {
            const auto result = _implCreateBindinds(info);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }

            inputData = info;
        }

        VkQueue queue{};
        std::vector<VkSemaphore> waitSemaphore{};
        std::vector<VkSemaphore> signalSemaphore{};
        std::vector<VkFence> fences{};
        std::vector<VkCommandBuffer> cmds{};

        VkCommandPool pool{};

        ~Stream()
        {
            processDelete();
        }

    private:

        void processCopy(Stream&&  input) noexcept
        {
            queue = std::exchange(input.queue , nullptr);
            pool = std::exchange(input.pool , nullptr);
            fences = std::move(input.fences);
            cmds = std::move(input.cmds);
            waitSemaphore = std::move(input.waitSemaphore);
            signalSemaphore = std::move(input.signalSemaphore);
            inputData = std::exchange(input.inputData , {});
        }

        void processDelete() noexcept
        {   
            if(inputData.logicalDevice)
            {
                deviceWaitIdle(inputData.logicalDevice);
                for(uint32_t i = 0 ; i < allocCount ; i++)
                {
                    vkDestroyFence(inputData.logicalDevice , fences[i], nullptr);
                    vkDestroySemaphore(inputData.logicalDevice , waitSemaphore[i], nullptr);
                    vkDestroySemaphore(inputData.logicalDevice , signalSemaphore[i], nullptr);
                }

                cmds.clear();
                vkDestroyCommandPool(inputData.logicalDevice , pool , nullptr);
            }
            queue = nullptr;
            inputData = {};
            pool = nullptr;
            cmds.clear();
            fences.clear();
            waitSemaphore.clear();
            signalSemaphore.clear();

        }   

        CreateInfo inputData{};

        [[nodiscard]] [[gnu::cold]] ErrorDataOutput<void> _implCreateBindinds( const CreateInfo input) noexcept
        {   

            DetailsCreateInfo details{};
            details.device = input.physicalDevice;
            details.surface = input.surface;
            const auto data =queueFamilyDetails::queryQueueFamilyDetails(details);

            if(type == StreamType::Compute)
            {
                vkGetDeviceQueue(input.logicalDevice, data.compute.value() , input.index , &queue);
            }
            else
            {
                vkGetDeviceQueue(input.logicalDevice, data.graphics.value() , input.index , &queue);
            }

            {
                VkCommandPoolCreateInfo info{};
                info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                
                if(type == StreamType::Compute)
                {
                    info.queueFamilyIndex = data.compute.value();
                }
                else
                {
                    info.queueFamilyIndex = data.graphics.value();
                }

                info.pNext = nullptr;

                const auto result = vkCreateCommandPool(input.logicalDevice , &info , nullptr , &pool);
                if(result != VK_SUCCESS )
                {
                    return std::unexpected(ErrorDataType(MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create Command Pool , error id -> {}" , static_cast<int32_t>(result))));
                }
            }

            {
                {
                    cmds.resize(allocCount);
                    VkCommandBufferAllocateInfo info{};
                    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                    info.pNext = nullptr;
                    info.commandPool = pool;
                    info.commandBufferCount = allocCount;
                    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

                    const auto result = vkAllocateCommandBuffers(input.logicalDevice , &info , cmds.data());
                    [[unlikely]] if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create command buffers , error ID -> {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }
                }
            }

            fences.resize(allocCount);
            waitSemaphore.resize(allocCount);
            signalSemaphore.resize(allocCount);

            VkFenceCreateInfo fence{};
            fence.flags = 0;
            fence.pNext = nullptr;
            fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

            VkSemaphoreCreateInfo sem{};
            sem.flags = 0;
            sem.pNext = nullptr;
            sem.sType  = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            signalSemaphore.resize(allocCount);
            fences.resize(allocCount);

            for(uint8_t i = 0 ; i < allocCount ; i++){
                {
                    const auto result = vkCreateFence(input.logicalDevice , &fence , nullptr , &fences[i]);
                    if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create fence , error ID -> {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }
                }

                {
                    const auto result = vkCreateSemaphore(input.logicalDevice , &sem , nullptr , &signalSemaphore[i]);
                    if(result != VK_SUCCESS){
                        const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create semaphores , error ID -> {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }
                }

            }

            return {};
        }
    };


    template<StreamType type , std::uint32_t allocCount>
    requires((allocCount > 0))
    using pStream = std::shared_ptr<Stream<type , allocCount>>;
}