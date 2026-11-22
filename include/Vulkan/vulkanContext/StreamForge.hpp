#pragma once

#include "Stream.hpp"
#include "SwapChain.hpp"
#include <functional>
#include <mutex>
#include <span>
#include <memory>
#include <vector>
#include <array>
#include "JobManager.hpp"

namespace mox
{

    // =============================== INFO ===============================
    // StreamForge - the jobs for the GPU , spread over the streams ( a stream = a queue , a pool , a fence , a semaphore )
    //
    //   addJob(function , type)            : the function that records the commands goes to the stream with the fewest jobs ( nothing is recorded yet , the GPU is not touched )
    //   executeJobs(type)                  : records every job of the type into its own command buffer , submits it , and returns WITHOUT waiting for the GPU
    //   executeJobsDetached(type)          : the same , but done by a thread of the JobManager ( the caller does not even wait for the recording )
    //   executeJobsWithSync(type)          : the streams run one AFTER another ( a stream waits for the semaphore of the previous one ) , and the call returns when everything is done
    //   executeSingular(function , type)   : one function , one command buffer , the call returns when the GPU has finished it
    //   waitForAll(type)                   : waits for everything that was submitted to the streams of the type
    //
    // the rules that the old version broke :
    //   -- a stream knows by itself if it has work on the GPU ( Lane::submitted ) , the fence is never waited for if nothing was submitted ( it would wait for ever )
    //   -- the command buffers of a submission are freed after its fence is waited for , the pool does not grow
    //   -- every stream has ITS OWN mutex : the compute streams and the graphics streams never block each other , and a detached compute and a detached graphics execution run at the same time
    //      ( the functions hold one lock at a time , only executeJobsWithSync holds all the lanes of ONE type , and it locks them in the order of the array , so there is no cycle )
    //   -- the mutex is NEVER held by the caller of a detached execution , the detached job takes it itself
    //   -- the last stream of a chain does not signal its semaphore ( nobody waits for it , and the next signal of a binary semaphore that is still signaled is an error )
    // =============================== END ===============================
    // gqCount / cqCount : how many graphics / compute streams ( queues ) the forge owns , the arrays have exactly this size

    using StreamForgeFunctionType = std::function<void(VkCommandBuffer&)>;

    template<uint32_t gqCount , uint32_t cqCount>
    requires(gqCount > 0 && cqCount > 0)
    class StreamForge final
    {
    public:


    private:

        struct Lane
        {
            std::shared_ptr<void> owner{};
            VkQueue queue{};
            VkCommandPool pool{};
            VkFence fence{};
            VkSemaphore signal{};

            // the lock of THIS stream only : the jobs , the fence , the pool and the in flight buffers of the lane are touched under it ( a pointer , because a mutex can not be moved )
            std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

            std::vector<StreamForgeFunctionType> jobs{};
            std::vector<VkCommandBuffer> inFlight{};
            bool submitted = false;
        };

    public:

        SetTypicalOperators(StreamForge)
        SetTypicalTransferOperators(StreamForge)

        struct CreateInfo
        {
            VkDevice logicalDevice{};
            VkPhysicalDevice physicalDevice{};
            VkSurfaceKHR surface{};
            VkInstance instance{};
            uint32_t cqOffset = 0;
            uint32_t gqOffset = 0;
        };

        explicit StreamForge(const CreateInfo& info)
        {
            const auto result = _implInit(info);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }


        void addJob(const StreamForgeFunctionType& function , const StreamType type) noexcept
        {
            _implAddJob(type , function);
        }

        [[nodiscard]] ErrorDataOutput<void> executeJobs(const StreamType type) noexcept
        {
            return _implExecuteJobs(type);
        }

        void executeJobsDetached(const StreamType type) noexcept
        {
            _implExecuteDetached(type);
        }

        // the streams one after another , returns when the GPU is done
        [[nodiscard]] ErrorDataOutput<void> executeJobsWithSync(const StreamType type) noexcept
        {
            return _implExecuteWithSync(type);
        }

        // one function , returns when the GPU is done
        [[nodiscard]] ErrorDataOutput<void> executeSingular(const StreamForgeFunctionType& function , const StreamType type) noexcept
        {
            return _implExecuteSingular(function , type);
        }

        void waitForAll(const StreamType type) noexcept
        {
            // the detached job first ( without the mutex : it takes it itself ) , then the GPU
            manager.waitFor(_implDetachedName(type));
            _implWaitForAll(type);
        }

        ~StreamForge() noexcept
        {
            processDelete();
        }

    private:

        void processDelete() noexcept
        {
            // the detached jobs have `this` inside : they have to end before anything of this object dies
            manager.waitForAll();

            if(device)
            {
                _implWaitForAll(StreamType::Compute);
                _implWaitForAll(StreamType::Graphics);
            }

            // the streams die with the lanes ( their destructor waits for the device )
            for(auto& l : cqLanes) l = {};
            for(auto& l : gqLanes) l = {};
            device = nullptr;
        }

        void processCopy(StreamForge&& input) noexcept
        {
            if(this == &input) return;
            processDelete();
            // the jobs of the other object have ITS `this` : they end before the move
            input.manager.waitForAll();
            // nobody uses the lanes of the other object now : its jobs ended , and a move is not allowed while another thread calls it
            cqLanes = std::move(input.cqLanes);
            gqLanes = std::move(input.gqLanes);
            device = std::exchange(input.device , nullptr);
        }

        std::array<Lane , cqCount> cqLanes{};
        std::array<Lane , gqCount> gqLanes{};
        JobManager manager{{}};
        VkDevice device{};

        // ------------------------------------------------------------------ helpers

        [[nodiscard]] std::span<Lane> _implLanes(const StreamType type) noexcept
        {
            return type == StreamType::Compute ? std::span<Lane>(cqLanes) : std::span<Lane>(gqLanes);
        }

        [[nodiscard]] static const char* _implDetachedName(const StreamType type) noexcept
        {
            return type == StreamType::Compute ? "StreamForge_execute_compute" : "StreamForge_execute_graphics";
        }

        template<StreamType type>
        static Lane _implMakeLane(const std::shared_ptr<Stream<type , 1>>& stream) noexcept
        {
            Lane lane{};
            lane.owner = stream;
            lane.queue = stream->queue;
            lane.pool = stream->pool;
            lane.fence = stream->fences[0];
            lane.signal = stream->signalSemaphore[0];
            return lane;
        }


        ErrorDataOutput<void> _implInit(const CreateInfo& input) noexcept
        {
            device = input.logicalDevice;
            try
            {
                {
                    // compute
                    Stream<StreamType::Compute , 1>::CreateInfo info{};
                    info.logicalDevice = input.logicalDevice;
                    info.physicalDevice = input.physicalDevice;
                    info.instance = input.instance;
                    info.surface = input.surface;
                    info.index = input.cqOffset;
                    for(auto& lane : cqLanes)
                    {
                        lane = _implMakeLane<StreamType::Compute>(std::make_shared<Stream<StreamType::Compute , 1>>(info));
                        info.index++;
                    }
                }

                {
                    // graphics : its own offset
                    Stream<StreamType::Graphics , 1>::CreateInfo info{};
                    info.logicalDevice = input.logicalDevice;
                    info.physicalDevice = input.physicalDevice;
                    info.instance = input.instance;
                    info.surface = input.surface;
                    info.index = input.gqOffset;
                    for(auto& lane : gqLanes)
                    {
                        lane = _implMakeLane<StreamType::Graphics>(std::make_shared<Stream<StreamType::Graphics , 1>>(info));
                        info.index++;
                    }
                }
            }
            catch(std::exception& e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create StreamForge"});
            }

            return {};
        }

        void _implWaitLane(Lane& lane) noexcept
        {
            if(!lane.submitted) return;

            vkWaitForFences(device , 1 , &lane.fence , VK_TRUE , UINT64_MAX);
            vkResetFences(device , 1 , &lane.fence);

            if(!lane.inFlight.empty())
            {
                vkFreeCommandBuffers(device , lane.pool , static_cast<uint32_t>(lane.inFlight.size()) , lane.inFlight.data());
                lane.inFlight.clear();
            }
            lane.submitted = false;
        }

        // every lane of the type , the other type is not touched . every lane is locked by itself , one at a time
        void _implWaitForAll(const StreamType type) noexcept
        {
            for(auto& lane : _implLanes(type))
            {
                std::lock_guard<std::mutex> guard(*lane.mutex);
                _implWaitLane(lane);
            }
        }

        // one command buffer for every function : allocate , begin , the function , end
        [[nodiscard]] ErrorDataOutput<std::vector<VkCommandBuffer>> _implRecord(Lane& lane , const std::vector<StreamForgeFunctionType>& jobs) noexcept
        {
            std::vector<VkCommandBuffer> cmds(jobs.size());

            {
                VkCommandBufferAllocateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                info.pNext = nullptr;
                info.commandPool = lane.pool;
                info.commandBufferCount = static_cast<uint32_t>(cmds.size());
                info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

                const auto result = vkAllocateCommandBuffers(device , &info , cmds.data());
                [[unlikely]] if(result != VK_SUCCESS)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create command buffers , error ID -> {}" , static_cast<int32_t>(result))});
                }
            }

            for(std::size_t i = 0 ; i < jobs.size() ; i++)
            {
                VkCommandBufferBeginInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                info.pNext = nullptr;
                info.pInheritanceInfo = nullptr;
                // every buffer is recorded and submitted exactly once
                info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

                VkResult result = vkBeginCommandBuffer(cmds[i] , &info);
                if(result == VK_SUCCESS)
                {
                    jobs[i](cmds[i]);
                    result = vkEndCommandBuffer(cmds[i]);
                }

                if(result != VK_SUCCESS)
                {
                    vkFreeCommandBuffers(device , lane.pool , static_cast<uint32_t>(cmds.size()) , cmds.data());
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to record the command buffer -> {} , error ID -> {}" , i , static_cast<int32_t>(result))});
                }
            }

            return cmds;
        }

        // vkQueueSubmit2 : the stages of the semaphores are written per semaphore . wait / signal can be null
        [[nodiscard]] ErrorDataOutput<void> _implSubmit(Lane& lane , std::vector<VkCommandBuffer>&& cmds , const VkSemaphore wait , const VkSemaphore signal) noexcept
        {
            std::vector<VkCommandBufferSubmitInfo> cmdInfos(cmds.size());
            for(std::size_t i = 0 ; i < cmds.size() ; i++)
            {
                cmdInfos[i] = {};
                cmdInfos[i].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
                cmdInfos[i].commandBuffer = cmds[i];
            }

            VkSemaphoreSubmitInfo waitInfo{};
            waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitInfo.semaphore = wait;
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            VkSemaphoreSubmitInfo signalInfo{};
            signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalInfo.semaphore = signal;
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            VkSubmitInfo2 info{};
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            info.pNext = nullptr;
            info.commandBufferInfoCount = static_cast<uint32_t>(cmdInfos.size());
            info.pCommandBufferInfos = cmdInfos.data();
            info.waitSemaphoreInfoCount = wait ? 1 : 0;
            info.pWaitSemaphoreInfos = wait ? &waitInfo : nullptr;
            info.signalSemaphoreInfoCount = signal ? 1 : 0;
            info.pSignalSemaphoreInfos = signal ? &signalInfo : nullptr;

            const auto result = vkQueueSubmit2(lane.queue , 1 , &info , lane.fence);
            if(result != VK_SUCCESS)
            {
                vkFreeCommandBuffers(device , lane.pool , static_cast<uint32_t>(cmds.size()) , cmds.data());
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to submit Queue , error ID -> {}" , static_cast<int32_t>(result))});
            }

            lane.inFlight = std::move(cmds);
            lane.submitted = true;
            return {};
        }

        // ------------------------------------------------------------------ add job : the stream with the fewest jobs . only the list is changed , a stream that is on the GPU is not touched

        void _implAddJob(const StreamType type , const StreamForgeFunctionType& function) noexcept
        {
            // the sizes are read one lock at a time ( never two locks at once ) : between the reading and the push another thread can add a job , the choice is only a hint for the balance
            Lane* best = nullptr;
            std::size_t bestSize = SIZE_MAX;
            for(auto& lane : _implLanes(type))
            {
                std::lock_guard<std::mutex> guard(*lane.mutex);
                if(lane.jobs.size() < bestSize)
                {
                    bestSize = lane.jobs.size();
                    best = &lane;
                }
            }

            std::lock_guard<std::mutex> guard(*best->mutex);
            best->jobs.push_back(function);
        }

        // ------------------------------------------------------------------ execute jobs : submit , do not wait for the GPU

        [[nodiscard]] ErrorDataOutput<void> _implExecuteJobs(const StreamType type) noexcept
        {
            for(auto& lane : _implLanes(type))
            {
                // one lane at a time : the compute lanes and the graphics lanes do not block each other
                std::lock_guard<std::mutex> guard(*lane.mutex);
                if(lane.jobs.empty()) continue;

                // the fence and the pool of the lane are free only when the previous submission is done
                _implWaitLane(lane);

                // the jobs are taken : if something fails , they are not repeated by the next call
                const std::vector<StreamForgeFunctionType> jobs = std::move(lane.jobs);
                lane.jobs.clear();

                auto cmds = _implRecord(lane , jobs);
                if(!cmds.has_value()) return std::unexpected(cmds.error());

                const auto result = _implSubmit(lane , std::move(cmds.value()) , nullptr , nullptr);
                MOX_CHECK_RESULT(result)
            }
            _implWaitForAll(type);

            return {};
        }


        void _implExecuteDetached(const StreamType type) noexcept
        {
            moxJobManagerFunctionType function = [this , type]() -> uint32_t
            {
                const auto result = _implExecuteJobs(type);
                if(!result.has_value())
                {
                    engineLogger(result);
                    return moxFunctionFinishedWithError;
                }
                return moxFunctionFinished;
            };

            // the caller holds NO mutex here : the job takes it in its own thread
            manager.addJob(_implDetachedName(type) , function);
        }


        [[nodiscard]] ErrorDataOutput<void> _implExecuteWithSync(const StreamType type) noexcept
        {
            // the chain needs all its lanes at once : they are locked ALWAYS in the order of the array ( the other functions hold one lock at a time , so no cycle is possible )
            std::vector<std::unique_lock<std::mutex>> locks{};
            for(auto& lane : _implLanes(type)) locks.emplace_back(*lane.mutex);

            // only the lanes that have work are in the chain
            std::vector<Lane*> chain{};
            for(auto& lane : _implLanes(type))
            {
                if(lane.jobs.empty()) continue;
                _implWaitLane(lane);
                chain.push_back(&lane);
            }

            VkSemaphore wait = nullptr;
            ErrorDataOutput<void> status{};

            for(std::size_t i = 0 ; i < chain.size() ; i++)
            {
                Lane& lane = *chain[i];

                const std::vector<StreamForgeFunctionType> jobs = std::move(lane.jobs);
                lane.jobs.clear();

                auto cmds = _implRecord(lane , jobs);
                if(!cmds.has_value())
                {
                    status = std::unexpected(cmds.error());
                    break;
                }

                const bool last = (i + 1 == chain.size());
                const VkSemaphore signal = last ? nullptr : lane.signal;

                const auto result = _implSubmit(lane , std::move(cmds.value()) , wait , signal);
                if(!result.has_value())
                {
                    status = std::unexpected(result.error());
                    break;
                }
                wait = signal;
            }

            // everything that was submitted is waited for ( also after an error : the semaphores of the lanes must be unsignaled again ) , the locks are already ours
            for(auto& lane : _implLanes(type)) _implWaitLane(lane);
            return status;
        }


        [[nodiscard]] ErrorDataOutput<void> _implExecuteSingular(const StreamForgeFunctionType& function , const StreamType type) noexcept
        {
            Lane& lane = _implLanes(type)[0];
            std::lock_guard<std::mutex> guard(*lane.mutex);
            _implWaitLane(lane);

            auto cmds = _implRecord(lane , std::vector<StreamForgeFunctionType>{function});
            if(!cmds.has_value()) return std::unexpected(cmds.error());

            const auto result = _implSubmit(lane , std::move(cmds.value()) , nullptr , nullptr);
            MOX_CHECK_RESULT(result)

            _implWaitLane(lane);
            return {};
        }
    };

    template<uint32_t gqCount , uint32_t cqCount>
    requires(gqCount > 0 && cqCount > 0)
    using pStreamForge = std::shared_ptr<StreamForge<gqCount , cqCount>>;
}
