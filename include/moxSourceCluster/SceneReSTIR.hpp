#pragma once

#include "../shaderStructures/shaderStructure.h"
#include "SourceCluster.hpp"

namespace mox
{
namespace Source
{
    /*

    SceneReSTIR.hlsl ( MOX_RESTIR_INCLUDED is the index of the set ) :

    [[vk::binding(0 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirGI> ReSTIRGIBuffer;       the reservoirs that THIS frame writes
    [[vk::binding(1 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirGI> prevReSTIRGIBuffer;   the reservoirs of the previous frame
    [[vk::binding(2 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirPT> ReSTIRPTBuffer;
    [[vk::binding(3 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirPT> prevReSTIRPTBuffer;
    [[vk::binding(4 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirSS> ReSTIRSSBuffer;       ReSTIR SSS ( the subsurface scattering )
    [[vk::binding(5 , MOX_RESTIR_INCLUDED)]] RWStructuredBuffer<ReservoirSS> prevReSTIRSSBuffer;

    ping pong : the pair is [0] and [1] , the descriptor buffer number id ( = frameIndex % 2 ) has the buffer [id] as the "next" one and [!id] as the "prev" one ,
    the same rule as moxPingPongIndex . the sources are ReSTIRGIBuffer_0 , ReSTIRGIBuffer_1 , ReSTIRPTBuffer_0 , ReSTIRPTBuffer_1 , ReSTIRSSBuffer_0 , ReSTIRSSBuffer_1 ( the name of the .mox file + _0 or _1 )

    */

    class SceneReSTIR final : public SourceCluster
    {
    public:

        SceneReSTIR() = delete;

        // device local , one reservoir per pixel of the screen . the CPU never reads them , so they are not readable ( a readable buffer would need the memory that is visible for the CPU too ,
        // and these are the biggest buffers of the engine ) . if the CPU has to read them one day , the type is the only thing to change
        using ReservoirBuffer = Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false>;

        std::array<ReservoirBuffer , 2> ReSTIRGIBuffer{};
        std::array<ReservoirBuffer , 2> ReSTIRPTBuffer{};
        std::array<ReservoirBuffer , 2> ReSTIRSSBuffer{};

        // the reservoirs have the FULL extent of the screen ( one reservoir per pixel ) : RayGen indexes them with bufferIndex ( the width of the screen ) . they were the half
        // of it when indirectImage was half . the cost at 1920 x 1080 : GI 64 B + PT 80 B + SS 80 B per pixel , x2 for the ping pong , about 930 MB
        VkExtent2D extent{};

        inline static const std::string Name = "SceneReSTIR";

        explicit SceneReSTIR(moxContext2* context) : SourceCluster(context , Name)
        {
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice | VkReqTypePhysicalDevice | VkReqTypeQueueManager | VkReqTypeSwapchain , context);
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

        ~SceneReSTIR() noexcept
        {
            processDelete();
        }

    private:

        void processDelete() noexcept override
        {
            for(auto& buffer : ReSTIRGIBuffer) buffer = nullptr;
            for(auto& buffer : ReSTIRPTBuffer) buffer = nullptr;
            for(auto& buffer : ReSTIRSSBuffer) buffer = nullptr;
            SourceCluster::processDelete();
        }

        // one buffer for every pixel of the half extent
        template<typename T>
        ReservoirBuffer _implCreateReservoirs() const
        {
            Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>::CreateInfo info{};
            info.context = context;
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            info.size = static_cast<VkDeviceSize>(sizeof(T)) * extent.width * extent.height;

            return std::make_shared<Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>>(info);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit() noexcept
        {
            try
            {
                // the same extent as the images of SceneImages ( at least 1 pixel )
                extent = {std::max(1u , context->sc->extent.width) , std::max(1u , context->sc->extent.height)};

                for(const auto& id : {0 , 1})
                {
                    ReSTIRGIBuffer[id] = _implCreateReservoirs<ReservoirGI>();
                    ReSTIRPTBuffer[id] = _implCreateReservoirs<ReservoirPT>();
                    ReSTIRSSBuffer[id] = _implCreateReservoirs<ReservoirSS>();
                }

                {
                    const std::array<Buffer::BufferBase2* , 6> all = {ReSTIRGIBuffer[0].get() , ReSTIRGIBuffer[1].get() , ReSTIRPTBuffer[0].get() , ReSTIRPTBuffer[1].get() , ReSTIRSSBuffer[0].get() , ReSTIRSSBuffer[1].get()};

                    StreamForgeFunctionType function = [&](VkCommandBuffer& cmd) -> void
                    {
                        std::array<VkBufferMemoryBarrier2 , 6> barriers{};
                        for(size_t i = 0 ; i < all.size() ; i++)
                        {
                            // translateBuffer also keeps the state of the buffer : the first barrier of NodePass will start from the transfer write
                            barriers[i] = all[i]->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
                        }

                        VkDependencyInfo info{};
                        info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                        info.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
                        info.pBufferMemoryBarriers = barriers.data();
                        vkCmdPipelineBarrier2(cmd , &info);

                        for(const auto& buffer : all)
                        {
                            vkCmdFillBuffer(cmd , buffer->buffer , 0 , buffer->getBufferSize() , 0);
                        }
                    };

                    const auto result = context->executeCommand(function);
                    if(!result.has_value()) return result;
                }

                // the sources : the pair is added once , the names are ReSTIRGIBuffer_0 , ReSTIRGIBuffer_1 and so on
                {
                    const auto result = _implAddSource(moxPingPongBufferBase2{ReSTIRGIBuffer[0].get() , ReSTIRGIBuffer[1].get()} , "ReSTIRGIBuffer");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(moxPingPongBufferBase2{ReSTIRPTBuffer[0].get() , ReSTIRPTBuffer[1].get()} , "ReSTIRPTBuffer");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(moxPingPongBufferBase2{ReSTIRSSBuffer[0].get() , ReSTIRSSBuffer[1].get()} , "ReSTIRSSBuffer");
                    MOX_CHECK_RESULT(result)
                }

                // the layout : 6 storage buffers
                {
                    std::vector<DescriptorLayoutData> data(6 , DescriptorLayoutData{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1});

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.resize(1);
                    dlayouts[0] = std::make_shared<DescriptorLayout>(info);
                }

                // two descriptor buffers , the number id is used in the frames with frameIndex % 2 == id : the buffer [id] is "next" and [!id] is "prev"
                dbuffers.resize(2);
                for(const auto& id : {0 , 1})
                {
                    std::vector<DescriptorData2> data{};
                    data.reserve(6);
                    data.push_back(DescriptorData2::createData(ReSTIRGIBuffer[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0));
                    data.push_back(DescriptorData2::createData(ReSTIRGIBuffer[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0));
                    data.push_back(DescriptorData2::createData(ReSTIRPTBuffer[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 2 , 0));
                    data.push_back(DescriptorData2::createData(ReSTIRPTBuffer[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0));
                    data.push_back(DescriptorData2::createData(ReSTIRSSBuffer[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0));
                    data.push_back(DescriptorData2::createData(ReSTIRSSBuffer[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 5 , 0));

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = dlayouts[0].get();

                    dbuffers[id] = std::make_shared<DescriptorBuffer2>(info);
                }

                containsPingPong = true;
            }
            catch(std::exception& e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create SceneReSTIR ( the reservoirs or the descriptor sets )"});
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(SceneReSTIR)
}
}
