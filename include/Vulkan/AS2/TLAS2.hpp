#pragma once

#include "BLAS2.hpp"

namespace mox
{

    class TLAS2 final : public AccelerationStructure
    {
    public:

        TLAS2() = delete;
        TLAS2(const TLAS2& input) = delete;
        TLAS2(TLAS2& input) = delete;
        TLAS2& operator=(const TLAS2& input) = delete;
        TLAS2& operator=(TLAS2& input) = delete;
        TLAS2& operator=(TLAS2&& input) = delete;

        
        TLAS2(moxContext2* context , std::vector<VkAccelerationStructureInstanceKHR>& instances) : AccelerationStructure(context)
        {   
            const auto result = _implCreateTLAS(instances);
            if(!result)
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        ~TLAS2()
        {
            processDelete();
        }

        MOX_NO_DISCARD ErrorDataOutput<void> update(VkCommandBuffer &cmd , std::vector<VkAccelerationStructureInstanceKHR> instances) noexcept
        {
            return _implUpdateTLAS(cmd , instances);
        }

    private:

        Buffer::pDeviceReadableBuffer buffer{};

        const uint32_t maxCount = 9999;

        void processDelete() noexcept
        {
            buffer = nullptr;
        }

        void processCopy(TLAS2&& input) noexcept
        {
            buffer = std::move(input.buffer);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implCreateTLAS(std::vector<VkAccelerationStructureInstanceKHR> &instances) noexcept
        {


            try{

                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
                    info.size = maxCount * sizeof(VkAccelerationStructureInstanceKHR);

                    buffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , Buffer::translateToSpan(instances));
                }

                const std::uint32_t primitiveCount = static_cast<uint32_t>(instances.size());

                VkAccelerationStructureGeometryKHR geometry{};
                geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                geometry.flags = 0;
                geometry.pNext = nullptr;
                geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
                geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
                geometry.geometry.instances.data.deviceAddress = buffer->pBuffer;
                geometry.geometry.instances.arrayOfPointers = VK_FALSE;
                geometry.geometry.instances.pNext = nullptr;
                

                VkAccelerationStructureBuildGeometryInfoKHR build{};
                build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                build.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                build.geometryCount = 1;
                build.dstAccelerationStructure = 0;
                build.srcAccelerationStructure = 0;
                build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                build.pGeometries = &geometry;
                build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

                VkAccelerationStructureBuildSizesInfoKHR sizes{};
                sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

                getAccelerationStructureBuildSizesKHR(device , VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR , &build ,&maxCount , &sizes  );

                {
                    Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.usage= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = sizes.accelerationStructureSize;

                    this->build = std::make_shared<Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>>(info);
                }

                {
                    Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>::CreateInfo info{};
                    info.context = context;
                    info.usage= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    info.size = align(std::max(sizes.buildScratchSize , sizes.updateScratchSize) , properties.value().minAccelerationStructureScratchOffsetAlignment) +properties.value().minAccelerationStructureScratchOffsetAlignment;

                    this->scratch = std::make_shared<Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>>(info);
                }

                build.scratchData.deviceAddress = alignedScratchAddress();

                {
                    VkAccelerationStructureCreateInfoKHR info{};
                    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                    info.createFlags = 0;
                    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
                    info.buffer = this->build->buffer;
                    info.size = sizes.accelerationStructureSize;
                    info.pNext = nullptr;
                    info.offset = 0;

                    auto result = createAccelerationStructureKHR(device , &info , nullptr , &AS);
                    if(result != VK_SUCCESS){
                        ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create BLAS as acceleration structure , error ID - > {}" , static_cast<int32_t>(result))};
                        return std::unexpected(error);
                    }

                    VkAccelerationStructureDeviceAddressInfoKHR add{};
                    add.accelerationStructure = AS;
                    add.pNext = nullptr;
                    add.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;

                    pAS = getAccelerationStructureDeviceAddressKHR(device , &add);
                }


                build.dstAccelerationStructure= AS;

                VkAccelerationStructureBuildRangeInfoKHR range{};

                range.firstVertex = 0;
                range.primitiveCount = primitiveCount;
                range.primitiveOffset = 0;
                range.transformOffset = 0;

                const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;

                context->queueManager->Begin();

                cmdFunctionType function = [&](VkCommandBuffer &cmd) -> void{
                    VkMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    barrier.pNext = nullptr;
                    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

                    vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);

                    mox::cmdBuildAccelerationStructuresKHR(cmd , 1 , &build , &ranges);

                    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

                    vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);
                };

                context->queueManager->recordCmd(function , VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);


                {
                    const auto result = context->queueManager->End();
                    if(!result.has_value()) return std::unexpected(result.error());
                }
            }
            catch(std::exception &e){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create BLAS, lol"});
            }

            return {};
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implUpdateTLAS(VkCommandBuffer &cmd , const std::vector<VkAccelerationStructureInstanceKHR> instances) noexcept
        {
            {
                const auto result = _implUpdateBuffer(cmd , instances);
                MOX_CHECK_RESULT(result)
            }
            
            VkAccelerationStructureGeometryKHR geometry{};
            geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometry.flags = 0;
            geometry.pNext = nullptr;
            geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            geometry.geometry.instances.data.deviceAddress = buffer->pBuffer;
            geometry.geometry.instances.arrayOfPointers = VK_FALSE;
            geometry.geometry.instances.pNext = nullptr;

            VkAccelerationStructureBuildGeometryInfoKHR build{};
            build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            build.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            build.geometryCount = 1;
            build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            build.pGeometries = &geometry;
            build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            build.scratchData.deviceAddress = alignedScratchAddress();

            build.dstAccelerationStructure= AS;
            build.srcAccelerationStructure = AS;

            VkAccelerationStructureBuildRangeInfoKHR range{};

            range.firstVertex = 0;
            range.primitiveCount = instances.size();
            range.primitiveOffset = 0;
            range.transformOffset = 0;

            const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;

            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.pNext = nullptr;
            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

            vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);

            mox::cmdBuildAccelerationStructuresKHR(cmd , 1 , &build , &ranges);

            barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

            vkCmdPipelineBarrier(cmd , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR , 0 , 1 , &barrier , 0 , nullptr , 0 , nullptr);
            return {};
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implUpdateBuffer(VkCommandBuffer& cmd, std::vector<VkAccelerationStructureInstanceKHR> instances) noexcept
        {

            if(instances.empty()) return {};

            buffer->updateBufferCmd(cmd , Buffer::translateToSpan(instances) , VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR , VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(TLAS2)
}