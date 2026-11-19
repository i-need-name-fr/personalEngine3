#pragma once

#include "../moxSourceCluster/SourceCluster.hpp"
#include "../Vulkan/AS2/TLAS2.hpp"
#include "../moxSourceCluster/LightHub.hpp"
#include "../moxSourceCluster/TextureHub.hpp"
#include <span>
#include <array>
#include <mutex>
#include <algorithm>
#include <limits>

namespace mox
{
    class Entity; // it records its data into the heap ( friend of VertexHeap )

    // NotStarted - Begin() was not called yet , Recording - between Begin() and End() ( append works ) , Finished - End() was called , the heap is closed
    enum class VertexHeapStatus : uint8_t
    {
        NotStarted = 0,
        Recording = 1,
        Finished = 2
    };

    struct VertexHeapStart
    {
        moxContext2* context  = nullptr;
        Source::TextureHub* txHub = nullptr;
        Source::LightHub* ltHub = nullptr;
    };

    #define MOX_MAX_INSTANCE_COUNT 9999ul
    
namespace Source
{
    class VertexHeap : public SourceCluster
    {
        // Entity does not inherit the heap ( the constructor of the heap works only after End() ) , it uses the static recording functions of the protected part
        friend class mox::Entity;

    public:

        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> vertexBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> indicesBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , true> instanceBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , true> instanceLayoutBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , true> LUT_buffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , true> MeshBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> meshletBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> meshletVertexBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> meshletIndexBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> meshletBoundBuffer{nullptr};
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> geometryBuffer{nullptr};
        inline static pTLAS2 tlas{nullptr};

        inline static std::vector<pBLAS2> blases{};

        MOX_NO_DISCARD inline static ErrorDataOutput<void> Begin(VertexHeapStart &input) noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            static bool firstIteration = true;
            if(firstIteration)
            {
                firstIteration = false;
                allowToRecord = true;
            }
            else
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to begin the recording of Vertex Heap , since the recodring is over"});
            }

            {
                const auto result = checkRequirements(VkReqTypeInstance | VkReqTypePhysicalDevice | VkReqTypeLogicalDevice , input.context);
                if(!result.has_value()) return result;
            }

            if(!input.ltHub)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to begin recording , since the input create info contains nullptr"});
            }

            if(!input.txHub)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to begin recording , since the input create info contains nullptr"});
            }

            txHub = input.txHub;
            ltHub = input.ltHub;
            context = input.context;

            startedToRecord = true;
            return {};
        }

        explicit VertexHeap() : SourceCluster(_implGetContext() , "VertexHeap")
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);

            const VertexHeapStatus status = getStatus();
            if(status != VertexHeapStatus::Finished)
            {
                const char* reason = (status == VertexHeapStatus::NotStarted) ? "Begin() was not called" : "End() was not called ( or it failed ) , the recording is still on";
                const ErrorDataType error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create Vertex Heap , it can be created only after a successful End() , reason -> {}" , reason)};
                engineLogger(&error);
                THROW_MESSAGE;
            }

            const auto result = _implInitDescriptors();
            if(!result.has_value())
            {
                engineLogger(&result.error());
                processDelete();
                THROW_MESSAGE;
            }
        }

        MOX_NO_DISCARD inline static ErrorDataOutput<void> End() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            if(!startedToRecord)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to end the recording of Vertex Heap , since the recording was not started or is already over"});
            }

            try
            {
                constexpr VkBufferUsageFlags meshletUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                constexpr VkBufferUsageFlags dynamicUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                // the static ones : the size is exactly the recorded data , they are written once
                vertexBuffer = _implCreateStatic(vertices , VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
                indicesBuffer = _implCreateStatic(indices , VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
                meshletBuffer = _implCreateStatic(meshlets , meshletUsage);
                meshletVertexBuffer = _implCreateStatic(mVertices , meshletUsage);
                meshletIndexBuffer = _implCreateStatic(mIndices , meshletUsage);
                meshletBoundBuffer = _implCreateStatic(bounds , meshletUsage);
                geometryBuffer = _implCreateStatic(geometries , meshletUsage);

                instanceLayoutBuffer = _implCreateDynamic<InstanceData>(dynamicUsage);
                MeshBuffer = _implCreateDynamic<MeshData>(dynamicUsage);
                instanceBuffer = _implCreateDynamic<glm::mat4>(dynamicUsage);
                LUT_buffer = _implCreateDynamic<LUT>(dynamicUsage);

                _implRecord(LUT_buffer , luts);
                _implRecord(MeshBuffer , meshs);
                _implRecord(instanceBuffer , instances);
                _implRecord(instanceLayoutBuffer , instanceLayout);

                for(const auto& layout : instanceLayout)
                {
                    const uint32_t id = layout.BLAS_index;

                    if(id >= asLayouts.size())
                    {
                        const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create BLAS in End() of Vertex Heap , the BLAS index is bigger than the number of AS layouts , BLAS index -> {} , AS layouts -> {}" , id , asLayouts.size())};
                        engineLogger(&error);
                        continue;
                    }

                    if(blases.size() <= id) blases.resize(id + 1);
                    if(blases[id]) continue;

                    BLAS2::CreateInfo info{};
                    info.context = context;
                    info.vertexBuffer = vertexBuffer.get();
                    info.indicesBuffer = indicesBuffer.get();
                    info.layout = asLayouts[id];

                    blases[id] = std::make_shared<BLAS2>(info);
                }

                {
                    auto data = _implGetStructures();
                    tlas = std::make_shared<TLAS2>(context , data);
                }

                needsUpdateFlag = false;
            }
            catch(std::exception &e)
            {
                Destroy();
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create Buffers and so on for VertexHeap"});
            }
            startedToRecord = false;
            allowToRecord = false;

            return {};
        }

        inline static void Update(VkCommandBuffer& cmd) noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            if(!cmd) return;
            if(!needsUpdateFlag) return;
            if(getStatus() != VertexHeapStatus::Finished) return;
            if(!LUT_buffer || !MeshBuffer || !instanceBuffer || !instanceLayoutBuffer || !tlas) return;

            constexpr VkAccessFlags2 read = VK_ACCESS_2_SHADER_READ_BIT;
            constexpr VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            LUT_buffer->updateBufferCmd(cmd , Buffer::translateToSpan(luts) , read , stage);
            MeshBuffer->updateBufferCmd(cmd , Buffer::translateToSpan(meshs) , read , stage);
            instanceBuffer->updateBufferCmd(cmd , Buffer::translateToSpan(instances) , read , stage);
            instanceLayoutBuffer->updateBufferCmd(cmd , Buffer::translateToSpan(instanceLayout) , read , stage);
            auto data = _implGetStructures();
            // TLAS2::update takes the vector by value : the move saves one copy ( up to MOX_MAX_INSTANCE_COUNT instances every update )
            const auto result = tlas->update(cmd , std::move(data));
            if(!result.has_value())
            {
                // the flag stays , the update is tried again in the next frame
                static uint32_t messagesLeft = 10;
                if(messagesLeft > 0)
                {
                    engineLogger(&result.error());
                    messagesLeft--;
                }
                return;
            }

            needsUpdateFlag = false;
        }

        // the buffers are static , so it is static too ( End() uses it if the creation fails ) . the descriptors of the cluster are cleaned by the SourceCluster itself
        inline static void Destroy() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            tlas = nullptr;
            blases.clear();
            vertexBuffer = nullptr;
            indicesBuffer = nullptr;
            LUT_buffer = nullptr;
            instanceBuffer = nullptr;
            instanceLayoutBuffer = nullptr;
            MeshBuffer = nullptr;
            meshletBuffer = nullptr;
            meshletBoundBuffer = nullptr;
            meshletVertexBuffer = nullptr;
            meshletIndexBuffer=  nullptr;
            geometryBuffer = nullptr;
        }

    protected:

        // getOffset , vertex , indices , meshet .... 

        inline static uint32_t vertexOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return vertices.size();
        }
        inline static uint32_t indicesOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return indices.size();
        }
        inline static uint32_t lutsOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return luts.size();
        }
        inline static uint32_t meshOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return meshs.size();
        }
        inline static uint32_t instanceOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return instances.size();
        }
        inline static uint32_t instanceLOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return instanceLayout.size();
        }
        inline static uint32_t geometryOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return geometries.size();
        }
        inline static uint32_t meshletOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return meshlets.size();
        }
        inline static uint32_t boundOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return bounds.size();
        }
        inline static uint32_t mVertexOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return mVertices.size();
        }
        inline static uint32_t mIndicesOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return mIndices.size();
        }
        inline static uint32_t asLayoutsOffset() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return asLayouts.size();
        }



        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendVertices(const std::span<const Vertex> data) noexcept {return appendTo(vertices , data , "vertices");}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendIndices(const std::span<const uint32_t> data) noexcept {return appendTo(indices , data , "indices");}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendLUT(const LUT& lut) noexcept {return appendDynamic(luts , std::span<const LUT>(&lut , 1) , "LUT" , MOX_MAX_INSTANCE_COUNT);}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendMesh(const MeshData& mesh) noexcept {return appendDynamic(meshs , std::span<const MeshData>(&mesh , 1) , "mesh" , MOX_MAX_INSTANCE_COUNT);}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendInstance(const glm::mat4& instance) noexcept {return appendDynamic(instances , std::span<const glm::mat4>(&instance , 1) , "instance" , MOX_MAX_INSTANCE_COUNT);}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendInstanceLayout(const InstanceData& layout) noexcept {return appendDynamic(instanceLayout , std::span<const InstanceData>(&layout , 1) , "instance layout" , MOX_MAX_INSTANCE_COUNT);}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendMeshlets(const std::span<const ms::Meshlet> data) noexcept {return appendTo(meshlets , data , "meshlets");}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendBounds(const std::span<const ms::MeshletBound> data) noexcept {return appendTo(bounds , data , "meshlet bounds");}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendMeshletVertices(const std::span<const uint32_t> data) noexcept {return appendTo(mVertices , data , "meshlet vertices");}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendMeshletIndices(const std::span<const uint32_t> data) noexcept {return appendTo(mIndices , data , "meshlet indices");}
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendASLayout(const BLAS2_layout& layout) noexcept {return appendTo(asLayouts , std::span<const BLAS2_layout>(&layout , 1) , "AS layout");}

        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendGeometry(const ms::GeometryData& geometry) noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            const auto result = appendTo(geometries , std::span<const ms::GeometryData>(&geometry , 1) , "geometry");
            if(!result.has_value()) return result; // nothing was added , so the maximum stays the same

            maxGeometry = std::max(maxGeometry , geometry.meshletCount);
            return result;
        }

        MOX_NO_DISCARD inline static ErrorDataOutput<void> changeMesh(const uint32_t index , const MeshData& mesh) noexcept {return changeIn(meshs , index , mesh , "mesh");}
        MOX_NO_DISCARD inline static ErrorDataOutput<void> changeLUT(const uint32_t index , const LUT& lut) noexcept {return changeIn(luts , index , lut , "LUT");}
        MOX_NO_DISCARD inline static ErrorDataOutput<void> changeInstance(const uint32_t index , const glm::mat4& instance) noexcept {return changeIn(instances , index , instance , "instance");}
        MOX_NO_DISCARD inline static ErrorDataOutput<void> changeInstanceLayout(const uint32_t index , const InstanceData& layout) noexcept {return changeIn(instanceLayout , index , layout , "instance layout");}

        inline static const auto& verticesAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return vertices.at(index);
        }
        inline static const auto& indicesAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return indices.at(index);
        }
        inline static const auto& lutsAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return luts.at(index);
        }
        inline static const auto& meshsAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return meshs.at(index);
        }
        inline static const auto& instancesAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return instances.at(index);
        }
        inline static const auto& instanceLayoutAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return instanceLayout.at(index);
        }
        inline static const auto& geometriesAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return geometries.at(index);
        }
        inline static const auto& meshletsAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return meshlets.at(index);
        }
        inline static const auto& boundsAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return bounds.at(index);
        }
        inline static const auto& mVerticesAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return mVertices.at(index);
        }
        inline static const auto& mIndicesAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return mIndices.at(index);
        }
        inline static const auto& asLayoutsAt(const uint32_t index)
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return asLayouts.at(index);
        }

        MOX_NO_DISCARD inline static VertexHeapStatus getStatus() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            if(startedToRecord) return VertexHeapStatus::Recording;
            return allowToRecord ? VertexHeapStatus::NotStarted : VertexHeapStatus::Finished;
        }

        MOX_NO_DISCARD inline static uint32_t getMaxGeometry() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return maxGeometry;
        }

        // needsUpdate : only LUT , mesh data , instance and instance layout have it ( append and change of them set it ) , the vertices , indices , meshlets and so on are uploaded once in End
        // the one who uploads the four buffers calls markUpdated() after the upload
        MOX_NO_DISCARD inline static bool needsUpdate() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return needsUpdateFlag;
        }

        inline static void markUpdated() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            needsUpdateFlag = false;
        }

        inline static moxContext2* context = nullptr;
        inline static Source::TextureHub* txHub = nullptr;
        inline static Source::LightHub* ltHub = nullptr;


    private:
        inline static std::vector<LUT> luts{};
        inline static std::vector<Vertex> vertices{};
        inline static std::vector<uint32_t> indices{};
        inline static std::vector<MeshData> meshs{};
        inline static std::vector<glm::mat4> instances{};
        inline static std::vector<InstanceData> instanceLayout{};
        inline static std::vector<ms::GeometryData> geometries{};
        inline static std::vector<ms::Meshlet> meshlets{};
        inline static std::vector<ms::MeshletBound> bounds{};
        inline static std::vector<uint32_t> mVertices;
        inline static std::vector<uint32_t> mIndices;
        inline static std::vector<BLAS2_layout> asLayouts{};  

        inline static bool startedToRecord = false;
        inline static bool allowToRecord = true;
        inline static uint32_t maxGeometry = 0;

        // true = LUT , mesh data , instance or instance layout got new data or was changed : their GPU buffers are older than the CPU copy ( all four are updated together ) , cleared only by markUpdated()
        inline static bool needsUpdateFlag = false;

        // recursive : the public functions call each other ( appendGeometry -> appendTo , End -> Destroy , the constructor -> getStatus ) and every one of them locks
        inline static std::recursive_mutex heapMutex{};

        template<typename T>
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendTo(std::vector<T>& storage , const std::span<const T> data , const char* name , const size_t limit = std::numeric_limits<uint32_t>::max()) noexcept
        {
            try
            {
                std::lock_guard<std::recursive_mutex> guard(heapMutex);
                if(!startedToRecord || !allowToRecord)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to append {} to Vertex Heap , since the recording is not started or is already over" , name)});
                }

                const size_t offset = storage.size();
                if(offset + data.size() > limit)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to append {} to Vertex Heap , since the limit is reached , size -> {} , adding -> {} , limit -> {}" , name , offset , data.size() , limit)});
                }

                if(!data.empty()) storage.insert(storage.end() , data.begin() , data.end());
                return static_cast<uint32_t>(offset);
            }
            catch(std::exception& e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to append {} to Vertex Heap , exception" , name)});
            }
        }

        // for the data that changes at runtime ( LUT , mesh data , instance , instance layout ) : the same as appendTo + the needsUpdate flag
        template<typename T>
        MOX_NO_DISCARD inline static ErrorDataOutput<uint32_t> appendDynamic(std::vector<T>& storage , const std::span<const T> data , const char* name , const size_t limit = std::numeric_limits<uint32_t>::max()) noexcept
        {
            // the append and the flag in ONE critical section , so Update() never sees the new data without the flag
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            const auto result = appendTo(storage , data , name , limit);
            if(result.has_value() && !data.empty()) needsUpdateFlag = true;
            return result;
        }

        template<typename T>
        MOX_NO_DISCARD inline static ErrorDataOutput<void> changeIn(std::vector<T>& storage , const uint32_t index , const T& value , const char* name) noexcept
        {
            try
            {
                std::lock_guard<std::recursive_mutex> guard(heapMutex);
                if(index >= storage.size())
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to change {} in Vertex Heap , since the index is out of range , index -> {} , size -> {}" , name , index , storage.size())});
                }

                storage[index] = value;
                needsUpdateFlag = true;
                return {};
            }
            catch(std::exception& e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to change {} in Vertex Heap , exception" , name)});
            }
        }

        template<typename T>
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , false> _implCreateStatic(std::vector<T>& data , const VkBufferUsageFlags usage)
        {
            Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>::CreateInfo info{};
            info.context = context;
            info.size = sizeof(T) * std::max<size_t>(data.size() , 1);
            info.usage = usage;

            return std::make_shared<Buffer::GeneralBuffer2<Buffer::BufferType::Device , false>>(info , Buffer::translateToSpan(data));
        }

        template<typename T>
        inline static Buffer::pGeneralBuffer2<Buffer::BufferType::Device , true> _implCreateDynamic(const VkBufferUsageFlags usage)
        {
            Buffer::GeneralBuffer2<Buffer::BufferType::Device , true>::CreateInfo info{};
            info.context = context;
            info.size = sizeof(T) * MOX_MAX_INSTANCE_COUNT;
            info.usage = usage;

            return std::make_shared<Buffer::GeneralBuffer2<Buffer::BufferType::Device , true>>(info);
        }

        template<typename T>
        inline static void _implRecord(Buffer::pGeneralBuffer2<Buffer::BufferType::Device , true>& buffer , std::vector<T>& data) noexcept
        {
            if(!buffer || data.empty()) return;
            buffer->recordData(Buffer::translateToSpan(data));
        }

        MOX_NO_DISCARD inline static moxContext2* _implGetContext() noexcept
        {
            std::lock_guard<std::recursive_mutex> guard(heapMutex);
            return context;
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInitDescriptors() noexcept
        {
            try
            {
                struct Slot
                {
                    Buffer::BufferBase2* buffer;
                    const char* name;
                };

                const std::array<Slot , 11> slots = {{
                    {vertexBuffer.get() , "vertices"},              // 0
                    {indicesBuffer.get() , "indices"},              // 1
                    {LUT_buffer.get() , "LUT_Buffer"},              // 2
                    {MeshBuffer.get() , "meshBuffer"},              // 3
                    {instanceBuffer.get() , "instanceBuffer"},      // 4
                    {instanceLayoutBuffer.get() , "instanceLayoutBuffer"}, // 5
                    {meshletVertexBuffer.get() , "mVertices"},      // 6
                    {meshletIndexBuffer.get() , "mIndices"},        // 7
                    {geometryBuffer.get() , "geometryBuffer"},      // 8
                    {meshletBuffer.get() , "meshlets"},             // 9
                    {meshletBoundBuffer.get() , "boundBuffer"}      // 10
                }};

                for(const auto& slot : slots)
                {
                    if(!slot.buffer)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the descriptors of Vertex Heap , since the buffer does not exist , name -> {}" , slot.name)});
                    }
                }

                // the TLAS is the binding after the last buffer ( 11 in VertexHeap.hlsl )
                constexpr uint32_t tlasBinding = static_cast<uint32_t>(std::tuple_size_v<decltype(slots)>);

                if(!tlas || !tlas->pAS)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the descriptors of Vertex Heap , since the TLAS does not exist"});
                }

                {
                    std::vector<DescriptorLayoutData> data(tlasBinding + 1 , DescriptorLayoutData{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .count = 1});

                    data[tlasBinding] = {.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR , .count = 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.resize(1);
                    dlayouts[0] = std::make_shared<DescriptorLayout>(info);
                }

                for(const auto& slot : slots)
                {
                    const auto result = _implAddSource(slot.buffer , slot.name);
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(tlas.get() , "TLAS");
                    MOX_CHECK_RESULT(result)
                }

                {
                    std::vector<DescriptorData2> data{};
                    data.reserve(tlasBinding + 1);
                    for(uint32_t binding = 0 ; binding < slots.size() ; binding++)
                    {
                        data.push_back(DescriptorData2::createData(slots[binding].buffer , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , binding , 0));
                    }

                    // push_back , not data[tlasBinding] = ... : the vector has only the buffers so far , reserve() does not make the elements
                    data.push_back(DescriptorData2::createData(tlas.get() , tlasBinding , 0));

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
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the descriptor set of Vertex Heap"});
            }

            return {};
        }


        inline static std::vector<VkAccelerationStructureInstanceKHR> _implGetStructures() noexcept
        {
            std::vector<VkAccelerationStructureInstanceKHR> data(instanceLayout.size());
            uint32_t slot = 0;

            // it runs on every update , so the messages are limited
            static uint32_t messagesLeft = 10;
            const auto broken = [&](const uint32_t layoutIndex , const char* what , const uint32_t badIndex , const size_t size) -> void
            {
                if(messagesLeft == 0) return;
                messagesLeft--;
                const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to put the instance into the TLAS , the {} index is out of range or the object is null , instance layout -> {} , index -> {} , size -> {}" , what , layoutIndex , badIndex , size)};
                engineLogger(&error);
            };

            for(uint32_t index = 0 ; index < instanceLayout.size() ; index++)
            {
                const InstanceData& d = instanceLayout[index];
                if(!d.needsToBeDraw) continue;

                if(d.BLAS_index >= blases.size() || !blases[d.BLAS_index]) {broken(index , "BLAS" , d.BLAS_index , blases.size()); continue;}
                if(d.meshDataIndex >= meshs.size()) {broken(index , "mesh data" , d.meshDataIndex , meshs.size()); continue;}
                if(d.LUT_index >= luts.size()) {broken(index , "LUT" , d.LUT_index , luts.size()); continue;}

                const MeshData& m = meshs[d.meshDataIndex];
                if(m.instanceIndex >= instances.size()) {broken(index , "instance matrix" , m.instanceIndex , instances.size()); continue;}

                const LUT& l = luts[d.LUT_index];

                VkAccelerationStructureInstanceKHR info{};
                info.accelerationStructureReference = blases[d.BLAS_index]->pAS;
                info.instanceCustomIndex = index;
                info.instanceShaderBindingTableRecordOffset = 0;
                info.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                // nothing is opaque : the BLAS has no opaque flag and every instance uses the any hit shader
                info.mask = (l.isLightSource ? moxInstanceMaskLight : moxInstanceMaskNonLight);
                info.transform = matrixTransform34(instances[m.instanceIndex]);

                data[slot] = info;
                slot++;
            }

            return data;
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(VertexHeap)
}
} 