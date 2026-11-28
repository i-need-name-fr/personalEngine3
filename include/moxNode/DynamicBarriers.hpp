#pragma once

#include "../Vulkan/DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../Vulkan/DescriptorLayout/DescriptorLayout.hpp"
#include <any>
#include <unordered_set>
#include "../moxSourceCluster/SourceCluster.hpp"

namespace mox
{

    enum moxNodeBindingType
    {
        moxNodeBindingSampler,
        moxNodeBindingImage, // for all Render Images and so on 
        moxNodeBindingBuffer, // doesnt matter if its Uniform ( Constant ) or storage Buffer
        moxNodeBindingAccelerationStructure, // pretty simple
        moxNodeBindingUndefined
    };

    enum moxNodePassRequestType
    {
        moxNodePassRequestAttachment,
        moxNodePassRequestSet,
        moxNodePassRequestUndefined
    };

    inline static std::string const moxNodePassUndefinedName = "__MX_GRAPH_RENDER_UNDEFINED_ATTACHMENT_NAME__";

    // request defines which type of attachment are required , sets also included
    struct moxNodePassRequest
    {
        std::string name{moxNodePassUndefinedName}; // doesnt really matter if its set or attachment , both has a name
        std::string sourceName{moxNodePassUndefinedName}; // only for attachment, probably will 
        moxNodePassRequestType requestType{moxNodePassRequestUndefined}; // defines what we are looking for , a set or an attachment
        moxNodeBindingType bindingType{moxNodeBindingUndefined}; // only for attachment
    };

    struct moxNodeRenderFeedback
    {   
        std::string name{};
        SourceCluster* set{nullptr};
        std::any attachment{};
        moxNodePassRequestType type{moxNodePassRequestUndefined};
    };

    struct moxNodePassAlternativeSource
    {
        SourceCluster* source;
        std::uint32_t index;
    };

    struct moxNodePassResource
    {
        std::any data{};
        VkDescriptorType type{};
        std::uint32_t binding{};
        std::uint32_t count{};
    };


namespace utils
{
    class DynamicBarriers final
    {
    private:

        struct BarrierSource
        {

            std::string source;
            std::string name;
            SourceCluster* cluster{};
            VkAccessFlags2 access;

        };

        struct moxBarriers
        {
            std::vector<VkImageMemoryBarrier2> imageBarriers;
            std::vector<VkBufferMemoryBarrier2> bufferBarriers;

            std::vector<BarrierSource> imageSources;
            std::vector<BarrierSource> bufferSources;

            VkDependencyInfo dep{};

            void addImage(const BarrierSource& source)
            {
                imageBarriers.push_back({});
                imageSources.push_back(source);
            }

            void addBuffer(const BarrierSource& source)
            {
                bufferBarriers.push_back({});
                bufferSources.push_back(source);
            }
        };

    public:

        struct CreateInfo 
        {
            std::unordered_map<std::string , moxNodePassResource> general;
            std::unordered_map<std::string , moxNodePassAlternativeSource> sources;
            FileData file;
            VkPipelineStageFlags2 stage;
            moxContext2* context;
        };

        DynamicBarriers() noexcept = delete;

        DynamicBarriers(const CreateInfo &input)
        {
            info = input;
            const auto result = construct();
            if(!result.has_value())
            {
                throw std::runtime_error(result.error().second);
            }
        }

        DynamicBarriers(const DynamicBarriers&) = delete;
        DynamicBarriers& operator=(const DynamicBarriers&) = delete;

        DynamicBarriers(DynamicBarriers&& input) noexcept
        {
            processCopy(std::move(input));
        }

        DynamicBarriers& operator=(DynamicBarriers&& input) noexcept
        {
            if(this != &input)
            {
                processDelete();
                processCopy(std::move(input));
            }
            return *this;
        }

        [[gnu::hot]] void execute(VkCommandBuffer& cmd , const VkPipelineStageFlags2 stage) noexcept
        {
            const uint32_t id = info.file.containsPingPong ? info.context->sc->frameIndex %2 : 0;
            _implFillBarriers(stage , id);
            dbarriers[id].dep.pBufferMemoryBarriers = dbarriers[id].bufferSources.size() > 0 ? dbarriers[id].bufferBarriers.data() : nullptr;
            dbarriers[id].dep.pImageMemoryBarriers = dbarriers[id].imageSources.size() > 0 ? dbarriers[id].imageBarriers.data() : nullptr;
            vkCmdPipelineBarrier2(cmd , &dbarriers[id].dep);
        }

        ~DynamicBarriers() noexcept
        {
            processDelete();
        }

        void deconstruct() noexcept
        {
            processDelete();
        }

        [[nodiscard]] bool isConstructed() const noexcept
        {
            return !dbarriers.empty();
        }

    private:

        CreateInfo info{};
        std::vector<moxBarriers> dbarriers{};

        [[nodiscard]] ErrorDataOutput<void> construct() noexcept
        {
            processDelete();

            const auto result = _implInit();
            if(!result.has_value())
            {
                processDelete();
                return std::unexpected(result.error());
            }

            return {};
        }
        
        [[nodiscard]] VkBufferMemoryBarrier2 _implGeneralBufferBarrier(const BarrierSource& source , const VkPipelineStageFlags2 stage) noexcept
        {
            const std::any& data = info.general.find(source.name)->second.data;
            return std::any_cast<Buffer::BufferBase2*>(data)->translateBuffer(source.access , stage);
        }

        [[nodiscard]] VkImageMemoryBarrier2 _implGeneralImageBarrier(const BarrierSource& source , const VkPipelineStageFlags2 stage) noexcept
        {
            const std::any& data = info.general.find(source.name)->second.data;

            if(data.type() == typeid(Image::RenderImage2*))
                return std::any_cast<Image::RenderImage2*>(data)->translateImage(source.access , stage);

            if(data.type() == typeid(Image::Skybox2*))
                return std::any_cast<Image::Skybox2*>(data)->translateImage(source.access , stage);

            return std::any_cast<Image::Texture2*>(data)->translateImage(source.access , stage);
        }

        void _implFillBarriers(const VkPipelineStageFlags2 stage , const uint32_t id) noexcept
        {

            for(auto&& [barrier , source] : std::views::zip(dbarriers[id].bufferBarriers , dbarriers[id].bufferSources) )
            {
                if(!source.cluster) // GENERAL has no cluster
                {
                    barrier = _implGeneralBufferBarrier(source , stage);
                }
                else
                {
                    barrier = std::get<VkBufferMemoryBarrier2>(source.cluster->requestBarrier(source.name , stage , source.access).value());
                }
            }

            for(auto&& [barrier , source] : std::views::zip(dbarriers[id].imageBarriers , dbarriers[id].imageSources) )
            {
                if(!source.cluster) // GENERAL has no cluster
                {
                    barrier = _implGeneralImageBarrier(source , stage);
                }
                else
                {
                    barrier = std::get<VkImageMemoryBarrier2>(source.cluster->requestBarrier(source.name , stage , source.access).value());
                }
            }

        }

        [[nodiscard]] ErrorDataOutput<void> _implCheckGeneralResource(const std::string& name , const bool isBuffer) const noexcept
        {
            const auto it = info.general.find(name);
            if(it == info.general.end())
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL set doesnt contain the attachment -> {}" , name));
            }

            const std::any& data = it->second.data;

            if(isBuffer)
            {
                if(data.type() != typeid(Buffer::BufferBase2*))
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL attachment -> {} is not a BufferBase2*" , name));
                }
                if(!std::any_cast<Buffer::BufferBase2*>(data))
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL buffer -> {} is nullptr" , name));
                }
                return {};
            }

            if(data.type() == typeid(Image::RenderImage2*))
            {
                if(!std::any_cast<Image::RenderImage2*>(data)) MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL image -> {} is nullptr ( RenderImage2* )" , name));
                return {};
            }
            if(data.type() == typeid(Image::Skybox2*))
            {
                if(!std::any_cast<Image::Skybox2*>(data)) MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL image -> {} is nullptr ( Skybox2* )" , name));
                return {};
            }
            if(data.type() == typeid(Image::Texture2*))
            {
                if(!std::any_cast<Image::Texture2*>(data)) MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL image -> {} is nullptr ( Texture2* )" , name));
                return {};
            }

            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL attachment -> {} is not RenderImage2* , Skybox2* or Texture2*" , name));
        }

        [[nodiscard]] ErrorDataOutput<void> _implInit() noexcept
        {

            dbarriers.resize(info.file.containsPingPong ? 2 : 1);


            for(auto& [sourceName , data] : info.file.sets)
            {

                if(sourceName == "GENERAL")
                {

                    for(auto& [attName , att] : data.attachments)
                    {

                        const bool isBuffer = (att.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || att.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
                        const bool isImage = (att.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || att.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);

                        if(!isBuffer && !isImage)
                        {
                            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the GENERAL attachment -> {} has a type that cannot have a barrier ( only buffers and images are allowed )" , att.name));
                        }

                        if(att.action == VK_ACCESS_2_NONE)
                        {
                            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the access of the GENERAL attachment -> {} is undefined" , att.name));
                        }

                        BarrierSource b{};
                        b.source = sourceName;
                        b.access = att.action;

                        if(att.pingPongSystem)
                        {

                            const std::string name0 = att.name + "_0";
                            const std::string name1 = att.name + "_1";

                            const auto result0 = _implCheckGeneralResource(name0 , isBuffer);
                            MOX_CHECK_RESULT(result0)
                            const auto result1 = _implCheckGeneralResource(name1 , isBuffer);
                            MOX_CHECK_RESULT(result1)

                            // CURRENT ( index 1 ) : frame 0 -> name_0 , frame 1 -> name_1
                            // PREV ( index 0 ) : frame 0 -> name_1 , frame 1 -> name_0
                            const std::size_t slotOfName0 = (att.pingPongIndex == 1) ? 0 : 1;

                            auto& first = dbarriers[slotOfName0];
                            auto& second = dbarriers[1 - slotOfName0];

                            b.name = name0;
                            if(isBuffer) first.addBuffer(b);
                            else first.addImage(b);

                            b.name = name1;
                            if(isBuffer) second.addBuffer(b);
                            else second.addImage(b);

                        }
                        else
                        {

                            const auto result = _implCheckGeneralResource(att.name , isBuffer);
                            MOX_CHECK_RESULT(result)

                            b.name = att.name;
                            // the same barrier for every slot ( 1 slot without ping pong , 2 with it )
                            for(auto& slot : dbarriers)
                            {
                                if(isBuffer) slot.addBuffer(b);
                                else slot.addImage(b);
                            }

                        }
                    }

                }
                else
                {
                    // current is 0 - 1
                    // prev is 1 - 0

                    if(!info.sources.contains(sourceName))
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the SourceCluster -> {} doesnt exist" , sourceName));
                    }

                    SourceCluster* const cluster = info.sources.at(sourceName).source;
                    if(!cluster)
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the SourceCluster -> {} is nullptr" , sourceName));
                    }

                    for(auto& [attName , att] : data.attachments)                    
                    {

                        // samplers and acceleration structures have no barrier
                        if(att.type == VK_DESCRIPTOR_TYPE_SAMPLER || att.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) continue;

                        if(att.action == VK_ACCESS_2_NONE)
                        {
                            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to init DynamicBarriers , the access of the attachment -> {} from -> {} is undefined" , attName , sourceName));
                        }

                        if(att.pingPongSystem)
                        {

                            const std::string name0 = attName + "_0";
                            const std::string name1 = attName + "_1";

                            const auto result = cluster->requestBarrierType(name1);

                            MOX_CHECK_RESULT(result)

                            if(std::holds_alternative<VkBufferMemoryBarrier2>(result.value()))
                            {

                                {
                                    BarrierSource b{};
                                    b.name = name0;
                                    b.source = sourceName;
                                    b.cluster = cluster;
                                    b.access = att.action;

                                    if(att.pingPongIndex == 1)
                                    {
                                        dbarriers[0].addBuffer(b); // to 0 , since its the first , when the bool will be 0 , it will choose the current , as 0 
                                        b.name = name1;
                                        dbarriers[1].addBuffer(b); 
                                    }
                                    else
                                    {
                                        dbarriers[1].addBuffer(b); // to 0 , since its the first , when the bool will be 0 , it will choose the current , as 0 
                                        b.name = name1;
                                        dbarriers[0].addBuffer(b); 
                                    }
                                }

                            }  
                            else
                            {

                                {
                                    BarrierSource b{};
                                    b.name = name0;
                                    b.source = sourceName;
                                    b.cluster = cluster;
                                    b.access = att.action;

                                    if(att.pingPongIndex == 1)
                                    {
                                        dbarriers[0].addImage(b); // to 0 , since its the first , when the bool will be 0 , it will choose the current , as 0 
                                        b.name = name1;
                                        dbarriers[1].addImage(b); 
                                    }
                                    else
                                    {
                                        dbarriers[1].addImage(b); // to 0 , since its the first , when the bool will be 0 , it will choose the current , as 0 
                                        b.name = name1;
                                        dbarriers[0].addImage(b); 
                                    }
                                }

                            }
                            continue;
                        }
                        else
                        {

                            const std::string name0 = attName;

                            const auto result = cluster->requestBarrierType(name0);

                            MOX_CHECK_RESULT(result)

                            if(std::holds_alternative<VkBufferMemoryBarrier2>(result.value()))
                            {

                                {
                                    BarrierSource b{};
                                    b.name = name0;
                                    b.source = sourceName;
                                    b.cluster = cluster;
                                    b.access = att.action;
                                    for(auto& slot : dbarriers) slot.addBuffer(b); // 1 slot without ping pong , 2 with it

                                }

                            }  
                            else
                            {
                                {
                                    BarrierSource b{};
                                    b.name = name0;
                                    b.source = sourceName;
                                    b.cluster = cluster;
                                    b.access = att.action;
                                    for(auto& slot : dbarriers) slot.addImage(b); // 1 slot without ping pong , 2 with it
                                }
                            }

                            continue;
                        }
                    }
                }

            }

            for(auto& d : dbarriers)
            {
                d.dep.dependencyFlags = 0;
                d.dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                d.dep.pNext = nullptr;
                d.dep.bufferMemoryBarrierCount = static_cast<uint32_t>(d.bufferBarriers.size());
                d.dep.pBufferMemoryBarriers = d.bufferBarriers.data();
                d.dep.imageMemoryBarrierCount  = static_cast<uint32_t>(d.imageBarriers.size());
                d.dep.pImageMemoryBarriers = d.imageBarriers.data();
                d.dep.memoryBarrierCount = 0;
                d.dep.pMemoryBarriers = nullptr;
            }

            return {};
        }

        void processDelete() noexcept
        {
            info = {};
            dbarriers = {};
        }

        void processCopy(DynamicBarriers&& input) noexcept
        {
            info = std::move(input.info);
            dbarriers = std::move(input.dbarriers);

            input.info = {};
            input.dbarriers = {};
        }

    };

    MOX_DECLARE_PTR_STRUCTURE(DynamicBarriers)
}
}