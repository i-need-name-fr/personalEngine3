#pragma once

#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include <any>
#include <unordered_set>
#include "../moxSourceCluster/SourceCluster.hpp"
#include "../ShaderPipe2/ShaderPipe2.hpp"
#include <typeindex>
namespace mox
{
    // currently The NodePass is different from other passes within a simple changes
    // now NodePass will automatically create DescriptorLayout for GENERAL set and bind other Descriptor sets from Chunks of data
    // but before that we need to create a structures for it to perform as we wish 

    // this is required to RenderGraph to analyze the data type and compare it to stored value
    enum moxNodeBindingType
    {
        // there are different types of binding, but we will be making only 5
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

    inline static std::string const moxNodeRenderGeneralSet = "__MX_DSET_GENERAL_SET__"; // this is 
    inline static std::string const moxNodePassDefaultName = "__MX_GRAPH_P3_DEFAULT_NAME__";

    template<const std::type_info& type>
    requires (type == typeid(Image::RenderImage2*))
    Image::RenderImage2* convertToOrigin(moxNodePassResource &data) noexcept
    {
        return std::any_cast<Image::RenderImage2*>(data.data);
    }

    template<const std::type_info& type>
    requires (type == typeid(Image::Skybox2*))
    Image::Skybox2* convertToOrigin(moxNodePassResource &data) noexcept
    {
        return std::any_cast<Image::Skybox2*>(data.data);
    }

    template<const std::type_info& type>
    requires (type == typeid(Image::Texture2*))
    Image::Texture2* convertToOrigin(moxNodePassResource &data) noexcept 
    {
        return std::any_cast<Image::Texture2*>(data.data);
    }

    template<const std::type_info& type>
    requires (type == typeid(Buffer::BufferBase2*))
    Buffer::BufferBase2* convertToOrigin(moxNodePassResource &data) noexcept
    {
        return std::any_cast<Buffer::BufferBase2*>(data.data);
    }

    // UPDATED: new NodePass now can pass the functions,  however the user needs to transfer it to the direct type , cannot be used directly
    class NodePass
    {
    public:

        SetTypicalOperators(NodePass)
        SetTypicalTransferOperators(NodePass)

        explicit NodePass(moxContext2* context , const std::string& name) noexcept __attribute__((nonnull))
        {
            this->context = context;
            this->passName = name;
        }

        virtual ~NodePass() noexcept
        {
            processDelete();
        }

        // sendRequest 
        MOX_NO_DISCARD ErrorDataOutput<std::vector<moxNodePassRequest>> sendRequest() noexcept
        {
            return _implSendRequestGR3();
        }

        // processFeedback

        MOX_NO_DISCARD ErrorDataOutput<void> analyzeFeedBack(const std::vector<moxNodeRenderFeedback> &feedback) noexcept
        {
            return _implProcessGRFeedback(feedback);
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> initPass() noexcept = 0;
        virtual bool requestExecution() const noexcept = 0;
        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept = 0;

        virtual std::string getPassName() const noexcept {return passName;}

    protected:

        virtual void processDelete() noexcept
        {
            generalAttachments = {};
            cSources = {};
            _dBarrier = {};
            moxFILE.reset();
            containsPingPong = false;
            passName = moxNodePassDefaultName;
            context = nullptr;
            __dbG.clear();
            __dbBindingInfo.clear();
            __dbSet.clear();
            __dbNames.clear();
            __dsets.clear();
        }

        virtual void processCopy(NodePass&& input) noexcept
        {
            context = std::exchange(input.context , nullptr);
            passName = std::move(input.passName);
            generalAttachments = std::move(input.generalAttachments);
            cSources = std::move(input.cSources);
            _dBarrier = std::move(input._dBarrier);
            moxFILE = std::move(input.moxFILE);
            containsPingPong = input.containsPingPong;

            input.passName = moxNodePassDefaultName;
            input.moxFILE.reset();
            input.containsPingPong = false;
        }

        // we need a structure that stores all dynamic barriers , for bindBarriers to work perfectly without need for allocating memory every frame
        struct moxDynamicBarrier
        {
            std::vector<VkBufferMemoryBarrier2> buffers{};
            std::vector<VkImageMemoryBarrier2> images{};
            VkDependencyInfo dep{};
        };

        moxContext2* context{nullptr};
        std::string passName{moxNodePassDefaultName};

        // create the descriptor Buffer for general layout
        // _implInitPass() 
        // move parts that are not nessesary if we have _implInitPass() 
        // process the feedback 

        // one for all the clusters
        // one for all attachments from general Set
        std::unordered_map<std::string , moxNodePassResource> generalAttachments{};
        std::unordered_map<std::string , moxNodePassAlternativeSource> cSources{};
        std::vector<std::optional<moxDynamicBarrier>> _dBarrier{}; // 1 element , or 2 for ping pong , chosen by frameIndex % 2

        // init 

        MOX_NO_DISCARD ErrorDataOutput<void> initFile(const std::filesystem::path& path) noexcept
        {
            // reading the file
            {
                const auto result = DescriptorFile::AnalyzeFile(path);
                if(!result.has_value()) return std::unexpected(result.error());
                moxFILE = result.value();
            }

            {
                const auto result = _implSetupForSourcesAndAttachments();
                if(!result.has_value()) return result;
            }

            return {};
        }

        // descriptor layout

        MOX_NO_DISCARD ErrorDataOutput<pDescriptorLayout> createGeneralDL(const VkShaderStageFlags stage) noexcept
        {
            return _implCreateGeneralDescriptorLayout(stage);
        }

        // descriptor buffer

        MOX_NO_DISCARD ErrorDataOutput<std::vector<pDescriptorBuffer2>> createdDBuffers(DescriptorLayout* layout) noexcept
        {
            return _implCreateDescriptorBuffers(layout);
        }   

        // barriers

        MOX_NO_DISCARD ErrorDataOutput<void> createBarriers(const VkPipelineStageFlags2 stage , const std::uint32_t frameID ) noexcept
        {
            return _implCreateDBarrier(stage , frameID);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> bindBarrier(VkCommandBuffer &cmd ,const VkPipelineStageFlags2 stage) noexcept
        {
            const auto result = createBarriers(stage , context->sc->frameIndex);
            if(!result.has_value()) return result;
            return _implBindBarriers(cmd);
        }

        MOX_NO_DISCARD pDescriptorBuffer2 getCurrentDBuffer(const std::vector<pDescriptorBuffer2> &buffers) const noexcept
        {   
            if(buffers.empty()) return nullptr;
            if(!context) return nullptr;

            if(!containsPingPong)
            {
                return buffers[0];
            }
            else
            {
                const uint32_t frame = context->sc->frameIndex % 2;
                return buffers[frame];
            }
        }

        MOX_NO_DISCARD ErrorDataOutput<void> processPipe(ShaderPipe2* pipe , VkShaderStageFlags stage , std::vector<VkPushConstantRange> pushs = {}) noexcept
        {

            // lets first of get from file the exact position of each set
            const std::size_t size = moxFILE.value().sets.size();
            std::vector<DescriptorLayout*> layouts(size);

            // the sets of the pipeline layout are numbered 0 .. N-1 without gaps ( GENERAL is the first one , 0 ) , a gap would be a nullptr layout , and an index >= N would write behind the vector
            {
                std::vector<uint8_t> used(size , 0);
                for(const auto& data : moxFILE.value().sets)
                {
                    const std::uint32_t index = (data.first == "GENERAL") ? 0u : data.second.index;
                    if(index >= size || used[index])
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process the pipe of NodePass -> {} , the SET_INDEX of the set {} is {} , but the {} sets must have the indices 0 .. {} without gaps and without repeats ( GENERAL is always 0 )" , passName , data.first , index , size , size - 1)});
                    }
                    used[index] = 1;
                }
            }
            for(auto& data : moxFILE.value().sets)
            {
                if(data.first == "GENERAL")
                {   
                    const auto result = createGeneralDL(stage);
                    MOX_CHECK_RESULT(result)
                    pipe->descriptorLayouts.push_back(result.value());
                    // since the general is always the first
                    layouts[0] = result.value().get();
                    {
                        const auto result_ = createdDBuffers(layouts[0]);
                        MOX_CHECK_RESULT(result_)
                        pipe->buffers = result_.value();
                    }
                }
                else
                {
                    layouts[cSources[data.second.source].index] = cSources[data.second.source].source->requestLayout();
                }
            }

            PipelineLayout::CreateInfo info{};
            info.layouts = layouts;
            info.ranges = pushs;
            info.context = context;
            pipe->layout = std::make_shared<PipelineLayout>(info);

            return {};
        }

        void bindAutoDescriptorBuffers(VkCommandBuffer& cmd , const VkPipelineBindPoint& point , PipelineLayout* layout) noexcept __attribute__((nonnull)) 
        {
            _implBindDescriptorBuffers(cmd , point , layout);
        }

    private:

        std::optional<mox::FileData> moxFILE{}; // file should be analyzed from start , given both .hlsl and .mox files 
        // we need functions to send data to the RenderGraph
        std::vector<DescriptorBuffer2*> __dbG{}; // general Descriptor buffers for automatic binding
        std::vector<DescriptorBuffer2*> __dbSet{}; // a dynamic set of buffers
        std::vector<uint32_t> __dsets{};
        std::vector<VkDeviceSize> __doffsets{};
        std::vector<std::string> __dbNames{}; // instruction to which db bind and at which position
        std::vector<VkDescriptorBufferBindingInfoEXT> __dbBindingInfo{};
        bool __hasGeneralSet{false};

        moxNodeBindingType _implTransferTypeToData(const VkDescriptorType type) const noexcept
        {
            if(type == VK_DESCRIPTOR_TYPE_SAMPLER)
            {
                return moxNodeBindingSampler;
            }

            if(type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
            {
                return moxNodeBindingImage;
            }

            if(type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            {
                return moxNodeBindingBuffer;
            }

            if(type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
            {
                return moxNodeBindingAccelerationStructure;
            }


            return moxNodeBindingUndefined;
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessSourceBarrier(const DescriptorAttachment& binding , VkPipelineStageFlags2 const stage , moxDynamicBarrier& barriers , const std::any& source) noexcept
        {
            if(binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            {
                if(source.type() == typeid(Buffer::BufferBase2*))
                {
                    const auto buffer = std::any_cast<Buffer::BufferBase2*>(source);
                    if(!buffer) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create dynamic barriers , since the DS attachment has null Buffer (BufferBase2*) , binding from -> {}, from -> {}" , binding.name , binding.source)});
                    const auto barrier = buffer->translateBuffer(DescriptorFile::TranslateToAccess(binding.action) , stage);
                    barriers.buffers.push_back(barrier);
                    return {};
                }

                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create dynamic barriers , since the DS attachment has different type of class from provided one , binding from -> {}, from -> {}" , binding.name , binding.source)});
            }
            if(binding.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || binding.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
            {
                if(source.type() == typeid(Image::RenderImage2*))
                {
                    const auto& image = std::any_cast<Image::RenderImage2*>(source);
                    if(!image) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create dynamic barriers , since the DS attachment has null image (RenderImage2*) , binding from -> {}, from -> {}" , binding.name , binding.source)});

                    const auto barrier = image->translateImage(DescriptorFile::TranslateToAccess(binding.action) , stage);
                    barriers.images.push_back(barrier);
                    return {};
                }

                if(source.type() == typeid(Image::Skybox2*))
                {
                    const auto& image = std::any_cast<Image::Skybox2*>(source);
                    if(!image) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create dynamic barriers , since the DS attachment has null image (Skybox2*) , binding from -> {}, from -> {}" , binding.name , binding.source)});

                    const auto barrier = image->translateImage(DescriptorFile::TranslateToAccess(binding.action) , stage);
                    barriers.images.push_back(barrier);
                    return {};
                }

                if(source.type() == typeid(Image::Texture2*))
                {
                    const auto& image = std::any_cast<Image::Texture2*>(source);
                    if(!image) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create dynamic barriers , since the DS attachment has null image (Texture2*) , binding from -> {}, from -> {}" , binding.name , binding.source)});

                    const auto barrier = image->translateImage(DescriptorFile::TranslateToAccess(binding.action) , stage);
                    barriers.images.push_back(barrier);
                    return {};
                }

                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create dynamic barriers , since the DS attachment has different type of class from provided one , binding from -> {}, from -> {}" , binding.name , binding.source)});
            }

            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process Source , since the source contains different type of data comparing to Descriptor request , binding -> {} , from -> {}" , binding.name , binding.source)});
        }

        MOX_NO_DISCARD static std::string _implResolveName(const DescriptorAttachment& binding , const uint32_t frameParity) noexcept
        {
            if(!binding.pingPongSystem) return binding.name;

            return moxPingPongName(binding.name , moxPingPongIndex(frameParity , binding.pingPongIndex));
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessAttachmentBarrier(const DescriptorAttachment& binding , VkPipelineStageFlags2 const stage , moxDynamicBarrier& barriers , const uint32_t frameParity) noexcept
        {
            const std::string resolvedName = _implResolveName(binding , frameParity);

            if(binding.type == VK_DESCRIPTOR_TYPE_SAMPLER || binding.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) return {};

            if(binding.action == moxDescriptorAccessUndefined) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process descriptor attachment , since the action is undefined , binding -> {}, from {}" , binding.name ,binding.source )});

            if(binding.source == DescriptorFile::_moxGeneralSet)
            {
                const auto it = generalAttachments.find(resolvedName);
                if(it == generalAttachments.end())
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create dynamic barriers , since the GENERAL set doesnt contain attachment -> {}" , resolvedName)});
                }

                // WARNING FOR BUFFERS
                // EACH BUFFER IS BUFFERBASE2 , NOT GENERALBUFFER2
                const auto result = _implProcessSourceBarrier(binding , stage , barriers , it->second.data);
                if(!result.has_value()) return result;
            }
            else
            {
                // we are looking in source clusters
                const auto clusterIt = cSources.find(binding.source);
                if(clusterIt == cSources.end())
                    return std::unexpected(mox::ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create Dynamic barriers , since the descriptor set requires for Source Cluster-> {} , which doesnt exists , binding -> {} , from -> {}" , binding.source, binding.name ,binding.source )});

                if(!clusterIt->second.source)
                {
                    return std::unexpected(mox::ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create Dynamic barriers , since the descriptor set requires for Source Cluster-> {} which is nullptr , binding -> {} , from -> {}" , binding.source, binding.name ,binding.source )});
                }
                const auto requestResult = clusterIt->second.source->requestBarrier(resolvedName , stage , DescriptorFile::TranslateToAccess(binding.action));

                if(!requestResult.has_value()) return std::unexpected(requestResult.error());

                // now we have to analyze the output of requestedResult , it should be either VkImageMemoryBarrier or VkBufferMemoryBarrier
                const auto variantData = requestResult.value();
                // its either image or buffer
                if(std::holds_alternative<VkImageMemoryBarrier2>(variantData))
                {
                    barriers.images.push_back(std::get<VkImageMemoryBarrier2>(variantData));
                }
                else
                {
                    barriers.buffers.push_back(std::get<VkBufferMemoryBarrier2>(variantData));
                }
            }

            return {};
        }

        // the barriers have to be rebuilt EVERY frame : translateImage / translateBuffer change the stored state ( currentAccess , layout ) of the resource
        // _dBarrier has 1 element , or 2 if the pass has ping pong , the element is chosen by frameIndex % 2
        // the vectors are cleared , not freed , so after the first frame there are no allocations
        MOX_NO_DISCARD ErrorDataOutput<void> _implCreateDBarrier(const VkPipelineStageFlags2 stage , const uint64_t frameIndex) noexcept
        {
            if(!moxFILE)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process NodePass dynamic Barriers , since the file is not being created"});
            }

            const size_t slots = containsPingPong ? 2 : 1;
            if(_dBarrier.size() != slots) _dBarrier.resize(slots);

            const uint32_t frameParity = static_cast<uint32_t>(frameIndex % 2);
            auto& slot = _dBarrier[containsPingPong ? frameParity : 0];

            size_t attachmentCount = 0;
            for(const auto& pset : moxFILE->sets) attachmentCount += pset.second.attachments.size();

            if(!slot) slot.emplace();
            moxDynamicBarrier& barriers = *slot;
            barriers.buffers.clear();
            barriers.images.clear();
            barriers.buffers.reserve(attachmentCount);
            barriers.images.reserve(attachmentCount);

            for(const auto& pset : moxFILE->sets)
            {
                for(const auto& att : pset.second.attachments)
                {
                    const auto result = _implProcessAttachmentBarrier(att.second , stage , barriers , frameParity);
                    if(!result.has_value())
                    {
                        slot.reset(); // half filled barrier must never be used
                        return result;
                    }
                }
            }

            // the pointers are taken only after everything is pushed , since push_back can move the data
            barriers.dep = {};
            barriers.dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            barriers.dep.pBufferMemoryBarriers = barriers.buffers.data();
            barriers.dep.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.buffers.size());
            barriers.dep.pImageMemoryBarriers = barriers.images.data();
            barriers.dep.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.images.size());

            return {};
        }

        MOX_NO_DISCARD ErrorDataOutput<std::vector<moxNodePassRequest>> _implSendRequestGR3() noexcept
        {

            if(!moxFILE) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to send request for graph render3 , since the fileData is empty"});

            // now we analyze the file
            
            std::vector<moxNodePassRequest> request{};
            // basically we need to request RenderGraph for sets and attachment
            // the flag comes from the file , so a ping pong attachment inside a SourceCluster set counts too
            containsPingPong = moxFILE->containsPingPong;
            for(const auto& setp : moxFILE.value().sets )
            {

                std::string sourceName{};
                // if the set is GENERAL , it means that it should be in generalBuffer of RenderGraph , not in SourceCluster, therefore it will look there
                if(setp.first == DescriptorFile::_moxGeneralSet)
                {
                    sourceName = moxNodeRenderGeneralSet;
                }
                else
                {
                    sourceName = setp.first;
                }

                // here we need to create first the set data
                // and only aftet that the binding data
                moxNodePassRequest data{};
                data.name = sourceName; // first is the NAME of the set 
                data.requestType = moxNodePassRequestSet;
                data.sourceName = sourceName; // if i forget to which exactly variable did i give name of the set
                data.bindingType = moxNodeBindingUndefined; // for sure to tell that this binding is SET
                request.push_back(data);
                // data.name = setp.second.source; // this will also work , sicne the source is also a name of the SET
                // now we analyze the given attachments for this set specifically 
                // we want to ask for GENERAL ONLY sets , since the SourceCluster will output all in one , no need to make a request for each
                if(sourceName == moxNodeRenderGeneralSet)
                {
                    // a ping pong resource can be declared twice ( "next" and "prev" ) , but RenderGraph has to be asked for its _0 and _1 only once
                    std::unordered_set<std::string> requestedPingPong{};

                    for(const auto& att : setp.second.attachments)
                    {
                        if(att.second.pingPongSystem)
                        {
                            if(!requestedPingPong.insert(att.second.name).second) continue;

                            moxNodePassRequest info{};
                            info.requestType = moxNodePassRequestAttachment;
                            info.name = moxPingPongName(att.second.name , 0); // the name of the attachment
                            info.sourceName = sourceName; // source where we want to take it from ( the set )
                            // here we need to translate descriptor type to moxType , since RenderGraph doesnt care about type of descriptor  , it requires type of DATA
                            info.bindingType = _implTransferTypeToData(att.second.type);

                            request.push_back(info); // for first _0
                            info.name = moxPingPongName(att.second.name , 1);
                            request.push_back(info); // for second _1
                        }
                        else
                        {
                            moxNodePassRequest info{};
                            info.requestType = moxNodePassRequestAttachment;
                            info.name = att.second.name; // the name of the attachment
                            info.sourceName = sourceName; // source where we want to take it from ( the set )
                            // here we need to translate descriptor type to moxType , since RenderGraph doesnt care about type of descriptor  , it requires type of DATA
                            info.bindingType = _implTransferTypeToData(att.second.type);
                            request.push_back(info);
                        }
                    }
                }

            }

            return request;
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implSetupForSourcesAndAttachments() noexcept
        {
            if(!moxFILE) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to send request for graph render3 , since the fileData is empty"});
            
            containsPingPong = moxFILE->containsPingPong;
            __hasGeneralSet = false; // if it contains general, it will add one more
            // now we create all the sources and add attachments
            __dbNames.resize(moxFILE.value().sets.size());

            // the indices of the sets have to be 0 .. N-1 , without gaps and without repeats : the position in __dbNames / __dbSet IS the number of the set , and the descriptor buffers are bound
            // as 0 .. N-1 in a row . N sets with N different indices that are all below N cannot have a gap , so two checks are enough . ( GENERAL is already checked to be 0 by the file reader )
            {
                const std::size_t setCount = moxFILE.value().sets.size();
                std::vector<std::string> owner(setCount);
                for(const auto& pset : moxFILE.value().sets)
                {
                    const std::uint32_t index = pset.second.index;
                    if(index >= setCount)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to setup the NodePass -> {} , the SET_INDEX of the set {} is {} , but the file has {} sets : the indices have to be 0 .. {} without gaps" , passName , pset.first , index , setCount , setCount - 1)});
                    }

                    if(!owner[index].empty())
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to setup the NodePass -> {} , the sets {} and {} have the same SET_INDEX {} : the indices have to be 0 .. {} without repeats" , passName , owner[index] , pset.first , index , setCount - 1)});
                    }
                    owner[index] = pset.first;
                }
            }

            for(auto& pset : moxFILE.value().sets)
            {
                if(pset.first == DescriptorFile::_moxGeneralSet)
                {
                    for(auto& binding : pset.second.attachments) 
                    {
                        // the rule : GENERAL contains only buffers and images , samplers and TLAS live in SourceCluster
                        

                        if(binding.second.type == VK_DESCRIPTOR_TYPE_SAMPLER || binding.second.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                        {
                            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to setup the NodePass , the GENERAL set can not contain a SAMPLER or an AS ( they belong to a SourceCluster ) , attachment -> {}" , binding.second.name)});
                        }

                        moxNodePassResource res{};
                        res.binding = binding.second.binding;
                        res.count = binding.second.count;
                        res.type = binding.second.type;
                        if(binding.second.pingPongSystem)
                        {
                            // one attachment in the file , but two resources : _0 and _1

                            generalAttachments.insert({moxPingPongName(binding.second.name , 0) , {res}});
                            generalAttachments.insert({moxPingPongName(binding.second.name , 1) , {res}});
                        }
                        else
                        {
                            generalAttachments.insert({binding.second.name , {res}});
                        }
                    }
                    __hasGeneralSet = true;
                }
                else
                {
                    cSources.insert({pset.first , {.source = nullptr , .index = pset.second.index}});
                }

                // the index is checked above : it is always inside of __dbNames and it is never used twice
                __dbNames[pset.second.index] = pset.first;
            }

            __dbSet.resize(moxFILE.value().sets.size());
            __dbBindingInfo.resize(__dbSet.size());
            __doffsets = std::vector<VkDeviceSize>(__dbBindingInfo.size() , 0);
            for(auto& b : __dbBindingInfo)
            {
                b.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                b.pNext = nullptr;
                b.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
                b.address = 0;
            }

            for(uint32_t i = 0 ; i < __dbBindingInfo.size() ; i++)
            {
                __dsets.push_back(i);
            }

            return {};
        }

        uint8_t containsPingPong = false;

        MOX_NO_DISCARD ErrorDataOutput<void> _implBindBarriers(VkCommandBuffer& cmd) noexcept
        {
            if(!cmd) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate barriers , since the command buffer is nullptr"});
            if(_dBarrier.empty()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate barriers , since the barriers are empty"});

            if(containsPingPong)
            {
                const uint32_t id = context->sc->frameIndex % 2;
                if(!_dBarrier[id].has_value()) 
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate barriers WITH ping pong , since the first barrier is not existing"});
                }
                vkCmdPipelineBarrier2(cmd , &_dBarrier[id].value().dep);
            }
            else
            {   
                if(!_dBarrier[0].has_value()) 
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to translate barriers WITHOUT ping pong , since the first barrier is not existing"});
                }
                vkCmdPipelineBarrier2(cmd , &_dBarrier[0].value().dep);
            }

            return {};
        }

        // feedback not includes GENERAL set , only general attachment and source cluster that are been requested
        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessGRFeedback(const std::vector<moxNodeRenderFeedback> &feedback) noexcept
        {   
            // the feedback can be empty : a pass with only the GENERAL set and no requested attachments
            if(!moxFILE) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to processRGFeedback , since the moxFILE isnt being created , graph name -> {}" , passName)});

            for(auto& source : feedback)
            {
                if(source.type == moxNodePassRequestSet)
                {
                    const auto it = cSources.find(source.name);
                    if(it != cSources.end())
                    {
                        // now we need to clarify that it contains a pointer
                        if(!source.set)
                        {
                            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process NodePass feedback of GR , since source Set contains nullptr , set -> {}" , source.name)});
                        }

                        it->second.source = source.set;
                        continue;
                    }

                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process NodePass feedback of GR , since feedback contains smth that wasnt requested , set -> {}" , source.name)});
                }
                if(source.type == moxNodePassRequestAttachment)
                {
                    
                    // if its attachment , it means its from general set , therefore , we need to add exactly to generalSource
                    const auto it = generalAttachments.find(source.name);
                    if(it !=  generalAttachments.end())
                    {
                        it->second.data = source.attachment;
                        continue;
                    }
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process NodePass feedback of GR , since feedback contains smth that wasnt requested , attachment -> {}" , source.name)});
                }
            }
            return {};
        }

        MOX_NO_DISCARD ErrorDataOutput<std::vector<DescriptorData2>> _implBuildGeneralDescriptorData(const uint32_t frameParity) const noexcept
        {
            if(!moxFILE) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to build the general descriptor data , since the moxFILE isnt being created , graph name -> {}" , passName)});

            const auto setIt = moxFILE->sets.find(DescriptorFile::_moxGeneralSet);
            if(setIt == moxFILE->sets.end()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to build the general descriptor data , since the file has no GENERAL set , graph name -> {}" , passName)});

            // the size is the count of attachments of the FILE , not of generalAttachments , since a ping pong attachment is there twice ( _0 and _1 ) but has only one binding
            const auto& attachments = setIt->second.attachments;
            std::vector<DescriptorData2> data(attachments.size());
            std::vector<uint8_t> filled(attachments.size() , 0);

            for(const auto& it : attachments)
            {
                const DescriptorAttachment& att = it.second;

                if(att.binding >= data.size())
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to build the general descriptor data for NodePass -> {} , the binding {} of {} is out of range , the bindings of GENERAL have to be 0 .. N-1" , passName , att.binding , it.first)});
                }

                if(filled[att.binding])
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to build the general descriptor data for NodePass -> {} , the binding {} is used twice" , passName , att.binding)});
                }

                const std::string resolvedName = _implResolveName(att , frameParity);
                const auto resourceIt = generalAttachments.find(resolvedName);
                if(resourceIt == generalAttachments.end())
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to build the general descriptor data for NodePass -> {} , there is no resource -> {}" , passName , resolvedName)});
                }

                const std::any& source = resourceIt->second.data;

                if(source.type() == typeid(Buffer::BufferBase2*))
                {
                    const auto d = std::any_cast<Buffer::BufferBase2*>(source);
                    if(!d) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding for NodePass , since the source is nullptr , name -> {}" , resolvedName)});
                    data[att.binding] = DescriptorData2::createData(d , att.type , att.binding , 0);
                }
                else if(source.type() == typeid(Image::RenderImage2*))
                {
                    const auto d = std::any_cast<Image::RenderImage2*>(source);
                    if(!d) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding for NodePass , since the source is nullptr , name -> {}" , resolvedName)});
                    data[att.binding] = DescriptorData2::createData(d , att.type , att.binding , 0);
                }
                else if(source.type() == typeid(Image::Skybox2*))
                {
                    const auto d = std::any_cast<Image::Skybox2*>(source);
                    if(!d) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding for NodePass , since the source is nullptr , name -> {}" , resolvedName)});
                    data[att.binding] = DescriptorData2::createData(d , att.type , att.binding , 0);
                }
                else if(source.type() == typeid(Image::Texture2*))
                {
                    const auto d = std::any_cast<Image::Texture2*>(source);
                    if(!d) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding for NodePass , since the source is nullptr , name -> {}" , resolvedName)});
                    data[att.binding] = DescriptorData2::createData(d , att.binding , 0);
                }
                else
                {
                    // an empty std::any means that RenderGraph didnt give this resource yet
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding for NodePass , the source has an unknown type or is empty , name -> {}" , resolvedName)});
                }

                filled[att.binding] = 1;
            }

            return data;
        }

        // creation for DescriptorLayout

        MOX_NO_DISCARD ErrorDataOutput<pDescriptorLayout> _implCreateGeneralDescriptorLayout(const VkShaderStageFlags stage) noexcept
        {
            try{
                if(!moxFILE) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create general layout , since the moxFILE isnt being created , graph name -> {}" , passName)});
                if(!__hasGeneralSet) return {};
                const auto setIt = moxFILE->sets.find(DescriptorFile::_moxGeneralSet);
                if(setIt == moxFILE->sets.end()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create general layout , since the file has no GENERAL set , graph name -> {}" , passName)});

                // one layout for both ping pong sides , so the layout is made from the attachments of the file ( once per binding ) , not from generalAttachments ( where ping pong is twice )
                const auto& attachments = setIt->second.attachments;
                std::vector<DescriptorLayoutData> data(attachments.size());
                std::vector<uint8_t> filled(attachments.size() , 0);
                for(const auto& it : attachments)
                {
                    const DescriptorAttachment& att = it.second;
                    if(att.binding >= data.size()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create Descriptor Layout for NodePass -> {}, since the binding {} is out of range , the bindings of GENERAL have to be 0 .. N-1" , passName , att.binding)});
                    if(filled[att.binding]) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create Descriptor Layout for NodePass -> {}, since the binding {} is used twice" , passName , att.binding)});

                    data[att.binding] = DescriptorLayoutData{.type = att.type , .count = att.count};
                    filled[att.binding] = 1;
                }

                DescriptorLayout::CreateInfo info{};
                info.context = context;
                info.data = data;
                info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                info.stage = stage;
                info.pNext = nullptr;

                return std::make_shared<DescriptorLayout>(info);
            }
            catch(std::exception &e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create General Descriptor Layout for NodePass"});
            }
        }

        // creation for Descriptor Buffer2
        // 1 buffer , or 2 for ping pong : the buffer number id is used in the frames with frameIndex % 2 == id , and it holds the resources which _implResolveName gives for that parity
        MOX_NO_DISCARD ErrorDataOutput<std::vector<pDescriptorBuffer2>> _implCreateDescriptorBuffers(DescriptorLayout* layout) noexcept __attribute__((nonnull))
        {
            if(!moxFILE) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create Descriptor buffers for general layout , since the moxFILE isnt being created , graph name -> {}" , passName)});

            try{
                if(!__hasGeneralSet) return {};
                const uint32_t count = containsPingPong ? 2 : 1;
                std::vector<pDescriptorBuffer2> buffers(count);
                __dbG.resize(count);
                for(uint32_t id = 0 ; id < count ; id++)
                {
                    auto data = _implBuildGeneralDescriptorData(id);
                    if(!data.has_value()) return std::unexpected(data.error());

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = std::move(data.value());
                    info.layout = layout;

                    buffers[id] = std::make_shared<DescriptorBuffer2>(info);
                    __dbG[id] = buffers[id].get();
                }

                return buffers;
            }
            catch(std::exception &e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create General Descriptor Buffers for NodePass , graph name -> {}" , passName)});
            }
        }

        void _implFill__dbSet()
        {
            const std::uint32_t id = containsPingPong ? context->sc->frameIndex % 2 : 0;
            [[unlikely]] if(__dbSet.empty()) __dbSet.resize(__dbBindingInfo.size());
            for(const auto& [location , name] : __dbNames | std::views::enumerate)
            {
                if(name == DescriptorFile::_moxGeneralSet) // if the set IS general
                {
                    __dbSet[location] = __dbG[id];
                }
                else
                {
                    __dbSet[location] = cSources[name].source->requestBuffer();
                }
            }
        }

        void _implBindDescriptorBuffers(VkCommandBuffer& cmd , const VkPipelineBindPoint& point , PipelineLayout* layout) noexcept __attribute__((nonnull))
        {
            if(__dbNames.empty() && !__hasGeneralSet)  return; // if there is NO source buffers required , and NO general set
            _implFill__dbSet();
            for(uint32_t i = 0 ; i < __dbSet.size() ; i++)
            {
                __dbBindingInfo[i].address = __dbSet[i]->pBuffer;
            }

            cmdBindDescriptorBuffersEXT(cmd , __dbBindingInfo.size() , __dbBindingInfo.data());
            cmdSetDescriptorBufferOffsetsEXT(cmd , point , layout->layout , 0 , __dbBindingInfo.size() , __dsets.data() , __doffsets.data());
        }
    };


}