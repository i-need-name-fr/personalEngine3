#pragma once

#include "NodePass.hpp"

namespace mox
{

    struct moxNodeRenderParameter
    {
        std::any source{};
        moxNodeBindingType type{};
    };

    template<typename T>
    concept _isPointer_ = std::is_pointer_v<T>;

    template<typename T>
    concept _moxNodeRenderType_ = _isPointer_<T> && (std::is_same_v<T , Image::RenderImage2*> || std::is_same_v<T , Image::Texture2*> || std::is_same_v<T , Image::Skybox2*> || std::is_same_v<T , Buffer::BufferBase2*>);

    class RenderGraph final
    {
    public:

        RenderGraph() = default;
        RenderGraph(RenderGraph& input) = delete;
        RenderGraph(const RenderGraph& input) = delete;
        RenderGraph& operator=(RenderGraph& input) = delete;
        RenderGraph& operator=(const RenderGraph& input) = delete;

        RenderGraph(RenderGraph&& input) noexcept
        {
            processCopy(std::move(input));
        }

        RenderGraph& operator=(RenderGraph&& input) noexcept
        {
            if(this == &input) return *this;
            processCopy(std::move(input));
            return *this;
        }

        template<typename T>
        requires _moxNodeRenderType_<T>
        MOX_NO_DISCARD ErrorDataOutput<void> bindSource(T source , const std::string name) noexcept 
        {
            return _implBindSource(source , name);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> bindSourceCluster(SourceCluster* source) noexcept __attribute__((nonnull))
        {
            return _implBindCluster(source);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> bindNodePass(NodePass* pass , const std::string& name) noexcept
        {
            return _implProcessNodePass(pass , name);
        }

        MOX_NO_DISCARD ErrorDataOutput<void> executePath(VkCommandBuffer &cmd , const std::string& path) noexcept
        {
            return _implExecutePath(cmd , path);
        }

    private:

        void processCopy(RenderGraph&& input) noexcept
        {
            processDelete();
            graphs = std::move(input.graphs);
            generalBuffer = std::move(input.generalBuffer);
            clusters = std::move(input.clusters);
        }

        void processDelete() noexcept
        {
            graphs.clear();
            generalBuffer.clear();
            clusters.clear();
        }

        std::unordered_map<std::string , std::vector<NodePass*>> graphs{};
        std::unordered_map<std::string , moxNodeRenderParameter> generalBuffer{};
        std::unordered_map<std::string ,SourceCluster*> clusters{};

        // add source

        template<typename T>
        requires _moxNodeRenderType_<T>
        MOX_NO_DISCARD ErrorDataOutput<void> _implBindSource(T source , const std::string name) noexcept
        {
            try{
                constexpr bool isBuffer = std::is_same_v<T , Buffer::BufferBase2*>;
                constexpr const char* typeName = isBuffer ? "BufferBase2" : "image";

                if(name.empty())
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process bind the {} for GR , since name is empty" , typeName)});
                }

                if(!source)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process bind the {} for GR , since the pointer is null , name -> {}" , typeName , name)});
                }

                // the required handle inside the class
                if constexpr(isBuffer)
                {
                    if(!source->buffer)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process bind the BufferBase2 for GR , since the VkBuffer is nullptr , name -> {}", name)});
                    }
                }
                else // RenderImage2 , Texture2 , Skybox2
                {
                    if(!source->view)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process bind the image for GR , since the image View is nullptr , name -> {}", name)});
                    }
                }

                const auto it = generalBuffer.find(name);
                if(it != generalBuffer.end()) // if it exists
                {
                    if(it->second.source.type() != typeid(T)) // if its not the same type
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process bind the {} for GR , since the generalBuffer already contains the name , but with a different type , consider to change the name of the binding , name -> {}", typeName , name)});
                    }

                    return {}; // if the type is the same , and it already exists
                }

                moxNodeRenderParameter data{};
                data.type = isBuffer ? moxNodeBindingBuffer : moxNodeBindingImage;
                data.source = source;

                generalBuffer.insert({name , std::move(data)});

                return {};
            }
            catch(std::exception &e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process bind the source for GR , exception , name -> {}" , name)});
            }
        }

        // add Source Cluster

        MOX_NO_DISCARD ErrorDataOutput<void> _implBindCluster(SourceCluster* source) noexcept
        {

            if(!source) 
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process GR Source cluster , since its null pointer"});
            }

            const std::string name = source->requestName();
            if(name.empty())
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process Source Cluster , since the name of it is nulltr"});
            }

            if(clusters.contains(name))
            {
                if(clusters[name] != source)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding the SOurce Graph for GR , since it already exists in GR , but different pointer , consider to rename the Source graph , name -> {}"  , name)});
                }
                return {};
            }

            clusters.insert(std::pair<std::string , SourceCluster*>(name , source));
            return {};
        }
        // add nodePass

        MOX_NO_DISCARD ErrorDataOutput<void> _implProcessNodePass(NodePass* pass , const std::string &path) noexcept
        {

            if(!pass)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process binding the NodePass , since it contains the nullptr"});
            }   

            if(path.empty())
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to process binding the nodePass , since the path of which GR to use , is null"});
            }

            const std::string name = pass->getPassName();
            if(name.empty())
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding the nodePass , since the name of pass itself is empty , source -> {}" , path)});
            }

            if(const auto pathIt = graphs.find(path); pathIt != graphs.end()) // the same pass can not be added twice to one path
            {
                const auto& list = pathIt->second;
                const auto found = std::find_if(list.begin() , list.end() , [&](const NodePass* other) -> bool {return other == pass || other->getPassName() == name;});
                if(found != list.end())
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding the nodePass , since the pass already exists in the path , pass -> {} , path -> {}" , name , path)});
                }
            }

            // now we have to find all the requested Attachments and Sources 

            if(!pass->is_complete())
            {
                {
                    const auto requestResult = pass->sendRequest();
                    if(!requestResult.has_value()) return std::unexpected(requestResult.error());
                    const auto& request = requestResult.value();
                    std::vector<moxNodeRenderFeedback> fd;
                    fd.reserve(request.size());
                    for(const auto& rq : request)
                    {
                        // request can be 2 types , from general buffer of GR , or from source cluster
                        if(rq.requestType == moxNodePassRequestSet)
                        {

                            if(rq.name == moxNodeRenderGeneralSet) continue;

                            const auto it = clusters.find(rq.name); // Cluster cannot have nullptr , since there is a check in adding the source
                            if(it == clusters.end())
                            {
                                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process NodePass creation , name -> {} , since the cluster requested - doesnt exist , Cluster Source Requested -> {}" , name , rq.name)});
                            }

                            moxNodeRenderFeedback data{};
                            data.type = moxNodePassRequestSet;
                            data.name = rq.name;
                            data.set = it->second;
                            data.attachment = {};

                            fd.push_back(data);
                        }   
                        else if(rq.requestType == moxNodePassRequestAttachment)
                        {
                            if(!generalBuffer.contains(rq.name)) // attachment doesnt exist in the general buffer of GR
                            {
                                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding for NodePass name -> {} , due to attachment not existing , attachment name -> {}" , name , rq.name)});
                            }

                            // now we check the type
                            const auto& contained = generalBuffer.at(rq.name);
                            if(contained.type != rq.bindingType) // the type isnt the same as the provided one
                            {
                                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to process binding for NodePass name -> {} , due to attachment being different type , attachment name -> {}" , name , rq.name)});
                            }

                            moxNodeRenderFeedback data{};
                            data.type = moxNodePassRequestAttachment;
                            data.name = rq.name;
                            data.attachment = contained.source;
                            data.set = nullptr;

                            fd.push_back(data);
                        }
                        else return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to analyze the requested NodePass , since the request type is undefined , pass -> {}" , name)});
                    }

                    {
                        const auto result = pass->analyzeFeedBack(fd);
                        if(!result.has_value()) return result;
                    }

                    {
                        const auto result = pass->initPass();
                        if(!result.has_value()) return result;
                    }
                }

                pass->mark_completed();
            }

            if(graphs.contains(path))
            {
                graphs[path].push_back(pass);
            }
            else
            {
                graphs.insert({path , {pass}});
            }

            return {};
        }

        // execute passes
        MOX_NO_DISCARD ErrorDataOutput<void> _implExecutePath(VkCommandBuffer &cmd , const std::string& path) noexcept
        {

            if(!graphs.contains(path))
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to execute GR , since the path is not existing -> {}" , path)});
            }

            for(auto& pass : graphs.at(path))
            {
                MOX_UNLIKELY if(!pass)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to execute one of the passes , path -> {}" , path)});
                }

                if(pass->requestExecution())
                {
                    const auto result = pass->executePass(cmd);
                    if(!result.has_value()) return result;
                }
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(RenderGraph)
}