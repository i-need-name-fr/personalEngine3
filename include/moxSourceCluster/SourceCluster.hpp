#pragma once

#include "../DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include <any>
// general idea of that class for GraphPass to get dlayout and dbuffers if it uses not only General one 

namespace mox
{

    // an attachment with PING_PONG_REQUIRED: 1 is written in the .mox file only once ( positionImage ) , but in the sources there are always TWO of them : positionImage_0 and positionImage_1
    // GraphPass3 and the one who fills the sources have to use this function , so the names always match
    inline std::string moxPingPongName(const std::string& baseName , const uint32_t index) noexcept
    {
        return baseName + "_" + std::to_string(index);
    }

    // which of the two resources ( _0 or _1 ) an attachment uses in the frame with the parity frameParity ( frameIndex % 2 )
    // pingPongCurrent is PING_PONG_CURRENT from the .mox file : 1 is the "next" one = the one which is written in this frame , 0 is the "prev" one = the other one
    // the barriers , the descriptor buffers of GraphPass3 and the descriptor buffers of a SourceCluster HAVE to use only this function
    inline uint32_t moxPingPongIndex(const uint32_t frameParity , const uint32_t pingPongCurrent) noexcept
    {
        return (frameParity + 1u + pingPongCurrent) % 2u;
    }

    using moxPingPongRenderImage2 = std::array<Image::pRenderImage2 , 2>;
    using moxPingPongSkybox2 = std::array<Image::pSkybox2 , 2>;
    using moxPingPongTexture2 = std::array<Image::pTexture2 , 2>;
    using moxPingPongBufferBase2 = std::array<Buffer::BufferBase2* , 2>;



    class SourceCluster
    {
    public:
        
        // no copy and no move : the cluster keeps raw pointers to the resources of its heirs , a moved cluster would point into the dead object
        SetTypicalOperators(SourceCluster)
        SourceCluster(SourceCluster&& input) = delete;
        SourceCluster& operator=(SourceCluster&& input) = delete;

        explicit SourceCluster(const std::string& name)
        {
            this->name = name;
        }

        MOX_NO_DISCARD virtual DescriptorBuffer2* requestBuffer() const noexcept
        {
            return _implRequestDSBuffer();
        }

        MOX_NO_DISCARD virtual DescriptorLayout* requestLayout() const noexcept
        {
            return _implRequestDSLayout();
        }

        MOX_NO_DISCARD ErrorDataOutput<std::variant<VkImageMemoryBarrier2 , VkBufferMemoryBarrier2>> requestBarrier(const std::string name , VkPipelineStageFlags2 const stage , VkAccessFlags2 const access ) noexcept
        {
            return _implAskForBarrier(name , stage , access);
        }

        const std::string requestName() const noexcept {return this->name;}

        virtual ~SourceCluster() noexcept
        {
            processDelete();
        }

    protected:

        // only the heirs can create it ( SourceBuffers and so on ) , the default constructor is deleted by SetTypicalOperators
        explicit SourceCluster(moxContext2* context , const std::string& name) noexcept
        {
            this->context = context;
            this->name = name;
        }

        virtual void processDelete() noexcept
        {
            dbuffers = {};
            dlayouts = {};
            sources = {};
            context = nullptr;
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<std::variant<VkImageMemoryBarrier2 , VkBufferMemoryBarrier2>> _implAskForBarrier(const std::string name , VkPipelineStageFlags2 const stage , VkAccessFlags2 const access) noexcept
        {
            if(!sources.contains(name)) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create Barrier for whatever this is , name -> {}" , name)});
            
            const auto data = sources.at(name);

            if(data.type() == typeid(Image::RenderImage2*))
            {
                const auto image = std::any_cast<Image::RenderImage2*>(data);
                if(!image)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create barrier for RenderImage2 in SourceCluster , since the image is nullptr , name -> {}", name)});
                }
                const auto barrier = image->translateImage(access , stage);
                return barrier;
            }

            if(data.type() == typeid(Image::Texture2*))
            {
                const auto image = std::any_cast<Image::Texture2*>(data);
                if(!image)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create barrier for Texture2 in SourceCluster , since the image is nullptr , name -> {}", name)});
                }
                const auto barrier = image->translateImage(access , stage);
                return barrier;
            }

            if(data.type() == typeid(Image::Skybox2*))
            {
                const auto image = std::any_cast<Image::Skybox2*>(data);
                if(!image)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create barrier for Skybox2 in SourceCluster , since the image is nullptr , name -> {}", name)});
                }
                const auto barrier = image->translateImage(access , stage);
                return barrier;
            }
            
            if(data.type() == typeid(Buffer::BufferBase2*))
            {
                const auto image = std::any_cast<Buffer::BufferBase2*>(data);
                if(!image)
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to create barrier for BufferBase in SourceCluster , since the image is nullptr , name -> {}", name)});
                }
                const auto barrier = image->translateBuffer(access , stage);
                return barrier;
            }

            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to creeate barrier for smth , due to SourceCluster containing undefined data , name -> {}", name)});
        }

        moxContext2* context{};
        uint8_t containsPingPong = false;
        std::vector<pDescriptorBuffer2> dbuffers{}; 
        std::vector<pDescriptorLayout> dlayouts{};
        std::unordered_map<std::string , std::any> sources{};
        std::string name{};

        // since the SourceCluster might have multiple ping pongs , we assume that there are only 2 , since i usually make only 2 , thats why its called Pingpong
        MOX_NO_DISCARD virtual DescriptorBuffer2* _implRequestDSBuffer() const noexcept
        {   
            if(dbuffers.empty()) return nullptr;

            if(containsPingPong)
            {
                const uint32_t id = context->sc->frameIndex % 2;
                return dbuffers[id].get();
            }
            else
            {
                return dbuffers[0].get();
            }
        }

        MOX_NO_DISCARD virtual DescriptorLayout* _implRequestDSLayout() const noexcept
        {   
            if(dlayouts.empty()) return nullptr;

            return dlayouts[0].get();
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(Buffer::BufferBase2* source , const std::string name) noexcept
        {
            if(!source) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to add Source for SourceCluster , since the pointer is nullptr , name -> {}", name )});
            if(sources.contains(name)) return {};
            sources.insert({name , source});
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(Image::RenderImage2* source , const std::string name) noexcept
        {
            if(!source) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to add Source for SourceCluster , since the pointer is nullptr , name -> {}", name )});
            if(sources.contains(name)) return {};
            sources.insert({name , source});
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(Image::Skybox2* source , const std::string name) noexcept
        {
            if(!source) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to add Source for SourceCluster , since the pointer is nullptr , name -> {}", name )});
            if(sources.contains(name)) return {};
            sources.insert({name , source});
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(AccelerationStructure* source , const std::string name) noexcept
        {
            if(!source) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to add Source for SourceCluster , since the pointer is nullptr , name -> {}", name )});
            if(sources.contains(name)) return {};
            sources.insert({name , source});
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(Image::Texture2* source , const std::string name) noexcept
        {
            if(!source) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to add Source for SourceCluster , since the pointer is nullptr , name -> {}", name )});
            if(sources.contains(name)) return {};
            sources.insert({name , source});
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(Sampler* source , const std::string name) noexcept
        {
            if(!source) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to add Source for SourceCluster , since the pointer is nullptr , name -> {}", name )});
            if(sources.contains(name)) return {};
            sources.insert({name , source});
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(moxPingPongRenderImage2 source , const std::string name) noexcept
        {
            {
                const auto result = _implAddSource(source[0].get() , name + "_0");
                if(!result.has_value()) return result;
            }

            {
                const auto result = _implAddSource(source[1].get() , name + "_1");
                if(!result.has_value()) return result;
            }
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(moxPingPongSkybox2 source , const std::string name) noexcept
        {
            {
                const auto result = _implAddSource(source[0].get() , name + "_0");
                if(!result.has_value()) return result;
            }

            {
                const auto result = _implAddSource(source[1].get() , name + "_1");
                if(!result.has_value()) return result;
            }
            return {};
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(moxPingPongTexture2 source , const std::string name) noexcept
        {
            {
                const auto result = _implAddSource(source[0].get() , name + "_0");
                if(!result.has_value()) return result;
            }

            {
                const auto result = _implAddSource(source[1].get() , name + "_1");
                if(!result.has_value()) return result;
            }
            return {};
        }

            MOX_NO_DISCARD virtual ErrorDataOutput<void> _implAddSource(moxPingPongBufferBase2 source , const std::string name) noexcept
        {
            {
                const auto result = _implAddSource(source[0] , name + "_0");
                if(!result.has_value()) return result;
            }

            {
                const auto result = _implAddSource(source[1] , name + "_1");
                if(!result.has_value()) return result;
            }
            return {};
        }
    };


    MOX_DECLARE_PTR_STRUCTURE(SourceCluster)
}