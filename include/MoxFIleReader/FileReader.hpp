#pragma once

#include "../vulkanContext/moxContext2.hpp"
#include <filesystem>
#include <unordered_map>
#include <fstream>
#include <cstdio>
#include <format>
namespace mox
{
    // general idea of this structure is to read data and output the required Sources

    inline static const std::string moxUndefinedBehavior = "3094t030fvhj0405";

    enum DescriptorAccess
    {
        moxDescriptorAccessUndefined,
        moxDescriptorAccessWrite,
        moxDescriptorAccessRead,
        moxDescriptorAccessBoth
    };

    #define MOX_GENERAL_SET_DEFAULT_INDEX 0ul

    struct DescriptorAttachment
    {
        // source
        std::string source{moxUndefinedBehavior};
        std::string name{moxUndefinedBehavior};
        VkDescriptorType type{};
        uint32_t pingPongSystem{0};
        uint32_t pingPongIndex{0};
        DescriptorAccess action{moxDescriptorAccessUndefined};
        uint32_t binding{0};
        // how many descriptors are in this binding , 1 for a simple one , more for an array
        uint32_t count{1};
    };

    struct SetData
    {
        std::string source{moxUndefinedBehavior};
        uint32_t index{0};
        // every attachment belongs to the set which is written above it in the file
        std::unordered_map<std::string , DescriptorAttachment> attachments{};
    };

    struct FileData
    {
        std::unordered_map<std::string , SetData> sets{};
        uint8_t containsPingPong{};
        uint32_t totalBindingUsage{};
    };

    class DescriptorFile final
    {
    public:
        inline static const std::string _moxGeneralSet = "GENERAL";

        MOX_NO_DISCARD inline static ErrorDataOutput<mox::FileData> AnalyzeFile(const std::filesystem::path &path) noexcept
        {
            return _implAnalyzeFile(path);
        }

        // the key of an attachment in SetData::attachments . the NAME is not enough : a ping pong resource can be declared twice in one set , as the "next" one ( PING_PONG_CURRENT: 1 ,
        // the one that is written in this frame ) and as the "prev" one ( PING_PONG_CURRENT: 0 ) , both with the same NAME , since the shader has two variables for it
        // so for a ping pong attachment PING_PONG_CURRENT is a part of the key . the real name is DescriptorAttachment::name , NOT the key
        inline static std::string AttachmentKey(const std::string& name , const uint32_t pingPongRequired , const uint32_t pingPongCurrent) noexcept
        {
            if(!pingPongRequired) return name;
            return name + "@" + std::to_string(pingPongCurrent);
        }

        inline static VkAccessFlags2 TranslateToAccess(DescriptorAccess const &access) noexcept
        {
            if(access == moxDescriptorAccessWrite) return VK_ACCESS_2_SHADER_WRITE_BIT;
            if(access == moxDescriptorAccessRead) return VK_ACCESS_2_SHADER_READ_BIT;
            if(access == moxDescriptorAccessBoth) return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            return VK_ACCESS_2_NONE;
        }

    private:

        // buffer
        inline static const std::string _DescriptorTypeUniformBuffer = "UNIFORM_BUFFER";
        inline static const std::string _DescriptorTypeStorageBuffer = "STORAGE_BUFFER";
        // image
        inline static const std::string _DescriptorTypeTexture = "TEXTURE";
        inline static const std::string _DescriptorTypeStorageImage = "STORAGE_IMAGE";
        // sampler
        inline static const std::string _DescriptorTypeSampler = "SAMPLER";
        // TLAS
        inline static const std::string _DescriptorTypeAS = "AS";

        inline static const std::string _moxActionTypeUndefined = "UNDEFINED";
        inline static const std::string _moxActionTypeWrite = "WRITE";
        inline static const std::string _moxActionTypeRead = "READ";
        inline static const std::string _moxActionTypeBoth = "BOTH";

        MOX_NO_DISCARD inline static ErrorDataOutput<DescriptorAccess> _implAnalyzeAction(const std::string action) noexcept
        {
            if(action == _moxActionTypeUndefined)
            {
                return moxDescriptorAccessUndefined;
            }
            else if(action == _moxActionTypeWrite)
            {
                return moxDescriptorAccessWrite;
            }
            else if(action == _moxActionTypeRead)
            {
                return moxDescriptorAccessRead;
            }
            else if(action == _moxActionTypeBoth)
            {
                return moxDescriptorAccessBoth;
            }

            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to analyze descriptor action , {}" , action)});
        }

        MOX_NO_DISCARD inline static ErrorDataOutput<VkDescriptorType> _implAnalyzeDescriptorType(std::string const type) noexcept
        {
            if(type == _DescriptorTypeUniformBuffer)
            {
                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            }
            else if(type == _DescriptorTypeStorageBuffer)
            {
                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

            }
            else if(type == _DescriptorTypeTexture)
            {
                return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

            }
            else if(type == _DescriptorTypeStorageImage)
            {
                return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

            }
            else if(type == _DescriptorTypeSampler)
            {
                return VK_DESCRIPTOR_TYPE_SAMPLER;

            }
            else if(type == _DescriptorTypeAS)
            {
                return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            }
            
            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to analyze descriptor type , {}" , type)});
        }

        MOX_NO_DISCARD inline static ErrorDataOutput<mox::FileData> _implAnalyzeFile(const std::filesystem::path& path) noexcept
        {
            namespace fs = std::filesystem;
            MOX_UNLIKELY if(!fs::exists(path))
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to process File | {} | , since it doesnt even exist" , path.string())});
            }

            mox::FileData data{};
            data.containsPingPong = false;
            data.totalBindingUsage = 0;
            std::ifstream file(path.string() , std::ios::in);
            MOX_UNLIKELY if(!file.is_open()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to process File | {} | , due to fail at opening the file" , path.string())});

            std::string line{};
            uint32_t lineNumber = 0;
            while(std::getline(file , line))
            {
                lineNumber++;
                // windows line ending and the spaces at the end do not matter
                while(!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
                if(line.empty()) continue;

                const char* l = line.c_str();
                int consumed = 0;

                if(line.starts_with("[[SET_DATA]]"))
                {
                    char name[64]{};
                    uint32_t index = 0;
                    // %n is not counted in the result , it only says how many characters were read , so we can check that nothing is left at the end
                    const int n = std::sscanf(l , "[[SET_DATA]] SET_NAME: %63s SET_INDEX: %u%n" , name , &index , &consumed);
                    if(n != 2 || l[consumed] != '\0')
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : wrong [[SET_DATA]] , expected : [[SET_DATA]] SET_NAME: <name> SET_INDEX: <index>" , path.string() , lineNumber)});
                    }

                    mox::SetData set{};
                    set.index = index;
                    set.source = name;
                    set.attachments = {};
                    if(data.sets.contains(set.source))
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to analyze the file , since Set has been declared 2 times , file -> {}" , path.string())});
                    }

                    if(set.source == _moxGeneralSet)
                    {
                        if(set.index != MOX_GENERAL_SET_DEFAULT_INDEX)
                        {
                            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to analyze the file , since the general set is NOT index 0 , general ID -> {} , file ->{}" , set.index , path.string())});
                        }
                    }

                    data.sets.insert(std::pair<std::string , mox::SetData>(set.source , set));
                }
                else if(line.starts_with("[[ATTACHMENT_DATA]]"))
                {
                    if(data.sets.empty())
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : [[ATTACHMENT_DATA]] before any [[SET_DATA]]" , path.string() , lineNumber)});
                    }

                    char source[128]{};
                    char name[128]{};
                    char type[128]{};
                    char action[128]{};
                    uint32_t pingPongRequired = 0;
                    uint32_t pingPongIndex = 0;
                    uint32_t binding = 0;
                    uint32_t count = 0;

                    const int n = std::sscanf(l , "[[ATTACHMENT_DATA]] DESCRIPTOR_SOURCE: %127s NAME: %127s DESCRIPTOR_TYPE: %127s PING_PONG_REQUIRED: %u PING_PONG_CURRENT: %u ACTION: %127s BINDING: %u COUNT: %u%n" ,
                                            source , name , type , &pingPongRequired , &pingPongIndex , action , &binding , &count , &consumed);
                    if(n != 8 || l[consumed] != '\0')
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : wrong [[ATTACHMENT_DATA]] , the fields have to be in the exact order of the format" , path.string() , lineNumber)});
                    }

                    if(pingPongRequired > 1 || pingPongIndex > 1)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : PING_PONG_REQUIRED and PING_PONG_CURRENT can only be 0 or 1" , path.string() , lineNumber)});
                    }

                    if(count == 0)
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : COUNT can not be 0" , path.string() , lineNumber)});
                    }

                    const auto dsTypeResult = _implAnalyzeDescriptorType(type);
                    if(!dsTypeResult.has_value()) return std::unexpected(dsTypeResult.error());

                    const auto accessResult = _implAnalyzeAction(action);
                    if(!accessResult.has_value()) return std::unexpected(accessResult.error());

                    // DESCRIPTOR_SOURCE is the name of the set the attachment belongs to , and the set has to be declared above
                    const auto setIt = data.sets.find(source);
                    if(setIt == data.sets.end())
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("File | {} | line {} : the attachment links to the Set {} which is not declared above it" , path.string() , lineNumber , source)});
                    }

                    mox::SetData& set = setIt->second;

                    const std::string key = AttachmentKey(name , pingPongRequired , pingPongIndex);
                    if(set.attachments.contains(key))
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : attachment {} is already declared in the set {}" , path.string() , lineNumber , name , set.source)});
                    }

                    // the same NAME twice is allowed only for a ping pong resource ( next + prev ) : both declarations have to be ping pong , and they differ by PING_PONG_CURRENT ( the key )
                    for(const auto& other : set.attachments)
                    {
                        if(other.second.name == name && !(other.second.pingPongSystem && pingPongRequired))
                        {
                            return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : attachment {} is already declared in the set {} , the same NAME twice is allowed only for a ping pong resource ( one PING_PONG_CURRENT: 1 and one PING_PONG_CURRENT: 0 )" , path.string() , lineNumber , name , set.source)});
                        }
                    }

                    const std::string sSource = source;
                    if(sSource == "GENERAL")
                    {
                        for(const auto& other : set.attachments)
                        {
                            if(other.second.binding == binding)
                            {
                                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : binding {} is already used in the set {}" , path.string() , lineNumber , binding , set.source)});
                            }
                        }
                    }

                    mox::DescriptorAttachment attachment{};
                    attachment.action = accessResult.value();
                    attachment.binding = binding;
                    attachment.count = count;
                    attachment.name = name;
                    attachment.pingPongIndex = pingPongIndex;
                    attachment.pingPongSystem = pingPongRequired;
                    attachment.source = source;
                    attachment.type = dsTypeResult.value();
                    set.attachments.emplace(key , std::move(attachment));

                    if(pingPongRequired)
                    {
                        data.containsPingPong = true;
                    }

                    data.totalBindingUsage++;
                }
                else
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("File | {} | line {} : unknown line , only [[SET_DATA]] and [[ATTACHMENT_DATA]] are allowed" , path.string() , lineNumber)});
                }
            }

            return data;
        }
    };

}