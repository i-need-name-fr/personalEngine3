#pragma once

#include "../MoxFIleReader/FileReader.hpp"


namespace mox
{
namespace MDL
{
    class Compiler final
    {
    public:

        inline static ErrorDataOutput<FileData> Compile(const std::filesystem::path &path) noexcept
        {
            return _implCompileFile(path);
        }

    private:

        inline static constexpr std::string_view StorageBuffer = "StorageBuffer";
        inline static constexpr std::string_view UniformBuffer = "UniformBuffer";
        inline static constexpr std::string_view StorageImage = "StorageImage";
        inline static constexpr std::string_view Texture = "Texture";
        inline static constexpr std::string_view ConstantBuffer = "ConstantBuffer";
        inline static constexpr std::string_view SampledImage = "SampledImage";

        inline static constexpr std::string_view ShaderRead = "ShaderRead";
        inline static constexpr std::string_view ShaderWrite = "ShaderWrite";
        inline static constexpr std::string_view ShaderBoth = "ShaderBoth";
        inline static constexpr std::string_view MemoryRead = "MemoryRead";
        inline static constexpr std::string_view MemoryWrite = "MemoryWrite";
        inline static constexpr std::string_view MemoryBoth = "MemoryBoth";
        inline static constexpr std::string_view ColorWrite = "ColorWrite";
        inline static constexpr std::string_view ColorRead = "ColorRead";
        inline static constexpr std::string_view DepthRead = "DepthRead";
        inline static constexpr std::string_view DepthBoth = "DepthBoth";
        inline static constexpr std::string_view ColorBoth = "ColorBoth";
        inline static constexpr std::string_view DepthWrite = "DepthWrite";

        inline static constexpr std::string_view PPCurrent = "CURRENT";
        inline static constexpr std::string_view PPPrev = "PREV";

        inline static VkDescriptorType TranslateToDescriptorType(const std::string_view name) noexcept
        {
            if(name == StorageBuffer) return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            if(name == UniformBuffer || name == ConstantBuffer) return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            if(name == StorageImage) return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            if(name == Texture || name == SampledImage) return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            return VK_DESCRIPTOR_TYPE_MAX_ENUM;   // unknown name
        }

        inline static VkAccessFlags2 TranslateToAccessFlags(const std::string_view name) noexcept
        {
            if(name == ShaderRead)  return VK_ACCESS_2_SHADER_READ_BIT;
            if(name == ShaderWrite) return VK_ACCESS_2_SHADER_WRITE_BIT;
            if(name == ShaderBoth)  return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

            if(name == MemoryRead)  return VK_ACCESS_2_MEMORY_READ_BIT;
            if(name == MemoryWrite) return VK_ACCESS_2_MEMORY_WRITE_BIT;
            if(name == MemoryBoth)  return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

            if(name == ColorRead)   return VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
            if(name == ColorWrite)  return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            if(name == ColorBoth)   return VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

            if(name == DepthRead)   return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            if(name == DepthWrite)  return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            if(name == DepthBoth)   return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            return VK_ACCESS_2_NONE;   // unknown name
        }

        inline static constexpr uint32_t maxBuffer = 256;

        inline static ErrorDataOutput<void> _implProcessAttachmentDeclaration(const std::string __declaration , DescriptorAttachment& __descriptor) noexcept
        {   
            if(__declaration.empty())
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to process file , since the descriptor in the file doesnt have a declaration <<>> , descriptor name -> {}" , __descriptor.name));
            }

            const std::string errFormat = std::format("failed to process descriptor in the mox file , since the declaration <<>> has unsuitable format , declaration [[{}]] , descriptor name -> {}" , __declaration , __descriptor.name);
            const bool general = (__descriptor.source == "GENERAL");

            int32_t binding = 999;
            char access[maxBuffer];
            char state[maxBuffer];
            bool pingPong = false;

            if(general)
            {
                if(sscanf(__declaration.c_str() , "%d , %255s , %255s" , &binding , state , access) == 3)
                {
                    pingPong = true;
                }
                else if(sscanf(__declaration.c_str() , "%d , %255s" , &binding , access) != 2)
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , errFormat);
                }
            }
            else
            {
                if(sscanf(__declaration.c_str() , "%255s , %255s" , state , access) == 2)
                {
                    pingPong = true;
                }
                else if(sscanf(__declaration.c_str() , "%255s" , access) != 1)
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , errFormat);
                }
            }

            __descriptor.pingPongSystem = 0;
            __descriptor.pingPongIndex = 0;
            if(pingPong)
            {
                const std::string ppState = state;
                if(ppState == PPCurrent)
                {
                    __descriptor.pingPongIndex = 1;
                }
                else if(ppState != PPPrev)
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , errFormat);
                }
                __descriptor.pingPongSystem = 1;
            }

            const std::string action = access;
            __descriptor.action = TranslateToAccessFlags(action);
            if(__descriptor.action == VK_ACCESS_2_NONE)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to process descriptor in the mox file , since the access [[{}]] is unknown , declaration [[{}]] , descriptor name -> {}" , action , __declaration , __descriptor.name));
            }

            __descriptor.binding = static_cast<uint32_t>(binding);
            __descriptor.count = 1;
            return {};
        }
        
        inline static ErrorDataOutput<void> _implProcessVariable( const std::string line ,DescriptorAttachment& descriptor) noexcept
        {
            char type[maxBuffer];
            char name[maxBuffer];
            char declaration[maxBuffer];
            int consumed = 0;
            // %s stops only on a space , so the name and the declaration have to be read with scansets
            if(sscanf(line.c_str() , "%255s %255[^ <] <<%255[^>]>>%n" , type , name , declaration , &consumed) == 3 && line[consumed] == '\0')
            {
                const std::string sType = type;
                const std::string sName = name;
                const std::string sDecleration = declaration;

                // the set already has source
                VkDescriptorType dType = TranslateToDescriptorType(type);
                if(dType == VK_DESCRIPTOR_TYPE_MAX_ENUM)
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to process descriptor , since the format is undefined , type -> {}" , sType));
                }

                descriptor.type = dType;
                descriptor.name = sName;
                {
                    const auto result = _implProcessAttachmentDeclaration(sDecleration , descriptor);
                    MOX_CHECK_RESULT(result)
                }

                return {};
            }
            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to process line , undefined format , line -> {}" , line));
        }

        inline static ErrorDataOutput<FileData> _implCompileFile(const std::filesystem::path filePath) noexcept
        {
            if(!std::filesystem::exists(filePath))
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to compile mox file , since it doesnt exist , file -> {}" , filePath.string()));
            }

            std::ifstream file(filePath.string(), std::ios::in);
            MOX_UNLIKELY if(!file.is_open()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to process File | {} | , due to fail at opening the file" , filePath.string())});

            uint32_t lineNumber = 0;
            std::optional<std::string> currentOpenedSource{};
            bool openedBrackets{false};
            std::string line{};

            FileData data{};

            /*
            
                struct SetData
                {
                    std::string source{moxUndefinedBehavior};
                    uint32_t index{0};
                    // every attachment belongs to the set which is written above it in the file
                    std::unordered_map<std::string , DescriptorAttachment> attachments{};
                };
            
            */

            const std::string fileName = filePath.string();

            while(std::getline(file , line))
            {
                lineNumber++;
                while(!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
                if(line.empty()) continue;

                if(line.starts_with("//")) continue;

                if(line.starts_with("#include"))
                {
                    char source[maxBuffer];
                    int32_t index = -1;
                    int consumed = 0;
                    if(sscanf(line.c_str() , "#include %255[^ <] <<%d>>%n" , source , &index , &consumed) != 2 || line[consumed] != '\0' || index < 0)
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : wrong #include , expected : #include <SetName><<index>> , line -> {}" , fileName , lineNumber , line));
                    }

                    const std::string sSource = source;

                    if(sSource == "GENERAL" && static_cast<uint64_t>(index) != MOX_GENERAL_SET_DEFAULT_INDEX)
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the general set has to have the index {} , but it is {}" , fileName , lineNumber , MOX_GENERAL_SET_DEFAULT_INDEX , index));
                    }

                    if(data.sets.contains(sSource))
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the set {} is already included" , fileName , lineNumber , sSource));
                    }

                    // the index of a set has to be unique , otherwise 2 sets would be bound to the same descriptor set
                    for(const auto& other : data.sets)
                    {
                        if(other.second.index == static_cast<uint32_t>(index))
                        {
                            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the index {} of the set {} is already used by the set {}" , fileName , lineNumber , index , sSource , other.second.source));
                        }
                    }

                    SetData set{};
                    set.attachments = {};
                    set.index = static_cast<uint32_t>(index);
                    set.source = sSource;

                    data.sets.insert({sSource , set});
                    continue;
                }

                if(line.starts_with("{"))
                {
                    // the source should be declared before the brackets , for example [SceneBuffers] and then {
                    if(!currentOpenedSource)
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the bracket {{ has no set before it , expected : [SetName] and then {{" , fileName , lineNumber));
                    }

                    if(openedBrackets)
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the bracket {{ is opened twice , the set {} is not closed" , fileName , lineNumber , currentOpenedSource.value()));
                    }

                    openedBrackets = true;

                    continue;
                }

                if(line.starts_with("}"))
                {
                    if(!currentOpenedSource || !openedBrackets)
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the bracket }} has no opened set , every }} needs [SetName] and {{ before it" , fileName , lineNumber));
                    }

                    openedBrackets = false;
                    currentOpenedSource.reset();

                    continue;
                }

                if(line.starts_with("["))
                {
                    char source[maxBuffer];
                    int consumed = 0;
                    if(sscanf(line.c_str() , "[%255[^]]]%n" , source , &consumed) != 1 || line[consumed] != '\0')
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : wrong set line , expected : [SetName] , line -> {}" , fileName , lineNumber , line));
                    }

                    if(currentOpenedSource)
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the set {} is not closed with }} , but the set [{}] starts" , fileName , lineNumber , currentOpenedSource.value() , source));
                    }

                    const std::string sSource = source;
                    if(!data.sets.contains(sSource))
                    {
                        if(sSource != "GENERAL")
                        {
                            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : no set {} was found , it has to be included above : #include {}<<index>>" , fileName , lineNumber , sSource , sSource));
                        }

                        // GENERAL can be used without #include , it is always the index 0
                        SetData general{};
                        general.source = sSource;
                        general.index = static_cast<uint32_t>(MOX_GENERAL_SET_DEFAULT_INDEX);
                        for(const auto& other : data.sets)
                        {
                            if(other.second.index == general.index)
                            {
                                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the index {} of the general set is already used by the set {}" , fileName , lineNumber , general.index , other.second.source));
                            }
                        }
                        data.sets.insert({sSource , general});
                    }

                    currentOpenedSource = sSource;

                    continue;
                }

                // other lines are about variables

                if(!currentOpenedSource || !openedBrackets)
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the variable has no set , it has to be inside [SetName] {{ ... }} , line -> {}" , fileName , lineNumber , line));
                }

                DescriptorAttachment descriptor{};
                descriptor.source = currentOpenedSource.value();

                {
                    const auto result = _implProcessVariable(line , descriptor );
                    if(!result.has_value())
                    {
                        // the error of the line gets the place in the file , the type stays the same
                        MOX_RETURN_ERROR(result.error().first , std::format("File | {} | line {} : {}" , fileName , lineNumber , result.error().second));
                    }
                }

                SetData& set = data.sets.at(descriptor.source);

                // the same NAME twice is allowed only for a ping pong resource ( CURRENT + PREV ) , they differ by the key
                const std::string key = DescriptorFile::AttachmentKey(descriptor.name , descriptor.pingPongSystem , descriptor.pingPongIndex);
                if(set.attachments.contains(key))
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the descriptor {} is already declared in the set {}" , fileName , lineNumber , descriptor.name , set.source));
                }

                for(const auto& other : set.attachments)
                {
                    if(other.second.name == descriptor.name && !(other.second.pingPongSystem && descriptor.pingPongSystem))
                    {
                        MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the descriptor {} is already declared in the set {} , the same name twice is allowed only for a ping pong resource ( CURRENT and PREV )" , fileName , lineNumber , descriptor.name , set.source));
                    }
                }

                // the binding is written in the file only in the GENERAL set , so only there it has to be unique
                if(descriptor.source == "GENERAL")
                {
                    for(const auto& other : set.attachments)
                    {
                        if(other.second.binding == descriptor.binding)
                        {
                            MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | line {} : the binding {} is already used in the GENERAL set by {} , descriptor -> {}" , fileName , lineNumber , descriptor.binding , other.second.name , descriptor.name));
                        }
                    }
                }

                if(descriptor.pingPongSystem) data.containsPingPong = true;
                data.totalBindingUsage++;

                set.attachments.insert({key , descriptor});
            }

            if(currentOpenedSource)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("File | {} | the set {} is not closed , the file ended without }}" , fileName , currentOpenedSource.value()));
            }

            file.close();
            return data;
        }
    };
}
}