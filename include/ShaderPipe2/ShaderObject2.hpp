#pragma once

#include "../vulkanContext/moxContext2.hpp"
#include <filesystem>
#include "../DescriptorLayout/DescriptorLayout.hpp"
#include "../DXC/DXCManager.hpp"

namespace mox{

    class ShaderObject2 final{
    public:

        ShaderObject2() = delete;
        ShaderObject2(ShaderObject2& input) = delete;
        ShaderObject2& operator=(ShaderObject2& input) = delete;
        ShaderObject2(ShaderObject2&& input) noexcept{
            performCopy(std::move(input));
        }

        ShaderObject2& operator=(ShaderObject2&& input) noexcept{
            if(this == &input) return *this;
            // performCopy only takes the handle of the other shader , the shader that this object held would stay alive for ever
            if(device && shader) destroyShaderEXT(device , shader , nullptr);
            shader = nullptr;
            performCopy(std::move(input));

            return *this;
        }

        struct CreateInfo{
            moxContext2* context;

            std::vector<DescriptorLayout*> layouts;
            std::vector<VkPushConstantRange> pushes;

            VkShaderStageFlagBits currentStage;
            VkShaderStageFlags nextStage;

            std::string filePath{};
        };

        ShaderObject2(const CreateInfo &info){
            const auto result = createShader(info);
            if(!result.has_value()){
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        ~ShaderObject2(){
            if(device){
                if(shader) destroyShaderEXT(device , shader , nullptr);
            }

            shader=  nullptr;
            device = nullptr;
        }

        VkShaderEXT shader = nullptr;
    private:

        void performCopy(ShaderObject2&& input) noexcept{
            shader = std::exchange(input.shader , nullptr);
            device = std::exchange(input.device,  nullptr);
        }

        [[nodiscard]] ErrorDataOutput<void> createShader(const CreateInfo &data) noexcept{

            if(!data.context){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the data context is nullptr"});
            }

            if(data.filePath.empty()){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the file path is nullptr"});
            }

            if(!data.context->logicalDevice){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , since the context logical device  is nullptr"});
            }

            device = data.context->logicalDevice->device;

            std::vector<VkDescriptorSetLayout> layouts{};
            for(auto& layout : data.layouts){
                if(!layout) continue;
                layouts.push_back(layout->layout);
            }

            if(layouts.size() != data.layouts.size()) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader object , due to some descriptor layouts being nullptr"});

            // every shader is compiled at every start ( no cache )
            return compileShaderObject(data, layouts);
        }

        VkDevice device = nullptr;

        [[nodiscard]] ErrorDataOutput<void> compileShaderObject(const CreateInfo &data , const std::vector<VkDescriptorSetLayout>& layouts) noexcept {

            // the passes can give the path with any extension ( getShaderPath ) : the source is always the .hlsl with the same name
            std::filesystem::path hlslPath(data.filePath);
            hlslPath.replace_extension(".hlsl");

            if(!std::filesystem::exists(hlslPath)){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to compile the shader , since the file name is not valid -> Path: {}" , hlslPath.string())});
            }

            const auto spirv = DXCManager::Compile(hlslPath);
            if(!spirv.has_value()) return std::unexpected(spirv.error());

            const std::vector<uint8_t>& buffer = spirv.value();

            VkShaderCreateInfoEXT info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
            info.pNext = nullptr;
            info.flags = 0;
            info.pSpecializationInfo = nullptr;
            info.codeSize = buffer.size();
            info.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
            info.nextStage = data.nextStage;
            info.pCode= buffer.data();
            info.stage = data.currentStage;
            info.setLayoutCount = static_cast<uint32_t>(data.layouts.size());
            info.pSetLayouts = layouts.data();
            info.pushConstantRangeCount = static_cast<uint32_t>(data.pushes.size());
            info.pPushConstantRanges = data.pushes.data();
            info.pName ="main";

            const auto result = createShaderEXT(device , 1 , &info , nullptr ,&shader);
            if(result != VK_SUCCESS){
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader object, Error ID -> {}", static_cast<int32_t>(result))});
            }
            return {};
        }

    };

    using pShaderObject2 = std::shared_ptr<ShaderObject2>;
}
