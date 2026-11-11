#pragma once

#include "../vulkanContext/moxContext2.hpp"
#include <filesystem>
#include "../DXC/DXCManager.hpp"

namespace mox{

    class ShaderModule2 final{
    private:

        [[nodiscard]] ErrorDataOutput<void> createShaderModule(const std::string path) noexcept {

            // the passes can give the path with any extension ( getShaderPath ) : the source is always the .hlsl with the same name
            std::filesystem::path hlslPath(path);
            hlslPath.replace_extension(".hlsl");

            if(!std::filesystem::exists(hlslPath)) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to compile the shader , since the file name is not valid -> Path: {}", hlslPath.string())});

            // every shader is compiled at every start ( no cache )
            const auto spirv = DXCManager::Compile(hlslPath);
            if(!spirv.has_value()) return std::unexpected(spirv.error());

            const std::vector<uint8_t>& buffer = spirv.value();

            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = buffer.size();
            info.flags = 0;
            info.pNext = nullptr;
            info.pCode = std::bit_cast<const uint32_t*>(buffer.data());

            auto result = vkCreateShaderModule(device , &info , nullptr , &module);
            if(result != VK_SUCCESS){
                ErrorDataType const error{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create shader module PATH -> {} : Error ID -> {}" , path , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }

            return {};
        }

        VkDevice device = nullptr;

        void clean() noexcept {
            if(device){
                if(module) vkDestroyShaderModule(device , module , nullptr);
            }

            module = nullptr;
            device = nullptr;
        }

        void processCopy(ShaderModule2&& input) noexcept{
            module = std::exchange(input.module,  nullptr);
            device = std::exchange(input.device , nullptr);
        }


    public:

        VkShaderModule module = nullptr;

        SetTypicalOperators(ShaderModule2)
        SetTypicalTransferOperators(ShaderModule2)

        ShaderModule2(moxContext2* context , const std::string path){
            if(!context){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader module , since the context is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }
            if(!context->logicalDevice){
                const ErrorDataType error(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create shader module , since the logical device  is nullptr");
                engineLogger(&error);
                THROW_MESSAGE;
            }

            this->device = context->logicalDevice->device;
            auto result = createShaderModule(path);
            if(result.has_value()){

                ErrorDataType const success{MOX_ERROR_TYPE_SUCCESS , std::format("shader module -> {} has been created" , path)};

            }else{
                engineLogger(&result.error());
                THROW_MESSAGE;
            }
        }

        ~ShaderModule2(){clean();}
    };

    using pShaderModule2 = std::shared_ptr<ShaderModule2>;

}
