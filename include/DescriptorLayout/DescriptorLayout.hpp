#pragma once

#include "../vulkanContext/moxContext2.hpp"
#include "../MoxFileReader/FileReader.hpp"

namespace mox{

    struct DescriptorLayoutData{
        VkDescriptorType type{};
        uint32_t count = 1;
    };


    class DescriptorLayout final{
    public:

        DescriptorLayout() = delete;
        DescriptorLayout(DescriptorLayout& input) = delete;
        DescriptorLayout(const DescriptorLayout& input) = delete;
        DescriptorLayout& operator=(DescriptorLayout& input) = delete;
        DescriptorLayout& operator=(const DescriptorLayout& input) = delete;

        DescriptorLayout(DescriptorLayout&& input) noexcept{
            if(layout == input.layout) return;
            processCopy(std::move(input));
        }
        DescriptorLayout& operator=(DescriptorLayout&& input) noexcept{
            if(layout == input.layout) return *this;
            processCopy(std::move(input));
            return *this;
        }

        struct CreateInfo{
            moxContext2* context = nullptr;
            std::vector<DescriptorLayoutData> data{};
            VkDescriptorSetLayoutCreateFlags flags{0};
            VkShaderStageFlags stage{VK_SHADER_UNUSED_KHR};
            void* pNext = nullptr;
        };

        explicit DescriptorLayout(const CreateInfo& info){
            {
                const auto result = checkRequirements(VkReqTypeLogicalDevice , info.context);
                if(!result.has_value()){
                    engineLogger(result.error());
                    THROW_MESSAGE;
                }
            }

            device = info.context->logicalDevice->device;
            const auto result = createLayout(info);
            if(!result.has_value()){
                engineLogger(result.error());
                processDelete();
                THROW_MESSAGE;
            }
        }

        MOX_NO_DISCARD inline static ErrorDataOutput<std::shared_ptr<DescriptorLayout>> CreateDescriptorLayoutFromFile(const DescriptorLayout::CreateInfo &info , const std::filesystem::path& path) noexcept
        {
            return _implCreateDescriptorLayoutFromFile(path , info);
        }

        MOX_NO_DISCARD inline static ErrorDataOutput<std::shared_ptr<DescriptorLayout>> CreateDescriptorLayoutFromFile( const DescriptorLayout::CreateInfo &info ,  mox::FileData& file) noexcept
        {
            return _implCreateDescriptorLayoutFromFile(file , info);
        }

        ~DescriptorLayout(){
            processDelete();
        }

        VkDescriptorSetLayout layout{nullptr};

    private:

        VkDevice device = nullptr;

        [[nodiscard]] ErrorDataOutput<void> createLayout(const CreateInfo &input) noexcept{

            std::vector<VkDescriptorSetLayoutBinding> bind{};
            {
                uint32_t i = 0 ;
                for(auto& dLayoutData : input.data){
                    VkDescriptorSetLayoutBinding binding{};
                    binding.binding = i;
                    binding.descriptorCount = dLayoutData.count;
                    binding.descriptorType = dLayoutData.type;
                    binding.pImmutableSamplers = nullptr;
                    binding.stageFlags = input.stage;

                    bind.push_back(std::move(binding));
                    i++;
                }
            }

            VkDescriptorSetLayoutCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            info.bindingCount = static_cast<uint32_t>(bind.size());
            info.pBindings = bind.data();
            info.flags = input.flags;
            info.pNext = input.pNext;
            
            const auto result = vkCreateDescriptorSetLayout(input.context->logicalDevice->device , &info , nullptr, &layout);
            if(result != VK_SUCCESS) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create descriptor set layout , error ID -> {}" , static_cast<int32_t>(result))});

            return {};
        }

        void processCopy(DescriptorLayout&& input) noexcept{
            processDelete();
            device = std::exchange(input.device , nullptr);
            layout = std::exchange(input.layout , nullptr);
        }

        void processDelete() noexcept{
            if(device && layout) vkDestroyDescriptorSetLayout(device , layout , nullptr);
            device = nullptr;
            layout = nullptr;
        }

        // CREATES ONLY GENERAL SET , NOT PERSONAL SET , LIKE SOURCE BUFFERS, ONLY LAYOUT FOR GENERAL!!!!
        MOX_NO_DISCARD inline static ErrorDataOutput<std::shared_ptr<DescriptorLayout>> _implCreateDescriptorLayoutFromFile(const std::filesystem::path& path , const DescriptorLayout::CreateInfo &addData) noexcept
        {
            try{

                const auto analyzedResult = DescriptorFile::AnalyzeFile(path);
                // now when we got the data , we need to check if there is a GENERAL
                if(!analyzedResult.has_value()) return std::unexpected(analyzedResult.error());
                auto data = analyzedResult.value();

 
                if(!data.sets.contains(DescriptorFile::_moxGeneralSet))
                {
                    const ErrorDataType error{MOX_ERROR_TYPE_WARNING , std::format("failed to create descriptor set layout , since it doesnt have GENERAL set, file -> {}" , path.string())};
                    engineLogger(error);
                    return nullptr;
                }

                std::vector<DescriptorLayoutData> layoutDatas{};

                // now we create DescriptorDataLayout per each attachments if its from GENERAL
                for(const auto& d : data.sets[DescriptorFile::_moxGeneralSet].attachments )
                {
                    DescriptorLayoutData l{};
                    l.count = d.second.count;
                    l.type = d.second.type;
                    layoutDatas.push_back(l);
                }

                DescriptorLayout::CreateInfo info{};
                info.context = addData.context;
                info.data = layoutDatas;
                info.flags = addData.flags;
                info.pNext = nullptr;
                info.stage = addData.stage;

                return std::make_shared<DescriptorLayout>(info);
            }
            catch(std::exception &e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create Descriptor set from file -> {}" , path.string())});
            }
        }

        MOX_NO_DISCARD inline static ErrorDataOutput<std::shared_ptr<DescriptorLayout>> _implCreateDescriptorLayoutFromFile(mox::FileData& data , const DescriptorLayout::CreateInfo &addData) noexcept
        {
            try{

                if(!data.sets.contains(DescriptorFile::_moxGeneralSet))
                {
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Descriptor layout from fileData structure"});
                }

                std::vector<DescriptorLayoutData> layoutDatas{};

                // now we create DescriptorDataLayout per each attachments if its from GENERAL
                for(const auto& d : data.sets[DescriptorFile::_moxGeneralSet].attachments )
                {
                    DescriptorLayoutData l{};
                    l.count = d.second.count;
                    l.type = d.second.type;
                    layoutDatas.push_back(l);
                }

                DescriptorLayout::CreateInfo info{};
                info.context = addData.context;
                info.data = layoutDatas;
                info.flags = addData.flags;
                info.pNext = nullptr;
                info.stage = addData.stage;

                return std::make_shared<DescriptorLayout>(info);
            }
            catch(std::exception &e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION , "failed to create Descriptor layout from fileData structure"});
            }
        }
    };

    using pDescriptorLayout = std::shared_ptr<DescriptorLayout>;
}