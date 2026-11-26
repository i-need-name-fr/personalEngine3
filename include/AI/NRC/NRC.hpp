#pragma once

#include "../NeuralBuffer/NeuralBuffer.hpp"
#include "../../moxSourceCluster/SourceCluster.hpp"

namespace mox
{
namespace Source
{
namespace AI
{
    class NeuralRadianceCache final : public SourceCluster
    {   
    public:

        // the NRC will have only 2 layers , first one is 64 x 16 , and second is 16 x 64 , but actually only 3
        enum class TranslateType
        {
            ToRowMajor,
            ToOptimal
        };

        inline static const std::string Name = "NRC";

        using requestTranslate = std::function<void(VkCommandBuffer& , const mox::Source::AI::NeuralRadianceCache::TranslateType)>;
        using requestAnnulment = std::function<void(VkCommandBuffer&)>;

        inline static const std::string requestTranslateName = "__MOX_NRC_REQUEST_TRANSLATE_TO_NAME__";
        inline static const std::string requestAnnulmentName = "__MOX_NRC_REQUEST_ANNULMENT__";

        explicit NeuralRadianceCache(moxContext2* context) : SourceCluster(context , Name)
        {
            const auto result = _implInit();
            if(!result.has_value())
            {   
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        inline static const uint8_t F = 2ul; // numbers per pos
        inline static const uint8_t L = 4ul; // num of layers

        Buffer::pNeuralBuffer<64 , 16 , "NRC_inputLayer"> inputLayer; //        done
        Buffer::pNeuralBuffer<16 , 64 , "NRC_outputLayer"> outputLayer; //      done

        Buffer::pDeviceReadableBuffer hashGridParamBuffer{}; //                 done
        Buffer::pDeviceReadableBuffer hashGridInfoBuffer{}; //                  done

        Buffer::pDeviceReadableBuffer NRCGeneralData{}; //                      done
        Buffer::pDeviceReadableBuffer trainingSampleBuffer{}; //                done

        ~NeuralRadianceCache() noexcept
        {
            processDelete();
        }

    private:

        void processDelete() noexcept override
        {
            inputLayer = nullptr;
            outputLayer = nullptr;
            hashGridInfoBuffer = nullptr;
            hashGridParamBuffer = nullptr;
            NRCGeneralData = nullptr;
            trainingSampleBuffer = nullptr;
            SourceCluster::processDelete();
        }

        [[gnu::cold]] MOX_NO_DISCARD ErrorDataOutput<void> _implSetupHashGrid() noexcept
        {
            try
            {

                const uint32_t baseT{4ul};
                const uint32_t baseCellSize{16ul};
                std::vector<::AI::Parameter> params{};
                std::vector<::AI::NRC::HashGridInfo> infos{};
                std::random_device rndDevice{};

                std::mt19937 engine(rndDevice());
                std::uniform_real_distribution<float> value(0.00001f , 0.001f);

                for(uint32_t i = 1 ; i <= L ; i++)
                {
                    ::AI::NRC::HashGridInfo gridInfo{};
                    gridInfo.offset = params.size();
                    gridInfo.cellSize = (float)baseCellSize / (float)(2 << i);
                    gridInfo.T = baseT * (2 << i);

                    infos.push_back(gridInfo);

                    for(uint32_t z = 0 ; z < gridInfo.T ; z++)
                    for(uint32_t y = 0 ; y < gridInfo.T ; y++)
                    for(uint32_t x = 0 ; x < gridInfo.T ; x++)
                    {
                        ::AI::Parameter info1{};
                        info1.grad = 0.0f;
                        info1.m = 0.0f;
                        info1.v = 0.0f;
                        info1.w = value(engine);

                        ::AI::Parameter info2{};
                        info2.grad = 0.0f;
                        info2.m = 0.0f;
                        info2.v = 0.0f;
                        info2.w = value(engine);

                        params.push_back(info1 );
                        params.push_back(info2 );
                    }
                }

                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(::AI::Parameter) * params.size();
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    hashGridParamBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , Buffer::translateToSpan(params));
                }

                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(::AI::NRC::HashGridInfo) * infos.size();
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    hashGridInfoBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info, Buffer::translateToSpan(infos));
                }

                const VkExtent2D extent = {context->getResolutionExtent().width / MOX_NRC_RESOLUTION_DIVISOR, context->getResolutionExtent().height / MOX_NRC_RESOLUTION_DIVISOR};

                {
                    std::vector<::AI::NRC::TrainingSample> trainings(extent.width  * extent.height);

                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(::AI::NRC::TrainingSample) * trainings.size();
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                    trainingSampleBuffer = std::make_shared<Buffer::DeviceReadableBuffer>(info , Buffer::translateToSpan(trainings));
                }

                {
                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.context = context;
                    info.size = sizeof(::AI::NRC::GeneralData);
                    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                    ::AI::NRC::GeneralData data{};
                    data.width = extent.width;
                    data.height = extent.height;
                    data.stepCount = 0;
                    data.validCount = 0;

                    NRCGeneralData = std::make_shared<Buffer::DeviceReadableBuffer>(info , &data);
                }

                {
                    const auto result = _implAddSource(hashGridParamBuffer.get() , "hashGridBuffer");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(hashGridInfoBuffer.get() , "HashGridInfos");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(trainingSampleBuffer.get() , "trainingBuffer");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddSource(NRCGeneralData.get() , "NRCGeneralData");
                    MOX_CHECK_RESULT(result)
                }
                
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create NRC buffers , please check the logs above");
            }
            return {};
        }
        
        [[gnu::cold]] MOX_NO_DISCARD ErrorDataOutput<void> _implSetupNeurons() noexcept
        {
            try
            {
                {
                    Buffer::NeuralBuffer<64 , 16 , "NRC_inputLayer">::CreateInfo info{};
                    info.context = context;
                    info.max = 0.001f;
                    info.min = 0.00001f;
                    info.needsToWriteFile = true;

                    inputLayer = std::make_shared<Buffer::NeuralBuffer<64 , 16 , "NRC_inputLayer">>(info);
                }

                {
                    Buffer::NeuralBuffer<16 , 64 , "NRC_outputLayer">::CreateInfo info{};
                    info.context = context;
                    info.max = 0.001f;
                    info.min = 0.00001f;
                    info.needsToWriteFile = true;

                    outputLayer = std::make_shared<Buffer::NeuralBuffer<16 , 64 , "NRC_outputLayer">>(info);
                }

                {
                    const auto result = _implAddSource(inputLayer->infoBuffer.get() , "inputOffsetBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(inputLayer->ptrBuffer.get() , "inputPtrBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(inputLayer->gradsBuffer.get() , "inputGradBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(inputLayer->stagingGradBuffer.get() , "inputStagingGradBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(inputLayer->weightBuffer.get() , "inputWeightBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(inputLayer->stagingWeightBuffer.get() , "inputStagingWeightBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(inputLayer->paramBuffer.get() , "inputParamBuffer");
                    MOX_CHECK_RESULT(result);
                }

                // output

                {
                    const auto result = _implAddSource(outputLayer->infoBuffer.get() , "outputOffsetBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(outputLayer->ptrBuffer.get() , "outputPtrBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(outputLayer->gradsBuffer.get() , "outputGradBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(outputLayer->stagingGradBuffer.get() , "outputStagingGradBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(outputLayer->weightBuffer.get() , "outputWeightBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(outputLayer->stagingWeightBuffer.get() , "outputStagingWeightBuffer");
                    MOX_CHECK_RESULT(result);
                }

                {
                    const auto result = _implAddSource(outputLayer->paramBuffer.get() , "outputParamBuffer");
                    MOX_CHECK_RESULT(result);
                }

            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create NRC buffers , please check the logs above");
            }
            return {};
        }

        [[gnu::hot]] void _implAnnulment(VkCommandBuffer& cmd) noexcept
        {
            inputLayer->annulment(cmd);
            outputLayer->annulment(cmd);
        }

        [[gnu::hot]] void _implTranslate(VkCommandBuffer& cmd , const TranslateType type) noexcept
        {
            if(type == TranslateType::ToRowMajor)
            {
                inputLayer->translate_RowMajor(cmd);
                outputLayer->translate_RowMajor(cmd);
            }
            else
            {
                inputLayer->translate_DriverOptimal(cmd);
                outputLayer->translate_DriverOptimal(cmd);
            }
        }

        [[gnu::cold]] MOX_NO_DISCARD ErrorDataOutput<void> _implInit() noexcept
        {   

            {
                const auto result = _implSetupHashGrid();
                MOX_CHECK_RESULT(result);
            }

            {
                const auto result = _implSetupNeurons();
                MOX_CHECK_RESULT(result);
            }

            try
            {

                {
                    std::vector<DescriptorLayoutData> data(18);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[5] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[6] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[7] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[8] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[9] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[10] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[11] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[12] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[13] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[14] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[15] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[16] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[17] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.push_back(std::move(std::make_shared<DescriptorLayout>(info)));
                }

                {
                    std::vector<DescriptorData2> data{};
                    data.push_back(mox::DescriptorData2::createData( hashGridParamBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0));
                    data.push_back(mox::DescriptorData2::createData( hashGridInfoBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0));
                    data.push_back(mox::DescriptorData2::createData( trainingSampleBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,2 , 0));
                    data.push_back(mox::DescriptorData2::createData( NRCGeneralData.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0));

                    inputLayer->addBindings(data , 4);
                    outputLayer->addBindings(data , 11);

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = dlayouts[0].get();

                    dbuffers.push_back(std::move(std::make_shared<DescriptorBuffer2>(info)));
                }
                containsPingPong = false;
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create DL or DB for NRC");
            }

            {
                // add 2 functions
                requestTranslate function = [&](VkCommandBuffer& cmd , const TranslateType type) ->void{_implTranslate(cmd , type);};
                const auto result = _implAddFunction(function , NeuralRadianceCache::requestTranslateName);
                MOX_CHECK_RESULT(result)
            }

            {
                NeuralRadianceCache::requestAnnulment function = [&](VkCommandBuffer& cmd) -> void{_implAnnulment(cmd);};
                const auto result = _implAddFunction(function , NeuralRadianceCache::requestAnnulmentName);
                MOX_CHECK_RESULT(result)
            }


            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(NeuralRadianceCache)
}
}
}