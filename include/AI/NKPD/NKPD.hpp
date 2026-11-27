#pragma once

#include "../NeuralBuffer/NeuralBuffer.hpp"
#include "../../moxSourceCluster/SourceCluster.hpp"
#include "../../moxSourceCluster/IndirectExtent.hpp"
namespace mox
{
namespace Source
{
namespace AI
{
    class NKPD final : public SourceCluster
    {
    public:

        // diffuse.xyz or indirect                      3
        // prev diffuse.xyz  or indirect                6
        // normal check diff                            7
        // depth diff                                   8
        // one check dissolution                        ... since this one doesnt go into actual vector
        // hit distance , roughness                     10


        /*

        struct NPKDOptimal
        {
            RWTexture<float4> inputImage;
            RWTexture<float4> ouputImage;
            RWTexture<float> hitDistanceImage;
            RWTexture<float> roughnessImage;
            RWTexture<float> pixelCheck
            RWTexture<float4> positionImage;
            RWTexture<float> depthImage;
            RWTexture<float4> normalImage;
            RWTexture<float2> countImage;
            // the diff and so on will be calculated in the ask
            bool validUV;
            uint2 prevCoord;
            uint2 currentCoord;
        };

        */

        // the same two layouts as NRC : the training works with the row major matrices , the inference with the ones the driver wants ( DriverOptimal )
        enum class TranslateType
        {
            ToRowMajor,
            ToOptimal
        };

        inline static const std::string Name = "NKPD";

        using requestTranslate = std::function<void(VkCommandBuffer& , const mox::Source::AI::NKPD::TranslateType)>;
        using requestAnnulment = std::function<void(VkCommandBuffer&)>;

        inline static const std::string requestTranslateName = "__MOX_NKPD_REQUEST_TRANSLATE_TO_NAME__";
        inline static const std::string requestAnnulmentName = "__MOX_NKPD_REQUEST_ANNULMENT__";

        explicit NKPD(moxContext2* context) : SourceCluster(context , Name)
        {
            const auto result = _implInit();
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }

        Buffer::pNeuralBuffer<128 , 272 , "NKPD_input_Layer"> inputLayer;
        Buffer::pNeuralBuffer<128 , 128 , "NKPD_hidden_layer"> hiddenLayer;
        Buffer::pNeuralBuffer<16 , 128 ,"NKPD_output_layer"> outputLayer;

        Buffer::pDeviceReadableBuffer NKPDGeneral{};
        std::array<Image::pRenderImage2 , 2> variancePool{}; // ping pong : [frameIndex % 2] is the current , the other one is the previous

        ~NKPD() noexcept
        {
            processDelete();
        }

    private:

        void processDelete() noexcept override
        {
            inputLayer = nullptr;
            hiddenLayer = nullptr;
            outputLayer = nullptr;
            NKPDGeneral = nullptr;
            variancePool[0] = nullptr;
            variancePool[1] = nullptr;
            SourceCluster::processDelete();
        }

        [[gnu::cold]] MOX_NO_DISCARD ErrorDataOutput<void> _implSetupGeneral() noexcept
        {
            try
            {
                Buffer::DeviceReadableBuffer::CreateInfo info{};
                info.context = context;
                info.size = sizeof(::AI::NKPD::GeneralData);
                info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

                ::AI::NKPD::GeneralData data{};
                data.width = context->getResolutionExtent().width;
                data.height = context->getResolutionExtent().height;
                data.stepCount = 0;
                data.validCount = 0;

                NKPDGeneral = std::make_shared<Buffer::DeviceReadableBuffer>(info , &data);

                const auto result = _implAddSource(NKPDGeneral.get() , "NKPDGeneral");
                MOX_CHECK_RESULT(result)
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create NKPDGeneral , please check the logs above");
            }
            return {};
        }

        // the 7 sources of one layer : the names are  <prefix><the name of the buffer> , prefix is input / hidden / output ( the names of NKPD.hlsl )
        template<typename Layer>
        MOX_NO_DISCARD ErrorDataOutput<void> _implAddLayerSources(Layer& layer , const std::string& prefix) noexcept
        {
            {
                const auto result = _implAddSource(layer->infoBuffer.get() , prefix + "OffsetBuffer");
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = _implAddSource(layer->ptrBuffer.get() , prefix + "PtrBuffer");
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = _implAddSource(layer->gradsBuffer.get() , prefix + "GradBuffer");
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = _implAddSource(layer->stagingGradBuffer.get() , prefix + "StagingGradBuffer");
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = _implAddSource(layer->weightBuffer.get() , prefix + "WeightBuffer");
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = _implAddSource(layer->stagingWeightBuffer.get() , prefix + "StagingWeightBuffer");
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = _implAddSource(layer->paramBuffer.get() , prefix + "ParamBuffer");
                MOX_CHECK_RESULT(result)
            }

            return {};
        }

        [[gnu::cold]] MOX_NO_DISCARD ErrorDataOutput<void> _implSetupNeurons() noexcept
        {
            try
            {
                // needsToWriteFile : the weights are written to the file when the net is destroyed , the same as NRC
                {
                    Buffer::NeuralBuffer<128 , 272 , "NKPD_input_Layer">::CreateInfo info{};
                    info.context = context;
                    info.needsToWriteFile = true;

                    inputLayer = std::make_shared<Buffer::NeuralBuffer<128 , 272 , "NKPD_input_Layer">>(info);
                }

                {
                    Buffer::NeuralBuffer<128 , 128 , "NKPD_hidden_layer">::CreateInfo info{};
                    info.context = context;
                    info.needsToWriteFile = true;

                    hiddenLayer = std::make_shared<Buffer::NeuralBuffer<128 , 128 , "NKPD_hidden_layer">>(info);
                }

                {
                    Buffer::NeuralBuffer<16 , 128 , "NKPD_output_layer">::CreateInfo info{};
                    info.context = context;
                    info.needsToWriteFile = true;

                    outputLayer = std::make_shared<Buffer::NeuralBuffer<16 , 128 , "NKPD_output_layer">>(info);
                }

                {
                    Image::RenderImage2::CreateInfo info{};
                    info.context = context;
                    info.imageFormat = VK_FORMAT_R32G32B32A32_SFLOAT;   // x , y : the moments of the luminance , z : the count of the history ( ping pong : no race of the read and the write )
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.imageUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;
                    info.mipMapCount = 1;
                    info.layerCount = 1;
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    VkExtent2D extent = context->getResolutionExtent(); // but this is only the resolution of diffuse image , we also need the indirect 
                    const VkExtent2D iextent = Source::indirectExtent(extent);
                    extent.width += iextent.width; // we dont care about the height , since it will max(height1 , heigh2) where resolution of diffuse wins
                    info.extent = extent;
                    variancePool[0] = std::make_shared<Image::RenderImage2>(info);
                    variancePool[1] = std::make_shared<Image::RenderImage2>(info);
                }

                {
                    const auto result = _implAddSource(variancePool , "variancePool"); // variancePool_0 , variancePool_1
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddLayerSources(inputLayer , "input");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddLayerSources(hiddenLayer , "hidden");
                    MOX_CHECK_RESULT(result)
                }

                {
                    const auto result = _implAddLayerSources(outputLayer , "output");
                    MOX_CHECK_RESULT(result)
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create NKPD neural buffers , please check the logs above");
            }
            return {};
        }

        // the gradients of all the 3 layers become zero ( the same as NRC , but 3 layers )
        [[gnu::hot]] void _implAnnulment(VkCommandBuffer& cmd) noexcept
        {
            inputLayer->annulment(cmd);
            hiddenLayer->annulment(cmd);
            outputLayer->annulment(cmd);
        }

        // all the 3 layers change the layout of the weights and gradients ( the same as NRC , but 3 layers )
        [[gnu::hot]] void _implTranslate(VkCommandBuffer& cmd , const TranslateType type) noexcept
        {
            if(type == TranslateType::ToRowMajor)
            {
                inputLayer->translate_RowMajor(cmd);
                hiddenLayer->translate_RowMajor(cmd);
                outputLayer->translate_RowMajor(cmd);
            }
            else
            {
                inputLayer->translate_DriverOptimal(cmd);
                hiddenLayer->translate_DriverOptimal(cmd);
                outputLayer->translate_DriverOptimal(cmd);
            }
        }

        [[gnu::cold]] MOX_NO_DISCARD ErrorDataOutput<void> _implInit() noexcept
        {
            {
                const auto result = _implSetupGeneral();
                MOX_CHECK_RESULT(result);
            }

            {
                const auto result = _implSetupNeurons();
                MOX_CHECK_RESULT(result);
            }

            try
            {
                // the layout : 0 NKPDGeneral , then 3 x 7 ( grad , weight , offset , staging grad , staging weight , ptr , param ) , the same order as NeuralBuffer::addBindings
                {
                    std::vector<DescriptorLayoutData> data{};
                    data.reserve(24);
                    data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1});

                    for(uint32_t layer = 0 ; layer < 3 ; layer++)
                    {
                        data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1});
                        data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1});
                        data.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1});
                        data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1});
                        data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1});
                        data.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1});
                        data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1});
                    }

                    // binding 22 : variancePool ( current ) , 23 : the previous one ( once , after the 3 layers )
                    data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1});
                    data.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1});

                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_ALL;

                    dlayouts.push_back(std::move(std::make_shared<DescriptorLayout>(info)));
                }

                // 2 descriptor buffers , the same as SceneImages : dbuffers[id] has variancePool[id] as the current ( 22 ) and variancePool[!id] as the previous ( 23 ) , id = frameIndex % 2
                for(const uint32_t id : {0u , 1u})
                {
                    std::vector<DescriptorData2> data{};
                    data.push_back(mox::DescriptorData2::createData(NKPDGeneral.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0));

                    inputLayer->addBindings(data , 1);
                    hiddenLayer->addBindings(data , 8);
                    outputLayer->addBindings(data , 15);

                    data.push_back(DescriptorData2::createData(variancePool[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 22 , 0));
                    data.push_back(DescriptorData2::createData(variancePool[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 23 , 0));

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = dlayouts[0].get();

                    dbuffers.push_back(std::move(std::make_shared<DescriptorBuffer2>(info)));
                }
                containsPingPong = true;
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create DL or DB for NKPD");
            }

            {
                NKPD::requestTranslate function = [&](VkCommandBuffer& cmd , const TranslateType type) -> void{_implTranslate(cmd , type);};
                const auto result = _implAddFunction(function , NKPD::requestTranslateName);
                MOX_CHECK_RESULT(result)
            }

            {
                NKPD::requestAnnulment function = [&](VkCommandBuffer& cmd) -> void{_implAnnulment(cmd);};
                const auto result = _implAddFunction(function , NKPD::requestAnnulmentName);
                MOX_CHECK_RESULT(result)
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(NKPD)
}
}
}
