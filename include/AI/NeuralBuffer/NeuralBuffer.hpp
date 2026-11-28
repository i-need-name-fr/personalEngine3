#pragma once

#include "../../Vulkan/vulkanContext/moxContext2.hpp"
#include "../../Vulkan/Buffer2/GeneralBuffer.hpp"
#include <source_location>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstring>
#include <bit>
#include <random>
#include <array>
#include <vector>
#include <string>
#include <string_view>
#include "../../Vulkan/DescriptorBuffer2/DescriptorBuffer2.hpp"
#include "../../Vulkan/ShaderPipe2/ShaderPipe2.hpp"

namespace mox
{
namespace Buffer
{
    // a string that can be a template parameter : std::string can not ( it is not a structural type ) , a char array inside of a class can
    // NeuralBuffer<64 , 16 , "hidden"> : N includes the zero at the end
    template<std::size_t N>
    struct FixedString
    {
        char value[N]{};

        constexpr FixedString(const char (&text)[N]) noexcept
        {
            for(std::size_t i = 0 ; i < N ; i++) value[i] = text[i];
        }

        constexpr bool empty() const noexcept { return N <= 1 || value[0] == '\0'; }
        constexpr std::string_view view() const noexcept { return std::string_view(value , N > 0 ? N - 1 : 0); }
    };

    // the name is the name of the files in Cache , so every NeuralBuffer must have its own
    template<std::uint32_t rows , std::uint32_t colons , FixedString bufferName>
    requires((rows % 16 ==0) && (colons % 16 == 0) && (!bufferName.empty()))
    class NeuralBuffer final
    {
    public:

        SetTypicalOperators(NeuralBuffer)
        SetTypicalTransferOperators(NeuralBuffer)

        struct CreateInfo
        {
            moxContext2* _Nonnull context;
            bool needsToWriteFile{false};
            // the range of the start weights . the weights must be around ZERO : an all-positive range makes the neurons almost the same
            // min >= max ( the default ) means Glorot : the range is +- sqrt( 6 / ( rows + colons ) )
            float min = 0.0f;
            float max = 0.0f;
            // the learning rate of Adam ( processAdam.hlsl ) : NKPD 1e-4 , NRC 1e-3
            float learningRate = 1e-4f;
        };

        pDeviceReadableBuffer gradsBuffer; // gradians                                  done
        pDeviceReadableBuffer weightBuffer; // INFERENCE + BIAS + TRAINING              done
        pDeviceReadableBuffer infoBuffer; // for offsets                                done
        pDeviceReadableBuffer stagingGradBuffer; //staging for TRAINING -> ROW MAJOR    done
        pDeviceReadableBuffer stagingWeightBuffer; // same                              done
        pDeviceReadableBuffer ptrBuffer; // for pointers                                done
        pDeviceReadableBuffer paramBuffer; // for Parameter                             done

        explicit NeuralBuffer(const CreateInfo &info)
        {
            const auto result = _implInit(info);
            if(!result.has_value())
            {
                // saveOnDestroy is set only after everything is created , so a failed object saves nothing
                engineLogger(result.error());
                processDelete();
                THROW_MESSAGE;
            }

            {
                const auto result = _implCreatePipe();
                if(!result.has_value())
                {
                    // saveOnDestroy is set only after everything is created , so a failed object saves nothing
                    engineLogger(result.error());
                    processDelete();
                    THROW_MESSAGE;
                }
            }
        }

        void translate_RowMajor(VkCommandBuffer& cmd) noexcept
        {
            _implTranslateGradsWeightsToRowMajor(cmd);
        }

        // the pipe of the optimizer ( processAdam.hlsl ) : once , before adamStep . NRC does not need it

        // Adam for this layer : buffer is NKPDGeneral ( validCount , stepCount ) , the staging buffers must be already in the row major ( translate_RowMajor )
        void adamStep(VkCommandBuffer& cmd , Buffer::BufferBase2* buffer) noexcept
        {
            if(!pipe) return;
            {
                VkDescriptorBufferBindingInfoEXT info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
                info.pNext = nullptr;
                info.address = pipe->buffers[0]->pBuffer;
                info.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;

                cmdBindDescriptorBuffersEXT(cmd , 1 , &info);

                VkDeviceSize offset[] = {0};
                const uint32_t indices[] = {0};
                cmdSetDescriptorBufferOffsetsEXT(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 0 , 1 , indices , offset );
            }

            {
                VkDescriptorBufferInfo buf{};
                buf.buffer = buffer->buffer;
                buf.offset = 0;
                buf.range = VK_WHOLE_SIZE;

                VkWriteDescriptorSet info{};
                info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                info.descriptorCount = 1;
                info.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                info.pBufferInfo = &buf;
                info.pImageInfo = nullptr;
                info.pTexelBufferView = nullptr;
                info.dstBinding = 0;
                info.dstSet = nullptr;
                cmdPushDescriptorSetKHR(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout->layout , 1 , 1 , &info);
            }

            {
                constexpr VkPipelineStageFlags2 cs = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                constexpr VkAccessFlags2 rd = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                constexpr VkAccessFlags2 rw = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

                struct Item { Buffer::BufferBase2* buf; VkAccessFlags2 access; };
                const std::array<Item , 6> items =
                {{
                    {stagingGradBuffer.get() , rd},     // the grads of the matrices ( converted )
                    {gradsBuffer.get() , rd},           // the grads of the bias ( half ) from backstep
                    {stagingWeightBuffer.get() , rw},   // adam writes the new weights here , the convert reads them later
                    {weightBuffer.get() , rw},          // adam writes the bias ( the atomics )
                    {paramBuffer.get() , rw},           // w , m , v , grad : the previous step wrote them
                    {buffer , rd}                       // NKPDGeneral : validCount , stepCount
                }};

                std::array<VkBufferMemoryBarrier2 , 6> barriers{};
                for(std::size_t i = 0 ; i < items.size() ; i++) barriers[i] = items[i].buf->translateBuffer(items[i].access , cs);

                const auto dep = TypicalDependency<6>(barriers.data());
                vkCmdPipelineBarrier2(cmd , &dep);
            }

            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            const struct { uint32_t M; uint32_t K; float lr; } push = {rows , colons , learningRate};
            vkCmdPushConstants(cmd , pipe->layout->layout , VK_SHADER_STAGE_COMPUTE_BIT , 0 , sizeof(push) , &push);

            const uint32_t x = ((rows * colons + rows) + 255) / 256;

            vkCmdDispatch(cmd , x , 1 , 1 );

            shaders[0] = nullptr;
            cmdBindShadersEXT(cmd , 1 , stage , shaders);
        }

        void addBindings(std::vector<DescriptorData2> &data , const uint32_t firstBinding) noexcept
        {
            // grad
            // weight
            // offset
            // staging grad
            // staging weight
            // ptr
            // param
            data.push_back(mox::DescriptorData2::createData(gradsBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , firstBinding , 0 ));
            data.push_back(mox::DescriptorData2::createData(weightBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , firstBinding + 1 , 0 ));
            data.push_back(mox::DescriptorData2::createData(infoBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , firstBinding + 2 , 0 ));
            data.push_back(mox::DescriptorData2::createData(stagingGradBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , firstBinding + 3 , 0 ));
            data.push_back(mox::DescriptorData2::createData(stagingWeightBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , firstBinding + 4 , 0 ));
            data.push_back(mox::DescriptorData2::createData(ptrBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , firstBinding + 5 , 0 ));
            data.push_back(mox::DescriptorData2::createData(paramBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , firstBinding + 6 , 0 ));

        }

        void translate_DriverOptimal(VkCommandBuffer& cmd) noexcept
        {
            _implTranslateGradsWeightsToDriver(cmd);
        }

        const ::AI::OffsetData getOffsets() const noexcept { return offsets;}

        void annulment(VkCommandBuffer& cmd) noexcept
        {   

            std::array<VkBufferMemoryBarrier2,2 > barriers{};
            barriers[0] = gradsBuffer->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
            barriers[1] = stagingGradBuffer->translateBuffer(VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

            const auto dep = TypicalDependency<2>(barriers.data());

            vkCmdPipelineBarrier2(cmd , &dep);

            gradsBuffer->fillBuffer(cmd);
            stagingGradBuffer->fillBuffer(cmd);
        }

        ~NeuralBuffer() noexcept
        {
            processDelete();
        }

    private:

        static constexpr std::size_t PARAMETER_COUNT = static_cast<std::size_t>(rows) * colons + rows;

        ::AI::OffsetData offsets{};
        ::AI::ptrData pointers{};
        inline static std::random_device rndDevice{};
        moxContext2* context{nullptr};
        float learningRate{1e-4f};
        std::filesystem::path weightFile{};
        std::filesystem::path paramsFile{}; 
        bool saveOnDestroy{false};
        std::array<VkConvertCooperativeVectorMatrixInfoNV , 2> _toTraining{}; // from row major -> inference and training optimal
        std::array<VkConvertCooperativeVectorMatrixInfoNV , 2> _fromTraining{};

        std::array<std::shared_ptr<std::uint64_t> , 4> ptrs{};

        pShaderPipe2 pipe{};

        static constexpr VkDeviceSize ALIGNMENT = 64;
        static constexpr VkDeviceSize _implAlignUp(const VkDeviceSize value) noexcept
        {
            return (value + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        }

        // float -> fp16 with the rounding to the nearest even , for the start value of the bias on the CPU
        static std::uint16_t _implFloatToHalf(const float value) noexcept
        {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
            const std::uint32_t sign = (bits >> 16) & 0x8000u;
            const std::uint32_t rawExponent = (bits >> 23) & 0xFFu;
            std::uint32_t mantissa = bits & 0x7FFFFFu;

            if(rawExponent == 0xFFu) return static_cast<std::uint16_t>(sign | 0x7C00u | (mantissa ? 0x200u : 0u)); // inf or nan

            const std::int32_t exponent = static_cast<std::int32_t>(rawExponent) - 127 + 15;
            if(exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u); // too big : inf

            if(exponent <= 0)
            {
                if(exponent < -10) return static_cast<std::uint16_t>(sign); // too small : zero
                mantissa |= 0x800000u; // the hidden one
                const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
                std::uint32_t half = mantissa >> shift;
                const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
                const std::uint32_t halfway = 1u << (shift - 1u);
                if(remainder > halfway || (remainder == halfway && (half & 1u))) half++;
                return static_cast<std::uint16_t>(sign | half);
            }

            std::uint32_t half = (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
            const std::uint32_t remainder = mantissa & 0x1FFFu;
            if(remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) half++; // the carry may raise the exponent : it is right
            return static_cast<std::uint16_t>(sign | half);
        }

        void processCopy(NeuralBuffer&& input) noexcept
        {
            // this object may already own a net : it is saved and released first
            processDelete();

            gradsBuffer = std::move(input.gradsBuffer);
            weightBuffer = std::move(input.weightBuffer);
            infoBuffer = std::move(input.infoBuffer);
            stagingGradBuffer = std::move(input.stagingGradBuffer);
            stagingWeightBuffer = std::move(input.stagingWeightBuffer);
            ptrBuffer = std::move(input.ptrBuffer);
            paramBuffer = std::move(input.paramBuffer);

            offsets = input.offsets;
            pointers = input.pointers;
            context = input.context;
            weightFile = std::move(input.weightFile);
            paramsFile = std::move(input.paramsFile);
            saveOnDestroy = input.saveOnDestroy;
            learningRate = input.learningRate;

            // the infos hold the device addresses ( the buffers are the same , so they stay valid ) and the pDstSize of the shared values
            _toTraining = input._toTraining;
            _fromTraining = input._fromTraining;
            ptrs = std::move(input.ptrs);
            pipe = std::move(input.pipe);

            // the moved object is empty and saves nothing
            input.saveOnDestroy = false;
            input.processDelete();
        }

        void processDelete() noexcept
        {
            deviceWaitIdle(context->logicalDevice->device);
            if(saveOnDestroy) _implSaveToFile();
            saveOnDestroy = false;

            gradsBuffer = nullptr;
            weightBuffer = nullptr;
            infoBuffer = nullptr;
            stagingGradBuffer = nullptr;
            stagingWeightBuffer = nullptr;
            ptrBuffer = nullptr;
            paramBuffer = nullptr;
            pipe = nullptr;
            offsets = {};
            pointers = {};
            context = nullptr;
            weightFile.clear();
            paramsFile.clear();

            // no device address and no pDstSize is left behind
            _toTraining = {};
            _fromTraining = {};
            ptrs = {};
        }

        ErrorDataOutput<void> _implCreatePipe() noexcept
        {
            const std::filesystem::path file = std::filesystem::path(std::source_location::current().file_name()).parent_path() / std::filesystem::path("processAdam.hlsl");
            try
            {
                pipe = std::make_shared<ShaderPipe2>(context);

                {
                    std::vector<DescriptorLayoutData> data(7);
                    data[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[3] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    data[5] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 1};
                    data[6] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    
                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->addDescriptorLayout(info);
                }

                {
                    DescriptorLayoutData data = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1};
                    DescriptorLayout::CreateInfo info{};
                    info.context = context;
                    info.data = {data};
                    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
                    info.pNext = nullptr;
                    info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

                    pipe->addDescriptorLayout(info);
                }

                VkPushConstantRange range{};
                range.offset = 0;
                range.size = sizeof(uint32_t) * 3;   // M , K , learning rate   
                range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

                {
                    PipelineLayout::CreateInfo info{};
                    info.ranges = {range};
                    info.context = context;
                    info.layouts =  {pipe->descriptorLayouts[0].get() , pipe->descriptorLayouts[1].get()};
                    pipe->layout = std::make_shared<PipelineLayout>(info);
                }

                {
                    std::vector<DescriptorData2> data(7);
                    data[0] = DescriptorData2::createData(gradsBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 0 , 0 );
                    data[1] = DescriptorData2::createData(weightBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 1 , 0 );
                    data[2] = DescriptorData2::createData(infoBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ,2 , 0 );
                    data[3] = DescriptorData2::createData(stagingGradBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 3 , 0 );
                    data[4] = DescriptorData2::createData(stagingWeightBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 4 , 0 );
                    data[5] = DescriptorData2::createData(ptrBuffer.get() , VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER , 5 , 0 );
                    data[6] = DescriptorData2::createData(paramBuffer.get() , VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , 6 , 0 );

                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = pipe->descriptorLayouts[0].get();

                    pipe->addDescriptorBuffer2(info);
                }

                {
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage = VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.pushes = {range};
                    info.filePath = file.string();
                    info.layouts = {pipe->descriptorLayouts[0].get() , pipe->descriptorLayouts[1].get()};

                    pipe->addShader(info);
                }
            }
            catch(std::exception& e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to creater pipe for neural buffer");
            }
            return {};
        }

        ErrorDataOutput<void> _implGetCoopSize(moxContext2* context , VkCooperativeVectorMatrixLayoutNV layout , VkDeviceSize& size) noexcept
        __attribute__((nonnull))
        {
            if(!convertCooperativeVectorMatrixNV)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to get the size of the cooperative matrix , since vkConvertCooperativeVectorMatrixNV is not loaded");
            }

            // a valid host address is given anyway , it is not read
            const std::vector<float> source(rows * colons , 0.0f);
            std::size_t dstSize = 0;

            VkConvertCooperativeVectorMatrixInfoNV info{};
            info.sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
            info.srcComponentType = VK_COMPONENT_TYPE_FLOAT32_KHR;
            info.srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
            info.srcStride = colons * sizeof(float);
            info.srcSize = source.size() * sizeof(float);
            info.srcData.hostAddress = source.data();
            info.numRows = rows;
            info.numColumns = colons;

            info.dstComponentType = VK_COMPONENT_TYPE_FLOAT16_NV;
            info.dstLayout = layout;
            info.dstStride = 0;
            info.dstData.hostAddress = nullptr;
            info.pDstSize = &dstSize;

            const VkResult result = convertCooperativeVectorMatrixNV(context->logicalDevice->device , &info);
            if(result != VK_SUCCESS || dstSize == 0)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , std::format("failed to get the size of the cooperative matrix {} x {} , result -> {} , size -> {}" , rows , colons , static_cast<int>(result) , dstSize));
            }

            size = dstSize;
            return {};
        }


        std::vector<uint8_t> _implReadFile(const std::filesystem::path& path) const noexcept
        {
            if(!std::filesystem::exists(path)) return {};
            std::ifstream file(path , std::ios::binary | std::ios::ate);
            if(!file.is_open()) return {};
            const std::size_t size = static_cast<std::size_t>(file.tellg());
            file.seekg(0);
            std::vector<uint8_t> buffer(size);
            file.read(reinterpret_cast<char*>(buffer.data()) , static_cast<std::streamsize>(size));
            return buffer;
        }

        void _implSaveFile(const std::vector<uint8_t>& data , const std::filesystem::path& path) const noexcept
        {
            std::error_code error;
            std::filesystem::create_directories(path.parent_path() , error);

            std::ofstream file(path , std::ios::binary);
            if(!file.is_open()) return;
            file.write(reinterpret_cast<const char*>(data.data()) , static_cast<std::streamsize>(data.size()));
        }

        void _implMakeFilePaths() noexcept
        {
            const std::string name(bufferName.view());
            const std::filesystem::path currentDir = std::filesystem::path(std::source_location::current().file_name()).parent_path();
            weightFile = currentDir / std::filesystem::path("Cache") /(name + ".txt");
            paramsFile = currentDir / std::filesystem::path("Cache") /(name + "Param.txt");
        }

        void _implSaveToFile() noexcept
        {
            if(!weightBuffer || !paramBuffer || !context) return;

            // the GPU must have finished the writes into the buffers
            deviceWaitIdle(context->logicalDevice->device);

            _implSaveFile(weightBuffer->getBuffetData<std::uint8_t>(static_cast<std::uint32_t>(weightBuffer->getBufferSize())) , weightFile);
            _implSaveFile(paramBuffer->getBuffetData<std::uint8_t>(static_cast<std::uint32_t>(PARAMETER_COUNT * sizeof(::AI::Parameter))) , paramsFile);
        }

        // ======================================== creation ========================================

        void _implFillRandom(const CreateInfo& input , std::vector<::AI::Parameter>& params) const noexcept
        {
            const float limit = std::sqrt(6.0f / static_cast<float>(rows + colons));
            float low = input.min;
            float high = input.max;
            if(!(low < high))
            {
                low = -limit;
                high = limit;
            }

            std::mt19937 engine(rndDevice());
            std::uniform_real_distribution<float> value(low , high);

            // only the matrix : the bias starts from zero , and so do m , v , grad
            for(std::uint32_t i = 0 ; i < rows * colons ; i++)
            {
                params[i] = {};
                params[i].w = value(engine);
            }
        }

        [[gnu::cold]] ErrorDataOutput<void> _implInit(const CreateInfo& input) noexcept
        {
            if(!input.context)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create NeuralBuffer , since the context is nullptr");
            }
            context = input.context;
            learningRate = input.learningRate;

            std::vector<::AI::Parameter> params(PARAMETER_COUNT);
            std::vector<uint8_t> savedWeights{};

            if(input.needsToWriteFile)
            {
                _implMakeFilePaths();

                if(utils::SystemCheck::similarSystem())
                {
                    const auto weightData = _implReadFile(weightFile);
                    const auto paramsData = _implReadFile(paramsFile);
                    // a file of ANOTHER size ( the net was changed : the number of neurons ) is not used : the copy would go over the end of the buffer
                    if(!weightData.empty() && !paramsData.empty() && paramsData.size() == params.size() * sizeof(::AI::Parameter))
                    {
                        std::memcpy(params.data() , paramsData.data() , paramsData.size());
                        savedWeights = weightData;
                    }
                }
            }

            if(savedWeights.empty()) _implFillRandom(input , params);

            const auto result = _implBuild(input , params , savedWeights);
            MOX_CHECK_RESULT(result)

            saveOnDestroy = input.needsToWriteFile;
            return {};
        }

        // creates all the buffers . random start : the weights go from the staging buffer into the optimal layouts . start from the file : the weights are taken from it as they are
        [[gnu::cold]] MOX_NO_DISCARD ErrorDataOutput<void> _implBuild(const CreateInfo& input , std::vector<::AI::Parameter>& params , const std::vector<uint8_t>& savedWeights) noexcept
        {
            const VkBufferUsageFlags flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

            // the matrix row major fp32 : the staging buffer of the weights starts with it , so the first conversion gives the right weights
            std::vector<float> weights(rows * colons);
            for(std::uint32_t i = 0 ; i < rows * colons ; i++) weights[i] = params[i].w;
            std::vector<float> zeros(rows * colons , 0.0f);

            VkDeviceSize weightsINFERENCE = 0;
            VkDeviceSize weightsTRAINING = 0; // size of the same matrix , therefore can be used for both neuronBuffer and Grads
            {
                const auto result = _implGetCoopSize(input.context , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV , weightsINFERENCE);
                MOX_CHECK_RESULT(result)
            }

            {
                const auto result = _implGetCoopSize(input.context , VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV , weightsTRAINING);
                MOX_CHECK_RESULT(result)
            }

            const VkDeviceSize biasSize = sizeof(uint16_t) * rows;

            offsets.BIAS_SIZE = biasSize;
            offsets.INFERENCE_WEIGHT_SIZE = weightsINFERENCE;
            offsets.TRAINING_WEIGHT_SIZE = weightsTRAINING;
            offsets.INFERENCE_W_OFFSET = 0;
            offsets.BIAS_OFFSET = _implAlignUp(weightsINFERENCE);
            offsets.TRAINING_W_OFFSET = _implAlignUp(offsets.BIAS_OFFSET + biasSize);
            offsets.BIAS_GRAD_OFFSET = _implAlignUp(weightsTRAINING);
            offsets.BIAS_PARAM_INDEX = rows * colons;

            // weightBuffer : [ INFERENCE matrix ][ bias ][ TRAINING matrix ] , gradsBuffer : [ TRAINING matrix ][ bias gradient ] . all the offsets are aligned
            const VkDeviceSize weightBufferSize = _implAlignUp(offsets.TRAINING_W_OFFSET + weightsTRAINING);

            // the file of the weights has to be exactly of this size , otherwise both files are from another net : the random start
            const bool useSavedWeights = !savedWeights.empty() && savedWeights.size() == weightBufferSize;
            if(!savedWeights.empty() && !useSavedWeights) _implFillRandom(input , params);
            const VkDeviceSize gradBufferSize = _implAlignUp(offsets.BIAS_GRAD_OFFSET + biasSize);

            // the start weight buffer : the bias is already inside ( fp16 ) , the matrices are written by the conversion at the end of this function
            std::vector<uint8_t> startWeightBuffer(weightBufferSize , 0);
            for(std::uint32_t i = 0 ; i < rows ; i++)
            {
                const std::uint16_t half = _implFloatToHalf(params[offsets.BIAS_PARAM_INDEX + i].w);
                std::memcpy(startWeightBuffer.data() + offsets.BIAS_OFFSET + i * sizeof(std::uint16_t) , &half , sizeof(half));
            }

            // start from the file : the whole content of the weight buffer ( the layouts of the driver and the bias )
            const bool fromFile = useSavedWeights;
            if(fromFile) std::memcpy(startWeightBuffer.data() , savedWeights.data() , savedWeights.size());

            std::vector<uint8_t> zeroGradBuffer(gradBufferSize , 0);

            try
            {
                { // now the buffers

                    Buffer::DeviceReadableBuffer::CreateInfo info{};
                    info.size = weightBufferSize;
                    info.context = input.context;
                    info.usage = flags;

                    weightBuffer  = std::make_shared<DeviceReadableBuffer>(info , translateToSpan(startWeightBuffer));
                }

                { // params
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = input.context;
                    info.usage = flags;
                    info.size = sizeof(::AI::Parameter) * PARAMETER_COUNT;
                    paramBuffer  = std::make_shared<DeviceReadableBuffer>(info , translateToSpan(params));
                }

                { // staging
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = input.context;
                    info.usage = flags;
                    info.size = sizeof(float) * rows * colons;
                    stagingGradBuffer  = std::make_shared<DeviceReadableBuffer>(info , translateToSpan(zeros));
                    stagingWeightBuffer  = std::make_shared<DeviceReadableBuffer>(info , translateToSpan(weights));
                }

                { // info
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = input.context;
                    info.usage = flags;
                    info.size = sizeof(::AI::OffsetData);
                    infoBuffer  = std::make_shared<DeviceReadableBuffer>(info , &offsets); // offsets are complete here
                }

                { // grads
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = input.context;
                    info.usage = flags;
                    info.size = gradBufferSize;
                    gradsBuffer = std::make_shared<DeviceReadableBuffer>(info , translateToSpan(zeroGradBuffer));
                }

                { // ptr
                    DeviceReadableBuffer::CreateInfo info{};
                    info.context = input.context;
                    info.usage = flags;
                    info.size = sizeof(::AI::ptrData);
                    ::AI::ptrData data{};
                    data.pGradBuffer = gradsBuffer->pBuffer;
                    data.pWeightBuffer = weightBuffer->pBuffer;
                    pointers = data;
                    ptrBuffer = std::make_shared<DeviceReadableBuffer>(info , &data);
                }
            }
            catch(const std::exception&)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create buffers for NeuralBuffer,  please check the logs above");
            }

            ptrs[0] = std::make_shared<std::uint64_t>(stagingWeightBuffer->getBufferSize());
            ptrs[1] = std::make_shared<std::uint64_t>(stagingGradBuffer->getBufferSize());
            ptrs[2] = std::make_shared<std::uint64_t>(offsets.INFERENCE_WEIGHT_SIZE);
            ptrs[3] = std::make_shared<std::uint64_t>(offsets.TRAINING_WEIGHT_SIZE);

            { // FROM TRAINING TO ROW MAJOR
            // for weights
                _fromTraining[0].sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;

                _fromTraining[0].srcSize = offsets.TRAINING_WEIGHT_SIZE;
                _fromTraining[0].srcComponentType = VK_COMPONENT_TYPE_FLOAT16_NV;
                _fromTraining[0].srcData.deviceAddress = weightBuffer->pBuffer + offsets.TRAINING_W_OFFSET;
                _fromTraining[0].srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV;
                _fromTraining[0].srcStride = 0;

                _fromTraining[0].numRows = rows;
                _fromTraining[0].numColumns = colons;

                _fromTraining[0].dstComponentType = VK_COMPONENT_TYPE_FLOAT32_NV;
                _fromTraining[0].dstData.deviceAddress = stagingWeightBuffer->pBuffer;
                _fromTraining[0].dstStride = colons * sizeof(float);
                _fromTraining[0].pDstSize = ptrs[0].get();
                _fromTraining[0].dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;

                // for grads
                _fromTraining[1].sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;

                _fromTraining[1].srcSize = offsets.TRAINING_WEIGHT_SIZE;
                _fromTraining[1].srcComponentType = VK_COMPONENT_TYPE_FLOAT16_NV;
                _fromTraining[1].srcData.deviceAddress = gradsBuffer->pBuffer;
                _fromTraining[1].srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV;
                _fromTraining[1].srcStride = 0;

                _fromTraining[1].numRows = rows;
                _fromTraining[1].numColumns = colons;

                _fromTraining[1].dstComponentType = VK_COMPONENT_TYPE_FLOAT32_NV;
                _fromTraining[1].dstData.deviceAddress = stagingGradBuffer->pBuffer;
                _fromTraining[1].dstStride = colons * sizeof(float);
                _fromTraining[1].pDstSize = ptrs[1].get();
                _fromTraining[1].dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
            }

            { // FROM ROW MAJOR TO INFERENCE AND TRAINING OPTIMAL
                _toTraining[0].sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;

                _toTraining[0].srcSize = stagingWeightBuffer->getBufferSize();
                _toTraining[0].srcComponentType = VK_COMPONENT_TYPE_FLOAT32_NV;
                _toTraining[0].srcData.deviceAddress = stagingWeightBuffer->pBuffer;
                _toTraining[0].srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
                _toTraining[0].srcStride = colons * sizeof(float);

                _toTraining[0].numRows = rows;
                _toTraining[0].numColumns = colons;

                _toTraining[0].dstComponentType = VK_COMPONENT_TYPE_FLOAT16_NV;
                _toTraining[0].dstData.deviceAddress = weightBuffer->pBuffer;
                _toTraining[0].dstStride = 0;
                _toTraining[0].pDstSize = ptrs[2].get();
                _toTraining[0].dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV;

                // for grads
                _toTraining[1].sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;

                _toTraining[1].srcSize = stagingWeightBuffer->getBufferSize();
                _toTraining[1].srcComponentType = VK_COMPONENT_TYPE_FLOAT32_NV;
                _toTraining[1].srcData.deviceAddress = stagingWeightBuffer->pBuffer;
                _toTraining[1].srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
                _toTraining[1].srcStride = colons * sizeof(float);

                _toTraining[1].numRows = rows;
                _toTraining[1].numColumns = colons;

                _toTraining[1].dstComponentType = VK_COMPONENT_TYPE_FLOAT16_NV;
                _toTraining[1].dstData.deviceAddress = weightBuffer->pBuffer + offsets.TRAINING_W_OFFSET;
                _toTraining[1].dstStride = 0;
                _toTraining[1].pDstSize = ptrs[3].get();
                _toTraining[1].dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV;
            }

            // the random start weights go from the staging buffer into the optimal layouts , the same way as every step later . the weights from the file are there already
            if(!fromFile)
            {
                const auto result = input.context->executeCommand([&](VkCommandBuffer& cmd) -> void
                {
                    _implTranslateGradsWeightsToDriver(cmd);
                });
                MOX_CHECK_RESULT(result)
            }

            return {};
        }

        [[gnu::hot]] void _implTranslateGradsWeightsToRowMajor(VkCommandBuffer& cmd) noexcept // translate for adam
        __attribute__((nonnull))
        {

            std::array<VkBufferMemoryBarrier2 , 4> barriers{};
            barriers[0] = weightBuffer->translateBuffer(VK_ACCESS_2_MEMORY_READ_BIT, VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV);
            barriers[1] = stagingWeightBuffer->translateBuffer(VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV);
            barriers[2] = gradsBuffer->translateBuffer(VK_ACCESS_2_MEMORY_READ_BIT, VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV);
            barriers[3] = stagingGradBuffer->translateBuffer(VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV);

            const auto dep = TypicalDependency<4>(barriers.data());
            vkCmdPipelineBarrier2(cmd , &dep);

            cmdConvertCooperativeVectorMatrixNV(cmd , 2 , _fromTraining.data());
        }

        [[gnu::hot]] void _implTranslateGradsWeightsToDriver(VkCommandBuffer& cmd) noexcept // translate as the end
        __attribute__((nonnull))
        {

            std::array<VkBufferMemoryBarrier2 , 2> barriers{};
            barriers[0] = weightBuffer->translateBuffer(VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV);
            barriers[1] = stagingWeightBuffer->translateBuffer(VK_ACCESS_2_MEMORY_READ_BIT, VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV);


            const auto dep = TypicalDependency<2>(barriers.data());
            vkCmdPipelineBarrier2(cmd , &dep);


            cmdConvertCooperativeVectorMatrixNV(cmd , 2 , _toTraining.data());
        }

    };

    template<std::uint32_t rows , std::uint32_t columns , FixedString bufferName>
    using pNeuralBuffer = std::shared_ptr<NeuralBuffer<rows , columns , bufferName>>;
}
}
