#pragma once

#include "SourceCluster.hpp"
#include "../Image2/MultiViewImage2.hpp"
#include "../utils/MultiScatterRoughness/MSR.hpp"

namespace mox
{
namespace Source
{
    class SceneImages final : public SourceCluster
    {
    public:

        SceneImages() = delete;
        SceneImages(const SceneImages& input) = delete;
        SceneImages(SceneImages& input) = delete;
        SceneImages& operator=(SceneImages& input) = delete;
        SceneImages& operator=(const SceneImages& input) = delete;
        SceneImages& operator=(SceneImages&& input) = delete;
        SceneImages(SceneImages&& input) = delete;

        ~SceneImages() noexcept
        {
            processDelete();
        }

        struct CreateInfo 
        {
            moxContext2* context{nullptr};
            std::filesystem::path MSR_path{moxUndefinedBehavior};
            // will add later translate the depth buffer to HiZ image and generate mips
            std::filesystem::path HiZ_mips;
            std::filesystem::path HiZ_translate;
        };

        explicit SceneImages(const CreateInfo& info) : SourceCluster(info.context , "SceneImages")
        {
            {
                const auto result = _implInit(info);
                if(!result.has_value())
                {
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = _implCreateDL();
                if(!result.has_value())
                {
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }

            {
                const auto result = _implCreateDB();
                if(!result.has_value())
                {
                    engineLogger(result);
                    processDelete();
                    THROW_MESSAGE;
                }
            }

            engineLogger("the init for Scene images is complete");
        }

        std::array<Image::pRenderImage2 , 2> positionImage; // done
        std::array<Image::pRenderImage2 , 2> depthImage; // done
        std::array<Image::pRenderImage2 , 2> normalImage; // done
        std::array<Image::pRenderImage2 , 2> diffuseImage; // done
        std::array<Image::pRenderImage2 , 2> diffuseDistance; // done
        std::array<Image::pRenderImage2 , 2> specularImage; // done
        std::array<Image::pRenderImage2 , 2> specularDistance; // done
        std::array<Image::pRenderImage2 , 2> indirectImage; // done
        std::array<Image::pRenderImage2 , 2> indirectDistance; // done
        
        // now the g buffer other sources
        Image::pRenderImage2 albedoImage; // done
        Image::pRenderImage2 roughnessImage;
        Image::pRenderImage2 metallicImage;
        Image::pRenderImage2 objectAOImage;

        // shadows and AO
        std::array<Image::pRenderImage2, 2> shadowImage;
        std::array<Image::pRenderImage2, 2> ambientImage;
        Image::pRenderImage2 fShadowImage;
        Image::pRenderImage2 fAmbientImage;

        // final Images
        std::array<Image::pRenderImage2, 2> HDRImage; // done
        Image::pRenderImage2 finalImage; // done
        Image::pRenderImage2 guiImage; // done 
        Image::pRenderImage2 guiCleanImage; // done

        // utils
        Image::pRenderImage2 pixelCheck; 
        Image::pRenderImage2 motionVector; // done
        Image::pRenderImage2 viewImage; // done

        Image::pMultiViewImage2<8 , Image::ViewPerType::Mip> HiZImage{nullptr};

        static constexpr uint32_t multiScatterRoughnessSize = 512;
        Image::pRenderImage2 MultiScatterRoughness{nullptr};

        MOX_NO_DISCARD VkImageMemoryBarrier2 requestHiZBarrier(const VkPipelineStageFlags2 stage , const VkAccessFlags2 access , const uint32_t mipCount , const uint32_t mipIndex) noexcept
        {
            return HiZImage->translateImage(access , stage , mipCount , mipIndex);
        }

        void cleanImages(VkCommandBuffer& cmd) noexcept
        {
            if(!cmd || !context || !context->sc) return;

            const uint32_t id = static_cast<uint32_t>(context->sc->frameIndex % 2);

            const std::array<Image::RenderImage2* , 23> images = {
                // the current half of the ping pong images
                positionImage[id].get() , normalImage[id].get() , depthImage[id].get() , diffuseImage[id].get() , diffuseDistance[id].get() ,
                specularImage[id].get() , specularDistance[id].get() , indirectImage[id].get() , indirectDistance[id].get() ,
                shadowImage[id].get() , ambientImage[id].get() , HDRImage[id].get() ,
                // the images without a pair
                albedoImage.get() , roughnessImage.get() , metallicImage.get() , objectAOImage.get() , fShadowImage.get() , fAmbientImage.get() ,
                finalImage.get() , motionVector.get() , viewImage.get() , guiImage.get() , guiCleanImage.get()
            };

            // one barrier for all the images : translateImage / translateImageUndefined also keep the state of the image
            const auto transition = [&](const bool fromUndefined , const VkAccessFlags2 access , const VkPipelineStageFlags2 stage) -> void
            {
                std::array<VkImageMemoryBarrier2 , images.size()> barriers{};
                uint32_t count = 0;
                for(Image::RenderImage2* image : images)
                {
                    if(!image) continue;
                    barriers[count] = fromUndefined ? image->translateImageUndefined(access , stage) : image->translateImage(access , stage);
                    count++;
                }

                if(count == 0) return;

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.imageMemoryBarrierCount = count;
                info.pImageMemoryBarriers = barriers.data();
                vkCmdPipelineBarrier2(cmd , &info);
            };

            // the old contents are not needed : the transition from UNDEFINED , and the clear after it
            transition(true , VK_ACCESS_2_TRANSFER_WRITE_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

            for(Image::RenderImage2* image : images)
            {
                if(image) image->cleanTheImage(cmd);
            }

            // from here it is an ordinary transition : the passes take the images from this state
            transition(false , VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        }

    private:

        // requestBarrier( name , stage , access ) of the base class knows RenderImage2 , Texture2 , Skybox2 and the buffers , but not MultiViewImage2 : HiZImage is the whole image ( all the mips )
        MOX_NO_DISCARD ErrorDataOutput<std::variant<VkImageMemoryBarrier2 , VkBufferMemoryBarrier2>> _implAskForBarrier(const std::string name , VkPipelineStageFlags2 const stage , VkAccessFlags2 const access) noexcept override
        {
            if(name == "HiZImage")
            {
                if(!HiZImage) return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create barrier for HiZImage in SceneImages , since the image is nullptr"});
                return HiZImage->translateWholeImage(access , stage);
            }

            return SourceCluster::_implAskForBarrier(name , stage , access);
        }

        void processDelete() noexcept
        {
            viewImage = nullptr;
            pixelCheck = nullptr;
            motionVector = nullptr;
            finalImage = nullptr;
            guiImage = nullptr;
            guiCleanImage = nullptr;
            roughnessImage = nullptr;
            metallicImage = nullptr;
            fShadowImage = nullptr;
            fAmbientImage = nullptr;
            objectAOImage = nullptr;
            albedoImage = nullptr;
            HiZImage = nullptr;
            MultiScatterRoughness = nullptr;
            for(const auto& id : { 0 ,1})
            {
                positionImage[id] = nullptr;
                depthImage[id] = nullptr;
                normalImage[id] = nullptr;

                diffuseImage[id] = nullptr;
                diffuseDistance[id] = nullptr;
                specularImage[id] = nullptr;
                specularDistance[id] = nullptr;
                indirectImage[id] = nullptr;
                indirectDistance[id] = nullptr;

                shadowImage[id] = nullptr;
                ambientImage[id] = nullptr;

                HDRImage[id] = nullptr;

            }
        }


        MOX_NO_DISCARD ErrorDataOutput<void> _implInit(const CreateInfo& input) noexcept
        {
            try
            {
                Image::RenderImage2::CreateInfo info{};
                info.context = this->context;
                info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                info.clear.float32[0] = 0.0f;
                info.clear.float32[1] = 0.0f;
                info.clear.float32[2] = 0.0f;
                info.clear.float32[3] = 0.0f;
                info.layerCount = 1;
                info.mipMapCount = 1;
                info.extent = context->sc->extent;
                info.imageType = VK_IMAGE_TYPE_2D;

                // all colored images except for final image / guiImage

                info.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
                normalImage[0] = std::make_shared<Image::RenderImage2>(info);
                normalImage[1] = std::make_shared<Image::RenderImage2>(info);

                diffuseImage[0] = std::make_shared<Image::RenderImage2>(info);
                diffuseImage[1] = std::make_shared<Image::RenderImage2>(info);


                specularImage[0] = std::make_shared<Image::RenderImage2>(info);
                specularImage[1] = std::make_shared<Image::RenderImage2>(info);

                albedoImage = std::make_shared<Image::RenderImage2>(info);

                HDRImage[0] = std::make_shared<Image::RenderImage2>(info);
                HDRImage[1] = std::make_shared<Image::RenderImage2>(info);

                indirectImage[0] = std::make_shared<Image::RenderImage2>(info);
                indirectImage[1] = std::make_shared<Image::RenderImage2>(info);
                
                info.imageFormat = VK_FORMAT_R8G8B8A8_UNORM;

                guiCleanImage = std::make_shared<Image::RenderImage2>(info);
                finalImage = std::make_shared<Image::RenderImage2>(info);
                guiImage = std::make_shared<Image::RenderImage2>(info);

                info.imageFormat = VK_FORMAT_R16G16_SFLOAT;
                motionVector = std::make_shared<Image::RenderImage2>(info);

                info.imageFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
                positionImage[0] = std::make_shared<Image::RenderImage2>(info);
                positionImage[1] = std::make_shared<Image::RenderImage2>(info);
                viewImage = std::make_shared<Image::RenderImage2>(info);

                info.imageFormat = VK_FORMAT_R32_SFLOAT;
                depthImage[0] = std::make_shared<Image::RenderImage2>(info);
                depthImage[1] = std::make_shared<Image::RenderImage2>(info);
                diffuseDistance[0] = std::make_shared<Image::RenderImage2>(info);
                diffuseDistance[1] = std::make_shared<Image::RenderImage2>(info);
                specularDistance[0] = std::make_shared<Image::RenderImage2>(info);
                specularDistance[1] = std::make_shared<Image::RenderImage2>(info);
                indirectDistance[0] = std::make_shared<Image::RenderImage2>(info);
                indirectDistance[1] = std::make_shared<Image::RenderImage2>(info);
                roughnessImage = std::make_shared<Image::RenderImage2>(info);
                metallicImage = std::make_shared<Image::RenderImage2>(info);
                pixelCheck = std::make_shared<Image::RenderImage2>(info);

                // the shadow has 2 channels : R is the shadow ( the clear value is 1 : no shadow ) , G is the number of the frames of the history ( 0 at the start ) , like in FrameTargets
                // the other images after it have one channel , so only R of the clear value matters for them
                info.clear.float32[0] = 1.0f;
                info.clear.float32[1] = 0.0f;
                info.clear.float32[2] = 0.0f;
                info.clear.float32[3] = 0.0f;

                info.imageFormat = VK_FORMAT_R32G32_SFLOAT;
                shadowImage[0] = std::make_shared<Image::RenderImage2>(info);
                shadowImage[1] = std::make_shared<Image::RenderImage2>(info);

                info.imageFormat = VK_FORMAT_R32_SFLOAT;
                ambientImage[0] = std::make_shared<Image::RenderImage2>(info);
                ambientImage[1] = std::make_shared<Image::RenderImage2>(info);
                fShadowImage = std::make_shared<Image::RenderImage2>(info);
                fAmbientImage = std::make_shared<Image::RenderImage2>(info);
                objectAOImage = std::make_shared<Image::RenderImage2>(info);

                {
                    Image::MultiViewImage2<8 , Image::ViewPerType::Mip>::CreateInfo info{};
                    info.context = context;
                    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    info.imageType = VK_IMAGE_TYPE_2D;
                    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    info.clear.float32[0] = 1.0f;
                    info.clear.float32[1] = 1.0f;
                    info.clear.float32[2] = 1.0f;
                    info.clear.float32[3] = 1.0f;
                    info.extent = context->sc->extent;
                    info.imageFormat = VK_FORMAT_R32_SFLOAT;

                    HiZImage = std::make_shared<Image::MultiViewImage2<8 , Image::ViewPerType::Mip>>(info);
                }

                {
                    // a 1D image : one row of texels ( the height is 1 ) and a 1D view
                    Image::RenderImage2::CreateInfo info{};
                    info.context = context;
                    info.imageFormat = VK_FORMAT_R16_SFLOAT;
                    info.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                    info.extent = {multiScatterRoughnessSize , 1};
                    info.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                    info.viewType = VK_IMAGE_VIEW_TYPE_1D;
                    info.imageType = VK_IMAGE_TYPE_1D;
                    info.layerCount = 1;
                    info.mipMapCount = 1;
                    info.clear.float32[0] = 0.0f;
                    info.clear.float32[1] = 0.0f;
                    info.clear.float32[2] = 0.0f;
                    info.clear.float32[3] = 0.0f;

                    MultiScatterRoughness = std::make_shared<Image::RenderImage2>(info);
                }

                {
                    const auto result = _implAddSource(positionImage , "positionImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(depthImage, "depthImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(normalImage, "normalImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(diffuseImage, "diffuseImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(diffuseDistance, "diffuseDistance");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(specularImage, "specularImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(specularDistance, "specularDistance");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(indirectImage, "indirectImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(indirectDistance, "indirectDistance");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(shadowImage, "shadowImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(ambientImage, "ambientImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(HDRImage, "HDRImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(albedoImage.get() , "albedoImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(roughnessImage.get() , "roughnessImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(objectAOImage.get() , "objectAOImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(metallicImage.get() , "metallicImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(pixelCheck.get() , "pixelCheck");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(motionVector.get() , "motionVector");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(fShadowImage.get() , "fShadowImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(fAmbientImage.get() , "fAmbientImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(finalImage.get() , "finalImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(guiCleanImage.get() , "guiCleanImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(guiImage.get() , "guiImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(viewImage.get() , "viewImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(HiZImage.get() , "HiZImage");
                    if(!result.has_value()) return result;
                }

                {
                    const auto result = _implAddSource(MultiScatterRoughness.get() , "MultiScatterRoughness");
                    if(!result.has_value()) return result;
                }


                containsPingPong = true;
            }
            catch(std::exception&e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create SceneImages , please check the logs"});
            }

            // here the MSR is created

            {
                const auto result = utils::MSR::Execute(context , MultiScatterRoughness.get() , input.MSR_path);
                MOX_CHECK_RESULT(result)
            }

            return {};
        }


        MOX_NO_DISCARD ErrorDataOutput<void> _implCreateDB() noexcept
        {
            try
            {
                dbuffers.resize(2);
                for(const auto& id : {0 , 1})
                {
                    std::vector<DescriptorData2> data(38);
                    data[0] = DescriptorData2::createData(positionImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 0 , 0);
                    data[1] = DescriptorData2::createData(positionImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1 , 0);

                    data[2] = DescriptorData2::createData(depthImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 2 , 0);
                    data[3] = DescriptorData2::createData(depthImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 3 , 0);

                    data[4] = DescriptorData2::createData(normalImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 4 , 0);
                    data[5] = DescriptorData2::createData(normalImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 5 , 0);

                    data[6] = DescriptorData2::createData(diffuseImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 6 , 0);
                    data[7] = DescriptorData2::createData(diffuseImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 7 , 0);

                    data[8] = DescriptorData2::createData(diffuseDistance[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 8 , 0);
                    data[9] = DescriptorData2::createData(diffuseDistance[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 9 , 0);

                    data[10] = DescriptorData2::createData(specularImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 10 , 0);
                    data[11] = DescriptorData2::createData(specularImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 11 , 0);

                    data[12] = DescriptorData2::createData(specularDistance[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 12 , 0);
                    data[13] = DescriptorData2::createData(specularDistance[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 13 , 0);

                    data[14] = DescriptorData2::createData(indirectImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 14 , 0);
                    data[15] = DescriptorData2::createData(indirectImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 15 , 0);

                    data[16] = DescriptorData2::createData(indirectDistance[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 16 , 0);
                    data[17] = DescriptorData2::createData(indirectDistance[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 17 , 0);

                    data[18] = DescriptorData2::createData(albedoImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 18 , 0);
                    data[19] = DescriptorData2::createData(roughnessImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 19 , 0);
                    data[20] = DescriptorData2::createData(metallicImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 20 , 0);
                    data[21] = DescriptorData2::createData(objectAOImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 21 , 0);

                    data[22] = DescriptorData2::createData(shadowImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 22 , 0);
                    data[23] = DescriptorData2::createData(shadowImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 23 , 0);

                    data[24] = DescriptorData2::createData(ambientImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 24, 0);
                    data[25] = DescriptorData2::createData(ambientImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 25 , 0);

                    data[26] = DescriptorData2::createData(fShadowImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 26 , 0);
                    data[27] = DescriptorData2::createData(fAmbientImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 27 , 0);

                    data[28] = DescriptorData2::createData(HDRImage[id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 28, 0);
                    data[29] = DescriptorData2::createData(HDRImage[!id].get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 29 , 0);

                    data[30] = DescriptorData2::createData(finalImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 30 , 0);
                    data[31] = DescriptorData2::createData(guiImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 31 , 0);
                    data[32] = DescriptorData2::createData(guiCleanImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 32 , 0);

                    data[33] = DescriptorData2::createData(pixelCheck.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 33 , 0);
                    data[34] = DescriptorData2::createData(motionVector.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 34 , 0);
                    data[35] = DescriptorData2::createData(viewImage.get() , VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 35 , 0);
                    data[36] = DescriptorData2::createData(HiZImage->fullView , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 36 , 0);
                    data[37] = DescriptorData2::createData(MultiScatterRoughness.get() , VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 37 , 0);
                    DescriptorBuffer2::CreateInfo info{};
                    info.context = context;
                    info.data = data;
                    info.layout = dlayouts[0].get();

                    dbuffers[id] = std::make_shared<DescriptorBuffer2>(info);
                }

            }
            catch(std::exception& e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create DB for SceneImages"});
            }
            return {};
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implCreateDL() noexcept
        {
            try
            {
                std::vector<DescriptorLayoutData> data(38);
                for(auto& d : data)
                {
                    d = {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , .count = 1};
                }

                data[36] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};
                data[37] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE , 1};

                DescriptorLayout::CreateInfo info{};
                info.context = context;
                info.data = data;
                info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
                info.pNext = nullptr;
                info.stage = VK_SHADER_STAGE_ALL;

                dlayouts.resize(1);
                dlayouts[0] = std::make_shared<DescriptorLayout>(info);
            }
            catch(std::exception& e)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to create DL for SceneImages"});
            }
            return {};
        }
    };
}
}