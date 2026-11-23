#pragma once

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../../imgui/imgui.h"
#include "../../imgui/imconfig.h"
#include "../../imgui/imgui_internal.h"
#include "../../imgui/backends/imgui_impl_glfw.h"
#include "../../imgui/backends/imgui_impl_vulkan.h"

namespace mox
{
namespace Passes
{
    class GUIPass : public NodePass
    {
    public:

        struct CreateInfo
        {
            // the path of the .mox without the extension ( Shaders/GUI/GUIPass ) , there is no .hlsl : ImGui has its own shaders
            std::string path_S;
            moxContext2* _Nonnull context;
        };

        SetTypicalOperators(GUIPass)

        explicit GUIPass(const CreateInfo& info) : NodePass(info.context , "GUIPass")
        {
            const std::filesystem::path fileMOX = info.path_S + ".mox";
            const auto result = initFile(fileMOX);
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }

            inputData = info;
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> initPass() noexcept override
        {
            return _implInit();
        }

        virtual bool requestExecution() const noexcept override
        {
            return true;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override
        {
            if(!vulkanBackend || !guiImage || !guiCleanImage) MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to execute GUIPass , since the vulkan backend of ImGui or the images do not exist");

            // the core has made the draw data with ImGui::Render ( ) : here it is only put into the command buffer
            ImDrawData* drawData = ImGui::GetDrawData();
            if(!drawData || drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) return {};

            // both images are the color attachments of the pass ( the layout is GENERAL : unified image layouts )
            {
                std::array<VkImageMemoryBarrier2 , 2> barriers{};
                barriers[0] = guiImage->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
                barriers[1] = guiCleanImage->translateImage(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT , VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

                const auto dependency = TypicalDependency<2>(barriers.data());
                vkCmdPipelineBarrier2(cmd , &dependency);
            }

            // the same draw data goes into two images : the second call only records the commands again ( the buffers of the vertices are already filled )
            const auto drawInto = [&](Image::RenderImage2* image , const VkAttachmentLoadOp loadOp)
            {
                VkRenderingAttachmentInfo color{};
                color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                color.pNext = nullptr;
                color.imageView = image->view;
                color.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                color.resolveMode = VK_RESOLVE_MODE_NONE;
                color.loadOp = loadOp;
                color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                // used only with LOAD_OP_CLEAR : the transparent background
                color.clearValue.color = {{0.0f , 0.0f , 0.0f , 0.0f}};

                VkRenderingInfo rendering{};
                rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                rendering.pNext = nullptr;
                rendering.flags = 0;
                rendering.renderArea.offset = {0 , 0};
                rendering.renderArea.extent = image->extent;
                rendering.layerCount = 1;
                rendering.viewMask = 0;
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments = &color;
                rendering.pDepthAttachment = nullptr;
                rendering.pStencilAttachment = nullptr;

                vkCmdBeginRendering(cmd , &rendering);
                ImGui_ImplVulkan_RenderDrawData(drawData , cmd);
                vkCmdEndRendering(cmd);
            };

            // the UI over the colour that HDR has written ( this image goes to the swapchain )
            drawInto(guiImage , VK_ATTACHMENT_LOAD_OP_LOAD);
            // the UI alone on the transparent background ( the layer for Streamline )
            drawInto(guiCleanImage , VK_ATTACHMENT_LOAD_OP_CLEAR);

            return {};
        }

        ~GUIPass() noexcept
        {
            processDelete();
        }

    private:

        // only one ImGui context can live in a process
        inline static bool alive = false;

        CreateInfo inputData{};
        VkDevice device{nullptr};

        // the images belong to SceneImages , the pass only uses them
        Image::RenderImage2* guiImage{nullptr};
        Image::RenderImage2* guiCleanImage{nullptr};

        // the format has to live as long as the pipeline of the backend is created ( the backend reads the pointer in Init )
        VkFormat colorFormat{VK_FORMAT_UNDEFINED};

        bool contextCreated{false};
        bool glfwBackend{false};
        bool vulkanBackend{false};

        void processDelete() noexcept override
        {
            // the backend frees buffers and textures that the GPU may still read
            if(device && vulkanBackend) mox::deviceWaitIdle(device);

            if(vulkanBackend) ImGui_ImplVulkan_Shutdown();
            if(glfwBackend) ImGui_ImplGlfw_Shutdown();
            if(contextCreated)
            {
                ImGui::DestroyContext();
                alive = false;
            }

            guiImage = nullptr;
            guiCleanImage = nullptr;
            device = nullptr;
            vulkanBackend = false;
            glfwBackend = false;
            contextCreated = false;
            inputData = {};
            NodePass::processDelete();
        }

        void processCopy(GUIPass&& input) noexcept
        {
            if(this == &input) return;
            processDelete();

            inputData = input.inputData;
            device = std::exchange(input.device , nullptr);
            guiImage = std::exchange(input.guiImage , nullptr);
            guiCleanImage = std::exchange(input.guiCleanImage , nullptr);
            colorFormat = input.colorFormat;

            // the ImGui context is global : it moves with the flags , and the moved from object must not shut it down
            contextCreated = std::exchange(input.contextCreated , false);
            glfwBackend = std::exchange(input.glfwBackend , false);
            vulkanBackend = std::exchange(input.vulkanBackend , false);
            NodePass::processCopy(std::move(input));
        }

        MOX_NO_DISCARD ErrorDataOutput<void> _implInit() noexcept
        {
            moxContext2* context = inputData.context;

            if(!context->window || !context->window->window || !context->surface || !context->graphicsQueue || !context->sc)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create GUIPass , since the context has no window , surface , graphics queue or swapchain");
            }

            if(alive)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create GUIPass , since another ImGui already exists ( ImGui has a single global context )");
            }

            // the images of the pass
            {
                const auto sceneIt = cSources.find(Source::SceneImages::Name);
                if(sceneIt == cSources.end() || !sceneIt->second.source)
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create GUIPass , since the SourceCluster SceneImages is not given");
                }

                const auto gui = sceneIt->second.source->requestStructure<Image::RenderImage2*>("guiImage");
                MOX_CHECK_RESULT(gui)
                guiImage = gui.value();

                const auto clean = sceneIt->second.source->requestStructure<Image::RenderImage2*>("guiCleanImage");
                MOX_CHECK_RESULT(clean)
                guiCleanImage = clean.value();
            }

            try
            {
                device = context->logicalDevice->device;

                // the context and the glfw backend
                IMGUI_CHECKVERSION();
                ImGui::CreateContext();
                contextCreated = true;
                alive = true;

                ImGuiIO& io = ImGui::GetIO();
                io.IniFilename = nullptr;
                io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

                // true = the backend installs its callbacks and CHAINS the ones that are already set ( the key and the mouse of the Player )
                if(!ImGui_ImplGlfw_InitForVulkan(context->window->window , true))
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to init the ImGui glfw backend");
                }
                glfwBackend = true;

                // the vulkan backend with the dynamic rendering : no render pass , the pipeline is made for the format of guiImage
                colorFormat = guiImage->format;

                DetailsCreateInfo details{};
                details.device = context->physicalDevice->device;
                details.surface = context->surface->surface;
                const auto family = queueFamilyDetails::queryQueueFamilyDetails(details);

                ImGui_ImplVulkan_InitInfo init{};
                init.ApiVersion = VK_API_VERSION_1_4;
                init.Instance = context->instance->instance;
                init.PhysicalDevice = context->physicalDevice->device;
                init.Device = device;
                init.QueueFamily = family.graphics.value();
                init.Queue = context->graphicsQueue->queue;
                init.DescriptorPoolSize = 32;
                init.MinImageCount = std::max(2u , context->sc->imageCount);
                init.ImageCount = std::max(init.MinImageCount , Swapchain::frames_in_flight * 2 + 2);

                init.UseDynamicRendering = true;
                init.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
                init.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;

                if(!ImGui_ImplVulkan_Init(&init))
                {
                    MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to init the ImGui vulkan backend");
                }
                vulkanBackend = true;
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the ImGui context or its backends");
            }

            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(GUIPass)
}
}
