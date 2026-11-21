#pragma once 

#include "../moxNode/NodePass.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../moxSourceCluster/SceneImages.hpp"

namespace mox
{
namespace Passes
{   

    template<ShaderType type>
    class ReLAX : public NodePass
    {
    public:

        struct CreateInfo
        {
            std::string path_S{};
            moxContext2* _Nonnull context ;
        };

        explicit ReLAX(const CreateInfo &info) : NodePass(info.context , (type == ShaderType::temporal ? "ReLAXT" : "ReLAXS"))
        {
            const std::filesystem::path path = info.path_S + ".mox";
            const auto result = initFile(path);
            if(!result.has_value())
            {
                engineLogger(result);
                NodePass::processDelete();
                THROW_MESSAGE;
            }

            input = info;
        }

        MOX_NO_DISCARD virtual ErrorDataOutput<void> initPass() noexcept override 
        {
            return _implCreatePipe();
        }

        virtual bool requestExecution() const noexcept override 
        {
            const auto features = requestFeatures();
            return features.ReLAX;
        }

        virtual ErrorDataOutput<void> executePass(VkCommandBuffer &cmd) noexcept override 
        __attribute__((nonnull))
        {   
            bindAutoDescriptorBuffers(cmd , VK_PIPELINE_BIND_POINT_COMPUTE , pipe->layout.get());
            const auto result = bindBarrier(cmd , VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            MOX_CHECK_RESULT(result)

            static const std::uint32_t x = ( input.context->sc->extent.width + 7) / 8;
            static const std::uint32_t y = ( input.context->sc->extent.height + 7) / 8;

            // now the shaders
            VkShaderEXT shaders[1] = {pipe->shaders[0]->shader};
            VkShaderStageFlagBits stage[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

            cmdBindShadersEXT(cmd , 1 , stage , shaders);
            vkCmdDispatch(cmd , x , y , 1);

            shaders[0] = nullptr;

            cmdBindShadersEXT(cmd , 1 , stage , shaders);

            return {};
        }

        ~ReLAX()
        {
            pipe = nullptr;
            input = {};
            requestFeatures = {};
        }

    private:

        Source::SceneBuffers::RequestFeaturesFunctionType requestFeatures{};
        pShaderPipe2 pipe{};
        CreateInfo input{};


        MOX_NO_DISCARD ErrorDataOutput<void> _implCreatePipe() noexcept
        {   
            try
            {
                pipe = std::make_shared<ShaderPipe2>(context);

                const auto result = processPipe(pipe.get() , VK_SHADER_STAGE_COMPUTE_BIT , {});
                MOX_CHECK_RESULT(result)

                {
                    ShaderObject2::CreateInfo info{};
                    info.context = context;
                    info.currentStage =VK_SHADER_STAGE_COMPUTE_BIT;
                    info.nextStage = 0;
                    info.filePath = input.path_S + ".hlsl";
                    info.layouts = pipe->getLayouts();
                    info.pushes = {};

                    pipe->addShader(info);
                }
            }
            catch(std::exception& e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL ,"failed to create pipe for ReLAX");
            }

            const auto result = cSources[Source::SceneBuffers::Name].source->template requestFunction<Source::SceneBuffers::RequestFeaturesFunctionType>(Source::SceneBuffers::RequestFeaturesFunctionName);
            MOX_CHECK_RESULT(result)
            requestFeatures = result.value();

            return {};
        }
    };
    template<ShaderType type>
    using pReLAX = std::shared_ptr<ReLAX<type>>;
}
}