#pragma once

// the SourceClusters : the sets of descriptors that the passes read and write ( every one is a SourceCluster )
#include "../moxSourceCluster/SourceCluster.hpp"
#include "../moxSourceCluster/SceneBuffers.hpp"
#include "../moxSourceCluster/SceneImages.hpp"
#include "../moxSourceCluster/SceneReSTIR.hpp"
#include "../moxSourceCluster/AtmoScat.hpp"
#include "../moxSourceCluster/LightHub.hpp"
#include "../moxSourceCluster/TextureHub.hpp"
#include "../moxSourceCluster/SharcHeap.hpp"
#include "../moxSourceCluster/VolSource.hpp"
#include "../moxObject/VertexHeap.hpp"
#include "../moxSourceCluster/BloomSource.hpp"
// all the passes
#include "../GeneralPasses/Passes.hpp"
#include "../moxObject/EntityForge.hpp"
#include "../utils/SkyManager/SkyManager.hpp"
#include "../moxNode/RenderGraph.hpp"

namespace mox
{
    class mCore final
    {
        inline static constexpr uint32_t RayTracingRequired = 2;
        inline static constexpr uint32_t PathTracingRequired = 3;
        inline static constexpr uint32_t RasterizationRequired = 1;
        inline static constexpr std::string RasterizationPath = "4356456sdf";
        inline static constexpr std::string RayTracingPath = "q3456wera";
        inline static constexpr std::string PathTracingPath = "f34q5rtggw45";

        struct {
            Source::pSceneBuffers buffers{};
            Source::pVolSource vol{};
            Source::pSceneImages images{};
            Source::pSceneReSTIR restir{};
            Source::pLightHub lights{};
            Source::pTextureHub textures{};
            Source::pAtmoScat atmo{};
            Source::pSharcHeap sharc{};
            Source::AI::pNeuralRadianceCache nrc{};
            Source::AI::pNKPD nkpd;
            Source::pBloomSource bloom{};
        } sources;


    public:

        mCore()
        {
            const auto result = init();
            if(!result.has_value())
            {
                engineLogger(result);
                processDelete();
                THROW_MESSAGE;
            }
        }   

        ~mCore() noexcept
        {
            processDelete();
        }

        void mainLoop(){
            float lastSecond = 0.0f;

            recordFunction = [this](VkCommandBuffer& cmd , const uint32_t id)-> ErrorDataOutput<void>{

                recordCmd(cmd , id);

                return {};
            };
            while(!glfwWindowShouldClose(context->window->window)){
                context->stream->beginFrame(static_cast<uint32_t>(context->sc->frameIndex));
                glfwPollEvents();
                const float time  = glfwGetTime();
                const uint32_t timeInt = static_cast<uint32_t>(time);
                const uint32_t lastTimeInt = static_cast<uint32_t>(lastSecond);
                dt = time - lastSecond;
                lastSecond = time;

                if(!context->user->openedMenu) context->user->assetinput(context->window->window , dt);
                context->stream->marker(sl::PCLMarker::eSimulationEnd);
                const auto result = context->processQueueLoop(recordFunction);
                if(!result.has_value()){
                    engineLogger(result);
                    return;
                }

                if(timeInt > lastTimeInt)
                {
                    const sl::DLSSGState state = context->stream->getStatus();
                    std::cout << "DLSS-G : status = " << static_cast<uint32_t>(state.status)
                              << " , presented last second = " << state.numFramesActuallyPresented
                              << " , max generated = " << state.numFramesToGenerateMax
                              << " , vsync support = " << (state.bIsVsyncSupportAvailable == sl::Boolean::eTrue) << "\n";
                }
            }
        }


    private:

        void processDelete() noexcept
        {
            // every step is written BEFORE it starts : if the process dies , the last line of the log is the step that did it
            const auto step = [](const char* what)
            {
                ErrorDataType message{MOX_ERROR_TYPE_NONE , std::string("shutdown : ") + what};
                engineLogger(&message);
            };

            step("wait for the GPU");
            if(context && context->logicalDevice && context->logicalDevice->device)
            {
                mox::deviceWaitIdle(context->logicalDevice->device);
            }

            step("the passes");
            guiPass = nullptr;
            rt = nullptr;
            pt = nullptr;
            sharcPass = nullptr;
            mrt = nullptr;
            pProcess = nullptr;
            volumentric = nullptr;
            evUpdate = nullptr;
            hdrae = nullptr;
            dlssPass = nullptr;
            hdr = nullptr;
            relaxs = nullptr;
            relaxt = nullptr;
            reblurs_indirect = nullptr;
            reblurt_indirect = nullptr;
            reblurs = nullptr;
            reblurt = nullptr;
            bf = nullptr;
            ao = nullptr;
            forward = nullptr;
            deferred = nullptr;
            nrct = nullptr;
            nrcb = nullptr;
            nrcp1 = nullptr;
            nrcp2 = nullptr;
            nrcp3 = nullptr;
            nkpdt = nullptr;
            nkpdb = nullptr;
            nkpdbi = nullptr;
            nkpdi = nullptr;
            nkpdp1 = nullptr;
            nkpdp2 = nullptr;
            nkpdp3 = nullptr;
            rtI = nullptr;
            ptI = nullptr;

            step("the graph");
            graph = RenderGraph{};
            recordFunction = {};
            sources.nkpd = nullptr;
            sources.nrc = nullptr;
            step("the sky manager");
            skyManager = nullptr;
            step("the entity forge ( the models and the heap )");
            eForge = nullptr;

            step("the sources");
            sources.sharc = nullptr;
            sources.vol = nullptr;
            sources.lights = nullptr;
            sources.atmo = nullptr;
            sources.textures = nullptr;
            sources.restir = nullptr;
            sources.images = nullptr;
            sources.buffers = nullptr;
            sources.bloom = nullptr;

            step("the context ( the swapchain , the queues , Streamline , the device )");
            context = nullptr;
            step("done");
        }

        // init

        MOX_NO_DISCARD ErrorDataOutput<void> init() noexcept
        {
            {
                const auto result = initSources();
                MOX_CHECK_RESULT(result);
            }

            {
                const auto result = createModels();
                MOX_CHECK_RESULT(result);
            }

            {
                const auto result = createLights();
                MOX_CHECK_RESULT(result);
            }

            {
                const auto result = initPasses();
                MOX_CHECK_RESULT(result);
            }

            setDefaultSettings();

            return {};
        }

        // all sources

        MOX_NO_DISCARD ErrorDataOutput<void> initSources() noexcept
        {   

            try
            {
                moxContext2::CreateInfo info{};
                // the queues of the multi threading are made by StreamForge ( moxContext2::forgeGraphicsCount / forgeComputeCount : 10 graphics + 5 compute ) , so the old counters are 0 . the graphics
                // family of the GPU has 16 queues : 2 ( the engine ) + 10 ( the forge ) + the queues of Streamline ( 1 - 2 ) , these two counters would be ADDED to them ( 2 + 15 + 10 = 27 > 16 )
                info.asyncMultiThreadQueueCount = 0;
                info.multithreadQueueCount = 0;
                context = std::make_shared<moxContext2>(info);

                {
                    moxContext2* ctx = context.get();

                    // the general buffers : generalBuffer , features , sunBuffer
                    sources.buffers = std::make_shared<Source::SceneBuffers>(ctx);

                    {
                        Source::SceneImages::CreateInfo images{};
                        images.context = ctx;
                        images.MSR_path = "../Shaders/MultiScatter/MultiScatter.hlsl";
                        images.HiZ_translate = "../Shaders/HiZShaders/translateToGeneralImage.hlsl";
                        images.HiZ_mips = "../Shaders/HiZShaders/generateMips.hlsl";
                        sources.images = std::make_shared<Source::SceneImages>(images);
                    }

                    sources.restir = std::make_shared<Source::SceneReSTIR>(ctx);

                    sources.textures = std::make_shared<Source::TextureHub>(ctx);

                    {
                        Source::LightHub::CreateInfo lights{};
                        lights.context = ctx;
                        lights.user = ctx->user.get();
                        lights.shaderPath = "../Shaders/lightGridUpdate/lightUpdate.hlsl";
                        sources.lights = std::make_shared<Source::LightHub>(lights);
                    }

                    sources.vol = std::make_shared<Source::VolSource>(ctx);

                    {
                        Source::SharcHeap::CreateInfo sharc{};
                        sharc.context = ctx;
                        sources.sharc = std::make_shared<Source::SharcHeap>(sharc);
                    }

                    {
                        utils::SkyManager::CreateInfo info{};
                        info.context = ctx;
                        info.sceneBuffers = sources.buffers.get();
                        info.IBL = "../Shaders/AtmoSphereScat/IBLcompute.hlsl";
                        info.MS = "../Shaders/AtmoSphereScat/multiScattering.hlsl";
                        info.transition = "../Shaders/AtmoSphereScat/transitionCube.hlsl";
                        info.transmission = "../Shaders/AtmoSphereScat/transmissionLUT.hlsl";
                        info.skyview = "../Shaders/AtmoSphereScat/SkyViewLUT.hlsl";
                        skyManager = std::make_shared<utils::SkyManager>(info);
                        sources.atmo = skyManager->source;
                    }

                    {
                        utils::EntityForge::CreateInfo info{};
                        info.context = ctx;
                        info.txHub = sources.textures.get();
                        info.lHub = sources.lights.get();
                        eForge = std::make_shared<utils::EntityForge>(info);
                    }

                    {
                        sources.nrc = std::make_shared<Source::AI::NeuralRadianceCache>(context.get());
                    }

                    {
                        sources.nkpd = std::make_shared<Source::AI::NKPD>(context.get());
                    }

                    {
                        Source::BloomSource::CreateInfo info{};
                        info.context = context.get();
                        info.upscal = "../Shaders/Bloom/Upsampler.hlsl";
                        info.downscale = "../Shaders/Bloom/downSampler.hlsl";

                        sources.bloom = std::make_shared<Source::BloomSource>(info);
                    }
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create the context or one of the sources , please check the logs above");
            }

            return {};
        }

        // all models

        [[nodiscard]] static glm::mat4 intelSponzaMatrix() noexcept{
            glm::mat4 model(1.0f);
            model = glm::translate(model , glm::vec3(-20.0f , 10.2f , 0.0f));
            model = glm::scale(model , glm::vec3(2.5f));
            return model;
        }

        [[nodiscard]] static glm::mat4 jungleRuinsMatrix() noexcept{
            return glm::translate(glm::mat4(1.0f) , glm::vec3(800.0f - 543.0f , 10.0f , 809.0f));
        }

        MOX_NO_DISCARD ErrorDataOutput<void> createModels() noexcept
        {

            {
                Model3::CreateInfo info{};
                info.color = glm::vec3(0.0f);
                info.modelPath = "../models/IntelSponza/Main/NewSponza_Main_glTF_003.gltf";
                info.metallic = 0.0f;
                info.roughness = 1.0f;
                info.skipBlend = true;
                info.randomSSS = true;
                // "exterior" is NOT skipped any more : the half covered tiles were the culling camera ( fixedPosition was inverted ) , not this node . without it the outer side of the building is open ( black floor from below , loose boxes of the windows )
                // info.skipNodes = {"exterior"};
                // models[0]->setMatrix(intelSponzaMatrix() , 0);
                utils::ForgeBluePrint print{};
                print.mInfo = info;
                print.name = "sponza";
                print.type = utils::moxBluePrintModel3;
                print.eInfo = {};
                eForge->addBluePrint(print);
            }

            // {
            //     Model3::CreateInfo info{};
            //     info.color = glm::vec3(0.0f);
            //     info.modelPath = "../models/Exterior_1k/BistroExterior.gltf";
            //     info.metallic = 0.0f;
            //     info.roughness = 1.0f;
            //     glm::mat4 model(1.0f);
            //     model = glm::translate(model , glm::vec3(100.0f ,10.0f, 0.0f));
            //     // models[1]->setMatrix(model , 0);
            //     utils::ForgeBluePrint print{};
            //     print.mInfo = info;
            //     print.name = "ext";
            //     print.type = utils::moxBluePrintModel3;
            //     print.eInfo = {};
            //     eForge->addBluePrint(print);

            // }


            bool first = true;
            for(const char* path : {"../models/IntelSponza/Curtains/NewSponza_Curtains_glTF.gltf" , "../models/IntelSponza/Ivy/NewSponza_IvyGrowth_glTF.gltf"}){
                Model3::CreateInfo info{};
                info.color = glm::vec3(0.0f);
                info.modelPath = path;
                info.metallic = 0.0f;
                info.roughness = 1.0f;
                // models[2]->setMatrix(intelSponzaMatrix() , 0);
                utils::ForgeBluePrint print{};
                print.mInfo = info;
                print.name = first ? "curtains" : "ivyGrowth";
                print.type = utils::moxBluePrintModel3;
                print.eInfo = {};
                eForge->addBluePrint(print);
                // the second model gets the name of the second : without this both were "curtains" and the forge refused the second one
                first = false;
            }

            const auto result = eForge->execute();
            MOX_CHECK_RESULT(result)

            glm::mat4 model(1.0f);
            model = glm::translate(model , glm::vec3(100.0f ,10.0f, 0.0f));

            eForge->setMatrixM("sponza" , intelSponzaMatrix() , 0);
            // eForge->setMatrixM("ext" , model , 0);
            eForge->setMatrixM("curtains" , intelSponzaMatrix() , 0);
            eForge->setMatrixM("ivyGrowth" , intelSponzaMatrix() , 0);

            return {};
        }

        // all lights

        [[nodiscard]] ErrorDataOutput<void> createLights() noexcept{
            // 3 lights
            {
                std::vector<lg::LightSource> lights(lightCount);

                for(auto& l : lights) l.type = MOX_AREA_LIGHT_TYPE_SPHERE;

                std::mt19937 rng{std::random_device{}()};
                const auto rnd = [&rng](const float low , const float high){
                    // the order of the limits does not matter ( the distribution asserts when low is bigger than high , and the program is aborted )
                    return std::uniform_real_distribution<float>(std::min(low , high) , std::max(low , high))(rng);
                };

                for(uint32_t i = 0 ; i < intelLampCount ; i++){
                    const float intensity = rnd(1.0f , 5.0f);
                    lights[i].diffuse = glm::vec3(rnd(0.4f , 1.0f) , rnd(0.4f , 1.0f) , rnd(0.4f , 1.0f)) * intensity;
                    lights[i].radius = rnd(15.0f , 30.0f);
                    lights[i].Linear = rnd(0.05f , 0.03f);
                    lights[i].Quadratic = rnd(0.02f , 0.004f);
                    lights[i].sourceRadius = rnd(1.0f , 2.5f);
                    lights[i].type = MOX_AREA_LIGHT_TYPE_SPHERE;
                    lights[i].direction = {0.0f , -1.0f , 0.0f};
                }

                lights[0].position = {-44.6 , 20.6 , -0.3f};
                lights[1].position = {6.1 , 21.1 , -0.9};
                lights[2].position = {-44.8 , 35.1 , 0.1};
                lights[3].position = {-19.0 , 20.1 , -14.7};
                lights[4].position = {-19.2 , 18.2 , 12.9};

                for(uint32_t i = 0 ; i < lights.size() ; i++){
                    sources.lights->addLight(lights[i] , std::format("light_{}" , i));
                }
            }
            return {};
        }
        
        MOX_NO_DISCARD ErrorDataOutput<void> initPasses() noexcept
        {
            if(!context || !sources.buffers || !sources.images || !sources.restir || !sources.lights || !sources.textures || !sources.atmo || !sources.vol || !sources.sharc)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create the passes , since the sources are not created ( initSources )");
            }

            if(!eForge || !eForge->heap)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create the passes , since VertexHeap does not exist ( the models have to be put together first )");
            }

            try
            {
                moxContext2* ctx = context.get();

                {
                    SourceCluster* clusters[] = {
                        sources.buffers.get() , sources.images.get() , sources.atmo.get() , eForge->heap.get() , sources.textures.get() ,
                        sources.lights.get() , sources.restir.get() , sources.sharc.get() , sources.vol.get() , sources.nrc.get() , sources.nkpd.get() , sources.bloom.get()
                    };

                    for(SourceCluster* cluster : clusters)
                    {
                        const auto result = graph.bindSourceCluster(cluster);
                        MOX_CHECK_RESULT(result)
                    }
                }

                {
                    Passes::mDeferredPass::CreateInfo info{};
                    info.context = ctx;
                    info.task_F = "../Shaders/DeferredRender/taskShader.hlsl";
                    info.mesh_F = "../Shaders/DeferredRender/meshShader.hlsl";
                    info.fragment_F = "../Shaders/DeferredRender/fragmentShader.hlsl";
                    info.general_F = "../Shaders/DeferredRender/general.mox";
                    deferred = std::make_shared<Passes::mDeferredPass>(info);
                }

                {
                    Passes::ForwardPlus::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/forwardPlus/forward";
                    forward = std::make_shared<Passes::ForwardPlus>(info);
                }

                {
                    Passes::AmbientOcclusion::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/AmbientOcclusion/AmbientOcclusion";
                    ao = std::make_shared<Passes::AmbientOcclusion>(info);
                }

                {
                    Passes::mRTPass::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/mRT/mRT";
                    mrt = std::make_shared<Passes::mRTPass>(info);
                }

                {
                    Passes::SHaRCPass::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/SHaRC/SHaRCPass";
                    sharcPass = std::make_shared<Passes::SHaRCPass>(info);
                }

                {
                    Passes::RayTracing::CreateInfo info{};
                    info.context = ctx;
                    info.directory = "../Shaders/RayTracing/";
                    info.moxName = "general";
                    rt = std::make_shared<Passes::RayTracing>(info);
                }

                {
                    Passes::PathTracing::CreateInfo info{};
                    info.context = ctx;
                    info.directory = "../Shaders/PathTracing/";
                    info.moxName = "pathTracing";
                    pt = std::make_shared<Passes::PathTracing>(info);
                }

                {
                    Passes::BF::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/BF_singular/bilateralFilteringSingular";
                    bf = std::make_shared<Passes::BF>(info);
                }

                {
                    Passes::ReBLUR<false , ShaderType::temporal>::CreateInfo t{};
                    t.context = ctx;
                    t.path_S = "../Shaders/denoiser/ReBLUR_temporal";
                    reblurt = std::make_shared<Passes::ReBLUR<false , ShaderType::temporal>>(t);

                    Passes::ReBLUR<false , ShaderType::Spatial>::CreateInfo s{};
                    s.context = ctx;
                    s.path_S = "../Shaders/denoiser/ReBLUR_spatial";
                    reblurs = std::make_shared<Passes::ReBLUR<false , ShaderType::Spatial>>(s);

                    Passes::ReBLUR<true , ShaderType::temporal>::CreateInfo ti{};
                    ti.context = ctx;
                    ti.path_S = "../Shaders/denoiser/ReBLUR_temporal";
                    reblurt_indirect = std::make_shared<Passes::ReBLUR<true , ShaderType::temporal>>(ti);

                    Passes::ReBLUR<true , ShaderType::Spatial>::CreateInfo si{};
                    si.context = ctx;
                    si.path_S = "../Shaders/denoiser/ReBLUR_spatial";
                    reblurs_indirect = std::make_shared<Passes::ReBLUR<true , ShaderType::Spatial>>(si);
                }

                {
                    Passes::ReLAX<ShaderType::temporal>::CreateInfo t{};
                    t.context = ctx;
                    t.path_S = "../Shaders/denoiser/ReLAX_temporal";
                    relaxt = std::make_shared<Passes::ReLAX<ShaderType::temporal>>(t);

                    Passes::ReLAX<ShaderType::Spatial>::CreateInfo s{};
                    s.context = ctx;
                    s.path_S = "../Shaders/denoiser/ReLAX_spatial";
                    relaxs = std::make_shared<Passes::ReLAX<ShaderType::Spatial>>(s);
                }

                {
                    Passes::Volumentric::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/Volumentric/Volumentric";
                    info.extent = sources.vol->extent;
                    info.layerCount = Source::VolSource::layerCount;
                    volumentric = std::make_shared<Passes::Volumentric>(info);
                }

                {
                    Passes::PostProcess::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/PostProcess/PostProcess";
                    pProcess = std::make_shared<Passes::PostProcess>(info);
                }

                {
                    Passes::HDRAE::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/HDR/HDRAE";
                    hdrae = std::make_shared<Passes::HDRAE>(info);
                }

                {
                    Passes::EVUpdatePass::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/HDR/updateEV";
                    evUpdate = std::make_shared<Passes::EVUpdatePass>(info);
                }

                {
                    Passes::HDR::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/HDR/HDR";
                    hdr = std::make_shared<Passes::HDR>(info);
                }

                {
                    Passes::DLSSPass::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/DLSS/DLSS";
                    info.depth = &deferred->depthBuffer;
                    dlssPass = std::make_shared<Passes::DLSSPass>(info);
                }

                {
                    Passes::NRCTrain::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NRC/trainingShader";
                    nrct = std::make_shared<Passes::NRCTrain>(info);
                }

                {
                    Passes::NRCBack::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NRC/backstepShader";
                    nrcb = std::make_shared<Passes::NRCBack>(info);
                }

                {
                    Passes::NRCApply1::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NRC/applyHidden";
                    nrcp1 = std::make_shared<Passes::NRCApply1>(info);
                }

                {
                    Passes::NRCApply2::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NRC/applyOutput";
                    nrcp2 = std::make_shared<Passes::NRCApply2>(info);
                }

                {
                    Passes::NRCApply3::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NRC/applyHash";
                    nrcp3 = std::make_shared<Passes::NRCApply3>(info);
                }

                // NKPD : the counters of the step , the backward ( two passes : the diffuse and the indirect signal , the push constant is the template argument ) and the Adam of the 3 layers
                {
                    Passes::NKPDTrain::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NKPD/trainingShader";
                    nkpdt = std::make_shared<Passes::NKPDTrain>(info);
                }

                {
                    Passes::NKPDBack<0>::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NKPD/backstepShader";
                    nkpdb = std::make_shared<Passes::NKPDBack<0>>(info);
                }

                {
                    Passes::NKPDBack<1>::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NKPD/backstepShader";
                    nkpdbi = std::make_shared<Passes::NKPDBack<1>>(info);
                }

                {
                    // NKPD of the indirect light : its own pass with the resolution of indirectImage ( the diffuse signal is denoised in PostProcess )
                    Passes::NKPDInference::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NKPD/inferenceIndirect";
                    nkpdi = std::make_shared<Passes::NKPDInference>(info);
                }

                {
                    Passes::NKPDApply1::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NKPD/applyInput";
                    nkpdp1 = std::make_shared<Passes::NKPDApply1>(info);
                }

                {
                    Passes::NKPDApply2::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NKPD/applyHidden";
                    nkpdp2 = std::make_shared<Passes::NKPDApply2>(info);
                }

                {
                    Passes::NKPDApply3::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/NKPD/applyOutput";
                    nkpdp3 = std::make_shared<Passes::NKPDApply3>(info);
                }

                {
                    Passes::GUIPass::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/GUI/GUIPass";
                    guiPass = std::make_shared<Passes::GUIPass>(info);
                }

                {
                    Passes::PathTracingIndirect::CreateInfo info{};
                    info.context = ctx;
                    info.directory = "../Shaders/PathTracingIndirect/";
                    info.moxName = "pathTracingIndirect";
                    ptI = std::make_shared<Passes::PathTracingIndirect>(info);
                }

                {
                    Passes::RayTracingIndirect::CreateInfo info{};
                    info.context = ctx;
                    info.path_S = "../Shaders/RayTracingIndirect/RayTracingIndirect";
                    rtI = std::make_shared<Passes::RayTracingIndirect>(info);
                }
                
                {
                    NodePass* order[] = {
                        deferred.get() , mrt.get() , forward.get(), nrct.get() , nrcb.get() , nrcp1.get() , nrcp2.get() , nrcp3.get() , sharcPass.get() , rtI.get() ,rt.get() , ao.get() , bf.get() ,
                        // the indirect one after the indirect spatial pass . the counters first , the Adam of the 3 layers ( and the translate of the weights ) after both backward passes
                        reblurt.get() , reblurs.get() , nkpdt.get() , nkpdb.get() ,
                        reblurt_indirect.get() , reblurs_indirect.get() , nkpdbi.get() ,
                        nkpdp1.get() , nkpdp2.get() , nkpdp3.get() ,
                        relaxt.get() , relaxs.get() , nkpdi.get() ,
                        // DLSS ( or the copy ) right after PostProcess : HDRAE and HDR read DLSSImage
                        volumentric.get() , pProcess.get() , dlssPass.get() , hdrae.get() , evUpdate.get() , hdr.get() , guiPass.get()
                    };

                    for(NodePass* pass : order)
                    {
                        const auto result = graph.bindNodePass(pass , RayTracingPath);
                        MOX_CHECK_RESULT(result)
                    }
                }

                // {
                //     NodePass* order[] = {
                //         deferred.get() , forward.get() , ao.get() , bf.get() , pProcess.get() , hdrae.get() , evUpdate.get() , hdr.get() , guiPass.get()
                //     };

                //     for(NodePass* pass : order)
                //     {
                //         const auto result = graph.bindNodePass(pass , RasterizationPath);
                //         MOX_CHECK_RESULT(result)
                //     }
                // }

                {
                    NodePass* order[] = {
                        deferred.get() , mrt.get() , forward.get(), nrct.get() , nrcb.get() , nrcp1.get() , nrcp2.get() , nrcp3.get() , sharcPass.get() , ptI.get() ,  pt.get() , ao.get() , bf.get() ,
                        reblurt.get() , reblurs.get() , nkpdt.get() , nkpdb.get() ,
                        reblurt_indirect.get() , reblurs_indirect.get() , nkpdbi.get() ,
                        nkpdp1.get() , nkpdp2.get() , nkpdp3.get() ,
                        relaxt.get() , relaxs.get() , nkpdi.get() ,
                        volumentric.get() , pProcess.get() , dlssPass.get() , hdrae.get() , evUpdate.get() , hdr.get() , guiPass.get()
                    };

                    for(NodePass* pass : order)
                    {
                        const auto result = graph.bindNodePass(pass , PathTracingPath);
                        MOX_CHECK_RESULT(result)
                    }
                }
            }
            catch(std::exception &e)
            {
                MOX_RETURN_ERROR(MOX_ERROR_TYPE_CRITICAL , "failed to create the passes , please check the logs above");
            }

            return {};
        }



        // update

        void update(VkCommandBuffer& cmd)
        {
            GeneralFeatures& features = featureSettings;
            // features : R did nothing
            features.currentMode = context->user->currentMode;
            FogVolume& fogData = fogSettings;
            // the time of the wind of the dust ( seconds , not too big : the float loses the fractions )
            fogData.dustTime = std::fmod(fogData.dustTime + dt , 3600.0f);
            SkyViewData& skyData = skySettings;
            SunSource& sun = sunSettings;
            GeneralInfo data = context->provideGeneralInfo(dt , lightCount , generalSettings);
            drawImGui(data , features , fogData , skyData , sun);
            generalSettings = data;

            if(generalSettings.prevViewProj == generalSettings.viewProj)
            {
                generalSettings.trainingIndex++;
            }
            else
            {
                generalSettings.trainingIndex = 0;
            }

            // the ground truth of the NKPD : the mean of the raw frames while the camera stands
            if(features.NKPDTrain && generalSettings.prevViewProj == generalSettings.viewProj) trainingFrames++;
            else trainingFrames = 0u;

            features.ReBLUR = data.ReBLUR;
            features.ReLAX = data.ReLAX;

            GeneralFeatures uploadedFeatures = features;
            GeneralInfo uploadedData = data;
            {
                bool warming = false;
                if(!features.NKPDTrain)
                {
                    nkpdTrainingTimer = MOX_TRAINING_START;
                }
                else if(nkpdTrainingTimer > 0.0f)
                {
                    nkpdTrainingTimer -= dt;
                    warming = true;
                }

                // the truth is not clean yet ( too few frames in the mean ) , or the camera moves : no steps
                if(features.NKPDTrain && trainingFrames < MOX_TRAINING_WARMUP_FRAMES + MOX_TRAINING_MIN_FRAMES) warming = true;

                if(warming)
                {
                    uploadedFeatures.NKPDTrain = 0u;
                    uploadedFeatures.ReBLUR = 1u;
                    uploadedData.ReBLUR = 1u;
                }
            }
            uploadedData.trainingIndex = (features.NKPDTrain && trainingFrames > MOX_TRAINING_WARMUP_FRAMES) ? trainingFrames : 0u;   // ReBLUR : the accumulation of the ground truth ( the frames )

            sl::Constants slData = context->convertToConstants(data);
            static uint8_t reset = true;
            if(reset)
            {
                slData.reset = sl::Boolean::eTrue;
                reset = false;
            }
            context->stream->setConstants(slData);
            {
                static uint32_t dlssgCounter = 0u;
                if((dlssgCounter++ % 120u) == 60u)
                {
                    const sl::DLSSGState state = context->stream->getStatus();
                    ErrorDataType message{MOX_ERROR_TYPE_NONE , std::format("DLSS-G : status {} , frames presented per frame {} , min size {}" , static_cast<int32_t>(state.status) , state.numFramesActuallyPresented , state.minWidthOrHeight)};
                    engineLogger(&message);
                }
            }
            translateRecords(deferred->depthBuffer.get() , sources.images->motionVector.get() , sources.images->finalImage.get() , sources.images->guiCleanImage.get());

            const auto frameGeneration = context->stream->setFrameGeneration();
            if(!frameGeneration.has_value()) engineLogger(frameGeneration);

            sources.buffers->generalBuffer->updateBufferCmd(cmd ,uploadedData , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            sources.buffers->sunBuffer->updateBufferCmd(cmd ,sun , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            sources.vol->fogBuffer->updateBufferCmd(cmd ,fogData , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            sources.atmo->skyBuffer->updateBufferCmd(cmd ,skyData , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            sources.buffers->features->updateBufferCmd(cmd ,uploadedFeatures , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT );
            sources.bloom->bloomBuffer->updateBufferCmd(cmd , bloomData , VK_ACCESS_2_SHADER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

            eForge->update(cmd);
            skyManager->update(cmd);
            sources.lights->lightShadowFade = features.lightShadowFade;
            sources.lights->updateGrid(cmd);
        }

        void setDefaultSettings() noexcept{
            generalSettings = GeneralInfo{};
            generalSettings.exposure = 1.6f;
            generalSettings.directWeight = 0.5f;
            generalSettings.indirectWeight = 0.5f;
            generalSettings.ambientInIndirect = 1;
            generalSettings.normalMapOpenGL = 1;
            generalSettings.ReBLUR = 1;
            generalSettings.ReLAX = 1;

            // features
            featureSettings = GeneralFeatures{};
            featureSettings.AO = SSAO_REQUIRED;
            featureSettings.AO_iteration = 16;
            featureSettings.AOradius = 0.6f;
            featureSettings.AO_intens = 1.0f;
            featureSettings.shadowThickness = 1.0f;
            featureSettings.RT = 1;
            featureSettings.PT = 0;
            featureSettings.currentMode = DEFAULT_MODE;
            featureSettings.randomValues = 1;
            featureSettings.volumentricFog = 0;
            featureSettings.IndirectShare = 0.5f;
            featureSettings.maxDepth = 50.0f;
            featureSettings.ReSTIRGI = true;
            featureSettings.ReSTIRGI_temporal = true;
            featureSettings.lightShadowFade = 0.5f;
            featureSettings.skyboxSampleCount = 4u;
            featureSettings.sunShadowFade = 200.0f;
            featureSettings.NRC = false;
            featureSettings.ReSTIRSSS = true;
            featureSettings.sssScale = 1.0f;
            featureSettings.NKPD = false;
            featureSettings.NKPDTrain = false;
            featureSettings.bloomRequired = 1u;

            featureSettings.lowCount = 0.65f;
            featureSettings.highCount = 0.90f;
            featureSettings.maxEV = 6.0f;
            featureSettings.minEV = -6.0f;


            bloomData = BloomInfo{};
            bloomData.threhold = 1.0f;
            bloomData.intensity = 0.3f;
            bloomData.range = 0.7f;

            fogSettings = FogVolume{};
            fogSettings.center = glm::vec3(0.0f);
            fogSettings.halfExtent = glm::vec3(1000.0f);
            fogSettings.albedo = glm::vec3(1.0f);
            fogSettings.density = 0.012f;
            fogSettings.feather = 0.0f;
            fogSettings.g = 0.45f;
            fogSettings.heightFallOff = 0.12f;
            fogSettings.fogBaseHeight = -2.0f;
            fogSettings.gridSize = static_cast<float>(Source::VolSource::layerCount);
            fogSettings.color = fogTint * fogGlow;

            // the dust ( off by default , the DUST button )
            featureSettings.dustMode = 0u;
            fogSettings.dustDensity = 0.006f;
            fogSettings.dustIntensity = 1.0f;
            fogSettings.dustScale = 0.3f;
            fogSettings.dustSpeed = 0.15f;
            fogSettings.dustTime = 0.0f;
            fogSettings.dustG = 0.6f;

            {
                const float elevation = glm::radians(40.0f);
                const float azimuth = glm::radians(215.0f);
                const glm::vec3 toSun(std::cos(elevation) * std::cos(azimuth) , std::sin(elevation) , std::cos(elevation) * std::sin(azimuth));

                sunSettings = SunSource{};
                sunSettings.sourceRadius = 0.1f;
                sunSettings.direction = -toSun;
                sunSettings.intensity = 4.0f;
                sunSettings.diffuse = glm::vec3(1.0f , 0.95f , 0.85f) * sunSettings.intensity;

                const float horizontal = std::sqrt(toSun.x * toSun.x + toSun.z * toSun.z);
                skySettings = SkyViewData{};
                skySettings.cameraHeight = 5.0f;
                skySettings.sunCosZenith = toSun.y;
                skySettings.sunCosAzimuth = toSun.x / horizontal;
                skySettings.sunSinAzimuth = toSun.z / horizontal;
            }

            requiredMode = RayTracingRequired;
        }

        void drawImGui(GeneralInfo &general , GeneralFeatures &features , FogVolume& fogData , SkyViewData &skyData , SunSource &sunData) noexcept {
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
        namespace gui = ImGui;
            
            gui::NewFrame();
            gui::Begin("generalSettings");
            gui::Text("FPS :  %.1f , ms : %.3f " , general.dt > 1e-6f ? 1.0f / general.dt : 0.0f , general.dt * 1000.0f);
            gui::Text("Player Position : X > %.1f  : Y > %.1f : Z >  %.1f" , context->user->position.x, context->user->position.y ,context->user->position.z);

            gui::Separator();

            if(gui::CollapsingHeader("General data")){
                drawSettingsForGeneralData(general);
            }

            gui::Separator();

            if(gui::CollapsingHeader("Fog settings")){
                drawSettingsForFog(fogData);
            }

            gui::Separator();

            if(gui::CollapsingHeader("Sun settings")){
                drawSettingsForSun(sunData,  skyData);
            }
            gui::Separator();

            if(gui::CollapsingHeader("Features")){
                drawSettingsForFeatures(features);
            }

            gui::Separator();

            if(gui::CollapsingHeader("Bloom")){
                drawSettingsForBloom(features , bloomData);
            }

            gui::Separator();

            applyNKPDRules(general , features);

            gui::End();
            gui::Render();
        }

        void applyNKPDRules(GeneralInfo &general , GeneralFeatures &features) noexcept{
            static uint32_t prevReBLUR = 1u;
            static uint32_t prevNKPD = 0u;

            if(features.NKPD && !prevNKPD)
            {
                general.ReBLUR = 0u;
                features.NKPDTrain = 0u;
            }
            else if(general.ReBLUR && !prevReBLUR)
            {
                features.NKPD = 0u;
                features.NKPDTrain = 0u;
            }

            // what always has to be true
            if(features.NKPD)
            {
                general.ReBLUR = 0u;
                features.NKPDTrain = 0u;
            }
            // the training needs ReBLUR ( its result is the ground truth , its temporal pass is the input of the net ) : without it the training is off , the net itself is off during the training
            if(!general.ReBLUR) features.NKPDTrain = 0u;
            if(features.NKPDTrain) features.NKPD = 0u;

            prevReBLUR = general.ReBLUR;
            prevNKPD = features.NKPD;
        }

        void drawSettingsForGeneralData(GeneralInfo &info) noexcept{
            namespace gui = ImGui;

            gui::Text("this settings are the basic, there is no really a big diff here");
            gui::Text("A general exposure , nothing really new");
            gui::SliderFloat("Exposure correction" , &info.exposure , 0.0f, 10.0f);
            gui::Separator();
            bool normalMapOpenGL = info.normalMapOpenGL != 0u;
            if(gui::Checkbox("Normal maps are OpenGL style" , &normalMapOpenGL)){
                info.normalMapOpenGL = normalMapOpenGL ? 1u : 0u;
            }

            bool blur = info.ReBLUR;
            bool lax = info.ReLAX;
            if(gui::Checkbox("Use ReBLUR" , &blur)){
                info.ReBLUR = blur ? 1u : 0u;
            }

            if(gui::Checkbox("Use ReLAX" , &lax)){
                info.ReLAX = lax ? 1u : 0u;
            }
        }

        void drawSettingsForBloom(GeneralFeatures &features , BloomInfo &info) noexcept{
            namespace gui = ImGui;

            bool enabled = features.bloomRequired != 0u;
            if(gui::Checkbox("Use Bloom" , &enabled)){
                features.bloomRequired = enabled ? 1u : 0u;
            }

            if(!enabled) return;

            gui::Separator();
            gui::Text("threshold : the brightness ( HDR , before the exposure ) , only what is brighter glows");
            gui::SliderFloat("Threshold" , &info.threhold , 0.0f , 10.0f , "%.2f" , ImGuiSliderFlags_Logarithmic);

            gui::Separator();
            gui::Text("intensity : how strong the glow is added to the picture");
            gui::SliderFloat("bloom Intensity" , &info.intensity , 0.0f , 2.0f , "%.3f");

            gui::Separator();
            gui::Text("range : how much of the wide levels is kept ( 0 : a tight glow , 1 : a wide one )");
            gui::SliderFloat("Range" , &info.range , 0.0f , 1.0f , "%.2f");
        }

        void drawSettingsForFog(FogVolume &info) noexcept{
            namespace gui = ImGui;

            gui::Text("this settings are for fog");
            
            gui::Text("Density : i think this doesnt require explanation");
            // a real fog is thin : 0.001 is a haze , 0.05 is a thick fog . the slider is logarithmic ( the step grows with the value ) and the range is small ,
            // so the small values are not squeezed into the first pixels of the slider . Ctrl + click types the number
            gui::SliderFloat("Density" , &info.density , 0.0f, 0.2f , "%.4f" , ImGuiSliderFlags_Logarithmic);

            gui::Separator();
            gui::Text("FallOff : how fast fog disapears per metr");
            gui::SliderFloat("HeightFallOff" , &info.heightFallOff , 0.0f, 1.0f , "%.3f" , ImGuiSliderFlags_Logarithmic);

            gui::Separator();
            gui::Text("baseHeight : at which point the fog is the most dens");
            // a drag : 0.05 m per pixel of the mouse , slow and precise
            gui::DragFloat("baseHeight" , &info.fogBaseHeight , 0.05f , -40.0f , 40.0f , "%.2f m");

            {
                gui::Separator();
                gui::Text("albedo : how does the light behaves in fog");
                // the colour is edited in place : the three floats of the vector
                gui::ColorEdit3("Albedo " , &info.albedo.x);
            }
            
            {
                // the glow of the fog is the product of the colour and the intensity : both are kept ( fogTint , fogGlow ) , so the colour is not lost when the glow is 0
                gui::Separator();
                gui::Text("fog Color : color of fog");
                bool changed = gui::ColorEdit3("Fog Color " , &fogTint.x);

                gui::Separator();
                gui::Text("fog Glow : same here , no explanation");
                changed |= gui::SliderFloat("Fog Glow" , &fogGlow , 0.0f , 2.0f , "%.3f" , ImGuiSliderFlags_Logarithmic);

                if(changed) info.color = fogTint * fogGlow;
            }

            gui::Separator();
            gui::Text("G : a twhich direction does the light go");
            gui::SliderFloat("G" , &info.g , -0.95f , 0.95f , "%.2f");

            // the dust ( the DUST button of the features ) : only the light of the sun and the lamps in the air , no fog
            gui::Separator();
            gui::Text("Dust ( god rays ) : the light of the sun and the lamps in the air , no fog");
            gui::SliderFloat("Dust density" , &info.dustDensity , 0.0f , 0.05f , "%.4f" , ImGuiSliderFlags_Logarithmic);
            gui::SliderFloat("Dust intensity" , &info.dustIntensity , 0.0f , 10.0f , "%.2f");
            gui::SliderFloat("Dust noise scale" , &info.dustScale , 0.02f , 2.0f , "%.2f" , ImGuiSliderFlags_Logarithmic);
            gui::SliderFloat("Dust wind speed" , &info.dustSpeed , 0.0f , 2.0f , "%.2f");
            gui::SliderFloat("Dust G" , &info.dustG , -0.95f , 0.95f , "%.2f");
        }
    
        void drawSettingsForSun(SunSource &sun , SkyViewData& sky) noexcept{

            float zenith = glm::degrees(std::acos(std::clamp(sky.sunCosZenith , -1.0f , 1.0f)));   // 0 .. 180 : 0 - overhead , 90 - horizon , more - under it
            float azimuth = glm::degrees(std::atan2(sky.sunSinAzimuth , sky.sunCosAzimuth));       // -180 .. 180
            static float powIntensity = 1.0f;
            if(azimuth < 0.0f) azimuth += 360.0f;                                                   // 0 .. 360 , 0 is +X

            static glm::vec3 diffuse = {1.0f , 1.0f , 1.0f};
            static float intensity = glm::sin(zenith);
            float color[3] = {diffuse.x , diffuse.y, diffuse.z};
            if(intensity > 1e-6f){
                color[0] = diffuse.x;
                color[1] = diffuse.y;
                color[2] = diffuse.z;
            }
            bool changed = false;
            namespace gui = ImGui;

            gui::Text("Zenith : the vertical orientation of the sun");
            changed |= gui::SliderFloat("Zenith" , &zenith , 0.0f , 97.5f , "%.1f deg");
            gui::Separator();

            gui::Text("Azimuth : the horizontal orientation of the sun");
            changed |= gui::SliderFloat("Azimuth" , &azimuth , 0.0f , 360.0f , "%.1f deg");
            gui::Separator();

            gui::Text("Sun color : the color of the sun");
            changed |= gui::ColorEdit3("Sun color" , color);
            gui::Separator();
            diffuse = {color[0] , color[1] , color[2]};
            // the size of the sun for the shadow : the half angle of the cone , radians ( the real sun is 0.00465 ) . the bigger , the softer the penumbra
            changed |= gui::SliderFloat("Sun source radius" , &sun.sourceRadius , 0.0f , 0.1f , "%.4f");
            gui::Separator();
            changed |= gui::SliderFloat("Intensity" , &powIntensity , 0.0f, 10.0f);
            if(changed){
                const float theta = glm::radians(zenith);
                const float phi = glm::radians(azimuth);

                const glm::vec3 toSun(std::sin(theta) * std::cos(phi) , std::cos(theta) , std::sin(theta) * std::sin(phi));
                intensity = glm::cos(glm::radians(glm::clamp(zenith , 0.0f  ,90.0f))) * powIntensity;
                sun.direction = -toSun;                 // the light travels away from the sun
                sun.intensity = intensity;
                sun.diffuse = glm::vec3(color[0] , color[1] , color[2]) * sun.intensity;

                sky.sunCosZenith = std::cos(theta);     // the same as toSun.y
                sky.sunCosAzimuth = std::cos(phi);
                sky.sunSinAzimuth = std::sin(phi);
            }

            sky.cameraHeight = std::max(context->user->position.y + 2.0f , 0.5f);
        }

        void drawSettingsForFeatures(GeneralFeatures &info) noexcept{
            namespace gui = ImGui;
            {
                gui::SliderFloat("min EV" , &info.minEV , -10.0f , 0.0f);
                gui::Separator();
                
                gui::Spacing();
                gui::SliderFloat("max EV" , &info.maxEV , 0.0f , 10.0f);
                gui::Spacing();
                gui::Separator();
            }

            {
                gui::SliderFloat("low Count" , &info.lowCount , 0.0f , 1.0f);
                gui::Separator();
                
                gui::Spacing();
                gui::SliderFloat("high Count" , &info.highCount , 0.0f , 1.0f);
                gui::Spacing();
                gui::Separator();
            }

            // AO
            gui::BeginDisabled(false);
            {
                // type
                if(gui::Button("SSAO")){
                    info.AO = SSAO_REQUIRED;
                }
                gui::SameLine();
                if(gui::Button("HBAO")){
                    info.AO = HBAO_REQUIRED;
                }
                gui::SameLine();
                if(gui::Button("NO AO")){
                    info.AO = AO_NOT_REQUIRED;
                }
                gui::BeginDisabled(info.AO == AO_NOT_REQUIRED);
                if(gui::CollapsingHeader("ao parameters"))
                {
                    gui::Separator();
                    ImGui::Spacing();
                    // intes
                    gui::SliderFloat("AO intesivity" , &info.AO_intens , 0.0f , 5.0f);
                    gui::Separator();
                    ImGui::Spacing();

                    // rate
                    gui::Separator();
                    gui::SliderInt("AO SampleCount" , std::bit_cast<int*>(&info.AO_iteration), 0 , 64);
                    gui::Separator();
                    ImGui::Spacing();

                    gui::Separator();
                    gui::SliderFloat("AO Radius" , &info.AOradius, 0 , 20.0f);
                    gui::Separator();
                }
                gui::EndDisabled();
            }

            gui::EndDisabled();
            gui::Separator();

            // shadows
            if(ImGui::CollapsingHeader("shader Parameters"))
            {
                {

                    gui::SliderFloat("shadow thickness" , &info.shadowThickness , 0.0f , 5.0f);
                    gui::Separator();
                    ImGui::Spacing();


                    gui::SliderFloat("light shadow fade" , &info.lightShadowFade , 0.0f , 2.0f);
                    gui::Separator();
                    gui::SliderFloat("sun shadow fade" , &info.sunShadowFade , 0.0f , 200.0f);
                    gui::Separator();
                }
            }
            gui::Separator();

            // RT PT rast

            {
                gui::SameLine();
                
                if(gui::Button("RAY TRACING")){
                    info.RT = 1;
                    info.PT = 0;
                    requiredMode = RayTracingRequired;
                }
                gui::SameLine();

                if(gui::Button("PATH TRACING")){
                    info.RT = 0;
                    info.PT = 1;
                    requiredMode = PathTracingRequired;
                }
            }
            gui::Separator();
            ImGui::Spacing();

            // SSR / SSGI
            {
                if(gui::Button(info.ReSTIRSSS ? "ReSTIRSSS ON" : "ReSTIRSSS OFF"))
                {
                    info.ReSTIRSSS = !info.ReSTIRSSS;
                }
                gui::SliderFloat("SSS scale" , &info.sssScale , 0.1f , 100.0f , "%.2f" , ImGuiSliderFlags_Logarithmic);
            }

            {
                {
                    if(gui::Button(info.NRC ? "NRC ON" : "NRC OFF"))
                    {
                        info.NRC = !info.NRC;
                    }
                }
                gui::Separator();
                gui::Spacing();
            }

            {
                // the training is on : NKPD can not be turned on
                gui::BeginDisabled(info.NKPDTrain != 0u);
                if(gui::Button(info.NKPD ? "NKPD USE ON" : "NKPD USE OFF"))
                {
                    info.NKPD = !info.NKPD;
                }
                gui::EndDisabled();

                // NKPD is on or ReBLUR is off : there is no ground truth to learn from
                gui::BeginDisabled(info.NKPD != 0u || info.ReBLUR == 0u);
                if(gui::Button(info.NKPDTrain ? "NKPD TRAIN ON" : "NKPD TRAIN OFF"))
                {
                    info.NKPDTrain = !info.NKPDTrain;
                }
                gui::EndDisabled();
                gui::Separator();
                gui::Spacing();
            }

            gui::Separator();

            if(info.RT)
            {
                if(gui::Button(Passes::mRTPass::enabled ? "G buffer : mRT ( ray tracing )" : "G buffer : raster ( mesh shaders )")){
                    Passes::mRTPass::enabled = !Passes::mRTPass::enabled;
                }
                gui::Separator();

                if(gui::CollapsingHeader("GI"))
                {
                    {
                        if(gui::Button(info.ReSTIRGI ? "ReSTIR GI : ON" : "ReSTIR GI : OFF")){
                            info.ReSTIRGI = !info.ReSTIRGI;
                            info.NRCFull = false;
                            info.SHaRC = false;
                        }
                    }
                    gui::Separator();

                    gui::BeginDisabled(!info.ReSTIRGI);
                    {
                        if(gui::Button(info.ReSTIRGI_temporal ? "ReSTIR GI temporal: ON" : "ReSTIR GI temporal: OFF")){
                            info.ReSTIRGI_temporal = !info.ReSTIRGI_temporal;
                        }
                    }
                    gui::EndDisabled();
                    gui::Separator();

                    {
                        if(gui::Button(info.SHaRC ? "SHaRC: ON" : "SHaRC: OFF")){
                            info.SHaRC = !info.SHaRC;
                            info.ReSTIRGI = false;
                            info.NRCFull = false;
                        }
                    }

                    gui::Separator();

                    gui::BeginDisabled(!info.RT);
                    {
                        if(gui::Button(info.NRCFull ? "NRCFull ON" : "NRCFull OFF"))
                        {
                            info.NRCFull = !info.NRCFull;
                            info.NRC = false;
                            info.SHaRC = false;
                        }
                    }
                    gui::EndDisabled();

                }
            }
            gui::Separator();




            // fog render
            gui::BeginDisabled(false);
            {
                if(gui::Button("VOLUMENTRIC RENDER")){
                    info.volumentricFog = !info.volumentricFog;
                }

                // the dust : god rays without the fog ( the light of the sun and the lamps in the air , added to the picture ) . it works without VOLUMENTRIC RENDER and replaces the fog while it is on
                if(gui::Button(info.dustMode ? "DUST ( GOD RAYS ) ON" : "DUST ( GOD RAYS ) OFF")){
                    info.dustMode = !info.dustMode;
                }
            }
            gui::EndDisabled();
        }


        void translateRecords(Image::RenderImage2* depth , Image::RenderImage2* motion , Image::RenderImage2* HUD , Image::RenderImage2* UI) noexcept
        {
            auto& stream = context->stream;
            if(!stream || !stream->pSetTagForFrame || !stream->currentFrame) return;

            sl::Resource depthRes = depth->translateToResource();
            sl::Resource motionRes = motion->translateToResource();
            sl::Resource hudRes = HUD->translateToResource();
            sl::Resource uiRes = UI->translateToResource();

            const sl::ResourceTag tags[4] = {
                sl::ResourceTag(&depthRes , sl::kBufferTypeDepth , sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&motionRes , sl::kBufferTypeMotionVectors , sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&hudRes , sl::kBufferTypeHUDLessColor , sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&uiRes , sl::kBufferTypeUIColorAndAlpha , sl::ResourceLifecycle::eValidUntilPresent)
            };

            const sl::Result result = stream->pSetTagForFrame(*stream->currentFrame , sl::ViewportHandle(0) , tags , 4 , nullptr);
            if(result != sl::Result::eOk) std::cout << "slSetTagForFrame failed , result = " << static_cast<int>(result) << "\n";
        }

        std::string lastFrameError{};

        void logFrameError(const ErrorDataOutput<void>& result) noexcept
        {
            if(result.has_value()) return;
            if(result.error().second == lastFrameError) return;
            lastFrameError = result.error().second;
            engineLogger(result);
        }

        void recordCmd(VkCommandBuffer& cmd , const uint32_t index) noexcept{
            {
                VkCommandBufferBeginInfo info{};
                info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                info.pInheritanceInfo = nullptr;
                info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                info.pNext = nullptr;

                const VkResult result = vkBeginCommandBuffer(cmd , &info);
                if(result != VK_SUCCESS)
                {
                    logFrameError(std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to begin the command buffer of the frame , error ID -> {}" , static_cast<int32_t>(result))}));
                }
            }
            update(cmd);

            sources.images->cleanImages(cmd);

            if(requiredMode == RasterizationRequired){
                const auto result = graph.executePath(cmd , RasterizationPath);
                if(!result.has_value()){
                    logFrameError(result);
                    vkEndCommandBuffer(cmd);
                    return;
                }
            }else if(requiredMode == RayTracingRequired){
                const auto result = graph.executePath(cmd , RayTracingPath);
                if(!result.has_value()){
                    logFrameError(result);
                    vkEndCommandBuffer(cmd);
                    return;
                }
            }else if(requiredMode == PathTracingRequired){
                const auto result = graph.executePath(cmd , PathTracingPath);
                if(!result.has_value()){
                    logFrameError(result);
                    vkEndCommandBuffer(cmd);
                    return;
                }
            }

            VkImageMemoryBarrier2 swapchainBarrier{};
            swapchainBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            swapchainBarrier.pNext = nullptr;
            swapchainBarrier.image = context->images[index];
            swapchainBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT , 0 , 1 , 0 , 1};

            // the final image is read by the copy , the swapchain image is written by it
            {
                std::array<VkImageMemoryBarrier2, 2> barriers;
                // guiImage : the colour with the UI over it . finalImage stays without the UI ( the HUDless of Streamline )
                barriers[0] = sources.images->guiImage->translateImage(VK_ACCESS_2_TRANSFER_READ_BIT , VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);

                barriers[1] = swapchainBarrier;
                barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[1].srcAccessMask = 0;
                barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 2;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = barriers.data();
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }

            {
                VkImageCopy region{};
                region.extent = {context->sc->extent.width , context->sc->extent.height , 1};
                region.dstOffset = {0 , 0};
                region.srcOffset = {0 , 0};
                region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.srcSubresource.baseArrayLayer = 0;
                region.srcSubresource.layerCount = 1;
                region.srcSubresource.mipLevel = 0;
                region.dstSubresource = region.srcSubresource;

                vkCmdCopyImage(cmd , sources.images->guiImage->image , VK_IMAGE_LAYOUT_GENERAL, context->images[index] , VK_IMAGE_LAYOUT_GENERAL , 1 , &region);
            }

            // the swapchain image is presented , and only PRESENT_SRC is a layout that can be presented
            {
                VkImageMemoryBarrier2 barrier = swapchainBarrier;
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
                barrier.dstAccessMask = 0;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

                VkDependencyInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                info.bufferMemoryBarrierCount = 0;
                info.imageMemoryBarrierCount = 1;
                info.pMemoryBarriers = nullptr;
                info.pBufferMemoryBarriers = nullptr;
                info.pImageMemoryBarriers = &barrier;
                info.memoryBarrierCount = 0;
                info.pNext = nullptr;
                info.dependencyFlags = 0;

                vkCmdPipelineBarrier2(cmd , &info);
            }

            const auto result = vkEndCommandBuffer(cmd);
        }

        pMoxContext2 context{};

        utils::pEntityForge eForge{};
        utils::pSkyManager skyManager{};

        Passes::pmDeferredPass deferred{};
        Passes::pForwardPlus forward{};
        Passes::pAmbientOcclusion ao{};
        Passes::pBF bf{};

        Passes::pReBLUR<false , ShaderType::temporal> reblurt;
        Passes::pReBLUR<false , ShaderType::Spatial> reblurs;
        Passes::pReBLUR<true , ShaderType::temporal> reblurt_indirect;
        Passes::pReBLUR<true , ShaderType::Spatial> reblurs_indirect;

        Passes::pReLAX<ShaderType::temporal> relaxt;
        Passes::pReLAX<ShaderType::Spatial> relaxs;
        Passes::pHDR hdr{};
        Passes::pHDRAE hdrae{};
        Passes::pDLSSPass dlssPass{};
        Passes::pEVUpdatePass evUpdate{};

        Passes::pPostProcess pProcess{};
        Passes::pVolumentric volumentric{};
        Passes::pSHaRCPass sharcPass{};
        Passes::pmRTPass mrt{};

        Passes::pRayTracing rt{};
        Passes::pPathTracingIndirect ptI{};
        Passes::pPathTracing pt{};
        Passes::pRayTracingIndirect rtI{};

        Passes::pNRCTrain nrct{};
        Passes::pNRCBack nrcb{};
        Passes::pNRCApply1 nrcp1{};
        Passes::pNRCApply2 nrcp2{};
        Passes::pNRCApply3 nrcp3{};

        Passes::pNKPDTrain nkpdt{};
        Passes::pNKPDBack<0> nkpdb{};
        Passes::pNKPDBack<1> nkpdbi{};
        Passes::pNKPDInference nkpdi{};
        Passes::pNKPDApply1 nkpdp1{};
        Passes::pNKPDApply2 nkpdp2{};
        Passes::pNKPDApply3 nkpdp3{};

        Passes::pGUIPass guiPass{};

        QueueRecordTypeFunction recordFunction{};

        static constexpr uint32_t intelLampCount = 5;
        static constexpr uint32_t lightCount = intelLampCount;
        float dt{0.0f};
        GeneralInfo generalSettings{};
        GeneralFeatures featureSettings{};
        FogVolume fogSettings{};
        SkyViewData skySettings{};
        SunSource sunSettings{};
        BloomInfo bloomData{};
        // the warm up of the NKPD training ( see update ) : the countdown ( seconds ) , is the history of ReBLUR still to be given its weight in the first frame of the training
        float nkpdTrainingTimer{MOX_TRAINING_START};
        // the frames of the accumulation of the ground truth : the training is on and the camera stands . 0 when the camera moves or the training is off
        uint32_t trainingFrames{0u};
        glm::vec3 fogTint{0.35f , 0.45f , 0.60f};
        float fogGlow{0.25f};
        uint32_t requiredMode = RasterizationRequired;
        RenderGraph graph{};
    };
}