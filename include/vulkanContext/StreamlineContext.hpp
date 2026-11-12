#pragma once

// BEFORE the headers of Streamline : sl_helpers_vk.h uses the types of Vulkan , sl_security.h uses the ones of Windows
#include <windows.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <iostream>
#include <sl.h>             // ядро: Preferences, slInit, slShutdown, ... (внутри уже тянет sl_consts.h, sl_core_api.h)
#include <sl_helpers_vk.h>    // sl::VulkanInfo, slSetVulkanInfo (по мере надобности)
#include <sl_dlss_g.h>        // Frame Generation: опции, состояние
#include <sl_reflex.h>        // Reflex (сам включает sl_pcl.h)
#include <sl_pcl.h>           // маркеры PCL
#include <sl_security.h>      
#include "../logger/logger.hpp"

#define SetTypicalOperators(type) type() = delete ;type(const type& input) = delete; type(type& input) = delete; type& operator=(type& input) = delete; type& operator=(const type& input) = delete;
#define SetTypicalTransferOperators(type) type(type&& input) noexcept{if(this == &input)return; processCopy(std::move(input));} type& operator=(type&& input) noexcept{if(this == &input)return *this; processCopy(std::move(input)); return *this;}
#define THROW_MESSAGE throw std::runtime_error("error occured , check logs")
#define MOX_DECLARE_PTR_STRUCTURE(type) using p##type = std::shared_ptr<type>;

namespace mox
{
    // the functions that Streamline intercepts ( sl_hooks.h : "Mandatory - Vulkan" and the two optional ones ) . every pointer is nullptr until StreamlineContext takes the PROXIES from sl.interposer.dll ,
    // the function under it ( the same name without Proxy , camelCase ) calls the proxy when there is one , and the usual function of Vulkan when there is not
    inline PFN_vkCreateInstance createInstanceProxy = nullptr;
    inline PFN_vkCreateDevice createDeviceProxy = nullptr;
    inline PFN_vkQueuePresentKHR queuePresentKHRProxy = nullptr;
    inline PFN_vkAcquireNextImageKHR acquireNextImageKHRProxy = nullptr;
    inline PFN_vkCreateSwapchainKHR createSwapchainKHRProxy = nullptr;
    inline PFN_vkDestroySwapchainKHR destroySwapchainKHRProxy = nullptr;
    inline PFN_vkGetSwapchainImagesKHR getSwapchainImagesKHRProxy = nullptr;
    inline PFN_vkDeviceWaitIdle deviceWaitIdleProxy = nullptr;
    #ifdef VK_USE_PLATFORM_WIN32_KHR
    inline PFN_vkCreateWin32SurfaceKHR createWin32SurfaceKHRProxy = nullptr;
    #endif
    inline PFN_vkDestroySurfaceKHR destroySurfaceKHRProxy = nullptr;
    inline PFN_vkGetInstanceProcAddr getInstanceProcAddress = nullptr;
    inline PFN_vkGetDeviceProcAddr getDeviceProcAddress = nullptr;

    // "createVkInstance" and not "createInstance" : Instance has a method with the name createInstance , inside of the class it would hide this function
    inline VkResult createVkInstance(const VkInstanceCreateInfo* info , const VkAllocationCallbacks* alloc , VkInstance* instance){
        if(createInstanceProxy) return createInstanceProxy(info , alloc , instance);
        return vkCreateInstance(info , alloc , instance);
    }

    inline VkResult createVkDevice(VkPhysicalDevice physicalDevice , const VkDeviceCreateInfo* info , const VkAllocationCallbacks* alloc , VkDevice* device){
        if(createDeviceProxy) return createDeviceProxy(physicalDevice , info , alloc , device);
        return vkCreateDevice(physicalDevice , info , alloc , device);
    }

    inline VkResult queuePresentKHR(VkQueue queue , const VkPresentInfoKHR* info){
        if(queuePresentKHRProxy) return queuePresentKHRProxy(queue , info);
        return vkQueuePresentKHR(queue , info);
    }

    inline VkResult acquireNextImageKHR(VkDevice device , VkSwapchainKHR swapchain , uint64_t timeout , VkSemaphore semaphore , VkFence fence , uint32_t* imageIndex){
        if(acquireNextImageKHRProxy) return acquireNextImageKHRProxy(device , swapchain , timeout , semaphore , fence , imageIndex);
        return vkAcquireNextImageKHR(device , swapchain , timeout , semaphore , fence , imageIndex);
    }

    inline VkResult createSwapchainKHR(VkDevice device , const VkSwapchainCreateInfoKHR* info , const VkAllocationCallbacks* alloc , VkSwapchainKHR* swapchain){
        if(createSwapchainKHRProxy) return createSwapchainKHRProxy(device , info , alloc , swapchain);
        return vkCreateSwapchainKHR(device , info , alloc , swapchain);
    }

    inline void destroySwapchainKHR(VkDevice device , VkSwapchainKHR swapchain , const VkAllocationCallbacks* alloc){
        if(destroySwapchainKHRProxy) destroySwapchainKHRProxy(device , swapchain , alloc);
        else vkDestroySwapchainKHR(device , swapchain , alloc);
    }

    inline VkResult getSwapchainImagesKHR(VkDevice device , VkSwapchainKHR swapchain , uint32_t* count , VkImage* images){
        if(getSwapchainImagesKHRProxy) return getSwapchainImagesKHRProxy(device , swapchain , count , images);
        return vkGetSwapchainImagesKHR(device , swapchain , count , images);
    }

    inline VkResult deviceWaitIdle(VkDevice device){
        if(deviceWaitIdleProxy) return deviceWaitIdleProxy(device);
        return vkDeviceWaitIdle(device);
    }

    #ifdef VK_USE_PLATFORM_WIN32_KHR
    inline VkResult createWin32SurfaceKHR(VkInstance instance , const VkWin32SurfaceCreateInfoKHR* info , const VkAllocationCallbacks* alloc , VkSurfaceKHR* surface){
        if(createWin32SurfaceKHRProxy) return createWin32SurfaceKHRProxy(instance , info , alloc , surface);
        return vkCreateWin32SurfaceKHR(instance , info , alloc , surface);
    }
    #endif

    inline void destroySurfaceKHR(VkInstance instance , VkSurfaceKHR surface , const VkAllocationCallbacks* alloc){
        if(destroySurfaceKHRProxy) destroySurfaceKHRProxy(instance , surface , alloc);
        else vkDestroySurfaceKHR(instance , surface , alloc);
    }

    // glm -> Streamline . the types are not related , there is no assignment between them
    // vectors : the same order of the components , only the type is different
    inline sl::float2 toSl(const glm::vec2& v) noexcept
    {
        return sl::float2(v.x , v.y);
    }

    inline sl::float3 toSl(const glm::vec3& v) noexcept
    {
        return sl::float3(v.x , v.y , v.z);
    }

    // the matrix : glm keeps COLUMNS ( m[column][row] ) , SL wants ROWS ( row major ) : the row r of the result is m[0][r] , m[1][r] , m[2][r] , m[3][r] . this is the transpose , it is
    // already inside of the indices , glm::transpose is not needed
    inline sl::float4x4 toSl(const glm::mat4& m) noexcept
    {
        sl::float4x4 out;
        for(uint32_t r = 0 ; r < 4 ; r++)
        {
            out.setRow(r , sl::float4(m[0][r] , m[1][r] , m[2][r] , m[3][r]));
        }
        return out;
    }

    class StreamlineContext final
    {
    public:

        struct CreateInfo
        {
            std::wstring folder;
        };

        static inline const std::wstring PATH = L"C:\\Users\\mosha\\sdk\\streamline-release-2.14.16\\extracted\\bin\\x64\\sl.interposer.dll";

        SetTypicalOperators(StreamlineContext)
        SetTypicalTransferOperators(StreamlineContext)

        explicit StreamlineContext(const CreateInfo& info)
        {
            const auto result = _implProcessInit(info);
            if(!result.has_value())
            {
                engineLogger(result);
                THROW_MESSAGE;
            }
        }

        const sl::FeatureRequirements acquireRequirements() const noexcept
        {
            return requirements;
        }

        void printRequirements() const noexcept
        {
            const sl::Feature list[3] = {sl::kFeatureDLSS_G , sl::kFeatureReflex , sl::kFeaturePCL};
            for(uint32_t i = 0 ; i < 3 ; i++)
            {
                const sl::FeatureRequirements& r = allRequirements[i];
                const uint32_t flags = static_cast<uint32_t>(r.flags);

                std::cout << "==== requirements of " << featureName(list[i]) << " ====\n";
                std::cout << "  vulkan supported : " << ((flags & static_cast<uint32_t>(sl::FeatureRequirementFlags::eVulkanSupported)) != 0) << "\n";
                std::cout << "  vsync has to be off : " << ((flags & static_cast<uint32_t>(sl::FeatureRequirementFlags::eVSyncOffRequired)) != 0) << "\n";
                std::cout << "  hardware gpu scheduling required : " << ((flags & static_cast<uint32_t>(sl::FeatureRequirementFlags::eHardwareSchedulingRequired)) != 0) << "\n";
                std::cout << "  max cpu threads : " << r.maxNumCPUThreads << " , max viewports : " << r.maxNumViewports << "\n";

                printVersion("os detected : " , r.osVersionDetected);
                printVersion("os required : " , r.osVersionRequired);
                printVersion("driver detected : " , r.driverVersionDetected);
                printVersion("driver required : " , r.driverVersionRequired);

                std::cout << "  required tags ( " << r.numRequiredTags << " ) :\n";
                for(uint32_t t = 0 ; t < r.numRequiredTags ; t++)
                {
                    std::cout << "    " << r.requiredTags[t] << " " << bufferName(r.requiredTags[t]) << "\n";
                }

                std::cout << "  extra queues : graphics " << r.vkNumGraphicsQueuesRequired << " , compute " << r.vkNumComputeQueuesRequired << " , optical flow " << r.vkNumOpticalFlowQueuesRequired << "\n";

                std::cout << "  instance extensions ( " << r.vkNumInstanceExtensions << " ) :\n";
                for(uint32_t e = 0 ; e < r.vkNumInstanceExtensions ; e++) std::cout << "    " << r.vkInstanceExtensions[e] << "\n";

                std::cout << "  device extensions ( " << r.vkNumDeviceExtensions << " ) :\n";
                for(uint32_t e = 0 ; e < r.vkNumDeviceExtensions ; e++) std::cout << "    " << r.vkDeviceExtensions[e] << "\n";

                std::cout << "  vulkan 1.2 features ( " << r.vkNumFeatures12 << " ) :\n";
                for(uint32_t e = 0 ; e < r.vkNumFeatures12 ; e++) std::cout << "    " << r.vkFeatures12[e] << "\n";

                std::cout << "  vulkan 1.3 features ( " << r.vkNumFeatures13 << " ) :\n";
                for(uint32_t e = 0 ; e < r.vkNumFeatures13 ; e++) std::cout << "    " << r.vkFeatures13[e] << "\n";
            }
        }

        // call it RIGHT AFTER the instance is created : the proxies that need the instance ( the surface ) . createInstanceProxy is loaded in the constructor ( the instance is made by SL ) ,
        // createDeviceProxy stays nullptr on purpose : the queues are added by hand ( the lists of the requirements ) and SL gets the device with slSetVulkanInfo
        void loadInstanceProxies(const VkInstance instance) noexcept
        {
            destroySurfaceKHRProxy = std::bit_cast<PFN_vkDestroySurfaceKHR>(getInstanceProcAddress(instance , "vkDestroySurfaceKHR"));
            #ifdef VK_USE_PLATFORM_WIN32_KHR
            createWin32SurfaceKHRProxy = std::bit_cast<PFN_vkCreateWin32SurfaceKHR>(getInstanceProcAddress(instance , "vkCreateWin32SurfaceKHR"));
            #endif
            std::cout << "SL proxies of the instance : destroySurface " << (destroySurfaceKHRProxy != nullptr)
            #ifdef VK_USE_PLATFORM_WIN32_KHR
                      << " , createWin32Surface " << (createWin32SurfaceKHRProxy != nullptr)
            #endif
                      << "\n";
        }

        // call it RIGHT AFTER the device is created : the proxies of the swapchain , of the present and of the wait idle
        void loadDeviceProxies(const VkDevice device) noexcept
        {
            queuePresentKHRProxy       = std::bit_cast<PFN_vkQueuePresentKHR>(getDeviceProcAddress(device , "vkQueuePresentKHR"));
            acquireNextImageKHRProxy   = std::bit_cast<PFN_vkAcquireNextImageKHR>(getDeviceProcAddress(device , "vkAcquireNextImageKHR"));
            createSwapchainKHRProxy    = std::bit_cast<PFN_vkCreateSwapchainKHR>(getDeviceProcAddress(device , "vkCreateSwapchainKHR"));
            destroySwapchainKHRProxy   = std::bit_cast<PFN_vkDestroySwapchainKHR>(getDeviceProcAddress(device , "vkDestroySwapchainKHR"));
            getSwapchainImagesKHRProxy = std::bit_cast<PFN_vkGetSwapchainImagesKHR>(getDeviceProcAddress(device , "vkGetSwapchainImagesKHR"));
            deviceWaitIdleProxy        = std::bit_cast<PFN_vkDeviceWaitIdle>(getDeviceProcAddress(device , "vkDeviceWaitIdle"));
            std::cout << "SL proxies of the device : present " << (queuePresentKHRProxy != nullptr)
                      << " , acquire " << (acquireNextImageKHRProxy != nullptr)
                      << " , createSwapchain " << (createSwapchainKHRProxy != nullptr)
                      << " , destroySwapchain " << (destroySwapchainKHRProxy != nullptr)
                      << " , getSwapchainImages " << (getSwapchainImagesKHRProxy != nullptr)
                      << " , deviceWaitIdle " << (deviceWaitIdleProxy != nullptr) << "\n";
        }

        // how many EXTRA queues the features want from the host ( the sum over DLSS_G , Reflex , PCL ) : the host creates them together with its own queues
        [[nodiscard]] uint32_t extraGraphicsQueues() const noexcept
        {
            uint32_t sum = 0;
            for(uint32_t i = 0 ; i < 3 ; i++) sum += allRequirements[i].vkNumGraphicsQueuesRequired;
            return sum;
        }

        [[nodiscard]] uint32_t extraComputeQueues() const noexcept
        {
            uint32_t sum = 0;
            for(uint32_t i = 0 ; i < 3 ; i++) sum += allRequirements[i].vkNumComputeQueuesRequired;
            return sum;
        }

        [[nodiscard]] ErrorDataOutput<void> setVulkanInfo(const VkInstance instance , const VkPhysicalDevice physicalDevice , const VkDevice device ,
                                                          const uint32_t graphicsFamily , const uint32_t graphicsIndex ,
                                                          const uint32_t computeFamily , const uint32_t computeIndex) noexcept
        {
            sl::VulkanInfo info{};
            info.instance = instance;
            info.physicalDevice = physicalDevice;
            info.device = device;
            info.graphicsQueueFamily = graphicsFamily;
            info.graphicsQueueIndex = graphicsIndex;
            info.computeQueueFamily = computeFamily;
            info.computeQueueIndex = computeIndex;
            info.useNativeOpticalFlowMode = false;

            const sl::Result result = pSetVulkanInfo(info);
            if(result != sl::Result::eOk)
            {
                std::cout << "slSetVulkanInfo failed , sl::Result = " << static_cast<uint32_t>(result) << "\n";
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "slSetVulkanInfo failed"});
            }
            std::cout << "slSetVulkanInfo ok : graphics family " << graphicsFamily << " from " << graphicsIndex << " , compute family " << computeFamily << " from " << computeIndex << "\n";

            const bool loaded = loadFeatureFunction(sl::kFeatureReflex , "slReflexSleep" , pReflexSleep) &&
                                loadFeatureFunction(sl::kFeatureReflex , "slReflexSetOptions" , pReflexSetOptions) &&
                                loadFeatureFunction(sl::kFeaturePCL , "slPCLSetMarker" , pPCLSetMarker) &&
                                loadFeatureFunction(sl::kFeatureDLSS_G , "slDLSSGSetOptions" , pDLSSGSetOptions) &&
                                loadFeatureFunction(sl::kFeatureDLSS_G , "slDLSSGGetState" , pDLSSGGetState);
            if(!loaded)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to get the functions of the features of SL ( Reflex , PCL , DLSS_G )"});
            }

            sl::ReflexOptions reflex{};
            reflex.mode = sl::ReflexMode::eLowLatency;
            const sl::Result reflexResult = pReflexSetOptions(reflex);
            if(reflexResult != sl::Result::eOk)
            {
                std::cout << "slReflexSetOptions failed , sl::Result = " << static_cast<uint32_t>(reflexResult) << "\n";
            }

            frameFunctionsReady = true;
            std::cout << "the frame functions of SL are ready\n";
            return {};
        }

        void beginFrame(const uint32_t frameIndex) noexcept
        {
            currentFrame = nullptr;
            if(!frameFunctionsReady) return;

            sl::FrameToken* token = nullptr;
            if(pGetNewFrameToken(token , &frameIndex) != sl::Result::eOk || !token) return;
            currentFrame = token;

            pReflexSleep(*currentFrame);
            pPCLSetMarker(sl::PCLMarker::eSimulationStart , *currentFrame);
        }

        void marker(const sl::PCLMarker marker) noexcept
        {
            if(currentFrame) pPCLSetMarker(marker , *currentFrame);
        }

        void setConstants(sl::Constants const data) noexcept
        {
            pSetConstants(data , *currentFrame, sl::ViewportHandle(0));
        }

        [[nodiscard]] ErrorDataOutput<void> setFrameGeneration() noexcept 
        {
            sl::DLSSGOptions options{};
            options.mode = sl::DLSSGMode::eOn;
            options.numFramesToGenerate = 3;
            options.enableUserInterfaceRecomposition = sl::Boolean::eTrue;
            options.numBackBuffers = 3;
            const auto result = pDLSSGSetOptions(sl::ViewportHandle(0) , options);
            if(result != sl::Result::eOk)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , std::format("failed to set options for DLSS G , error ID -> {}" , static_cast<int32_t>(result))});
            }
            return {};
        }

        sl::DLSSGState getStatus() const noexcept
        {
            sl::DLSSGOptions options{};
            options.mode = sl::DLSSGMode::eOn;
            options.numFramesToGenerate = 3;
            options.enableUserInterfaceRecomposition = sl::Boolean::eTrue;
            options.numBackBuffers = 3;

            sl::DLSSGState state{};
            pDLSSGGetState(sl::ViewportHandle(0), state, &options);
            return state;
        }

        ~StreamlineContext() noexcept
        {
            processDelete();
        }

        PFun_slSetVulkanInfo* pSetVulkanInfo = std::nullptr_t();

        PFun_slGetNewFrameToken* pGetNewFrameToken = std::nullptr_t();
        PFun_slGetFeatureFunction* pGetFeatureFunction = std::nullptr_t();
        PFun_slSetConstants* pSetConstants = std::nullptr_t();
        PFun_slSetTagForFrame* pSetTagForFrame = std::nullptr_t();
        PFun_slReflexSleep* pReflexSleep = std::nullptr_t();
        PFun_slReflexSetOptions* pReflexSetOptions = std::nullptr_t();
        PFun_slPCLSetMarker* pPCLSetMarker = std::nullptr_t();
        PFun_slDLSSGSetOptions* pDLSSGSetOptions = std::nullptr_t();
        PFun_slDLSSGGetState* pDLSSGGetState = std::nullptr_t();

        sl::FrameToken* currentFrame = nullptr;
        PFun_slShutdown* pShutdown = std::nullptr_t();
        PFun_slInit* pInit = std::nullptr_t();
        PFun_slGetFeatureRequirements* getFeatureRequirements = std::nullptr_t();
        HMODULE module{};

    private:

        bool slInitialized = false;
        bool frameFunctionsReady = false;

        template<typename Function>
        [[nodiscard]] bool loadSlFunction(Function*& function , const char* name) noexcept
        {
            function = std::bit_cast<Function*>(GetProcAddress(module , name));
            return function != nullptr;
        }

        // a function of a feature : SL gives its address by the name , only after slSetVulkanInfo
        template<typename Function>
        [[nodiscard]] bool loadFeatureFunction(const sl::Feature feature , const char* name , Function*& function) noexcept
        {
            void* address = nullptr;
            if(pGetFeatureFunction(feature , name , address) != sl::Result::eOk || !address) return false;
            function = reinterpret_cast<Function*>(address);
            return true;
        }

        std::wstring pluginsFolder;
        const wchar_t* pluginPaths[1]{};
        sl::Feature features[3]{};
        sl::FeatureRequirements requirements{};
        // the answers for features[0] ( DLSS_G ) , features[1] ( Reflex ) , features[2] ( PCL ) . the pointers inside ( extension names , tags ) belong to the DLL of SL and live until slShutdown
        sl::FeatureRequirements allRequirements[3]{};

        static const char* featureName(const sl::Feature feature) noexcept
        {
            if(feature == sl::kFeatureDLSS_G) return "DLSS_G";
            if(feature == sl::kFeatureReflex) return "Reflex";
            if(feature == sl::kFeaturePCL)    return "PCL";
            return "unknown feature";
        }

        static const char* bufferName(const sl::BufferType type) noexcept
        {
            if(type == sl::kBufferTypeDepth)           return "Depth";
            if(type == sl::kBufferTypeMotionVectors)   return "MotionVectors";
            if(type == sl::kBufferTypeHUDLessColor)    return "HUDLessColor";
            if(type == sl::kBufferTypeUIColorAndAlpha) return "UIColorAndAlpha";
            if(type == sl::kBufferTypeUIAlpha)         return "UIAlpha";
            return "other ( see sl_core_types.h )";
        }

        static void printVersion(const char* label , const sl::Version& v) noexcept
        {
            std::cout << "  " << label << v.major << "." << v.minor << "." << v.build << "\n";
        }

        static inline const std::wstring LOGS = L"C:\\Users\\mosha\\personalEngine2\\build\\sl_logs";

        static void logCallback(sl::LogType type , const char* msg)
        {
            std::cout << "[SL] " << msg;
        }

        void processDelete() noexcept
        {
            // SL has to be shut down BEFORE its DLL is unloaded
            if(slInitialized && pShutdown)
            {
                pShutdown();
                slInitialized = false;
            }
            // the proxies point INTO the DLL : after it is unloaded they would be dangling , the wrappers of Instance.hpp fall back to the usual functions of Vulkan
            createInstanceProxy = nullptr;
            createDeviceProxy = nullptr;
            queuePresentKHRProxy = nullptr;
            acquireNextImageKHRProxy = nullptr;
            createSwapchainKHRProxy = nullptr;
            destroySwapchainKHRProxy = nullptr;
            getSwapchainImagesKHRProxy = nullptr;
            deviceWaitIdleProxy = nullptr;
            #ifdef VK_USE_PLATFORM_WIN32_KHR
            createWin32SurfaceKHRProxy = nullptr;
            #endif
            destroySurfaceKHRProxy = nullptr;
            getInstanceProcAddress = nullptr;
            getDeviceProcAddress = nullptr;

            if(module) FreeLibrary(module);
            module = nullptr;
            pShutdown = nullptr;
            pInit = nullptr;
            pSetVulkanInfo = nullptr;

            frameFunctionsReady = false;
            currentFrame = nullptr;
            pGetNewFrameToken = nullptr;
            pGetFeatureFunction = nullptr;
            pSetConstants = nullptr;
            pSetTagForFrame = nullptr;
            pReflexSleep = nullptr;
            pReflexSetOptions = nullptr;
            pPCLSetMarker = nullptr;
            pDLSSGSetOptions = nullptr;
            pDLSSGGetState = nullptr;
        }

        void processCopy(StreamlineContext&& input) noexcept
        {
            
        }

        [[nodiscard]] ErrorDataOutput<void> _implProcessInit(const CreateInfo& input) noexcept
        {
            // DEBUG ( remove later ) : every output goes out at once , so a crash does not eat the last lines of the console
            std::cout << std::unitbuf;

            const std::wstring path = input.folder.empty() ? PATH : input.folder + L"\\sl.interposer.dll";

            module = LoadLibraryW(path.c_str());
            if(!module)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load library for FG"});
            }
            else
            {
                std::cout << "lib loaded\n";
            }
            pShutdown = std::bit_cast<PFun_slShutdown*>(GetProcAddress(module , "slShutdown"));
            if(!pShutdown)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load library slShutdown"});
            }

            pInit = std::bit_cast<PFun_slInit*>(GetProcAddress(module , "slInit"));
            if(!pInit)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load library slInit"});
            }

            getFeatureRequirements = std::bit_cast<PFun_slGetFeatureRequirements*>(GetProcAddress(module , "slGetFeatureRequirements"));
            if(!getFeatureRequirements)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load library slGetFeatureRequirements"});
            }

            pluginsFolder = path.substr(0 , path.find_last_of(L"\\/"));

            getInstanceProcAddress = std::bit_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(module , "vkGetInstanceProcAddr"));
            if(!getInstanceProcAddress)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load vkGetInstanceProcAddr from the interposer"});
            }

            getDeviceProcAddress = std::bit_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(module , "vkGetDeviceProcAddr"));
            if(!getDeviceProcAddress)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load vkGetDeviceProcAddr from the interposer"});
            }

            pSetVulkanInfo = std::bit_cast<PFun_slSetVulkanInfo*>(GetProcAddress(module , "slSetVulkanInfo"));
            if(!pSetVulkanInfo)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load slSetVulkanInfo"});
            }

            if(!loadSlFunction(pGetNewFrameToken , "slGetNewFrameToken") || !loadSlFunction(pGetFeatureFunction , "slGetFeatureFunction") ||
               !loadSlFunction(pSetConstants , "slSetConstants") || !loadSlFunction(pSetTagForFrame , "slSetTagForFrame"))
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load the frame functions of SL ( slGetNewFrameToken , slGetFeatureFunction , slSetConstants , slSetTagForFrame )"});
            }

            const auto initResult = _implProcessSlInit();
            if(!initResult.has_value()) return initResult;

            createInstanceProxy = std::bit_cast<PFN_vkCreateInstance>(getInstanceProcAddress(VK_NULL_HANDLE , "vkCreateInstance"));
            if(!createInstanceProxy)
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to load the vkCreateInstance proxy of SL"});
            }
            std::cout << "SL proxy of vkCreateInstance loaded\n";
            return {};
        }

        [[nodiscard]] ErrorDataOutput<void> _implProcessSlInit() noexcept
        {
            CreateDirectoryW(LOGS.c_str() , nullptr);

            features[0] = sl::kFeatureDLSS_G;
            features[1] = sl::kFeatureReflex;
            features[2] = sl::kFeaturePCL;
            pluginPaths[0] = pluginsFolder.c_str();

            sl::Preferences prefs{};
            prefs.showConsole = false;
            prefs.logLevel = sl::LogLevel::eVerbose;
            prefs.pathsToPlugins = pluginPaths;
            prefs.numPathsToPlugins = 1;
            prefs.pathToLogsAndData = LOGS.c_str();
            prefs.logMessageCallback = &logCallback;
            prefs.featuresToLoad = features;
            prefs.numFeaturesToLoad = 3;
            prefs.engine = sl::EngineType::eCustom;
            prefs.engineVersion = "1.0";
            prefs.projectId = "5b7f0c3e-8a21-4c6d-9d1e-2f4a6b8c0d12";
            prefs.renderAPI = sl::RenderAPI::eVulkan;
            prefs.flags = prefs.flags | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
            prefs.flags = prefs.flags | sl::PreferenceFlags::eUseManualHooking;

            // the second argument has to be given : the default value exists only in the declaration of slInit , not in the type of the pointer
            const sl::Result result = pInit(prefs , sl::kSDKVersion);
            if(result != sl::Result::eOk)
            {
                std::cout << "slInit failed , sl::Result = " << static_cast<uint32_t>(result) << "\n";
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "slInit failed"});
            }

            slInitialized = true;
            std::cout << "slInit ok\n";
            // the requirements of EVERY feature that was loaded : each one can ask for its own extensions and queues , the results are added together
            for(uint32_t i = 0 ; i < 3 ; i++)
            {
                const sl::Result result = getFeatureRequirements(features[i] , allRequirements[i]);
                if(result != sl::Result::eOk)
                {
                    std::cout << "failed to get requirements of " << featureName(features[i]) << " , sl::Result = " << static_cast<uint32_t>(result) << "\n";
                    return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_CRITICAL , "failed to get requirements"});
                }
            }
            // features[0] is DLSS_G : the old name keeps working
            requirements = allRequirements[0];
            printRequirements();
            return {};
        }
    };

    MOX_DECLARE_PTR_STRUCTURE(StreamlineContext)
}