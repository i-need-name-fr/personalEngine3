#pragma once

#include "Window.hpp"
#include "Instance.hpp"

namespace mox{

    class Surface final{
    private:

        VkInstance instance = nullptr;

        ErrorDataOutput<void> init(GLFWwindow* window){
            if(!instance || !window){
                ErrorDataType error = {MOX_ERROR_TYPE_FAILED_CREATION , "failed to create the surface , since the window or instance are null"};
                return std::unexpected(error);
            }
            // the surface is made by hand with createWin32SurfaceKHR ( the proxy of Streamline when it is loaded ) , NOT with glfwCreateWindowSurface : SL learns which window the swapchain belongs to
            // only from this call , without it vkCreateSwapchainKHR fails with "Could not find a window corresponding to this application"
            VkWin32SurfaceCreateInfoKHR surfaceInfo{};
            surfaceInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
            surfaceInfo.hinstance = GetModuleHandleW(nullptr);
            surfaceInfo.hwnd = glfwGetWin32Window(window);
            auto result = createWin32SurfaceKHR(instance , &surfaceInfo , nullptr , &surface);
            if(result != VK_SUCCESS){
                ErrorDataType error = {MOX_ERROR_TYPE_FAILED_CREATION , std::format("failed to create the surface, ERROR {}" , static_cast<int32_t>(result))};
                return std::unexpected(error);
            }

            return {};
        }

        void clean(){
            if(surface){
                destroySurfaceKHR(instance , surface , nullptr);
                ErrorDataType data = {MOX_ERROR_TYPE_NONE , "destroys the surface"};
                engineLogger(&data);
            }
        }

    public:
        // init
        Surface() = delete;

        struct SurfaceCreateInfo{
            VkInstance instance;
            GLFWwindow* window;
        };

        Surface(const SurfaceCreateInfo& info){
            if(!info.instance || !info.window ) throw std::runtime_error("info must have valid window and instance, surface creation failure");\
            instance = info.instance;
            auto result = init(info.window);
            if(result.has_value()){
                ErrorDataType res = {MOX_ERROR_TYPE_SUCCESS , "surface has been created"};
                engineLogger(&res);
            }else{
                auto error = result.error();
                engineLogger(&error);
                THROW_MESSAGE;
            }
        }

        Surface(Surface& input) = delete;
        Surface(Surface&& input){
            surface = std::exchange(input.surface , nullptr);
            instance = std::exchange(input.instance , nullptr);
        }

        Surface& operator=(Surface& input) = delete;
        Surface& operator=(Surface&& input){
            surface = std::exchange(input.surface , nullptr);
            instance = std::exchange(input.instance , nullptr);
            return *this;
        }

        VkSurfaceKHR surface = nullptr;

        ~Surface(){
            clean();
            ErrorDataType res{MOX_ERROR_TYPE_NONE , "destroying the surface"};
            engineLogger(&res);
        }
    };
}