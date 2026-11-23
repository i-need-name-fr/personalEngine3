#define GLFW_INCLUDE_VULKAN
#define GLFW_EXPOSE_NATIVE_WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define VMA_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define STB_IMAGE_IMPLEMENTATION
#define SPIRV_COMPILATION_REQUIRED

#include <iostream>
#include <csignal>
#include <cstdlib>
#include "include/Core/Core.hpp"

// ---------------------------------------------------------------- the crash is written into the log : an access violation , an abort ( assert of VMA , std::terminate ) do not throw a C++ exception ,
// so the catch below sees nothing . the handler writes the code , the address and the stack with the NAME OF THE MODULE of every frame ( project.exe , nvoglv64.dll , sl.dlss_g.dll , vulkan-1.dll ... ) :
// it says whose code died . the log is written at once ( the same file as all the messages )
static void moxLogCrash(const std::string& title)
{
    std::string text = "[CRASH] " + title;
    void* frames[48]{};
    const USHORT count = RtlCaptureStackBackTrace(0 , 48 , frames , nullptr);
    for(USHORT i = 0 ; i < count ; i++)
    {
        HMODULE module = nullptr;
        char path[MAX_PATH]{};
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT , static_cast<LPCSTR>(frames[i]) , &module);
        if(module) GetModuleFileNameA(module , path , MAX_PATH);
        const std::string file = path[0] ? std::string(path) : std::string("?");
        const std::size_t slash = file.find_last_of("\\/");
        text += std::format("\n    #{} {}  +0x{:x}" , i , slash == std::string::npos ? file : file.substr(slash + 1) , static_cast<std::size_t>(reinterpret_cast<const char*>(frames[i]) - reinterpret_cast<const char*>(module)));
    }
    mox::ErrorDataType message{mox::MOX_ERROR_TYPE_CRITICAL , text};
    mox::engineLogger(&message);
    std::cerr << text << std::endl;
}

static LONG WINAPI moxExceptionFilter(EXCEPTION_POINTERS* info)
{
    moxLogCrash(std::format("unhandled exception 0x{:08x} at address {}" , static_cast<unsigned>(info->ExceptionRecord->ExceptionCode) , static_cast<const void*>(info->ExceptionRecord->ExceptionAddress)));
    return EXCEPTION_CONTINUE_SEARCH;
}

static void moxAbortHandler(int)
{
    moxLogCrash("abort ( an assert , std::terminate )");
    std::_Exit(3);
}

int main(){
    SetUnhandledExceptionFilter(moxExceptionFilter);
    std::signal(SIGABRT , moxAbortHandler);
    std::set_terminate([]{ moxLogCrash("std::terminate : an exception that nobody caught"); std::_Exit(3); });
    std::cout << "hello world\n";
    try
    {

        mox::mCore core{};
        core.mainLoop();

    }
    catch(std::exception &e){
        std::cerr << e.what() << '\n';
        return 1;
    }


    return 0;
}