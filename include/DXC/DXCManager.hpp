#pragma once

#include <iostream>
#include <fstream>
#include <optional>
#include <vector>
#include <string>
#include <format>
#include <bit>
#include <cstring>
#include <expected>
#include <windows.h>
#include <unknwn.h>
#include <libloaderapi.h>
#include <dxc/dxcapi.h>
#include <wrl/client.h>
#include <filesystem>
#include "../logger/logger.hpp"

// clang for MinGW ignores __declspec(uuid) of dxcapi.h , so __uuidof(IDxcX) (and IID_PPV_ARGS) has no GUID and the linker fails on __mingw_uuidof<IDxcX>
// the GUIDs are copied from dxcapi.h ( CROSS_PLATFORM_UUIDOF )
extern "C++"
{
    template<> inline const GUID& __mingw_uuidof<IDxcCompiler3>()
    {
        static constexpr GUID guid = {0x228B4687 , 0x5A6A , 0x4730 , {0x90 , 0x0C , 0x97 , 0x02 , 0xB2 , 0x20 , 0x3F , 0x54}};
        return guid;
    }

    template<> inline const GUID& __mingw_uuidof<IDxcUtils>()
    {
        static constexpr GUID guid = {0x4605C4CB , 0x2019 , 0x492A , {0xAD , 0xA4 , 0x65 , 0xF2 , 0x0B , 0xB7 , 0xD6 , 0x7F}};
        return guid;
    }

    template<> inline const GUID& __mingw_uuidof<IDxcResult>()
    {
        static constexpr GUID guid = {0x58346CDA , 0xDDE7 , 0x4497 , {0x94 , 0x61 , 0x6F , 0x87 , 0xAF , 0x5E , 0x06 , 0x59}};
        return guid;
    }

    template<> inline const GUID& __mingw_uuidof<IDxcBlob>()
    {
        static constexpr GUID guid = {0x8BA5FB08 , 0x5195 , 0x40E2 , {0xAC , 0x58 , 0x0D , 0x98 , 0x9C , 0x3A , 0x01 , 0x02}};
        return guid;
    }

    template<> inline const GUID& __mingw_uuidof<IDxcBlobUtf8>()
    {
        static constexpr GUID guid = {0x3DA636C9 , 0xBA71 , 0x4024 , {0xA3 , 0x01 , 0x30 , 0xCB , 0xF1 , 0x25 , 0x30 , 0x5B}};
        return guid;
    }
}

namespace mox
{

    class DXCManager final
    {   
    public:

        [[nodiscard]] inline static ErrorDataOutput<std::vector<uint8_t>> Compile(const std::filesystem::path &path) noexcept
        {
            return _implCompileShader(path);
        }

    private:

        inline static Microsoft::WRL::ComPtr<IDxcCompiler3> compiler{};
        inline static Microsoft::WRL::ComPtr<IDxcUtils> utils{};
        inline static Microsoft::WRL::ComPtr<IDxcIncludeHandler> handler{};

        inline static uint8_t initCompleted = false;
        
        [[nodiscard]] inline static ErrorDataOutput<void> _implInitCompiler() noexcept
        {

            if(!initCompleted)
            {
                {
                    HRESULT result = DxcCreateInstance(CLSID_DxcCompiler , IID_PPV_ARGS(compiler.GetAddressOf()));
                    if(FAILED(result))
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , "failed to create instance required for DXC real time compilation"});
                    }
                }

                {
                    HRESULT result = DxcCreateInstance(CLSID_DxcUtils , IID_PPV_ARGS(utils.GetAddressOf()));
                    if(FAILED(result))
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , "failed to create utils required for DXC real time compilation"});
                    }
                }

                {
                    HRESULT result = utils->CreateDefaultIncludeHandler(&handler);
                    if(FAILED(result))
                    {
                        return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , "failed to create include handler required for DXC real time compilation"});
                    }
                }
                initCompleted = true;
            }

            return {};
        }

        inline static std::vector<int8_t> readFile(const std::filesystem::path &path) noexcept
        {   
            
            if(!std::filesystem::exists(path)) return {};

            std::ifstream file(path.string() , std::ios::ate | std::ios::binary);
            if(!file.is_open())
            {
                mox::ErrorDataType const error{MOX_ERROR_TYPE_FAILED , std::format("failed to read file for compilation , name -> {}" , path.string())};
                return {};
            }
            const auto size = file.tellg();
            std::vector<int8_t> buffer(size);
            file.seekg(0);
            file.read(std::bit_cast<char*>(buffer.data()) , size);
            file.close();
            return buffer;
        }
        
        [[nodiscard]] inline static std::vector<std::wstring> _implGetFlagsForPath(const std::filesystem::path& file) noexcept 
        {
            return {
                file.wstring(),                                      
                L"-spirv",
                L"-T",  L"lib_6_7",
                L"-HV", L"202x",
                L"-fspv-target-env=vulkan1.3",
                L"-fspv-extension=SPV_EXT_descriptor_indexing",
                L"-fspv-extension=SPV_KHR_ray_tracing",
                L"-fspv-extension=SPV_KHR_ray_query",
                L"-fspv-extension=SPV_EXT_mesh_shader",
                L"-fspv-extension=SPV_KHR_physical_storage_buffer",
                L"-enable-16bit-types",
                L"-I",  L"../include/",
            };
        }


        [[nodiscard]] inline static ErrorDataOutput<std::vector<uint8_t>> _implCompileShader(const std::filesystem::path path) noexcept
        {
            if(!initCompleted)
            {
                const auto result = _implInitCompiler();
                if(!result.has_value()) return std::unexpected(result.error());
            }

            const auto folderName = path.parent_path();
            const auto fsfileName = path.stem();
            const auto extension = path.extension();
            if(extension.string() != ".hlsl") return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to compile shader , Name -> {} , since the format isnt .hlsl" , fsfileName.string())});
            const std::string fileName = fsfileName.string() + ".spv";
            const std::filesystem::path outputFilePath = folderName.string() + "/" + fileName; // should be smth like path..../fileName.spv

            // ================================ LEGACY ================================
            // this part is no more required , since the output created SPIRV code


            // if(std::filesystem::exists(outputFilePath))
            // {
            //     std::cout << std::format("deleting the compiled file -> {}" , fileName) << '\n';
            //     try{
            //         std::filesystem::remove(outputFilePath);
            //     }
            //     catch(std::filesystem::filesystem_error &e)
            //     {
            //         return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to delete file for some reason -> {}"  , e.what()) });
            //     }
            // }
            
            auto data = readFile(path);
            if(data.empty())
            {
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("for some reason file is empty , buffer for DxcBuffer is nullptr , file Name -> {}" , path.string())});
            }

            DxcBuffer buffer{};
            buffer.Encoding = DXC_CP_UTF8;
            buffer.Ptr = data.data();
            buffer.Size = static_cast<uint32_t>(data.size());

            // now we need to create all the flags for our compilation
            // ray query
            // ray tracing
            // coop vec
            // descriptor indexing
            // descriptor buffer indexing
            // physical storage buffer
            // mesh shading
            // and I ../include/

            const auto flags = _implGetFlagsForPath(path);
            IDxcResult *dxcResult = nullptr;
            std::vector<LPCWSTR> flagPtrs;
            flagPtrs.reserve(flags.size());
            for (const std::wstring& f : flags)
                flagPtrs.push_back(f.c_str());
            HRESULT hr = compiler->Compile(&buffer,
                            flagPtrs.data(),
                            static_cast<uint32_t>(flagPtrs.size()),
                            handler.Get(),
                            __uuidof(IDxcResult),
                            reinterpret_cast<void**>(&dxcResult));

            if (FAILED(hr) || !dxcResult)
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION, "DXC Compile call failed"});

            Microsoft::WRL::ComPtr<IDxcBlobUtf8> error;
            dxcResult->GetOutput(DXC_OUT_ERRORS,
                                __uuidof(IDxcBlobUtf8),
                                reinterpret_cast<void**>(error.GetAddressOf()),
                                nullptr);
            std::string errorText;
            if (error && error->GetStringLength() > 0)
                errorText = error->GetStringPointer();

            HRESULT status;
            dxcResult->GetStatus(&status);

            if (FAILED(status))
            {
                dxcResult->Release();      
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED_CREATION,
                    std::format("failed to compile shader {} : {}", fileName, errorText)});
            }
            Microsoft::WRL::ComPtr<IDxcBlob> object;


            dxcResult->GetOutput(DXC_OUT_OBJECT , __uuidof(IDxcBlob) , reinterpret_cast<void**>(object.GetAddressOf()) ,nullptr);
            if(!object)
            {   
                dxcResult->Release();
                return std::unexpected(ErrorDataType{MOX_ERROR_TYPE_FAILED , std::format("failed to create object for DXC compiling  file Name -> {}" , path.string())});       
            }
            std::vector<uint8_t> spirv(object->GetBufferSize());
            memcpy(spirv.data(), object->GetBufferPointer(), object->GetBufferSize());
            
            dxcResult->Release(); 
            return spirv;
        }
    };

}