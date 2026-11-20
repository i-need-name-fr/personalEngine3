#pragma once

#include <vulkan/vulkan.h>
#include <source_location>
#include <filesystem>
#include <fstream>
#include <vector>
#include <optional>
#include <bit>
#include <string>

namespace mox
{
namespace utils
{
    class SystemCheck final
    {
    public:

        inline static void init(VkPhysicalDevice device) noexcept
        {
            if(!similar.has_value()) _implCheckSimilar(device);
        }

        inline static bool similarSystem() noexcept
        {
            return similar.has_value() ? similar.value() : false;
        }

    private:

        inline static const std::filesystem::path file{std::filesystem::path(std::source_location::current().file_name()).parent_path() / std::filesystem::path("Cache") / std::filesystem::path("System.txt")};
        inline static std::optional<bool> similar{};

        inline static const std::string fileNotFounded{"__file_not_founded__"};

        inline static void _implCheckSimilar(VkPhysicalDevice device) noexcept
        __attribute__((nonnull))
        {
            VkPhysicalDeviceDriverProperties driver{};
            driver.pNext = nullptr;
            driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;

            VkPhysicalDeviceProperties2 info{};
            info.pNext = &driver;
            info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

            vkGetPhysicalDeviceProperties2(device , &info);

            const std::string data = std::string(driver.driverName) + driver.driverInfo; // we dont care about how exactly we write
            const auto prev = readFile();
            if(prev == fileNotFounded) 
            {
                similar = false;
            }
            else
            {
                similar = data == prev;
            }

            if(similar.has_value() && similar.value() == false) fileSave(data);
        }

        inline static const std::string readFile() noexcept
        {
            if(!std::filesystem::exists(file)) return {fileNotFounded};
            std::ifstream check(file.string() , std::ios::binary | std::ios::ate);
            if(!check.is_open()) return {fileNotFounded};
            const std::size_t size = check.tellg();
            check.seekg(0);
            std::vector<std::uint8_t> buffer(size);
            check.read(std::bit_cast<char*>(buffer.data()) , size);
            check.close();
            return std::string(buffer.begin() , buffer.end());
        }

        inline static void fileSave(const std::string &data) noexcept
        {
            // the folder Cache is not in git ( it is empty and ignored ) , so after a clone it does not exist
            std::error_code error;
            std::filesystem::create_directories(file.parent_path() , error);

            std::ofstream check(file.string() , std::ios::binary);
            if(!check.is_open()) return;
            std::vector<std::uint8_t> buffer(data.begin() , data.end());
            check.write(std::bit_cast<char*>(buffer.data()) , buffer.size());
            check.close();
        }
    };  
}
}