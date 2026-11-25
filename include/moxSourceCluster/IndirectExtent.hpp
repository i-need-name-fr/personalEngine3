#pragma once

#include <algorithm>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "../shaderStructures/shaderStructure.h"

namespace mox
{
namespace Source
{
    inline VkExtent2D indirectExtent(const VkExtent2D screen) noexcept
    {
        const std::uint32_t divisor = std::max<std::uint32_t>(1u , static_cast<std::uint32_t>(MOX_INDIRECT_DIVISOR_RESOLUTION));
        return {std::max(1u , screen.width / divisor) , std::max(1u , screen.height / divisor)};
    }
}
}
