#pragma once

#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>

// the common part of the passes of NKPD ( the inference of the diffuse and of the indirect light , the training )
namespace mox
{
namespace Passes
{
    // the passes of the net ( a-trous ) : the step of the neighbours of every dispatch . the first one is the temporal one , the others read the result of the previous one ( tmpDiffuseImage ) , the LAST one writes the final image
    inline static constexpr std::array<std::uint32_t , 2> NKPD_PASS_STEPS = {1u , 2u};

    struct NKPDPassPush
    {
        std::uint32_t stepSize;     // the step of the neighbours
        std::uint32_t last;         // 1 : the last pass , it writes the final image
    };

    // the dispatches of the passes depend on each other ( a pass reads the neighbours that the previous one has written ) : the barrier of the memory between them
    inline void nkpdPassBarrier(VkCommandBuffer cmd) noexcept
    {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd , &dependency);
    }
}
}
