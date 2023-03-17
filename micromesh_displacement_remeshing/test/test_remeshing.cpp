//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <array>
#include <cstdlib>
#include <random>
#include <vector>

#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_utils.h>
#include <micromesh/micromesh_format_types.h>
#include <micromesh/micromesh_displacement_remeshing.h>


// Ensure assertions are always checked even in release mode
#undef NDEBUG

using namespace micromesh;
std::default_random_engine gen(1);  // Ensure the RNG is seeded so tests are reproducible

#define PRINT_AND_ASSERT_FALSE(...)                                                                                    \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
        assert(false);                                                                                                 \
    }

#define TEST_TRUE(a)                                                                                                   \
    if(!(a))                                                                                                           \
    {                                                                                                                  \
        PRINT_AND_ASSERT_FALSE("Test failed: " #a "\n");                                                               \
        return false;                                                                                                  \
    }

#define TEST_SUCCESS(a)                                                                                                \
    if((a) != Result::eSuccess)                                                                                        \
    {                                                                                                                  \
        PRINT_AND_ASSERT_FALSE("Test did not return Result::eSuccess: " #a " \n");                                     \
        return false;                                                                                                  \
    }

void basicMessageCallback(MessageSeverity severity, const char* message, uint32_t threadIndex, const void* userData)
{
    if(severity == MessageSeverity::eInfo)
    {
        printf("INFO: %s\n", message);
    }
    else if(severity == MessageSeverity::eWarning)
    {
        printf("WARNING: %s\n", message);
    }
    else if(severity == MessageSeverity::eError)
    {
        PRINT_AND_ASSERT_FALSE("ERROR: %s\n", message);
    }
}

const MessageCallbackInfo messenger{basicMessageCallback, nullptr};

typedef void* PseudoGpuHandle;

PseudoGpuHandle pseudoCreateResource(const gpu::ResourceAllocInfo& info)
{
    return nullptr;
}

PseudoGpuHandle pseudoCreatePipelineLayout(const gpu::PipelineLayoutInfo& info)
{
    return nullptr;
}

PseudoGpuHandle pseudoCreatePipeline(const gpu::PipelineInfo& info, PseudoGpuHandle pipelineLayout)
{
    return nullptr;
}

void* pseudoReadData(PseudoGpuHandle resource, uint64_t& size)
{
    return nullptr;
}

bool pseudoGpuExample()
{
    Result                                               result;
    typedef gpu::ResourceInfo<gpu::GpuRemeshingResource> GpuRemeshingResourceInfo;

    gpu::GpuRemeshing system;
    gpu::SetupInfo    setup;
    struct SystemData
    {
        std::vector<PseudoGpuHandle>          scratchPersistentResources;
        std::vector<PseudoGpuHandle>          pipelineLayouts;
        std::vector<PseudoGpuHandle>          pipelines;
        std::vector<PseudoGpuHandle>          userPipelines;
        std::vector<GpuRemeshingResourceInfo> readResourceInfos;
        std::vector<void*>                    readResourceDatas;
        std::vector<uint64_t>                 readResourceSizes;
        PseudoGpuHandle                       globalConstantBuffer = nullptr;
    } sysData;

    {
        // setup system

        gpu::AvailableShaderCodeTypes availableTypes;
        TEST_SUCCESS(micromeshGpuRemeshingGetAvailableShaderCodeTypes(&availableTypes));
        assert(availableTypes.isAvailable[gpu::eShaderCodeSPIRV]);

        gpu::GpuRemeshing_config config;
        RemeshingMode            mode = RemeshingMode::eRelax;
        config.codeType               = gpu::eShaderCodeSPIRV;
        config.supportedModeCount     = 1;
        config.supportedModes         = &mode;
        TEST_SUCCESS(gpu::micromeshGpuRemeshingCreate(&config, &system, &messenger));

        // main info
        TEST_SUCCESS(gpu::micromeshGpuRemeshingGetSetup(system, &setup));

        // constant buffer
        if(setup.globalConstantBuffer.buffer.size)
        {
            sysData.globalConstantBuffer = pseudoCreateResource(setup.globalConstantBuffer);
        }

        // read resources
        sysData.readResourceDatas.resize(setup.readResourcesMaxCount);
        sysData.readResourceSizes.resize(setup.readResourcesMaxCount);
        sysData.readResourceInfos.resize(setup.readResourcesMaxCount);

        std::vector<gpu::ResourceAllocInfo> scratchPersistentAllocs(setup.scratchPersistentCount);
        gpu::PersistentResourceInfo         persistent;
        persistent.scratchPersistentCount  = setup.scratchPersistentCount;
        persistent.scratchPersistentAllocs = scratchPersistentAllocs.data();
        TEST_SUCCESS(gpu::micromeshGpuRemeshingGetPersistent(system, &persistent));

        sysData.scratchPersistentResources.resize(setup.scratchPersistentCount);
        for(uint32_t i = 0; i < setup.scratchPersistentCount; i++)
        {
            sysData.scratchPersistentResources[i] = pseudoCreateResource(scratchPersistentAllocs[i]);
        }

        sysData.pipelineLayouts.resize(setup.pipelineLayoutCount);
        for(uint32_t i = 0; i < setup.pipelineLayoutCount; i++)
        {
            gpu::PipelineLayoutInfo pipeLayoutInfo;
            TEST_SUCCESS(gpu::micromeshGpuRemeshingGetPipelineLayout(system, i, &pipeLayoutInfo));
            sysData.pipelineLayouts[i] = pseudoCreatePipelineLayout(pipeLayoutInfo);
        }

        sysData.pipelines.resize(setup.pipelineCount);
        for(uint32_t i = 0; i < setup.pipelineLayoutCount; i++)
        {
            gpu::PipelineInfo pipeInfo;
            TEST_SUCCESS(gpu::micromeshGpuRemeshingGetPipeline(system, i, &pipeInfo));
            sysData.pipelines[i] = pseudoCreatePipeline(pipeInfo, sysData.pipelineLayouts[pipeInfo.pipelineLayoutIndex]);
        }

        sysData.userPipelines.resize(setup.userPipelineCount);
        // user would have to setup their own pipelines
        sysData.userPipelines[gpu::GpuRemeshingUserPipeline::eGpuRemeshingUserMergeVertices] = nullptr;
    }

    {
        struct TaskData
        {
            std::vector<PseudoGpuHandle>                     scratchTaskResources;
            std::vector<PseudoGpuHandle>                     allResourceHandles;
            SystemData*                                      sysData;
            gpu::ReadResourceData<gpu::GpuRemeshingResource> readData;
            bool                                             hadRead = false;
        } taskData;

        taskData.readData.resources         = sysData.readResourceInfos.data();
        taskData.readData.resourceDataSizes = sysData.readResourceSizes.data();
        taskData.readData.resourceDatas     = sysData.readResourceDatas.data();

        taskData.allResourceHandles.resize(gpu::eGpuRemeshingScratchStart + setup.scratchPersistentCount + setup.scratchTaskCount);
        for(uint32_t i = 0; i < setup.scratchPersistentCount; i++)
        {
            taskData.allResourceHandles[i + gpu::eGpuRemeshingScratchStart] = sysData.scratchPersistentResources[i];
        }

        // user would fill in various custom handles
        taskData.allResourceHandles[gpu::eGpuRemeshingMeshVertexPositionsBuffer] = nullptr;


        OpRemeshing_settings     settings;
        gpu::GpuRemeshing_input  input;
        gpu::GpuRemeshing_output output;
        gpu::GpuRemeshingTask    task;

        std::vector<gpu::ResourceAllocInfo> scratchTaskResources(setup.scratchTaskCount);
        output.scratchTaskCount  = setup.scratchTaskCount;
        output.scratchTaskAllocs = scratchTaskResources.data();

        TEST_SUCCESS(gpu::micromeshGpuRemeshingBeginTask(system, &settings, &input, &output, &task));

        // prepare task specific scratch resources
        taskData.scratchTaskResources.resize(setup.scratchPersistentCount);
        for(uint32_t i = 0; i < setup.scratchTaskCount; i++)
        {
            // allocate
            taskData.scratchTaskResources[i] = pseudoCreateResource(scratchTaskResources[i]);
            // update task table for easier resolving
            taskData.allResourceHandles[i + gpu::eGpuRemeshingScratchStart + setup.scratchPersistentCount] =
                taskData.scratchTaskResources[i];
        }

        gpu::CommandSequenceInfo<gpu::GpuRemeshingResource> seq;
        seq.previousReadData = nullptr;
        seq.userData         = &taskData;

        auto fnCommandGenerator = [](gpu::CommandType cmdType, const void* cmdData, void* userData) {
            TaskData* data = reinterpret_cast<TaskData*>(userData);

            switch(cmdType)
            {
            case gpu::CommandType::eBindPipeline: {
                const auto*     bindPipeline = reinterpret_cast<const gpu::CmdBindPipeline*>(cmdData);
                PseudoGpuHandle pipeline     = data->sysData->pipelines[bindPipeline->pipelineIndex];
            }
            break;
            case gpu::CommandType::eBindUserPipeline: {
                const auto* bindUserPipeline =
                    reinterpret_cast<const gpu::CmdBindUserPipeline<gpu::GpuRemeshingUserPipeline>*>(cmdData);
                PseudoGpuHandle pipeline = data->sysData->userPipelines[bindUserPipeline->userPipelineIndex];
            }
            break;
            case gpu::CommandType::eBindResources: {
                const auto* bindResources = reinterpret_cast<const gpu::CmdBindResources<gpu::GpuRemeshingResource>*>(cmdData);
                // prepare descriptor set and bind it
                // there will be maximum of setup.descriptorSetAllocationInfo.setMaxCount
                // many eBindResources per sequence.
            }
            break;
            case gpu::CommandType::eClearResources:
                break;
            case gpu::CommandType::eReadResources: {
                const auto* readResources = reinterpret_cast<const gpu::CmdReadResources<gpu::GpuRemeshingResource>*>(cmdData);

                data->hadRead                = true;
                data->readData.resourceCount = readResources->resourceCount;
                memcpy(data->readData.resources, readResources->resources,
                       sizeof(GpuRemeshingResourceInfo) * readResources->resourceCount);
            }

            break;
            case gpu::CommandType::eGlobalConstants:
                break;
            case gpu::CommandType::eLocalConstants:
                break;
            case gpu::CommandType::eBarrier:
                break;
            case gpu::CommandType::eDispatch:
                break;
            case gpu::CommandType::eDispatchIndirect:
                break;
            case gpu::CommandType::eBeginLabel:
                break;
            case gpu::CommandType::eEndLabel:
                break;
            }
        };
        seq.pfnGenerateGpuCommand = fnCommandGenerator;

        // setup & execute task
        do
        {
            taskData.hadRead = false;

            result = gpu::micromeshGpuRemeshingContinueTask(system, task, &seq);
            if(result == Result::eSuccess || result == Result::eContinue)
            {
                // submit generated command buffer

                if(taskData.hadRead)
                {
                    for(uint32_t i = 0; i < taskData.readData.resourceCount; i++)
                    {
                        taskData.readData.resourceDatas[i] =
                            pseudoReadData(taskData.allResourceHandles[taskData.readData.resources[i].resourceIndex],
                                           taskData.readData.resourceDataSizes[i]);
                    }

                    // readback and setup for next
                    seq.previousReadData = &taskData.readData;
                }
                else
                {
                    seq.previousReadData = nullptr;
                }
            }
        } while(result == Result::eContinue);

        TEST_SUCCESS(gpu::micromeshGpuRemeshingEndTask(system, task, &output));
    }

    return true;
}

int main(int argc, const char** argv)
{
    OpContext context;
    {
        OpConfig config;
        config.threadCount = 2;
        TEST_SUCCESS(micromeshCreateOpContext(&config, &context, &messenger));
    }

    micromeshDestroyOpContext(context);

    printf("All tests passed.\n");
    return 0;
}