//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <cassert>
#include <algorithm>

#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_utils.h>

// nvpro_core-style logging macros. These include information for determining
// where the error originated without a breakpoint within the callback.
// Context-based variants, e.g. "log info"
#define LOGI(ctx, msg, ...)                                                                                            \
    {                                                                                                                  \
        contextLog((ctx), 0, MessageSeverity::eInfo, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__);          \
    }
#define LOGW(ctx, msg, ...)                                                                                            \
    {                                                                                                                  \
        contextLog((ctx), 0, MessageSeverity::eWarning, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__);       \
    }
#define LOGE(ctx, msg, ...)                                                                                            \
    {                                                                                                                  \
        contextLog((ctx), 0, MessageSeverity::eError, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__);         \
    }
// Message callback variants, e.g. "message log info"
#define MLOGI(callbacks, msg, ...)                                                                                     \
    {                                                                                                                  \
        messageLog((callbacks), 0, MessageSeverity::eInfo, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__);    \
    }
#define MLOGW(callbacks, msg, ...)                                                                                     \
    {                                                                                                                  \
        messageLog((callbacks), 0, MessageSeverity::eWarning, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
    }
#define MLOGE(callbacks, msg, ...)                                                                                     \
    {                                                                                                                  \
        messageLog((callbacks), 0, MessageSeverity::eError, "%s (%d): " msg, __FUNCTION__, __LINE__, ##__VA_ARGS__);   \
    }

// Functions for reducing lines of code when checking if arguments are null.
#define CHECK_NONNULL(ctx, argument)                                                                                   \
    {                                                                                                                  \
        if(!(argument))                                                                                                \
        {                                                                                                              \
            LOGE((ctx), "Non-optional argument `" #argument "` was null.");                                            \
            return Result::eInvalidValue;                                                                              \
        }                                                                                                              \
    }
#define CHECK_NONNULLM(callbacks, argument)                                                                            \
    {                                                                                                                  \
        if(!(argument))                                                                                                \
        {                                                                                                              \
            MLOGE((callbacks), "Non-optional argument `" #argument "` was null.");                                     \
            return Result::eInvalidValue;                                                                              \
        }                                                                                                              \
    }
#define CHECK_ARRAYVALID(ctx, argument)                                                                                \
    {                                                                                                                  \
        if(!arrayIsValid((const ArrayInfo&)(argument)))                                                                \
        {                                                                                                              \
            LOGE((ctx), "Non-optional array `" #argument "` was invalid.");                                            \
            return Result::eInvalidValue;                                                                              \
        }                                                                                                              \
    }
#define CHECK_ARRAYVALIDM(callbacks, argument)                                                                         \
    {                                                                                                                  \
        if(!arrayIsValid((const ArrayInfo&)(argument)))                                                                \
        {                                                                                                              \
            MLOGE((callbacks), "Non-optional array `" #argument "` was invalid.");                                     \
            return Result::eInvalidValue;                                                                              \
        }                                                                                                              \
    }
#define CHECK_ARRAYVALIDTYPED(ctx, argument)                                                                           \
    {                                                                                                                  \
        if(!arrayTypedIsValid((argument)))                                                                             \
        {                                                                                                              \
            LOGE((ctx), "Non-optional array `" #argument "` was invalid or had wrong format.");                        \
            return Result::eInvalidValue;                                                                              \
        }                                                                                                              \
    }
#define CHECK_ARRAYVALIDTYPEDM(callbacks, argument)                                                                    \
    {                                                                                                                  \
        if(!arrayTypedIsValid((argument)))                                                                             \
        {                                                                                                              \
            MLOGE((callbacks), "Non-optional array `" #argument "` was invalid or had wrong format.");                 \
            return Result::eInvalidValue;                                                                              \
        }                                                                                                              \
    }
#define CHECK_CTX_NONNULL(ctx)                                                                                         \
    if(!(ctx))                                                                                                         \
    {                                                                                                                  \
        return Result::eInvalidValue;                                                                                  \
    };
#define CHECK_CTX_BEGIN(ctx)                                                                                           \
    if((ctx)->m_opNextSequenceFn)                                                                                      \
    {                                                                                                                  \
        LOGE((ctx), "Attempted to call this function in the middle of a Begin/End sequence.");                         \
        return Result::eInvalidOperationOrder;                                                                         \
    };
#define CHECK_CTX_END(ctx, fn)                                                                                         \
    if((ctx)->m_opNextSequenceFn != fn)                                                                                \
    {                                                                                                                  \
        LOGE((ctx), "Attempted to call this function without matching Begin.");                                        \
        return Result::eInvalidOperationOrder;                                                                         \
    };

namespace micromesh
{
struct OpContext_s
{
    OpContext_s(const OpConfig& config)
    {
        m_config  = config;
        m_threads = new std::thread[config.threadCount];
    }

    ~OpContext_s()
    {
        resetSequence();
        delete[] m_threads;
    }

    OpConfig     m_config;
    std::thread* m_threads;

    void*                      m_opNextSequenceFn = nullptr;
    void*                      m_opPayload        = nullptr;
    std::function<void(void*)> m_opPayloadDeleter = nullptr;

    MessageCallbackInfo m_messageCallbackInfo = {};

    template <typename T>
    void setNextSequenceFn(T fptr)
    {
        m_opNextSequenceFn = reinterpret_cast<void*&>(fptr);
    }

    void setPayload(void* data, std::function<void(void*)> fnDeleter)
    {
        m_opPayload        = data;
        m_opPayloadDeleter = fnDeleter;
    }

    void resetSequence()
    {
        m_opNextSequenceFn = nullptr;
        if(m_opPayloadDeleter)
        {
            m_opPayloadDeleter(m_opPayload);
        }

        m_opPayload        = nullptr;
        m_opPayloadDeleter = nullptr;
    }


    // route through here in case we do want to selectively speed this up with
    // threading one day, and to support variable byte strides

    template <class T>
    void arrayCopy(ArrayInfo& dst, const ArrayInfo& src)
    {
        assert(dst.count == src.count);
        for(uint64_t i = 0; i < src.count; i++)
        {
            arraySetV(dst, i, arrayGetV<T>(src, i));
        }
    }

    uint32_t getThreadCount() const { return m_config.threadCount; }

    typedef std::function<Result(OpContext ctx, uint32_t threadIdx, void* userData)> FnSerial;

    typedef std::function<void(uint64_t itemIdx, uint32_t threadIdx, void* userData)> FnParallel;

    typedef std::function<void(uint64_t itemFirst, uint64_t itemLast, uint32_t threadIdx, void* userData)> FnParallelRanges;


    // TODO
    // note for deferred contexts we need a different system that isn't as immediate.
    // need a scheduler object of sorts in which parallel and serial jobs can be
    // enqueued. A serial task after a parallel section must be able to spawn new
    // parallel jobs.
    // The final serial job also needs to set the function's result code.
    //
    // Reason is that the developer will run "join" only once per thread, we need
    // distribute the jobs and handle the partial results etc. all to the very
    // end.
#if 0
    struct Scheduler
    {
        struct Task
        {
            uint64_t batchSize = 0;
            uint64_t itemCount = 0;

            void* userData = nullptr;

            FnSerial          serial          = nullptr;
            FnParallelSingles parallelSingles = nullptr;
            FnParallelRanges  paralellRanges  = nullptr;

            Task* nextTask = nullptr;
        };

        std::atomic_uint64_t batchCounter = 0;

        std::mutex taskMutex;
        Task*      taskHead;
        Task*      taskTail;

        void enqueue(Task task);

        // keeps popping tasks from list, distributes task
        // result code from last task (must be serial)
        Result join(uint32_t threadIndex);
    };

    Scheduler   scheduler;
#endif

    // distributes individual items using batches of `batchSize` items across
    // multiple threads. `itemCount` reflects the total number
    // of items to process.
    // returns number of threads used.
    uint32_t parallel_items(uint64_t itemCount, FnParallel fn, void* userData = nullptr, uint64_t batchSize = 1024)
    {
        uint32_t numThreads = m_config.threadCount;

        if(numThreads <= 1 || itemCount < numThreads || itemCount < batchSize)
        {
            for(uint64_t idx = 0; idx < itemCount; idx++)
            {
                fn(idx, 0, userData);
            }

            return 1;
        }
        else
        {
            std::atomic_uint64_t counter = 0;

            auto worker = [&](uint32_t threadIdx) {
                uint64_t first;
                while((first = counter.fetch_add(batchSize)) < itemCount)
                {
                    uint64_t last = std::min(itemCount, first + batchSize);
                    for(uint64_t i = first; i < last; i++)
                    {
                        fn(i, threadIdx, userData);
                    }
                }
            };

            for(uint32_t i = 1; i < numThreads; i++)
            {
                m_threads[i] = std::thread(worker, i);
            }

            worker(0);

            for(uint32_t i = 1; i < numThreads; i++)
            {
                m_threads[i].join();
                m_threads[i] = std::thread();
            }

            return numThreads;
        }
    }

    // distributes batches of loop ranges using `batchSize` items across
    // multiple threads. `itemCount` reflects the total number
    // of items to process.
    // returns number of threads used.
    uint32_t parallel_item_ranges(uint64_t itemCount, FnParallelRanges fn, void* userData = nullptr, uint64_t batchSize = 1024)
    {
        uint32_t numThreads = m_config.threadCount;

        if(numThreads <= 1 || itemCount < numThreads || itemCount < batchSize)
        {
            fn(0, itemCount, 0, userData);

            return 1;
        }
        else
        {
            std::atomic_uint64_t counter = 0;

            auto worker = [&](uint32_t threadIdx) {
                uint64_t first;
                while((first = counter.fetch_add(batchSize)) < itemCount)
                {
                    uint64_t last = std::min(itemCount, first + batchSize);
                    fn(first, last, threadIdx, userData);
                }
            };

            for(uint32_t i = 1; i < numThreads; i++)
            {
                m_threads[i] = std::thread(worker, i);
            }

            worker(0);

            for(uint32_t i = 1; i < numThreads; i++)
            {
                m_threads[i].join();
                m_threads[i] = std::thread();
            }

            return numThreads;
        }
    }
};

// Logs a message through only the OpContext callback info.
// messageCallbackInfo may be null, in which case no message is printed.
void messageLog(const MessageCallbackInfo* messageCallbackInfo,
                uint32_t                   threadIndex,
                const MessageSeverity      severity,
#ifdef _MSC_VER
                _Printf_format_string_
#endif
                const char* fmt,
                ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 4, 5)));
#endif
;

// Logs a message through an OpContext. Checks that ctx != nullptr.
void contextLog(const OpContext       ctx,
                uint32_t              threadIndex,
                const MessageSeverity severity,
#ifdef _MSC_VER
                _Printf_format_string_
#endif
                const char* fmt,
                ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 4, 5)));
#endif
;

}  // namespace micromesh
