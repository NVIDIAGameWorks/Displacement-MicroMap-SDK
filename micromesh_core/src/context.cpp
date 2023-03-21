//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#include <micromesh_internal/micromesh_context.h>

#include <cstdarg>
#include <cstdio>
#include <string>

namespace micromesh
{
static_assert(sizeof(bool) == 1, "bool must be 1 byte in size");

// nvprintf2-like function, but for micromesh logging.
void logPrintfSeverity(const MessageCallbackInfo* messageCallbackInfo,
                       uint32_t                   threadIndex,
                       const MessageSeverity      severity,
                       const char*                fmt,
                       std::va_list               vlist)
{
    if(!messageCallbackInfo || !messageCallbackInfo->pfnCallback)
        return;

    // Format the inputs into an std::string.
    // Copy vlist as it may be modified by vsnprintf.
    std::string  formattedStr;
    std::va_list vlistCopy;
    va_copy(vlistCopy, vlist);
    const int charactersNeeded = std::vsnprintf(nullptr, 0, fmt, vlistCopy);
    va_end(vlistCopy);
    if((charactersNeeded < 0) || (size_t(charactersNeeded) > formattedStr.max_size() - 1))
    {
        // Formatting error
        messageCallbackInfo->pfnCallback(MessageSeverity::eError, "Internal message formatting error.", threadIndex,
                                         messageCallbackInfo->userData);
        return;
    }

    // Resize the string; add 1, because vsnprintf doesn't count the
    // terminating null character. This can potentially throw an exception.
    try
    {
        formattedStr.resize(charactersNeeded + 1);
    }
    catch(...)
    {
        messageCallbackInfo->pfnCallback(MessageSeverity::eError, "Error resizing buffer to hold message.", threadIndex,
                                         messageCallbackInfo->userData);
        return;
    }

    // Format it and send it to the message callback!
    (void)std::vsnprintf(formattedStr.data(), formattedStr.size(), fmt, vlist);
    messageCallbackInfo->pfnCallback(severity, formattedStr.c_str(), threadIndex, messageCallbackInfo->userData);
}

// Logs a message through only the OpContext callback info.
void messageLog(const MessageCallbackInfo* messageCallbackInfo,
                uint32_t                   threadIndex,
                const MessageSeverity      severity,
#ifdef _MSC_VER
                _Printf_format_string_
#endif
                const char* fmt,
                ...)
{
    std::va_list vlist;
    va_start(vlist, fmt);
    logPrintfSeverity(messageCallbackInfo, threadIndex, severity, fmt, vlist);
    va_end(vlist);
}

// Logs a message through an OpContext. Checks that ctx != nullptr.
void contextLog(const OpContext       ctx,
                uint32_t              threadIndex,
                const MessageSeverity severity,
#ifdef _MSC_VER
                _Printf_format_string_
#endif
                const char* fmt,
                ...)
{
    if(!ctx)
        return;
    std::va_list vlist;
    va_start(vlist, fmt);
    logPrintfSeverity(&ctx->m_messageCallbackInfo, threadIndex, severity, fmt, vlist);
    va_end(vlist);
}

MICROMESH_API Result MICROMESH_CALL micromeshCreateOpContext(const OpConfig* config, OpContext* pContext, const MessageCallbackInfo* messageCallback)
{
    CHECK_NONNULLM(messageCallback, config);

    if(config->contextType != OpContextType::eImmediateAutomaticThreading)
    {
        MLOGE(messageCallback, "config->contextType must be eImmediateAutomaticThreading, but it was %u.",
              (uint32_t)config->contextType);
        return Result::eInvalidValue;
    }

    if(config->threadCount == 0)
    {
        MLOGE(messageCallback, "config->threadCount must be non-zero, but it was.");
        return Result::eInvalidValue;
    }

    *pContext = new OpContext_s(*config);

    // Set the default message callback by default if provided in case the
    // developer doesn't set it. It can be cleared by using
    // micromeshOpContextSetMessageCallback(context, {}).
    if(messageCallback)
        micromeshOpContextSetMessageCallback(*pContext, *messageCallback);

    return Result::eSuccess;
}

MICROMESH_API void MICROMESH_CALL micromeshDestroyOpContext(OpContext context)
{
    if(context)
    {
        delete context;
    }
}

MICROMESH_API void MICROMESH_CALL micromeshOpContextAbort(OpContext context)
{
    if(!context)
        return;

    context->resetSequence();
}

MICROMESH_API void MICROMESH_CALL micromeshOpContextSetMessageCallback(OpContext context, const MessageCallbackInfo info)
{
    if(!context)
        return;
    context->m_messageCallbackInfo = info;
}

MICROMESH_API MessageCallbackInfo MICROMESH_CALL micromeshOpContextGetMessageCallback(OpContext context)
{
    if(!context)
    {
        return MessageCallbackInfo();
    }
    return context->m_messageCallbackInfo;
}

MICROMESH_API OpConfig MICROMESH_CALL micromeshOpContextGetConfig(OpContext context)
{
    if(!context)
    {
        return OpConfig();
    }

    return context->m_config;
}

MICROMESH_API Result MICROMESH_CALL micromeshOpDistributeWork(OpContext ctx, const OpDistributeWork_input* input, uint64_t totalWorkload)
{
    CHECK_CTX_NONNULL(ctx);
    CHECK_NONNULL(ctx, input);
    CHECK_CTX_BEGIN(ctx);

    if(totalWorkload)
    {
        if(input->pfnGenericSingleWorkload)
        {
            ctx->parallel_items(totalWorkload, input->pfnGenericSingleWorkload, input->userData, input->batchSize);
        }
        else if(input->pfnGenericRangeWorkload)
        {
            ctx->parallel_item_ranges(totalWorkload, input->pfnGenericRangeWorkload, input->userData, input->batchSize);
        }
    }

    return Result::eSuccess;
}

MICROMESH_API OpConfig MICROMESH_CALL micromeshGetDefaultOpConfig()
{
    micromesh::OpConfig config;
    config.contextType = micromesh::OpContextType::eImmediateAutomaticThreading;
    config.threadCount = std::thread::hardware_concurrency();
    return config;
}

}  // namespace micromesh
