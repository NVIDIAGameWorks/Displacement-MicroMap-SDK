/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.

  Clear the hash map using 128-bit operations for efficiency

*/


#undef MAIN_NAME
#define MAIN_NAME remeshClearHashMap


#include "remesh_common.h"

#define BATCH_SIZE 1

MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index      = uint(gl_GlobalInvocationID.x);
    uint uintCount  = RM_CONSTANTS.hashMapSize * 2;
    uint uvec4Count = uintCount / 4;

    uint batchCount = uvec4Count / BATCH_SIZE;

    for(uint i = 0; i < BATCH_SIZE; i++)
    {
        if((index + i) >= uvec4Count)
            return;

        RM_DATA(scratchHashMap128)[index * BATCH_SIZE + i] = uvec4(0);
    }
}
