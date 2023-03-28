/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* Once vertices have been marked for merging, issue merge orders for the 
* external merge kernel
* 
*/

#undef MAIN_NAME
#define MAIN_NAME remeshCollapseResolve

#include "remesh_common.h"


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;
    uint index = getActiveVertex(gl_GlobalInvocationID.x);
    if(index == ~0u)
        return;


    if(hasFlag(rvGetFlags(index), RM_V_ORPHAN))
        return;

    RM_DATA(vertexDebug)[index] = rvGetFlags(index);

    // Return if that vertex is not involved in a merging operation
    if(rvGetMergingWith(index) == ~0u)
        return;

    RM_DATA(vertexDebug)[rvGetMergingWith(index)] = rvGetFlags(index);

    if(RM_CONSTANTS.iterationIndex > 1)
    {
        // Block volume preservation at seams to avoid watertightness issues
        mergeOutputVertices(index, rvGetMergingWith(index),
                            hasFlag(rvGetFlags(index), RM_V_MARKED) && !hasFlag(rvGetFlags(index), RM_V_EDGE));
    }
}
