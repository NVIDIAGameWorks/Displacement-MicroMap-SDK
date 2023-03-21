/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.

  Clear the micromesh-related displacement bounds per vertex and per triangle

*/


#undef MAIN_NAME
#define MAIN_NAME remeshClearMicromeshData

#include "remesh_common.h"


MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    uint index = uint(gl_GlobalInvocationID.x);

    if(index >= RM_CONSTANTS.indexCount / 3)
        return;

    rtResetDisplacementBounds(index);

    for(uint i = 0; i < 3; i++)
    {
        uint vIndex = RM_DATA(triangles)[3 * index + i];
        rvResetMinMaxDisplacement(vIndex);
    }
}
