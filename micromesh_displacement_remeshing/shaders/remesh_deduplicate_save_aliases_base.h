/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
* 
* 
* Keep track of the vertex merging history: when two vertices are merged the 
* discarded vertex keeps the index of the preserved vertex at its index in 
* scratchVertexAliases
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshDeduplicateSaveAliasesBase

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

    uint h = rvGetHashIndex(index);
    if(h == RM_H_NOT_FOUND)
    {
        RM_DATA(currentState).errorState = eRemesherErrorVertexHashNotFound;
        return;
    }
    uint dedupIndex = rhGetStoredIndex(h);
    if(dedupIndex == ~0u)
    {
        return;
    }
    // If the vertex has been discarded in the last merge, store the index of the
    // vertex surviving the merge
    if(dedupIndex != index)
    {
        if(RM_DATA(scratchVertexAliases)[index] == ~0u)
            RM_DATA(scratchVertexAliases)[index] = dedupIndex;
        else
        {
            // Do nothing, keep the existing alias for bookkeeping
        }
    }
    else
    {
        // Make sure the selected vertex is currently used in the mesh
        // FIXME: redundant test (see construction in deduplicate_base)
        if(RM_DATA(scratchVertexAliases)[index] != ~0u)
            RM_DATA(currentState).errorState = eRemesherErrorDebug;
    }
}
