/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#undef MAIN_NAME
#define MAIN_NAME remeshCompactVertices

#define REMESHER_OVERRIDE_BLOCK_SIZE
#include "remesh_common.h"

// A large block size strongly benefits performance (tested on RTX3090Ti)
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

// The compaction uses an auxiliary buffer. Instead of allocating a buffer for that
// usage we reuse the hash map used for vertex/edge deduplication
#define compactionAuxData RM_DATA(scratchHashMap)

// Process 4 triangles per thread for performance
#define COMPACTION_ENTRIES_PER_THREAD 4

// Check whether the vertex at index is an orphan. In that case
// it is not needed by the final decimated mesh, and can be
// safely overwritten
bool checkValue(uint index)
{
    return !(hasFlag(rvGetFlags(index), RM_V_ORPHAN));
}

// The compaction auxiliary data stores a flag indicating whether the corresponding
// entry at index is valid or not. This gains performance by avoiding reading again
// the contents of the input buffer, while the aux data is read/written in any case
bool checkValueFlag(uint index)
{
    return (compactionAuxData[index] >> 31) == 1;
}

// Issue a copy request to the external application, using the
// special weight of 1 to force a copy instead of a merge
void copyData(uint dst, uint src, bool invalidate)
{
    RM_DATA(vertexMerges)[3 * src + 0] = dst;
    RM_DATA(vertexMerges)[3 * src + 1] = src;
    RM_DATA(vertexMerges)[3 * src + 2] = floatBitsToUint(1.f);

    RM_DATA(vertexDebug)[dst] = RM_DATA(vertexDebug)[src];

    // Sanity check FIXME to remove
    if(dst != ~0u && src < dst)
    {
        RM_DATA(currentState).errorState = eRemesherErrorDebug;
        RM_DATA(currentState).debug.x    = 2;
        return;
    }
}


uint vertexCount()
{
    return RM_DATA(scratchMetadata).uncompactedVertexCount;
}

MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    // First pass, count the number of valid entries and store the validity flags
    if(RM_CONSTANTS.compactionPass == 0)
    {
        uint index = uint(gl_GlobalInvocationID.x) * COMPACTION_ENTRIES_PER_THREAD;

        uint validEntries = 0;

        for(uint i = 0; i < COMPACTION_ENTRIES_PER_THREAD; i++)
        {
            if(index + i >= vertexCount())
                break;
            bool isValid = checkValue(index + i);
            if(isValid)
            {
                validEntries++;
            }
            compactionAuxData[index + i] = (isValid ? (1 << 31) : 0);
        }
        atomicAdd(RM_DATA(scratchMetadata).vertexCompactionValidEntries, validEntries);
        return;
    }

    // Second pass, enqueue the required number of free slots (i.e. corresponding to invalid
    // entries) by adding their indices in the auxiliary buffer
    if(RM_CONSTANTS.compactionPass == 1)
    {
        uint index = uint(gl_GlobalInvocationID.x) * COMPACTION_ENTRIES_PER_THREAD;

        uint8_t invalidEntries[COMPACTION_ENTRIES_PER_THREAD];
        uint    invalidEntryCount = 0;

        for(uint i = 0; i < COMPACTION_ENTRIES_PER_THREAD; i++)
        {
            if(index + i >= RM_DATA(scratchMetadata).vertexCompactionValidEntries)
                break;

            bool isValid = checkValueFlag(index + i);
            if(!isValid)
            {
                uint slot            = invalidEntryCount++;
                invalidEntries[slot] = uint8_t(i);
            }
            // Make invalid copy request for the vertices before the cut line
            copyData(~0u, index + i, false);
        }
        uint begin = atomicAdd(RM_DATA(scratchMetadata).vertexCompactionCurrentInvalidEntry, invalidEntryCount);
        for(uint i = 0; i < invalidEntryCount; i++)
        {
            compactionAuxData[begin + i] |= index + uint(invalidEntries[i]);
        }
        return;
    }

    // Final pass, for each valid entry with an index beyond the final number of valid entries,
    // we reserve a free slot and copy the vertex data into it.
    if(RM_CONSTANTS.compactionPass == 2)
    {
        uint validEntries = RM_DATA(scratchMetadata).vertexCompactionValidEntries;

        uint index = uint(gl_GlobalInvocationID.x) * COMPACTION_ENTRIES_PER_THREAD + validEntries;

        // We set the merge count to the total vertex count, so the merging kernel will consider
        // all vertices. The ones with an invalid merge index ~0u will be ignored, and the others
        // will trigger the necessary copies
        if(gl_GlobalInvocationID.x == 0)
        {
            RM_DATA(currentState).mergeCount = vertexCount();
        }
        for(uint i = 0; i < COMPACTION_ENTRIES_PER_THREAD; i++)
        {
            if(index + i >= vertexCount())
                break;

            bool isValid = checkValueFlag(index + i);

            if(isValid && (index + i < vertexCount()))
            {
                uint auxEntryId = atomicAdd(RM_DATA(scratchMetadata).vertexCompactionCurrentValidEntry, 1);
                uint slot       = compactionAuxData[auxEntryId];
                slot            = slot & (~(1 << 31));
                copyData(slot, index + i, true);
            }
            else
            {
                copyData(~0u, index + i, true);
            }
        }
    }
}
