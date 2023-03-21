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
* Compact the index buffer after edge collapse
* 
*/


#undef MAIN_NAME
#define MAIN_NAME remeshCompactIndices

#define REMESHER_OVERRIDE_BLOCK_SIZE
#include "remesh_common.h"

// A large block size strongly benefits performance (tested on RTX3090Ti)
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

// The compaction uses an auxiliary buffer. Instead of allocating a buffer for that
// usage we reuse the hash map used for vertex/edge deduplication
#define compactionAuxData RM_DATA(scratchHashMap)

// Process 4 triangles per thread for performance
#define COMPACTION_ENTRIES_PER_THREAD 4

// Check whether the triangle at index is valid or degenerate
bool checkValue(uint index)
{
    // Check for duplicate indices
    uvec3 indices;

    for(uint i = 0; i < 3; i++)
    {
        indices[i] = RM_DATA(triangles)[3 * index + i];
    }
    if(!isValid(indices))
        return false;

    // Check for duplicate coordinates
    vec3 pos[3];
    for(uint i = 0; i < 3; i++)
    {
        pos[i] = getOutputVertex(indices[i]);
    }

    if(pos[0] == pos[1] || pos[0] == pos[2] || pos[1] == pos[2])
        return false;
    return true;
}

// Update the index values after compacting the vertex buffer
// For this we use the contents of the last vertex merging orders,
// where the index r of the vertex replacing the one at vertex i is
// r = vertexMerges[3 * i + 0]
void updateIndices(uint index)
{
    bool isInvalid = false;
    for(uint i = 0; i < 3; i++)
    {
        uint vertexIndex = RM_DATA(triangles)[3 * index + i];

        if(vertexIndex >= RM_DATA(currentState).mergeCount)
        {
            RM_DATA(currentState).errorState = eRemesherErrorDebug;
            return;
        }
        // Find the replacement of the current vertex
        uint replacement = vertexMerges[3 * vertexIndex + 0];

        if(vertexIndex >= RM_DATA(scratchMetadata).uncompactedVertexCount)
        {
            isInvalid = true;
        }

        // If the vertex has indeed been moved by the vertex buffer compaction,
        // update the index stored in the index buffer for that vertex
        if(replacement != ~0u)
        {
            if(isInvalid || replacement >= RM_DATA(currentState).vertexCount || vertexIndex <= replacement)
            {
                isInvalid                        = true;
                RM_DATA(currentState).errorState = eRemesherErrorDebug;
                return;
            }
            RM_DATA(triangles)[3 * index + i] = replacement;
        }
        else
        {
            if(vertexIndex >= RM_DATA(currentState).vertexCount)
            {
                RM_DATA(currentState).errorState = eRemesherErrorDebug;
            }
        }
    }
}
// The compaction auxiliary data stores a flag indicating whether the corresponding
// entry at index is valid or not. This gains performance by avoiding reading again
// the contents of the input buffer, while the aux data is read/written in any case
bool checkValueFlag(uint index)
{
    return (compactionAuxData[index] >> 31) == 1;
}

// Copy the triangle data from src to dst, including its per-triangle
// subdivision info and collapse counter
void copyData(uint dst, uint src, bool invalidate)
{
    for(uint i = 0; i < 3; i++)
    {
        RM_DATA(triangles)[dst * 3 + i] = RM_DATA(triangles)[src * 3 + i];
        if(invalidate)
            RM_DATA(triangles)[src * 3 + i] = 0;
    }
    rmmCopy(dst, src);
    rtSetAggregatedCounter(dst, rtGetAggregatedCounter(src));
}

MAIN
{
    if(RM_DATA(currentState).errorState != eRemesherErrorNone)
        return;

    // First pass, count the number of valid entries and store the validity flags
    if(RM_CONSTANTS.compactionPass == 0)
    {
        uint index        = uint(gl_GlobalInvocationID.x) * COMPACTION_ENTRIES_PER_THREAD;
        uint validEntries = 0;

        for(uint i = 0; i < COMPACTION_ENTRIES_PER_THREAD; i++)
        {
            if(index + i >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
                break;
            // We update the indices only when the vertex buffer has just been compacted,
            // at the end of the decimation process
            if(RM_CONSTANTS.isFinalCompaction == 1)
                updateIndices(index + i);

            bool isValid = checkValue(index + i);
            if(isValid)
            {
                validEntries++;
            }
            // Store the entry validity flag
            compactionAuxData[index + i] = isValid ? (1 << 31) : 0;
        }
        atomicAdd(RM_DATA(scratchMetadata).indexCompactionValidEntries, validEntries);
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
            if(index + i >= RM_DATA(scratchMetadata).indexCompactionValidEntries)
                break;

            bool isValid = checkValueFlag(index + i);
            if(!isValid)
            {
                uint slot            = invalidEntryCount++;
                invalidEntries[slot] = uint8_t(i);
            }
        }
        uint begin = atomicAdd(RM_DATA(scratchMetadata).indexCompactionCurrentInvalidEntry, invalidEntryCount);
        for(uint i = 0; i < invalidEntryCount; i++)
        {
            compactionAuxData[begin + i] |= index + invalidEntries[i];
        }
        return;
    }

    // Final pass, for each valid entry with an index beyond the final number of valid entries,
    // we reserve a free slot and copy the triangle data into it.
    if(RM_CONSTANTS.compactionPass == 2)
    {
        uint validEntries = RM_DATA(scratchMetadata).indexCompactionValidEntries;

        uint index = uint(gl_GlobalInvocationID.x) * COMPACTION_ENTRIES_PER_THREAD + validEntries;
        for(uint i = 0; i < COMPACTION_ENTRIES_PER_THREAD; i++)
        {
            if(index + i >= RM_DATA(scratchMetadata).uncompactedTriangleCount)
                break;

            bool isValid = checkValueFlag(index + i);

            if(isValid)
            {
                uint auxEntryId = atomicAdd(RM_DATA(scratchMetadata).indexCompactionCurrentValidEntry, 1);
                uint slot       = compactionAuxData[auxEntryId];
                slot            = slot & (~(1 << 31));
                copyData(slot, index + i, true);
            }
        }
    }
}
