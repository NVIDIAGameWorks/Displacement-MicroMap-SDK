/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// This file includes useful functions for debugging and printing information
// about internal displacement encoder structures. It's mainly used for the
// tests, but it can also be included in the library to debug issues (and then
// should be removed once debugging is complete).

#pragma once


#include <cassert>
#include <micromesh_internal/micromesh_containers.h>
#include <micromesh/micromesh_utils.h>

template <class T>
void printBitsIfVerbose(const micromesh::container::vector<T> data)
{
#ifdef DIAG_VERBOSE
    printf("Values (most significant byte and bit first):\n");
    const int64_t tBits = int64_t(8 * sizeof(T));
    for(int64_t i = int64_t(data.size()) - 1; i >= 0; i--)
    {
        const T& v = data[i];
        for(int64_t bit = tBits - 1; bit >= 0; bit--)
        {
            printf("%c", (size_t(v) & (1ULL << bit)) ? '1' : '0');
        }
        // Print index and line break every 4 Ts
        if(i % 4 == 0)
        {
            printf(" (%zu)\n", i * tBits);
        }
        else
        {
            printf(" ");
        }
    }
#endif
}

void printBlockArrayIfVerbose(const uint16_t* data, const uint32_t subdivLevel, micromesh::PFN_getMicroVertexIndex getMicroVertexIndex = nullptr)
{
#ifdef DIAG_VERBOSE
    const uint32_t numSegments = 1u << subdivLevel;
    printf("Block values (u down, v to the right):\n");
    for(uint32_t u = 0; u < numSegments + 1; u++)
    {
        for(uint32_t v = 0; v < numSegments + 1; v++)
        {
            if(u + v < numSegments + 1)
            {
                uint32_t index = 0;
                if(getMicroVertexIndex)
                {
                    index = getMicroVertexIndex(u, v, subdivLevel, nullptr);
                }
                else
                {
                    index = micromesh::umajorUVtoLinear(u, v, subdivLevel);
                }
                printf("%-4u ", data[index]);
            }
        }
        printf("\n");
    }
#endif
}

void printBlockArrayIfVerbose(const micromesh::container::vector<uint16_t> data, const uint32_t subdivLevel)
{
    assert(data.size() == micromesh::subdivLevelGetVertexCount(subdivLevel));
    printBlockArrayIfVerbose(data.data(), subdivLevel);
}