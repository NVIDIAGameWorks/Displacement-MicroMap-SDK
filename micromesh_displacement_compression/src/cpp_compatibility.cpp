/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "cpp_compatibility.h"

#ifndef NVDISP_COMPAT_USE_CPP20

#ifdef _MSC_VER  // MSVC
#include <intrin.h>
#endif

namespace micromesh::compat
{
int countl_zero(unsigned int x) noexcept
{
#ifdef _MSC_VER  // MSVC
    unsigned long result = 0;
    if(_BitScanReverse(&result, x) == 0)
    {
        // No set bits were found
        return 32;
    }
    return static_cast<int>(31 - result);
#else  // GCC, clang
    if(x != 0)
    {
        return __builtin_clz(x);
    }
    return 32;
#endif
}

int popcount(unsigned int x) noexcept
{
#ifdef _MSC_VER  // MSVC
    return static_cast<int>(__popcnt(x));
#else  // GCC, clang
    return __builtin_popcount(x);
#endif
}

}  // namespace micromesh::compat

#endif
