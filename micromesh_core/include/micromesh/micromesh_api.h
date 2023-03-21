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

#if defined(_MSC_VER)
#define MICROMESH_CALL __fastcall
#elif !defined(__aarch64__) && !defined(__x86_64) && (defined(__GNUC__) || defined(__clang__))
#define MICROMESH_CALL __attribute__((fastcall))
#else
#define MICROMESH_CALL
#endif

// anticipate dll etc.

#ifndef MICROMESH_API
#define MICROMESH_API extern "C"
#endif
