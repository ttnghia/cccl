//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_STANDARD_LAYOUT_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_STANDARD_LAYOUT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/remove_all_extents.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_STANDARD_LAYOUT) && !defined(_LIBCUDACXX_USE_IS_STANDARD_LAYOUT_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_standard_layout : public integral_constant<bool, _CCCL_BUILTIN_IS_STANDARD_LAYOUT(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_standard_layout_v = _CCCL_BUILTIN_IS_STANDARD_LAYOUT(_Tp);

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_standard_layout : integral_constant<bool, is_scalar<remove_all_extents_t<_Tp>>::value>
{};

template <class _Tp>
inline constexpr bool is_standard_layout_v = is_standard_layout<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_STANDARD_LAYOUT) && !defined(_LIBCUDACXX_USE_IS_STANDARD_LAYOUT_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_STANDARD_LAYOUT_H
