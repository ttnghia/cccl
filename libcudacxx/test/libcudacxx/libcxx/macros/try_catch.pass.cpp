//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/exceptions.h>
#include <cuda/std/cassert>

#include <nv/target>

struct ExceptionBase
{
  int value;
};

struct Exception : ExceptionBase
{};

void test_host()
{
  _CCCL_TRY
  {
    throw Exception{};
    assert(false);
  }
  _CCCL_CATCH (const Exception & e1)
  {
    assert(e1.value == 0);
    assert(true);
  }
  _CCCL_CATCH (const ExceptionBase & e2)
  {
    assert(e2.value == 0);
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
}

__device__ void test_device()
{
  _CCCL_TRY
  {
    assert(true);
  }
  _CCCL_CATCH (const Exception & e1)
  {
    assert(e1.value == 0);
    assert(false);
  }
  _CCCL_CATCH (const ExceptionBase & e2)
  {
    assert(e2.value == 0);
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
}

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test_host();), (test_device();))
  return 0;
}
