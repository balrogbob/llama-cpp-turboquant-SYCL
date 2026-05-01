//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TURBO_WHT_HPP
#define GGML_SYCL_TURBO_WHT_HPP

#include "common.hpp"

void ggml_sycl_op_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_TURBO_WHT_HPP
