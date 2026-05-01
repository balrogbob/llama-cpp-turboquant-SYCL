//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cstring>
#include "turbo_wht.hpp"

static constexpr float turbo_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static constexpr float turbo_wht_s2[128] = {
     1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
     1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
     1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
     1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

static inline void turbo_fwht(float * x, int group_size) {
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

static void turbo_wht_sycl(const float * src, float * dst, const float * scale_inv,
                           int64_t n_heads, int64_t head_dim, int group_size, int direction,
                           queue_ptr stream) {
    const int64_t groups_per_head = head_dim / group_size;
    const int64_t n_groups = n_heads * groups_per_head;
    const float inv_sqrt = 1.0f / sycl::sqrt((float) group_size);

    stream->parallel_for(sycl::range<1>((size_t) n_groups), [=](sycl::id<1> idx) {
        const int64_t g = (int64_t) idx[0];
        const int64_t head_idx = g / groups_per_head;
        const int64_t grp_in_head = g % groups_per_head;
        const int64_t base = head_idx * head_dim + grp_in_head * group_size;

        float x[128];
        const float * in = src + base;

        if (direction == 0 && scale_inv != nullptr) {
            for (int i = 0; i < group_size; ++i) {
                x[i] = in[i] * scale_inv[i % group_size];
            }
        } else {
            for (int i = 0; i < group_size; ++i) {
                x[i] = in[i];
            }
        }

        for (int i = 0; i < group_size; ++i) {
            x[i] *= (direction == 0 ? turbo_wht_s1[i] : turbo_wht_s2[i]);
        }

        turbo_fwht(x, group_size);

        float * out = dst + base;
        for (int i = 0; i < group_size; ++i) {
            float val = x[i] * inv_sqrt * (direction == 0 ? turbo_wht_s2[i] : turbo_wht_s1[i]);
            if (direction == 1 && scale_inv != nullptr) {
                val *= scale_inv[i % group_size];
            }
            out[i] = val;
        }
    });
}

void ggml_sycl_op_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * scale = dst->src[1];

    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction = 0;
    int group_size = 0;
    memcpy(&direction, dst->op_params + 0, sizeof(int));
    memcpy(&group_size, dst->op_params + sizeof(int), sizeof(int));

    GGML_ASSERT(direction == 0 || direction == 1);
    GGML_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
    GGML_ASSERT(src->ne[0] % group_size == 0);

    const int64_t n_heads = ggml_nelements(src) / src->ne[0];
    turbo_wht_sycl((const float *) src->data, (float *) dst->data,
                   scale ? (const float *) scale->data : nullptr,
                   n_heads, src->ne[0], group_size, direction, ctx.stream());
}
