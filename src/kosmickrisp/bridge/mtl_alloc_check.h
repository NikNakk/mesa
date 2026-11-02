/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_ALLOC_CHECK_H
#define MTL_ALLOC_CHECK_H 1

#include <stdbool.h>
#include <stdint.h>

bool mtl_alloc_check_enabled(void);
void mtl_alloc_check_begin(const void *command_buffer, const void *allocator);
void mtl_alloc_check_end(const void *command_buffer);
void mtl_alloc_check_reset(const void *allocator);
void mtl_alloc_check_commit(void *options, void *const *command_buffers,
                            uint32_t count);

#endif
