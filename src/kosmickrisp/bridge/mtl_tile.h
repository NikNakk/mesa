/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_TILE_H
#define MTL_TILE_H 1

#include "mtl_format.h"
#include "mtl_types.h"

#include <stdbool.h>
#include <stdint.h>

mtl_render_pipeline_state *
mtl_new_tile_render_pipeline(mtl_compiler *compiler,
                             mtl_function_descriptor *function,
                             uint32_t sample_count,
                             const enum mtl_pixel_format *formats,
                             uint32_t format_count);

void mtl_render_pass_descriptor_set_tile_size(
   mtl_render_pass_descriptor *descriptor, uint32_t width, uint32_t height);

void mtl_dispatch_threads_per_tile(mtl_render_encoder *encoder, uint32_t width,
                                   uint32_t height);

enum mtl_render_encoder_options {
   MTL_RENDER_ENCODER_SUSPENDING = 1u << 0,
   MTL_RENDER_ENCODER_RESUMING = 1u << 1,
};

void mtl_render_pass_descriptor_set_visibility_accumulate(
   mtl_render_pass_descriptor *descriptor, bool accumulate);

mtl_render_encoder *mtl_new_render_command_encoder_with_options(
   mtl_command_buffer *command_buffer, mtl_render_pass_descriptor *descriptor,
   uint32_t options);

#endif
