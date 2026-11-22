/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "mtl_rate_map.h"

mtl_rate_map *
mtl_new_rate_map(mtl_device *device, uint32_t screen_width,
                 uint32_t screen_height, uint32_t layer_count,
                 uint32_t horizontal_count, uint32_t vertical_count,
                 const float *horizontal, const float *vertical)
{
   return NULL;
}

void
mtl_rate_map_physical_size(mtl_rate_map *map, uint32_t layer, uint32_t *width,
                           uint32_t *height)
{
   *width = 0;
   *height = 0;
}

uint64_t
mtl_rate_map_parameter_size(mtl_rate_map *map)
{
   return 0;
}

void
mtl_rate_map_copy_parameters(mtl_rate_map *map, mtl_buffer *buffer,
                             uint64_t offset)
{
}

void
mtl_render_pass_descriptor_set_rate_map(mtl_render_pass_descriptor *descriptor,
                                        mtl_rate_map *map)
{
}

mtl_rate_map *mtl_texture_copy_rate_map(mtl_texture *texture) { return NULL; }
