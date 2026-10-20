/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_METAL_OBJECTS_H
#define MTL_METAL_OBJECTS_H 1

#include "mtl_types.h"

#include <stdbool.h>
#include <stdint.h>

struct kk_image_layout;

struct mtl_texture_mismatch {
   bool device, type, format, extent, levels, samples, layers;
   uint32_t missing_usage;
};

void mtl_texture_check_layout(mtl_texture *texture, mtl_device *device,
                              const struct kk_image_layout *layout,
                              struct mtl_texture_mismatch *out);

mtl_texture *mtl_new_texture_with_iosurface(mtl_device *device,
                                            const struct kk_image_layout *layout,
                                            void *iosurface);

void *mtl_texture_get_iosurface(mtl_texture *texture);

uint64_t mtl_shared_event_device_registry_id(mtl_shared_event *event);

#endif /* MTL_METAL_OBJECTS_H */
