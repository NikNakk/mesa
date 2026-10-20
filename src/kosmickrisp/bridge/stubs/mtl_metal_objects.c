/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "mtl_metal_objects.h"

#include <string.h>

void
mtl_texture_check_layout(mtl_texture *texture, mtl_device *device,
                         const struct kk_image_layout *layout,
                         struct mtl_texture_mismatch *out)
{
   memset(out, 0, sizeof(*out));
   out->device = true;
}

mtl_texture *
mtl_new_texture_with_iosurface(mtl_device *device,
                               const struct kk_image_layout *layout,
                               void *iosurface)
{
   return NULL;
}

void *
mtl_texture_get_iosurface(mtl_texture *texture)
{
   return NULL;
}

uint64_t
mtl_shared_event_device_registry_id(mtl_shared_event *event)
{
   return 0;
}
