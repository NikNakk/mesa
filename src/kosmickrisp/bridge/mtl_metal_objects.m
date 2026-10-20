/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "mtl_metal_objects.h"

#include "kk_image_layout.h"

#include <IOSurface/IOSurfaceRef.h>
#include <Metal/MTLDevice.h>
#include <Metal/MTLEvent.h>
#include <Metal/MTLTexture.h>

void
mtl_texture_check_layout(mtl_texture *texture, mtl_device *device,
                         const struct kk_image_layout *layout,
                         struct mtl_texture_mismatch *out)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      id<MTLDevice> dev = (id<MTLDevice>)device;
      *out = (struct mtl_texture_mismatch){0};
      out->device = tex.device.registryID != dev.registryID;
      out->type = tex.textureType != (MTLTextureType)layout->type;
      out->format = tex.pixelFormat != (MTLPixelFormat)layout->format.mtl;
      out->extent = tex.width != layout->width_px ||
                    tex.height != layout->height_px ||
                    tex.depth != layout->depth_px;
      out->levels = tex.mipmapLevelCount != layout->levels;
      out->samples = tex.sampleCount != layout->sample_count_sa;
      out->layers = tex.arrayLength != layout->layers;
      out->missing_usage = (uint32_t)(layout->usage & ~tex.usage);
   }
}

mtl_texture *
mtl_new_texture_with_iosurface(mtl_device *device,
                               const struct kk_image_layout *layout,
                               void *iosurface)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor new] autorelease];
      descriptor.textureType = (MTLTextureType)layout->type;
      descriptor.pixelFormat = layout->format.mtl;
      descriptor.width = layout->width_px;
      descriptor.height = layout->height_px;
      descriptor.depth = 1;
      descriptor.mipmapLevelCount = 1;
      descriptor.sampleCount = 1;
      descriptor.arrayLength = 1;
      descriptor.usage = (MTLTextureUsage)layout->usage;
      descriptor.storageMode = MTLStorageModeShared;
      return [dev newTextureWithDescriptor:descriptor
                                 iosurface:(IOSurfaceRef)iosurface
                                     plane:0];
   }
}

void *
mtl_texture_get_iosurface(mtl_texture *texture)
{
   @autoreleasepool {
      return (void *)((id<MTLTexture>)texture).iosurface;
   }
}

uint64_t
mtl_shared_event_device_registry_id(mtl_shared_event *event)
{
   @autoreleasepool {
      return ((id<MTLSharedEvent>)event).device.registryID;
   }
}
