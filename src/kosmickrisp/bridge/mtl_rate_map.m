/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "mtl_rate_map.h"

#include <Metal/MTL4RenderPass.h>
#include <Metal/MTLDevice.h>
#include <Metal/MTLTexture.h>
#include <objc/runtime.h>
#include <Metal/MTLRasterizationRate.h>

mtl_rate_map *
mtl_new_rate_map(mtl_device *device, uint32_t screen_width,
                 uint32_t screen_height, uint32_t layer_count,
                 uint32_t horizontal_count, uint32_t vertical_count,
                 const float *horizontal, const float *vertical)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      if (![dev supportsRasterizationRateMapWithLayerCount:layer_count])
         return NULL;
      MTLRasterizationRateMapDescriptor *desc = [MTLRasterizationRateMapDescriptor
         rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(screen_width,
                                                                  screen_height, 0)];
      for (uint32_t l = 0; l < layer_count; l++) {
         MTLRasterizationRateLayerDescriptor *layer =
            [[[MTLRasterizationRateLayerDescriptor alloc]
               initWithSampleCount:MTLSizeMake(horizontal_count, vertical_count, 0)
                        horizontal:horizontal + l * horizontal_count
                          vertical:vertical + l * vertical_count] autorelease];
         [desc setLayer:layer atIndex:l];
      }
      return [dev newRasterizationRateMapWithDescriptor:desc];
   }
}

void
mtl_rate_map_physical_size(mtl_rate_map *map, uint32_t layer, uint32_t *width,
                           uint32_t *height)
{
   @autoreleasepool {
      MTLSize size = [(id<MTLRasterizationRateMap>)map physicalSizeForLayer:layer];
      *width = size.width;
      *height = size.height;
   }
}

uint64_t
mtl_rate_map_parameter_size(mtl_rate_map *map)
{
   @autoreleasepool {
      return ((id<MTLRasterizationRateMap>)map).parameterBufferSizeAndAlign.size;
   }
}

void
mtl_rate_map_copy_parameters(mtl_rate_map *map, mtl_buffer *buffer,
                             uint64_t offset)
{
   @autoreleasepool {
      [(id<MTLRasterizationRateMap>)map copyParameterDataToBuffer:(id<MTLBuffer>)buffer
                                                           offset:offset];
   }
}

void
mtl_render_pass_descriptor_set_rate_map(mtl_render_pass_descriptor *descriptor,
                                        mtl_rate_map *map)
{
   @autoreleasepool {
      ((MTL4RenderPassDescriptor *)descriptor).rasterizationRateMap =
         (id<MTLRasterizationRateMap>)map;
   }
}

#ifdef THRESHOLD_SHARED_FOVEATION
#include <threshold_foveation.h>

__attribute__((visibility("default"))) int
kk_metal_foveation_set_texture(void *texture, void *map)
{
   return tf_metal_set_texture_map(texture, map);
}

mtl_rate_map *
mtl_texture_copy_rate_map(mtl_texture *texture)
{
   return tf_metal_copy_texture_map(texture);
}
#else
static char rate_map_key;

static id<MTLTexture>
rate_map_root(id<MTLTexture> texture)
{
   while (texture.parentTexture)
      texture = texture.parentTexture;
   return texture;
}

__attribute__((visibility("default"))) int
kk_metal_foveation_set_texture(void *texture, void *map)
{
   if (!texture)
      return -1;
   id<MTLTexture> root = rate_map_root((id<MTLTexture>)texture);
   id<MTLRasterizationRateMap> rate = (id<MTLRasterizationRateMap>)map;
   if (rate && (rate.device != root.device || rate.layerCount != 1 ||
                rate.screenSize.width != root.width || rate.screenSize.height != root.height))
      return -1;
   objc_setAssociatedObject(root, &rate_map_key, rate, OBJC_ASSOCIATION_RETAIN);
   return 0;
}

mtl_rate_map *
mtl_texture_copy_rate_map(mtl_texture *texture)
{
   return [objc_getAssociatedObject(rate_map_root((id<MTLTexture>)texture), &rate_map_key) retain];
}

#endif
