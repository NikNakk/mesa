/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_IMAGE_H
#define KK_IMAGE_H 1

#include "kk_private.h"

struct kk_bo;

#include "kk_device_memory.h"
#include "kk_image_layout.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "vk_image.h"

struct kk_device_memory;
struct kk_physical_device;
struct kk_queue;

VkFormatFeatureFlags2
kk_get_image_format_features(struct kk_physical_device *pdevice,
                             VkFormat format, VkImageTiling tiling,
                             uint64_t drm_format_mod);

uint32_t kk_image_max_dimension(const struct kk_physical_device *pdev,
                                VkImageType image_type);

struct kk_image_plane {
   struct kk_image_layout layout;
   struct kk_device_memory *mem;
   uint64_t mem_offset_B;
   // TODO_KOSMICKRISP Only have one handle since we will only create 2D arrays
   // anyway
   /* Metal handle with original handle type */
   mtl_texture *mtl_handle;
   /* Metal handle with 2D array type for 3D images */
   mtl_texture *mtl_handle_array;
   uint64_t addr;
};

struct kk_fdm_params;

struct kk_image {
   struct vk_image vk;

   /** True if the planes are bound separately
    * * This is set based on VK_IMAGE_CREATE_DISJOINT_BIT
    */
   bool disjoint;

   /* Forces alpha value to be one when creating views. Needed for WSI */
   bool wsi_opaque_alpha;

   bool metal_imported;
   bool wsi_direct;
   void *wsi_drawable;

   struct {
      uint8_t *cpu;
      uint32_t width, height, layers;
      uint32_t version;
      void *map;
      uint32_t map_version, map_width, map_height, map_layers;
      int16_t map_shift[8][2];
      uint64_t serial;
      uint32_t params_bytes;
      struct kk_fdm_params *params;
   } fdm;

   /* Implicit foveation: a multisampled eye-sized attachment (kk_fdm.h). */
   bool implicit_candidate;
   bool implicit_fdm;
   /* Its hidden single-sample image in the same foveated layout: the pass resolves
    * into it and it is then blitted, through the map, into the application's
    * ordinary resolve target. */
   VkImage implicit_resolve_image;
   VkImageView implicit_resolve_view;
   VkDeviceMemory implicit_resolve_memory;
   /* Single-sample foveated images: an ordinary copy made before a raw copy reads it. */
   VkImage implicit_unwarp_image;
   VkDeviceMemory implicit_unwarp_memory;
   bool implicit_unwarp_logged;
   bool implicit_poison; /* shares a pass with an image that cannot be foveated */
   struct kk_bo *rate_params;
   /* The parameters (and map) the image's contents are laid out in, referenced. */
   struct kk_fdm_params *rate_ref;
   const void *rate_params_owner;
   uint64_t rate_params_serial;
   uint32_t rate_params_version;

   uint8_t plane_count;
   struct kk_image_plane planes[3];
};

#define kk_foreach_slice(ndx, image, subresource_member)                       \
   for (uint32_t ndx = region->subresource_member.baseArrayLayer;              \
        ndx < (region->subresource_member.baseArrayLayer +                     \
               vk_image_subresource_layer_count(&image->vk,                    \
                                                &region->subresource_member)); \
        ++ndx)

static inline mtl_resource *
kk_image_to_mtl_resource(const struct kk_image *image, int plane)
{
   if (image != NULL) {
      assert(plane < ARRAY_SIZE(image->planes));
      return (mtl_resource *)image->planes[plane].mtl_handle;
   }
   return NULL;
}

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)

static inline uint64_t
kk_image_plane_base_address(const struct kk_image_plane *plane)
{
   return plane->addr;
}

static inline uint64_t
kk_image_base_address(const struct kk_image *image, uint8_t plane)
{
   return kk_image_plane_base_address(&image->planes[plane]);
}

static inline uint8_t
kk_image_aspects_to_plane(ASSERTED const struct kk_image *image,
                          VkImageAspectFlags aspectMask)
{
   /* Memory planes are only allowed for memory operations */
   assert(!(aspectMask & (VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT |
                          VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT |
                          VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT |
                          VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT)));

   /* Verify that the aspects are actually in the image */
   assert(!(aspectMask & ~image->vk.aspects));

   /* Must only be one aspect unless it's depth/stencil */
   assert(aspectMask ==
             (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) ||
          util_bitcount(aspectMask) == 1);

   switch (aspectMask) {
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
      return 1;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
      return 2;
   default:
      return 0;
   }
}

static inline uint8_t
kk_image_memory_aspects_to_plane(ASSERTED const struct kk_image *image,
                                 VkImageAspectFlags aspectMask)
{
   if (aspectMask & (VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT |
                     VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT |
                     VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT |
                     VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT)) {
      /* We don't support DRM format modifiers on anything but single-plane
       * color at the moment.
       */
      assert(aspectMask == VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT);
      return 0;
   } else {
      return kk_image_aspects_to_plane(image, aspectMask);
   }
}

mtl_texture *kk_image_plane_create_texture(struct kk_image_plane *plane,
                                           struct kk_image_layout *layout,
                                           uint64_t offset_B);

void kk_image_set_label(struct kk_image *image, const char *label);

#define KK_RATE_PARAMS_HEADER_B 256
#define KK_RATE_PARAMS_SIZE_B   (64 * 1024)
#define KK_RATE_PARAMS_MAX_LAYERS 8

struct kk_rate_params_header {
   uint32_t valid;
   uint32_t layers;
   uint32_t screen[2];
   uint32_t physical[KK_RATE_PARAMS_MAX_LAYERS][2];
};

void kk_fdm_capture(struct kk_image *image, const uint8_t *src,
                    uint32_t row_length, uint32_t image_height,
                    const VkImageSubresourceLayers *subresource,
                    VkOffset3D offset, VkExtent3D extent);
VkResult kk_fdm_image_init(struct kk_device *dev, struct kk_image *image);
void kk_fdm_image_finish(struct kk_device *dev, struct kk_image *image);

#ifdef VK_USE_PLATFORM_METAL_EXT
VkResult kk_image_import_metal(struct kk_device *dev, struct kk_image *image,
                               const VkImageCreateInfo *info);
void kk_image_release_metal_import(struct kk_device *dev,
                                   struct kk_image *image);
#endif

#endif
