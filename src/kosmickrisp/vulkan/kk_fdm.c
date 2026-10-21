/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "kk_fdm.h"

#include "kk_bo.h"
#include "kk_debug.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_image.h"
#include "kk_image_view.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_rate_map.h"

#include "util/log.h"
#include "util/u_debug.h"
#include "vk_util.h"

#include <stdlib.h>
#include <string.h>

static bool
kk_fdm_debug(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("KK_FDM_DEBUG", false);
   return enabled;
}

void
kk_fdm_capture(struct kk_image *image, const uint8_t *src, uint32_t row_length,
               uint32_t image_height, const VkImageSubresourceLayers *subresource,
               VkOffset3D offset, VkExtent3D extent)
{
   if (!KK_EXPERIMENTAL(FDM_RATE) ||
       !(image->vk.usage & VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT) ||
       image->vk.format != VK_FORMAT_R8G8_UNORM || src == NULL)
      return;

   const uint32_t w = image->vk.extent.width;
   const uint32_t h = image->vk.extent.height;
   const uint32_t layers = image->vk.array_layers;
   if (image->fdm.cpu == NULL) {
      image->fdm.cpu = calloc((size_t)w * h * layers, 2);
      if (image->fdm.cpu == NULL)
         return;
      image->fdm.width = w;
      image->fdm.height = h;
      image->fdm.layers = layers;
   }

   if (row_length == 0)
      row_length = extent.width;
   if (image_height == 0)
      image_height = extent.height;

   const uint32_t layer_count = vk_image_subresource_layer_count(&image->vk, subresource);
   for (uint32_t l = 0; l < layer_count; l++) {
      const uint32_t layer = subresource->baseArrayLayer + l;
      if (layer >= layers)
         break;
      for (uint32_t y = 0; y < extent.height && offset.y + y < h; y++) {
         uint32_t copy_w = MIN2(extent.width, w - MIN2((uint32_t)offset.x, w));
         memcpy(image->fdm.cpu +
                   (((size_t)layer * h + offset.y + y) * w + offset.x) * 2,
                src + (((size_t)l * image_height + y) * row_length) * 2,
                (size_t)copy_w * 2);
      }
   }
   image->fdm.version++;
}

VkResult
kk_fdm_image_init(struct kk_device *dev, struct kk_image *image)
{
   if (!KK_EXPERIMENTAL(FDM_RATE) ||
       !(image->vk.create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT))
      return VK_SUCCESS;

   VkResult result = kk_alloc_bo(dev, &image->vk.base, KK_RATE_PARAMS_SIZE_B, 0,
                                 &image->rate_params);
   if (result != VK_SUCCESS)
      return result;
   memset(image->rate_params->cpu, 0, KK_RATE_PARAMS_HEADER_B);
   return VK_SUCCESS;
}

void
kk_fdm_image_finish(struct kk_device *dev, struct kk_image *image)
{
   if (image->rate_params)
      kk_destroy_bo(dev, image->rate_params);
   if (image->fdm.params)
      kk_destroy_bo(dev, image->fdm.params);
   if (image->fdm.map)
      mtl_release(image->fdm.map);
   free(image->fdm.cpu);
}

static float
kk_fdm_density(const struct kk_image *density, uint32_t layer, uint32_t x,
               uint32_t y, uint32_t channel)
{
   const uint8_t *texel =
      density->fdm.cpu +
      (((size_t)layer * density->fdm.height + y) * density->fdm.width + x) * 2;
   return MAX2(texel[channel] / 255.0f, 1.0f / 16.0f);
}

static void *
kk_fdm_get_map(struct kk_device *dev, struct kk_image *density,
               VkExtent2D screen)
{
   if (density->fdm.cpu == NULL)
      return NULL;

   if (density->fdm.map && density->fdm.map_version == density->fdm.version &&
       density->fdm.map_width == screen.width &&
       density->fdm.map_height == screen.height)
      return density->fdm.map;

   const uint32_t mw = density->fdm.width, mh = density->fdm.height;
   const uint32_t layers = MIN2(density->fdm.layers, KK_RATE_PARAMS_MAX_LAYERS);
   const uint32_t nh = 2 * mw + 1, nv = 2 * mh + 1;
   float *horizontal = calloc((size_t)nh * layers, sizeof(float));
   float *vertical = calloc((size_t)nv * layers, sizeof(float));
   if (horizontal == NULL || vertical == NULL) {
      free(horizontal);
      free(vertical);
      return NULL;
   }

   for (uint32_t l = 0; l < layers; l++) {
      for (uint32_t i = 0; i < nh; i++) {
         uint32_t a = i / 2, b = i / 2;
         if (!(i & 1)) {
            a = i ? i / 2 - 1 : 0;
            b = MIN2(i / 2, mw - 1);
         }
         float q = 0;
         for (uint32_t y = 0; y < mh; y++) {
            q = MAX2(q, kk_fdm_density(density, l, a, y, 0));
            q = MAX2(q, kk_fdm_density(density, l, b, y, 0));
         }
         horizontal[l * nh + i] = q;
      }
      for (uint32_t i = 0; i < nv; i++) {
         uint32_t a = i / 2, b = i / 2;
         if (!(i & 1)) {
            a = i ? i / 2 - 1 : 0;
            b = MIN2(i / 2, mh - 1);
         }
         float q = 0;
         for (uint32_t x = 0; x < mw; x++) {
            q = MAX2(q, kk_fdm_density(density, l, x, a, 1));
            q = MAX2(q, kk_fdm_density(density, l, x, b, 1));
         }
         vertical[l * nv + i] = q;
      }
   }

   void *map = mtl_new_rate_map(dev->mtl_handle, screen.width, screen.height,
                                layers, nh, nv, horizontal, vertical);
   free(horizontal);
   free(vertical);
   if (map == NULL)
      return NULL;

   if (mtl_rate_map_parameter_size(map) >
       KK_RATE_PARAMS_SIZE_B - KK_RATE_PARAMS_HEADER_B) {
      mtl_release(map);
      return NULL;
   }
   if (density->fdm.params == NULL &&
       kk_alloc_bo(dev, &density->vk.base, KK_RATE_PARAMS_SIZE_B, 0,
                   &density->fdm.params) != VK_SUCCESS) {
      mtl_release(map);
      return NULL;
   }

   struct kk_rate_params_header *header = density->fdm.params->cpu;
   memset(header, 0, KK_RATE_PARAMS_HEADER_B);
   mtl_rate_map_copy_parameters(map, density->fdm.params->map,
                                KK_RATE_PARAMS_HEADER_B);
   header->layers = layers;
   header->screen[0] = screen.width;
   header->screen[1] = screen.height;
   for (uint32_t l = 0; l < layers; l++)
      mtl_rate_map_physical_size(map, l, &header->physical[l][0],
                                 &header->physical[l][1]);
   header->valid = 1;

   if (density->fdm.map)
      mtl_release(density->fdm.map);
   density->fdm.map = map;
   density->fdm.map_version = density->fdm.version;
   density->fdm.map_width = screen.width;
   density->fdm.map_height = screen.height;
   density->fdm.map_layers = layers;

   if (kk_fdm_debug())
      mesa_logi("kk-fdm: rate map %ux%u from %ux%ux%u density map, layer 0 "
                "physical %ux%u (%.1f%%)",
                screen.width, screen.height, mw, mh, layers,
                header->physical[0][0], header->physical[0][1],
                100.0 * header->physical[0][0] * header->physical[0][1] /
                   ((double)screen.width * screen.height));
   return map;
}

static bool
kk_fdm_subsampled(const struct kk_image_view *iview)
{
   return iview == NULL ||
          (iview->vk.image->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT);
}

static bool
kk_fdm_all_subsampled(const struct kk_rendering_state *render)
{
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (!kk_fdm_subsampled(render->color_att[i].iview) ||
          (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE &&
           !kk_fdm_subsampled(render->color_att[i].resolve_iview)))
         return false;
   }
   return kk_fdm_subsampled(render->depth_att.iview) &&
          kk_fdm_subsampled(render->stencil_att.iview) &&
          (render->depth_att.resolve_mode == VK_RESOLVE_MODE_NONE ||
           kk_fdm_subsampled(render->depth_att.resolve_iview)) &&
          (render->stencil_att.resolve_mode == VK_RESOLVE_MODE_NONE ||
           kk_fdm_subsampled(render->stencil_att.resolve_iview));
}

static void
kk_fdm_publish(struct kk_image *density, struct kk_image_view *iview)
{
   if (iview == NULL)
      return;
   struct kk_image *image = container_of(iview->vk.image, struct kk_image, vk);
   if (image->rate_params == NULL)
      return;
   if (image->rate_params_owner == density->fdm.map &&
       image->rate_params_version == density->fdm.map_version)
      return;
   memcpy(image->rate_params->cpu, density->fdm.params->cpu,
          KK_RATE_PARAMS_SIZE_B);
   image->rate_params_owner = density->fdm.map;
   image->rate_params_version = density->fdm.map_version;
}

void
kk_fdm_setup_rendering(struct kk_cmd_buffer *cmd,
                       const VkRenderingInfo *pRenderingInfo,
                       mtl_render_pass_descriptor *descriptor,
                       VkExtent2D screen)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_rendering_state *render = &gfx->render;
   struct kk_image *density = NULL;

   if (KK_EXPERIMENTAL(FDM_RATE)) {
      const VkRenderingFragmentDensityMapAttachmentInfoEXT *fdm_info =
         vk_find_struct_const(pRenderingInfo->pNext,
                              RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO_EXT);
      if (fdm_info && fdm_info->imageView != VK_NULL_HANDLE) {
         VK_FROM_HANDLE(kk_image_view, view, fdm_info->imageView);
         density = container_of(view->vk.image, struct kk_image, vk);
      } else if (gfx->fdm.carry) {
         density = gfx->fdm.density;
      }
   }
   gfx->fdm.carry = false;

   void *map = density ? kk_fdm_get_map(dev, density, screen) : NULL;
   if (map && !kk_fdm_all_subsampled(render)) {
      static uint32_t logged;
      if (kk_fdm_debug() && logged++ < 8)
         mesa_logi("kk-fdm: full rate, pass has a non-subsampled attachment");
      map = NULL;
   }

   gfx->fdm.density = map ? density : NULL;
   uint64_t params = 0;
   if (map) {
      mtl_render_pass_descriptor_set_rate_map(descriptor, map);
      params = density->fdm.params->gpu;
      for (uint32_t i = 0; i < render->color_att_count; i++) {
         kk_fdm_publish(density, render->color_att[i].iview);
         kk_fdm_publish(density, render->color_att[i].resolve_iview);
      }
      kk_fdm_publish(density, render->depth_att.iview);
      kk_fdm_publish(density, render->depth_att.resolve_iview);
      kk_fdm_publish(density, render->stencil_att.iview);
      kk_fdm_publish(density, render->stencil_att.resolve_iview);
   }

   struct kk_descriptor_state *desc = &gfx->descriptors;
   if (desc->root.fdm_rate_params != params) {
      desc->root.fdm_rate_params = params;
      desc->root_dirty = true;
   }
}

void
kk_fdm_carry_to_next_rendering(struct kk_cmd_buffer *cmd)
{
   cmd->state.gfx.fdm.carry = cmd->state.gfx.fdm.density != NULL;
}
