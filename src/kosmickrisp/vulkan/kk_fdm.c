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
#include "kosmickrisp/bridge/mtl_encoder.h"
#include "kosmickrisp/bridge/mtl_rate_map.h"

#include "util/log.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vk_util.h"
#include "vk_cmd_queue.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void
kk_fdm_execute_commands(struct kk_cmd_buffer *cmd, struct vk_cmd_queue *queue)
{
   if (kk_cmd_buffer_device(cmd)->vk.enabled_extensions.EXT_fragment_density_map_offset &&
       debug_get_bool_option("KK_FDM_OFFSET_DEFER", true)) {
      cmd->fdm_scan_head = &queue->cmds;
      cmd->fdm_scan_next = queue->cmds.next;
   }
   vk_cmd_queue_execute(queue, kk_cmd_buffer_to_handle(cmd),
                        &kk_cmd_buffer_device(cmd)->vk.dispatch_table);
   cmd->fdm_scan_head = NULL;
   cmd->fdm_scan_next = NULL;
}

static void
kk_fdm_scan_offsets(struct kk_cmd_buffer *cmd)
{
   memset(cmd->fdm_offsets, 0, sizeof(cmd->fdm_offsets));
   if (!cmd->fdm_scan_next)
      return;
   const struct list_head *p = cmd->fdm_scan_next;
   while (p != cmd->fdm_scan_head) {
      const struct vk_cmd_queue_entry *entry =
         list_entry(p, const struct vk_cmd_queue_entry, cmd_link);
      p = p->next;
      if (entry->type == VK_CMD_END_RENDERING2_KHR) {
         const VkRenderingEndInfoKHR *end = entry->u.end_rendering2_khr.rendering_end_info;
         const VkRenderPassFragmentDensityMapOffsetEndInfoEXT *offset =
            vk_find_struct_const(end->pNext, RENDER_PASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_EXT);
         if (offset) {
            uint32_t count = MIN2(offset->fragmentDensityOffsetCount, ARRAY_SIZE(cmd->fdm_offsets));
            memcpy(cmd->fdm_offsets, offset->pFragmentDensityOffsets,
                   count * sizeof(VkOffset2D));
         }
         break;
      }
      if (entry->type == VK_CMD_END_RENDERING)
         break;
   }
   cmd->fdm_scan_next = p;
}

static simple_mtx_t kk_fdm_lock = SIMPLE_MTX_INITIALIZER;

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
   simple_mtx_lock(&kk_fdm_lock);
   if (image->fdm.cpu == NULL) {
      image->fdm.cpu = calloc((size_t)w * h * layers, 2);
      if (image->fdm.cpu == NULL) {
         simple_mtx_unlock(&kk_fdm_lock);
         return;
      }
      image->fdm.width = w;
      image->fdm.height = h;
      image->fdm.layers = layers;
   }

   if (row_length == 0)
      row_length = extent.width;
   if (image_height == 0)
      image_height = extent.height;

   bool changed = image->fdm.version == 0;
   const uint32_t layer_count = vk_image_subresource_layer_count(&image->vk, subresource);
   for (uint32_t l = 0; l < layer_count; l++) {
      const uint32_t layer = subresource->baseArrayLayer + l;
      if (layer >= layers)
         break;
      for (uint32_t y = 0; y < extent.height && offset.y + y < h; y++) {
         uint32_t copy_w = MIN2(extent.width, w - MIN2((uint32_t)offset.x, w));
         uint8_t *dst = image->fdm.cpu +
                        (((size_t)layer * h + offset.y + y) * w + offset.x) * 2;
         const uint8_t *row = src + (((size_t)l * image_height + y) * row_length) * 2;
         if (memcmp(dst, row, (size_t)copy_w * 2)) {
            memcpy(dst, row, (size_t)copy_w * 2);
            changed = true;
         }
      }
   }
   if (changed)
      image->fdm.version++;
   if (kk_fdm_debug()) {
      bool same = layers < 2 || !memcmp(image->fdm.cpu, image->fdm.cpu + (size_t)w * h * 2,
                                        (size_t)w * h * 2);
      uint32_t full[2] = {0, 0};
      for (uint32_t l = 0; l < MIN2(layers, 2); l++)
         for (uint32_t i = 0; i < w * h; i++)
            full[l] += image->fdm.cpu[((size_t)l * w * h + i) * 2] == 255;
      mesa_logi("kk-fdm: density upload %ux%ux%u %s, layers %s, full-rate texels %u/%u",
                extent.width, extent.height, layer_count,
                changed ? "changed" : "unchanged", same ? "identical" : "differ",
                full[0], full[1]);
   }
   simple_mtx_unlock(&kk_fdm_lock);
}

void
kk_fdm_note_use(const struct vk_image *image, const char *what)
{
   if (image == NULL ||
       !(image->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT) ||
       !KK_EXPERIMENTAL(FDM_RATE) || !kk_fdm_debug())
      return;
   static struct {
      const struct vk_image *image;
      const char *what;
   } seen[256];
   static uint32_t seen_count;
   simple_mtx_lock(&kk_fdm_lock);
   for (uint32_t i = 0; i < seen_count; i++) {
      if (seen[i].image == image && seen[i].what == what) {
         simple_mtx_unlock(&kk_fdm_lock);
         return;
      }
   }
   if (seen_count < ARRAY_SIZE(seen)) {
      seen[seen_count].image = image;
      seen[seen_count].what = what;
      seen_count++;
      mesa_logi("kk-fdm: use %s: image %p format %d samples %u %ux%ux%u usage 0x%x",
                what, (const void *)image, image->format, image->samples,
                image->extent.width, image->extent.height, image->array_layers,
                (unsigned)image->usage);
   }
   simple_mtx_unlock(&kk_fdm_lock);
}

void
kk_fdm_note_view(const struct vk_image_view *view, const char *what)
{
   if (view == NULL ||
       !(view->image->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT) ||
       !KK_EXPERIMENTAL(FDM_RATE) || !kk_fdm_debug())
      return;
   static struct {
      const struct vk_image_view *view;
      const char *what;
   } seen[256];
   static uint32_t seen_count;
   simple_mtx_lock(&kk_fdm_lock);
   for (uint32_t i = 0; i < seen_count; i++) {
      if (seen[i].view == view && seen[i].what == what) {
         simple_mtx_unlock(&kk_fdm_lock);
         return;
      }
   }
   if (seen_count < ARRAY_SIZE(seen)) {
      seen[seen_count].view = view;
      seen[seen_count].what = what;
      seen_count++;
      mesa_logi("kk-fdm: view %s: view type %d layers %u+%u image %ux%ux%u format %d",
                what, view->view_type, view->base_array_layer, view->layer_count,
                view->image->extent.width, view->image->extent.height,
                view->image->array_layers, view->format);
   }
   simple_mtx_unlock(&kk_fdm_lock);
}

VkResult
kk_fdm_image_init(struct kk_device *dev, struct kk_image *image)
{
   bool discard_depth = debug_get_bool_option("KK_FDM_DISCARD_DEPTH", false) &&
                        image->vk.samples == VK_SAMPLE_COUNT_1_BIT &&
                        (image->vk.aspects & VK_IMAGE_ASPECT_DEPTH_BIT);
   if (!KK_EXPERIMENTAL(FDM_RATE) ||
       (!(image->vk.create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT) &&
        !discard_depth))
      return VK_SUCCESS;

   VkResult result = kk_alloc_bo(dev, &image->vk.base, KK_RATE_PARAMS_SIZE_B, 0,
                                 &image->rate_params);
   if (result != VK_SUCCESS)
      return result;
   memset(image->rate_params->cpu, 0, KK_RATE_PARAMS_HEADER_B);
   return VK_SUCCESS;
}

static void
kk_fdm_params_unref(struct kk_fdm_params *params)
{
   assert(params->refs > 0);
   if (--params->refs)
      return;
   kk_destroy_bo(params->dev, params->bo);
   free(params);
}

void
kk_fdm_cmd_release(struct kk_cmd_buffer *cmd)
{
   if (!util_dynarray_num_elements(&cmd->fdm_params, struct kk_fdm_params *))
      return;
   simple_mtx_lock(&kk_fdm_lock);
   util_dynarray_foreach(&cmd->fdm_params, struct kk_fdm_params *, params)
      kk_fdm_params_unref(*params);
   simple_mtx_unlock(&kk_fdm_lock);
   util_dynarray_clear(&cmd->fdm_params);
}

static void
kk_fdm_cmd_ref(struct kk_cmd_buffer *cmd, struct kk_fdm_params *params)
{
   util_dynarray_foreach(&cmd->fdm_params, struct kk_fdm_params *, held) {
      if (*held == params)
         return;
   }
   struct kk_fdm_params **slot =
      util_dynarray_grow(&cmd->fdm_params, struct kk_fdm_params *, 1);
   if (slot == NULL) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   *slot = params;
   params->refs++;
}

void
kk_fdm_image_finish(struct kk_device *dev, struct kk_image *image)
{
   if (image->rate_params)
      kk_destroy_bo(dev, image->rate_params);
   if (image->fdm.params) {
      simple_mtx_lock(&kk_fdm_lock);
      kk_fdm_params_unref(image->fdm.params);
      simple_mtx_unlock(&kk_fdm_lock);
      image->fdm.params = NULL;
   }
   if (image->fdm.map)
      mtl_release(image->fdm.map);
   free(image->fdm.cpu);
}

#define KK_FDM_TEXEL_SIZE 32

static struct {
   bool valid;
   float x, y;
} kk_fdm_gaze[KK_RATE_PARAMS_MAX_LAYERS], kk_fdm_latched[KK_RATE_PARAMS_MAX_LAYERS];
static uint64_t kk_fdm_serial;

PUBLIC void
kk_fdm_set_gaze(uint32_t view, float x, float y, bool valid)
{
   if (view >= KK_RATE_PARAMS_MAX_LAYERS)
      return;
   simple_mtx_lock(&kk_fdm_lock);
   kk_fdm_gaze[view].valid = valid;
   kk_fdm_gaze[view].x = CLAMP(x, 0.0f, 1.0f);
   kk_fdm_gaze[view].y = CLAMP(y, 0.0f, 1.0f);
   simple_mtx_unlock(&kk_fdm_lock);
}

/* KK_FDM_GAZE simulates gaze: "x,y" for a fixed point, or "sweep[:seconds]"
 * for a point that circles the view. */
static bool
kk_fdm_live_gaze(uint32_t layer, float *x, float *y)
{
   static const char *sim;
   static bool init;
   if (!init) {
      sim = getenv("KK_FDM_GAZE");
      init = true;
   }
   if (sim && sim[0]) {
      if (!strncmp(sim, "sweep", 5)) {
         double period = sim[5] == ':' ? atof(sim + 6) : 4.0;
         if (period <= 0)
            period = 4.0;
         double t = os_time_get_nano() / 1e9 / period * 2.0 * M_PI;
         *x = 0.5f + 0.3f * (float)cos(t);
         *y = 0.5f + 0.25f * (float)sin(t);
         return true;
      }
      return sscanf(sim, "%f,%f", x, y) == 2;
   }
   if (!kk_fdm_gaze[layer].valid)
      return false;
   *x = kk_fdm_gaze[layer].x;
   *y = kk_fdm_gaze[layer].y;
   return true;
}

static bool
kk_fdm_get_gaze(uint32_t layer, float *x, float *y)
{
   *x = kk_fdm_latched[layer].x;
   *y = kk_fdm_latched[layer].y;
   return kk_fdm_latched[layer].valid;
}

void
kk_fdm_latch_gaze(void)
{
   static uint64_t last_ns;
   if (!KK_EXPERIMENTAL(FDM_RATE))
      return;
   uint64_t now = os_time_get_nano();
   simple_mtx_lock(&kk_fdm_lock);
   if (now - last_ns >= 10000000ull) {
      last_ns = now;
      for (uint32_t l = 0; l < KK_RATE_PARAMS_MAX_LAYERS; l++) {
         float x = 0.5f, y = 0.5f;
         kk_fdm_latched[l].valid = kk_fdm_live_gaze(l, &x, &y);
         kk_fdm_latched[l].x = x;
         kk_fdm_latched[l].y = y;
      }
   }
   simple_mtx_unlock(&kk_fdm_lock);
}

/* Moves a rate profile so that the middle of its highest-rate band lands on
 * position target, in profile entries. Returns the shift applied. */
static int
kk_fdm_shift_profile(float *profile, uint32_t n, float target)
{
   float best = 0;
   for (uint32_t i = 0; i < n; i++)
      best = MAX2(best, profile[i]);
   int first = -1, last = -1;
   for (uint32_t i = 0; i < n; i++) {
      if (profile[i] >= best * 0.999f) {
         if (first < 0)
            first = i;
         last = i;
      }
   }
   if (first < 0)
      return 0;
   int shift = (int)lroundf(target - 0.5f * (first + last));
   if (shift == 0)
      return 0;
   float tmp[2 * 256 + 1];
   if (n > ARRAY_SIZE(tmp))
      return 0;
   for (uint32_t i = 0; i < n; i++)
      tmp[i] = profile[CLAMP((int)i - shift, 0, (int)n - 1)];
   memcpy(profile, tmp, n * sizeof(float));
   return shift;
}

static float
kk_fdm_density(const struct kk_image *density, uint32_t layer, int32_t x,
               int32_t y, uint32_t channel, const VkOffset2D *offset)
{
   if (offset) {
      x -= offset->x / KK_FDM_TEXEL_SIZE;
      y -= offset->y / KK_FDM_TEXEL_SIZE;
   }
   x = CLAMP(x, 0, (int32_t)density->fdm.width - 1);
   y = CLAMP(y, 0, (int32_t)density->fdm.height - 1);
   const uint8_t *texel =
      density->fdm.cpu +
      (((size_t)layer * density->fdm.height + y) * density->fdm.width + x) * 2;
   return MAX2(texel[channel] / 255.0f, 1.0f / 16.0f);
}

static void *
kk_fdm_get_map(struct kk_cmd_buffer *cmd, struct kk_image *density,
               VkExtent2D screen, VkRect2D area)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   bool app_offset = dev->vk.enabled_extensions.EXT_fragment_density_map_offset;
   if (density->fdm.cpu == NULL)
      return NULL;

   const uint32_t mw = DIV_ROUND_UP(screen.width, KK_FDM_TEXEL_SIZE);
   const uint32_t mh = DIV_ROUND_UP(screen.height, KK_FDM_TEXEL_SIZE);
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
      const VkOffset2D *offset = app_offset ? &cmd->fdm_offsets[l] : NULL;
      for (uint32_t i = 0; i < nh; i++) {
         uint32_t a = i / 2, b = i / 2;
         if (!(i & 1)) {
            a = i ? i / 2 - 1 : 0;
            b = MIN2(i / 2, mw - 1);
         }
         float q = 0;
         for (uint32_t y = 0; y < mh; y++) {
            q = MAX2(q, kk_fdm_density(density, l, a, y, 0, offset));
            q = MAX2(q, kk_fdm_density(density, l, b, y, 0, offset));
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
            q = MAX2(q, kk_fdm_density(density, l, x, a, 1, offset));
            q = MAX2(q, kk_fdm_density(density, l, x, b, 1, offset));
         }
         vertical[l * nv + i] = q;
      }
   }

   int16_t shift[KK_RATE_PARAMS_MAX_LAYERS][2] = {{0}};
   for (uint32_t l = 0; l < layers; l++) {
      if (app_offset) {
         shift[l][0] = cmd->fdm_offsets[l].x / 16;
         shift[l][1] = cmd->fdm_offsets[l].y / 16;
         continue;
      }
      float gx, gy;
      if (!kk_fdm_get_gaze(l, &gx, &gy))
         continue;
      /* Profile entry 2k + 1 is the middle of map texel k. */
      float px = area.offset.x + gx * area.extent.width;
      float py = area.offset.y + gy * area.extent.height;
      shift[l][0] = kk_fdm_shift_profile(&horizontal[l * nh], nh,
                                         2.0f * px / KK_FDM_TEXEL_SIZE);
      shift[l][1] = kk_fdm_shift_profile(&vertical[l * nv], nv,
                                         2.0f * py / KK_FDM_TEXEL_SIZE);
   }

   if (density->fdm.map && density->fdm.map_version == density->fdm.version &&
       density->fdm.map_width == screen.width &&
       density->fdm.map_height == screen.height &&
       !memcmp(density->fdm.map_shift, shift, sizeof(shift))) {
      free(horizontal);
      free(vertical);
      return density->fdm.map;
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
   struct kk_fdm_params *params = calloc(1, sizeof(*params));
   if (params == NULL) {
      mtl_release(map);
      return NULL;
   }
   if (kk_alloc_bo(dev, &density->vk.base, KK_RATE_PARAMS_SIZE_B, 0,
                   &params->bo) != VK_SUCCESS) {
      free(params);
      mtl_release(map);
      return NULL;
   }
   params->dev = dev;
   params->refs = 1;
   if (density->fdm.params)
      kk_fdm_params_unref(density->fdm.params);
   density->fdm.params = params;
   density->fdm.serial = ++kk_fdm_serial;
   density->fdm.params_bytes =
      KK_RATE_PARAMS_HEADER_B + mtl_rate_map_parameter_size(map);
   memcpy(density->fdm.map_shift, shift, sizeof(shift));

   struct kk_rate_params_header *header = density->fdm.params->bo->cpu;
   memset(header, 0, KK_RATE_PARAMS_HEADER_B);
   mtl_rate_map_copy_parameters(map, density->fdm.params->bo->map,
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

   if (kk_fdm_debug()) {
      static uint32_t logged;
      if (logged++ < 16 || (logged & (logged - 1)) == 0)
         mesa_logi("kk-fdm: rate map %u: %ux%u from %ux%ux%u density map, layer 0 "
                   "physical %ux%u (%.1f%%), shift %d,%d / %d,%d texels/2",
                   logged, screen.width, screen.height, mw, mh, layers,
                   header->physical[0][0], header->physical[0][1],
                   100.0 * header->physical[0][0] * header->physical[0][1] /
                      ((double)screen.width * screen.height),
                   shift[0][0], shift[0][1], shift[1][0], shift[1][1]);
   }
   return map;
}

static bool
kk_fdm_subsampled(const struct kk_image_view *iview)
{
   return iview == NULL ||
          (iview->vk.image->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT);
}

static bool
kk_fdm_depth_compatible(const struct kk_rendering_state *render,
                        const struct kk_attachment *attachment)
{
   if (kk_fdm_subsampled(attachment->iview))
      return true;
   if (!debug_get_bool_option("KK_FDM_DISCARD_DEPTH", false) ||
       !(attachment->iview->vk.image->aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ||
       attachment->iview->vk.image->samples != VK_SAMPLE_COUNT_1_BIT ||
       (attachment->load_op != VK_ATTACHMENT_LOAD_OP_CLEAR &&
        attachment->load_op != VK_ATTACHMENT_LOAD_OP_DONT_CARE) ||
       attachment->store_op != VK_ATTACHMENT_STORE_OP_DONT_CARE ||
       attachment->resolve_mode != VK_RESOLVE_MODE_SAMPLE_ZERO_BIT ||
       !attachment->resolve_iview ||
       !kk_fdm_subsampled(attachment->resolve_iview))
      return false;
   const struct vk_image_view *view = &attachment->iview->vk;
   return view->base_mip_level == 0 && view->base_array_layer == 0 &&
          view->layer_count == view->image->array_layers &&
          render->area.offset.x == 0 && render->area.offset.y == 0 &&
          render->area.extent.width == view->image->extent.width &&
          render->area.extent.height == view->image->extent.height &&
          (render->view_mask
              ? render->view_mask == BITFIELD64_MASK(view->layer_count)
              : render->layer_count == view->layer_count);
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
   return kk_fdm_depth_compatible(render, &render->depth_att) &&
          kk_fdm_depth_compatible(render, &render->stencil_att) &&
          (render->depth_att.resolve_mode == VK_RESOLVE_MODE_NONE ||
           kk_fdm_subsampled(render->depth_att.resolve_iview)) &&
          (render->stencil_att.resolve_mode == VK_RESOLVE_MODE_NONE ||
           kk_fdm_subsampled(render->stencil_att.resolve_iview));
}

static void
kk_fdm_log_view(const char *what, uint32_t i, const struct kk_image_view *iview)
{
   if (iview == NULL)
      return;
   const struct vk_image *img = iview->vk.image;
   mesa_logi("kk-fdm:   %s %u: format %d samples %u %ux%ux%u usage 0x%x %s", what,
             i, img->format, img->samples, img->extent.width, img->extent.height,
             img->array_layers, (unsigned)img->usage,
             (img->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT)
                ? "subsampled"
                : "NOT subsampled");
}

static void
kk_fdm_log_pass(const struct kk_rendering_state *render, const char *why)
{
   mesa_logi("kk-fdm: %s flags=0x%x depth load=%u store=%u resolve=0x%x",
             why, render->flags, render->depth_att.load_op,
             render->depth_att.store_op, render->depth_att.resolve_mode);
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      kk_fdm_log_view("color", i, render->color_att[i].iview);
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         kk_fdm_log_view("color resolve", i, render->color_att[i].resolve_iview);
   }
   if (render->color_att_count && render->color_att[0].iview)
      mesa_logi("kk-fdm: color image %p load=%u store=%u depth image %p",
                (void *)render->color_att[0].iview->vk.image,
                render->color_att[0].load_op, render->color_att[0].store_op,
                render->depth_att.iview
                   ? (void *)render->depth_att.iview->vk.image : NULL);
   kk_fdm_log_view("depth", 0, render->depth_att.iview);
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      kk_fdm_log_view("depth resolve", 0, render->depth_att.resolve_iview);
   if (render->stencil_att.iview != render->depth_att.iview)
      kk_fdm_log_view("stencil", 0, render->stencil_att.iview);
}

static void
kk_fdm_publish(struct kk_cmd_buffer *cmd, struct kk_image *density,
               struct kk_image_view *iview)
{
   if (iview == NULL)
      return;
   struct kk_image *image = container_of(iview->vk.image, struct kk_image, vk);
   if (image->rate_params == NULL)
      return;
   /* Copied in every pass that renders the image rather than once: the
    * command buffer holding an earlier copy may be re-recorded or dropped. */
   uint32_t bytes = MIN2(density->fdm.params_bytes, KK_RATE_PARAMS_SIZE_B);
   if (cmd->metal.render == NULL)
      mtl_copy_from_buffer_to_buffer(cs_get_compute(cmd), density->fdm.params->bo->map,
                                     0, image->rate_params->map, 0, bytes);
   else
      memcpy(image->rate_params->cpu, density->fdm.params->bo->cpu, bytes);
   image->rate_params_owner = density;
   image->rate_params_serial = density->fdm.serial;
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
   kk_fdm_scan_offsets(cmd);

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

   if (!gfx->in_custom_resolve)
      gfx->seq_log = false;
   if (render->samples > 1 && !gfx->in_custom_resolve) {
      static int64_t start = -2;
      static uint32_t ms_passes;
      if (start == -2)
         start = debug_get_num_option("KK_SEQ_LOG", -1);
      gfx->seq_log = false;
      if (start >= 0) {
         simple_mtx_lock(&kk_fdm_lock);
         uint32_t n = ms_passes++;
         simple_mtx_unlock(&kk_fdm_lock);
         gfx->seq_log = n >= start && n < start + 3;
      }
   }

   simple_mtx_lock(&kk_fdm_lock);
   void *map = density ? kk_fdm_get_map(cmd, density, screen, render->area) : NULL;
   if (map && !kk_fdm_all_subsampled(render)) {
      static uint32_t logged;
      if (kk_fdm_debug() && logged++ < 8)
         kk_fdm_log_pass(render, "full rate, non-subsampled attachment");
      map = NULL;
   }
   if (density && kk_fdm_debug()) {
      static struct {
         const void *density, *image;
         uint32_t sw, sh, ax, ay, aw, ah;
      } combos[64];
      static uint32_t combo_count;
      const void *first = render->color_att_count && render->color_att[0].iview
                             ? (const void *)render->color_att[0].iview->vk.image
                             : NULL;
      bool found = false;
      for (uint32_t i = 0; i < combo_count && !found; i++)
         found = combos[i].density == density && combos[i].image == first &&
                 combos[i].sw == screen.width && combos[i].sh == screen.height &&
                 combos[i].ax == (uint32_t)render->area.offset.x &&
                 combos[i].ay == (uint32_t)render->area.offset.y &&
                 combos[i].aw == render->area.extent.width &&
                 combos[i].ah == render->area.extent.height;
      if (!found && combo_count < ARRAY_SIZE(combos)) {
         combos[combo_count].density = density;
         combos[combo_count].image = first;
         combos[combo_count].sw = screen.width;
         combos[combo_count].sh = screen.height;
         combos[combo_count].ax = render->area.offset.x;
         combos[combo_count].ay = render->area.offset.y;
         combos[combo_count].aw = render->area.extent.width;
         combos[combo_count].ah = render->area.extent.height;
         combo_count++;
         mesa_logi("kk-fdm: pass density %ux%u screen %ux%u area %d,%d %ux%u views 0x%x %s",
                   density->vk.extent.width, density->vk.extent.height, screen.width,
                   screen.height, render->area.offset.x, render->area.offset.y,
                   render->area.extent.width, render->area.extent.height,
                   render->view_mask, map ? "mapped" : "full rate");
         kk_fdm_log_pass(render, "  attachments:");
      }
      static uint32_t applied, fallback;
      if (map)
         applied++;
      else
         fallback++;
      if (((applied + fallback) & 4095) == 0)
         mesa_logi("kk-fdm: %u FDM passes rate mapped, %u at full rate", applied,
                   fallback);
   }

   gfx->fdm.density = map ? density : NULL;
   render->fdm_discard_depth = map &&
      (!kk_fdm_subsampled(render->depth_att.iview) ||
       !kk_fdm_subsampled(render->stencil_att.iview));
   if (map && render->stitch) {
      if (gfx->stitch.fdm_serial && gfx->stitch.fdm_serial != density->fdm.serial)
         gfx->stitch.fdm_mixed = true;
      gfx->stitch.fdm_serial = density->fdm.serial;
   }
   uint64_t params = 0;
   /* Resumed segments of a stitched pass may not start with another
    * encoder; the segment that began the pass published the parameters. */
   bool publish = !(render->flags & VK_RENDERING_RESUMING_BIT);
   if (map) {
      mtl_render_pass_descriptor_set_rate_map(descriptor, map);
      kk_fdm_cmd_ref(cmd, density->fdm.params);
      params = density->fdm.params->bo->gpu;
      for (uint32_t i = 0; publish && i < render->color_att_count; i++) {
         kk_fdm_publish(cmd, density, render->color_att[i].iview);
         kk_fdm_publish(cmd, density, render->color_att[i].resolve_iview);
      }
      if (publish) {
         kk_fdm_publish(cmd, density, render->depth_att.iview);
         kk_fdm_publish(cmd, density, render->depth_att.resolve_iview);
         kk_fdm_publish(cmd, density, render->stencil_att.iview);
         kk_fdm_publish(cmd, density, render->stencil_att.resolve_iview);
      }
   }
   simple_mtx_unlock(&kk_fdm_lock);

   struct kk_descriptor_state *desc = &gfx->descriptors;
   if (desc->root.fdm_rate_params != params) {
      desc->root.fdm_rate_params = params;
      desc->root_dirty = true;
   }
}

void
kk_seq_log(struct kk_cmd_buffer *cmd, const char *fmt, ...)
{
   if (!cmd->state.gfx.seq_log)
      return;
   char buf[256];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   mesa_logi("kk-seq %p: %s", (void *)cmd, buf);
}

void
kk_fdm_clear_image_mapping(struct kk_cmd_buffer *cmd, struct kk_image *image)
{
   if (!image || !image->rate_params)
      return;
   kk_cmd_write(cmd, (struct libkk_imm_write){image->rate_params->gpu, 0});
   mtl_barrier_after_encoder_stages(cs_get_compute(cmd), MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);
   image->rate_params_owner = NULL;
   image->rate_params_serial = 0;
}

void
kk_fdm_carry_to_next_rendering(struct kk_cmd_buffer *cmd)
{
   cmd->state.gfx.fdm.carry = cmd->state.gfx.fdm.density != NULL;
}
