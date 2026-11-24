/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "kk_fdm.h"
#ifdef THRESHOLD_SHARED_FOVEATION
#include <threshold_foveation.h>
#endif

#include "kk_bo.h"
#include "kk_debug.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_image.h"
#include "kk_image_view.h"
#include "kk_entrypoints.h"
#include "kk_descriptor_set.h"
#include "kk_descriptor_set_layout.h"
#include "kk_descriptor_types.h"
#include "kk_shader.h"
#include "vk_format.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_encoder.h"
#include "kosmickrisp/bridge/mtl_rate_map.h"

#include "util/log.h"
#include <sys/stat.h>
#include "util/u_math.h"
#include "util/half_float.h"
#include "util/u_atomic.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vk_util.h"
#include "vk_cmd_queue.h"

#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


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
static float kk_scaled_min = 0.4f;


/* Images with rate parameters, by the parameters' GPU address, so a dispatch can tell
 * from a storage descriptor which image it writes (KK_IMPLICIT_COMPACT). */
static simple_mtx_t kk_rate_registry_lock = SIMPLE_MTX_INITIALIZER;
static struct kk_image **kk_rate_registry;
static uint32_t kk_rate_registry_count, kk_rate_registry_cap;

static void
kk_rate_registry_add(struct kk_image *image)
{
   simple_mtx_lock(&kk_rate_registry_lock);
   if (kk_rate_registry_count == kk_rate_registry_cap) {
      uint32_t cap = MAX2(64, kk_rate_registry_cap * 2);
      struct kk_image **grown = realloc(kk_rate_registry, cap * sizeof(*grown));
      if (grown) {
         kk_rate_registry = grown;
         kk_rate_registry_cap = cap;
      }
   }
   if (kk_rate_registry_count < kk_rate_registry_cap)
      kk_rate_registry[kk_rate_registry_count++] = image;
   simple_mtx_unlock(&kk_rate_registry_lock);
}

static void
kk_rate_registry_remove(struct kk_image *image)
{
   simple_mtx_lock(&kk_rate_registry_lock);
   for (uint32_t i = 0; i < kk_rate_registry_count; i++) {
      if (kk_rate_registry[i] == image) {
         kk_rate_registry[i] = kk_rate_registry[--kk_rate_registry_count];
         break;
      }
   }
   simple_mtx_unlock(&kk_rate_registry_lock);
}

static struct kk_image *
kk_rate_registry_find_locked(uint64_t gpu)
{
   for (uint32_t i = 0; i < kk_rate_registry_count; i++)
      if (kk_rate_registry[i]->rate_params->gpu == gpu)
         return kk_rate_registry[i];
   return NULL;
}

/* Sampled view ids of foveatable images, so a diagnostic can name the image behind a
 * descriptor that carries no map. */
static struct {
   uint64_t id;
   struct kk_image *image;
} kk_view_ids[8192];
static uint32_t kk_view_id_count;

/* Diagnostic: sampled view ids listed (hex) in a "watch-resources" file in Monado's
 * session recording folder. Their images are logged at view creation, kept here for
 * the mark logs and dumps, and traced. */
static struct {
   uint64_t ids[16];
   uint32_t count;
   int loaded;
   struct { uint64_t id; struct kk_image *image; } views[64];
   uint32_t view_count;
} kk_watched_res;
static struct kk_image *kk_watched_res_traced[8];

static bool
kk_watched_res_listed(uint64_t id)
{
   if (!kk_watched_res.loaded) {
      kk_watched_res.loaded = 1;
      const char *session = getenv("XRT_SESSION_RECORD_DIR");
      char path[1024];
      if (session) {
         snprintf(path, sizeof(path), "%s/watch-resources", session);
         FILE *f = fopen(path, "r");
         if (f) {
            char word[64];
            while (kk_watched_res.count < ARRAY_SIZE(kk_watched_res.ids) &&
                   fscanf(f, " %63s", word) == 1)
               kk_watched_res.ids[kk_watched_res.count++] = strtoull(word, NULL, 16);
            fclose(f);
            mesa_logi("kk-res: watching %u resource id(s)", kk_watched_res.count);
         }
      }
   }
   for (uint32_t i = 0; i < kk_watched_res.count; i++)
      if (kk_watched_res.ids[i] == id)
         return true;
   return false;
}

/* Diagnostic: every sampled view id with its image's description (copied, so it can be
 * printed without touching the image), for the mark logs. */
static struct {
   uint64_t id;
   const void *image;
   uint32_t w, h, levels;
   int format, candidate;
} kk_names[8192];
static uint32_t kk_name_count;

static void
kk_name_note_locked(uint64_t id, const struct kk_image *image)
{
   if (kk_name_count < ARRAY_SIZE(kk_names))
      kk_names[kk_name_count++] = (__typeof__(kk_names[0])){
         id, image, image->vk.extent.width, image->vk.extent.height, image->vk.mip_levels,
         image->vk.format, image->implicit_candidate};
}

void
kk_implicit_note_view_id(uint64_t id, struct kk_image *image)
{
   if (id && image) {
      simple_mtx_lock(&kk_rate_registry_lock);
      kk_name_note_locked(id, image);
      simple_mtx_unlock(&kk_rate_registry_lock);
   }
   if (id && image && kk_watched_res_listed(id)) {
      mesa_logi("kk-res: view %llx is image %p %ux%u fmt %d usage 0x%x flags 0x%x mips %u "
                "layers %u samples %u candidate %d", (unsigned long long)id, (void *)image,
                image->vk.extent.width, image->vk.extent.height, image->vk.format,
                (unsigned)image->vk.usage, (unsigned)image->vk.create_flags, image->vk.mip_levels,
                image->vk.array_layers, (unsigned)image->vk.samples, image->implicit_candidate);
      simple_mtx_lock(&kk_rate_registry_lock);
      if (kk_watched_res.view_count < ARRAY_SIZE(kk_watched_res.views)) {
         kk_watched_res.views[kk_watched_res.view_count].id = id;
         kk_watched_res.views[kk_watched_res.view_count++].image = image;
      }
      for (uint32_t i = 0; i < ARRAY_SIZE(kk_watched_res_traced); i++) {
         if (!kk_watched_res_traced[i] || kk_watched_res_traced[i] == image) {
            kk_watched_res_traced[i] = image;
            break;
         }
      }
      simple_mtx_unlock(&kk_rate_registry_lock);
   }
   if (!id || !image || !image->rate_params)
      return;
   simple_mtx_lock(&kk_rate_registry_lock);
   if (kk_view_id_count < ARRAY_SIZE(kk_view_ids)) {
      kk_view_ids[kk_view_id_count].id = id;
      kk_view_ids[kk_view_id_count++].image = image;
   }
   simple_mtx_unlock(&kk_rate_registry_lock);
}

void
kk_implicit_forget_view_id(uint64_t id)
{
   if (!id)
      return;
   simple_mtx_lock(&kk_rate_registry_lock);
   for (uint32_t i = 0; i < kk_name_count; i++) {
      if (kk_names[i].id == id) {
         kk_names[i] = kk_names[--kk_name_count];
         break;
      }
   }
   for (uint32_t i = 0; i < kk_watched_res.view_count; i++) {
      if (kk_watched_res.views[i].id == id) {
         kk_watched_res.views[i] = kk_watched_res.views[--kk_watched_res.view_count];
         break;
      }
   }
   for (uint32_t i = 0; i < kk_view_id_count; i++) {
      if (kk_view_ids[i].id == id) {
         kk_view_ids[i] = kk_view_ids[--kk_view_id_count];
         break;
      }
   }
   simple_mtx_unlock(&kk_rate_registry_lock);
}

/* The live image behind a sampled view id, or NULL. Called with kk_rate_registry_lock. */
static struct kk_image *
kk_view_id_find_locked(uint64_t id)
{
   for (uint32_t i = 0; i < kk_watched_res.view_count; i++)
      if (kk_watched_res.views[i].id == id)
         return kk_watched_res.views[i].image;
   for (uint32_t i = 0; i < kk_view_id_count; i++) {
      if (kk_view_ids[i].id != id)
         continue;
      for (uint32_t j = 0; j < kk_rate_registry_count; j++)
         if (kk_rate_registry[j] == kk_view_ids[i].image)
            return kk_view_ids[i].image;
   }
   return NULL;
}

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


#define KK_IMPLICIT_SAMPLES 64
struct kk_implicit_profile {
   uint32_t count; /* samples across one eye, per axis */
   float horizontal[2][KK_IMPLICIT_SAMPLES], vertical[KK_IMPLICIT_SAMPLES];
   uint64_t serial;
};
static struct {
   int state; /* 0 unread, 1 on, -1 off */
   bool automatic;
   uint32_t view_width, view_height; /* one eye */
   struct kk_implicit_profile pending, latched;
   struct {
      uint32_t width, height, eyes;
      struct kk_image *density;
   } targets[8];
   uint32_t target_count;
} kk_implicit;

bool
kk_implicit_fdm_enabled(void)
{
   if (kk_implicit.state == 0) {
      const char *text = getenv("KK_IMPLICIT_FDM");
      uint32_t width = 0, height = 0;
      float outer = 0.3f, core = 0.35f, edge = 0.75f;
      if (text && KK_EXPERIMENTAL(FDM_RATE) && !strcmp(text, "auto")) {
         kk_implicit.automatic = true;
         kk_implicit.state = 1;
         mesa_logi("kk-implicit: waiting for the OpenXR layer's view size and profile");
      } else if (text && KK_EXPERIMENTAL(FDM_RATE) &&
                 sscanf(text, "%ux%u,%f,%f,%f", &width, &height, &outer, &core, &edge) >= 2 &&
                 width && height) {
         /* Fixed: both eyes side by side in a WIDTHxHEIGHT target. */
         outer = CLAMP(outer, 0.1f, 1.0f);
         edge = MAX2(edge, core + 0.05f);
         struct kk_implicit_profile *p = &kk_implicit.pending;
         p->count = KK_IMPLICIT_SAMPLES;
         const float half = width / 4.0f;
         for (uint32_t i = 0; i < p->count; i++) {
            float along = (i + 0.5f) / p->count;
            float dist[2] = {fabsf(along - 0.5f) * 2.0f, fabsf(along - 0.5f) * height / half};
            for (uint32_t c = 0; c < 2; c++) {
               float s = CLAMP((dist[c] - core) / (edge - core), 0.0f, 1.0f);
               s = s * s * (3.0f - 2.0f * s);
               float rate = 1.0f + (outer - 1.0f) * s;
               if (c == 0)
                  p->horizontal[0][i] = p->horizontal[1][i] = rate;
               else
                  p->vertical[i] = rate;
            }
         }
         p->serial = 1;
         kk_implicit.view_width = width / 2;
         kk_implicit.view_height = height;
         kk_implicit.state = 1;
         mesa_logi("kk-implicit: foveating multisampled %ux%u targets, outer %.2f core %.2f edge %.2f",
                   width, height, outer, core, edge);
      } else {
         kk_implicit.state = -1;
      }
   }
   return kk_implicit.state > 0;
}

/* From the OpenXR layer in the same process: one eye's size and the rate (0..1] at
 * `count` evenly spaced positions across each eye, left to right and top to bottom of
 * the image as this driver stores it. Takes effect at the next frame of each target.
 * A profile equal to the pending one changes nothing: a new serial would rebuild every
 * target's density and rate map for identical rates. Returns whether it changed. */
static bool
kk_implicit_take_profile_locked(uint32_t view_width, uint32_t view_height, uint32_t count,
                                const float *horizontal0, const float *horizontal1,
                                const float *vertical, const char *source)
{
   count = MIN2(count, KK_IMPLICIT_SAMPLES);
   struct kk_implicit_profile next = {.count = count};
   for (uint32_t i = 0; i < count; i++) {
      next.horizontal[0][i] = CLAMP(horizontal0[i], 0.0625f, 1.0f);
      next.horizontal[1][i] = CLAMP(horizontal1[i], 0.0625f, 1.0f);
      next.vertical[i] = CLAMP(vertical[i], 0.0625f, 1.0f);
   }
   struct kk_implicit_profile *p = &kk_implicit.pending;
   if (p->serial && p->count == count && kk_implicit.view_width == view_width &&
       kk_implicit.view_height == view_height &&
       !memcmp(p->horizontal, next.horizontal, sizeof(next.horizontal)) &&
       !memcmp(p->vertical, next.vertical, sizeof(next.vertical)))
      return false;
   next.serial = p->serial + 1;
   *p = next;
   if (kk_implicit.view_width != view_width || kk_implicit.view_height != view_height)
      mesa_logi("kk-implicit: view size %ux%u from %s", view_width, view_height, source);
   kk_implicit.view_width = view_width;
   kk_implicit.view_height = view_height;
   return true;
}

PUBLIC void
kk_implicit_fdm_set_profile(uint32_t view_width, uint32_t view_height, uint32_t count,
                            const float *horizontal0, const float *horizontal1,
                            const float *vertical)
{
   if (!kk_implicit_fdm_enabled() || !kk_implicit.automatic || !count || !view_width || !view_height)
      return;
   simple_mtx_lock(&kk_fdm_lock);
   kk_implicit_take_profile_locked(view_width, view_height, count, horizontal0, horizontal1, vertical,
                                   "the OpenXR layer");
   simple_mtx_unlock(&kk_fdm_lock);
}

/*
 * Runtime profiles (threshold-foveation API level 3). An OpenXR runtime in this process
 * that binds a rate map on each swapchain image may also publish the per-view profile it
 * was built from, for whatever renders the application's frames. With
 * KK_IMPLICIT_FDM=auto the driver reads it from the runtime images it imported (the
 * swapchain images an application or a layer such as Wine's OpenXR bridge renders or
 * copies into) and takes the newest, so no layer has to call kk_implicit_fdm_set_profile.
 * A profile covering both eyes is preferred: an image of a one-eye swapchain carries
 * only its own eye's, whose horizontal rates are wrong for the other eye. A revision is
 * published at each acquire, which may come in the middle of the application's frame,
 * so it never ends a frame by itself; a new profile is taken at the next frame start
 * (kk_implicit_density). KK_IMPLICIT_RUNTIME_FLIP=1 reverses the vertical
 * profile for an application whose images are bottom-up (OpenGL through Zink).
 */
#define KK_RUNTIME_IMAGES 16
static struct {
   struct kk_image *images[KK_RUNTIME_IMAGES];
   uint32_t count;
   uint64_t revision;
} kk_runtime;

void
kk_implicit_note_import(struct kk_image *image)
{
   simple_mtx_lock(&kk_fdm_lock);
   if (kk_runtime.count < KK_RUNTIME_IMAGES)
      kk_runtime.images[kk_runtime.count++] = image;
   simple_mtx_unlock(&kk_fdm_lock);
}

static void
kk_implicit_forget_import(struct kk_image *image)
{
   simple_mtx_lock(&kk_fdm_lock);
   for (uint32_t i = 0; i < kk_runtime.count; i++) {
      if (kk_runtime.images[i] == image) {
         kk_runtime.images[i] = kk_runtime.images[--kk_runtime.count];
         break;
      }
   }
   simple_mtx_unlock(&kk_fdm_lock);
}

/* Takes a newer runtime profile, if one of the imported images carries it. Called with
 * kk_fdm_lock held. */
static void
kk_implicit_poll_runtime_locked(void)
{
#ifdef THRESHOLD_SHARED_FOVEATION
   typedef int (*copy_profile_fn)(void *, struct tf_image_profile *);
   typedef int (*copy_process_fn)(struct tf_image_profile *);
   static copy_profile_fn copy_profile;
   static copy_process_fn copy_process;
   static int looked;
   static int flip = -1;
   if (!looked) {
      copy_profile = (copy_profile_fn)dlsym(RTLD_DEFAULT, "tf_metal_copy_texture_profile");
      copy_process = (copy_process_fn)dlsym(RTLD_DEFAULT, "tf_copy_process_profile");
      looked = 1;
      if (!copy_profile)
         mesa_logi("kk-implicit: threshold-foveation has no image profiles (API level 3)");
   }
   if (flip < 0)
      flip = debug_get_bool_option("KK_IMPLICIT_RUNTIME_FLIP", false);
   /* The newest profile among the images, for both eyes and for one eye. */
   struct tf_image_profile stereo = {0}, mono = {0}, process = {0}, profile;
   for (uint32_t i = 0; copy_profile && i < kk_runtime.count; i++) {
      memset(&profile, 0, sizeof(profile));
      profile.struct_size = sizeof(profile);
      if (copy_profile(kk_runtime.images[i]->planes[0].mtl_handle, &profile) == 0) {
         struct tf_image_profile *slot = profile.view_count > 1 ? &stereo : &mono;
         if (profile.revision > slot->revision)
            *slot = profile;
      }
   }
   /* The runtime's profile for the process (API level 4) covers both eyes. It serves
    * output that reaches the runtime only through another API (Zink's OpenGL frames
    * that an OpenVR layer copies through D3D11), and one-eye swapchains. */
   if (copy_process) {
      process.struct_size = sizeof(process);
      if (copy_process(&process) != 0)
         process.revision = 0;
   }
   /* Both eyes from the images first, then from the process, then one eye. */
   struct tf_image_profile best = stereo;
   const char *source = "the OpenXR runtime's images";
   if (!best.revision && process.revision && process.view_count > 1) {
      best = process;
      source = "the OpenXR runtime's process profile";
   } else if (!best.revision && mono.revision) {
      best = mono;
      source = "the OpenXR runtime's one-eye images";
   } else if (!best.revision && process.revision) {
      best = process;
      source = "the OpenXR runtime's process profile";
   }
   static int reported_none;
   if (!best.revision && !reported_none) {
      reported_none = 1;
      mesa_logi("kk-implicit: %u imported runtime image(s) and no process profile yet",
                kk_runtime.count);
   }
   if (best.revision <= kk_runtime.revision)
      return;
   const bool first = !kk_runtime.revision;
   kk_runtime.revision = best.revision;
   const struct tf_image_profile_view *left = &best.views[0];
   const struct tf_image_profile_view *right = &best.views[best.view_count > 1 ? 1 : 0];
   /* One vertical profile serves both eyes: take the higher rate of the two. */
   float vertical[KK_IMPLICIT_SAMPLES];
   const uint32_t count = MIN2(best.sample_count, KK_IMPLICIT_SAMPLES);
   for (uint32_t i = 0; i < count; i++) {
      const uint32_t row = flip ? count - 1 - i : i;
      vertical[i] = MAX2(left->vertical[row], right->vertical[row]);
   }
   kk_implicit_take_profile_locked(left->width, left->height, count, left->horizontal, right->horizontal,
                                   vertical, source);
   static const char *reported_source;
   if (first || source != reported_source)
      mesa_logi("kk-implicit: profiles from %s (%u views%s)", source, best.view_count,
                flip ? ", bottom-up" : "");
   reported_source = source;
#endif
}

/* Set while the driver creates its own images (the unwarp copies): never candidates. */
static _Thread_local bool kk_implicit_internal_image;

/* Formats left unfoveated: VkFormat numbers in a "nofoveate-formats" file in Monado's
 * session recording folder at launch (a diagnostic, to find the image whose foveation
 * breaks an effect). */
static bool
kk_implicit_format_excluded(VkFormat format)
{
   static int loaded;
   static uint32_t formats[16], count;
   if (!loaded) {
      loaded = 1;
      const char *session = getenv("XRT_SESSION_RECORD_DIR");
      char path[1024];
      if (session) {
         snprintf(path, sizeof(path), "%s/nofoveate-formats", session);
         FILE *f = fopen(path, "r");
         if (f) {
            unsigned v;
            while (count < ARRAY_SIZE(formats) && fscanf(f, " %u", &v) == 1)
               formats[count++] = v;
            fclose(f);
            mesa_logi("kk-implicit: %u format(s) left unfoveated (%s)", count, path);
         }
      }
   }
   for (uint32_t i = 0; i < count; i++)
      if (formats[i] == (uint32_t)format)
         return true;
   return false;
}

bool
kk_implicit_fdm_image(const VkImageCreateInfo *info)
{
   if (kk_implicit_internal_image || kk_implicit_format_excluded(info->format))
      return false;
   /* Every multisampled attachment is a candidate: the size that marks an eye target
    * may only be known later, and descriptors copy the parameter address when written. */
   if (!kk_implicit_fdm_enabled() || info->imageType != VK_IMAGE_TYPE_2D ||
       info->arrayLayers != 1 ||
       (info->mipLevels != 1 && info->samples != VK_SAMPLE_COUNT_1_BIT) ||
       !(info->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)))
      return false;
   if (info->samples != VK_SAMPLE_COUNT_1_BIT)
      return info->mipLevels == 1;
   /* KK_IMPLICIT_SINGLE=1: single-sample eye targets too (The Talos Principle VR). They
    * stay warped: shader reads are translated, and raw copies read an unwarped copy
    * (kk_implicit_unwarp_for_copy). Storage access is not translated, so such images
    * are left alone; colour targets must be sampleable for the unwarp blit. Not for
    * applications whose MSAA resolve targets are read outside Vulkan (Earth VR). */
   static int single = -1;
   if (single < 0)
      single = debug_get_bool_option("KK_IMPLICIT_SINGLE", false);
   /* Mip levels below the first are made by blits of the image into itself, which keep
    * the warped layout at every level (kk_implicit_set_raw). */
   if (!single || ((info->usage & VK_IMAGE_USAGE_STORAGE_BIT) && !kk_implicit_storage_enabled()) ||
       (info->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_ALIAS_BIT)))
      return false;
   return (info->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ||
          (info->usage & VK_IMAGE_USAGE_SAMPLED_BIT);
}

static uint32_t
kk_implicit_eyes(const struct kk_image *image)
{
   if (!kk_implicit.view_width)
      return 0;
   /* KK_IMPLICIT_SKIP=WxH[:FORMAT][,...] (diagnostic): sizes, or sizes in one VkFormat,
    * never foveated. */
   const char *skip = getenv("KK_IMPLICIT_SKIP");
   if (skip) {
      char want[2][40];
      snprintf(want[0], sizeof(want[0]), "%ux%u", image->vk.extent.width, image->vk.extent.height);
      snprintf(want[1], sizeof(want[1]), "%ux%u:%d", image->vk.extent.width,
               image->vk.extent.height, (int)image->vk.format);
      for (unsigned w = 0; w < 2; w++)
         for (const char *p = skip; (p = strstr(p, want[w])); p++)
            if ((p == skip || p[-1] == ',') &&
                (p[strlen(want[w])] == 0 || p[strlen(want[w])] == ','))
               return 0;
   }
   /* KK_IMPLICIT_HALF=1 (single-sample route): half-size targets too, each with its own
    * map of the same profile (half-resolution effects; The Talos Principle VR). */
   static int half = -1;
   if (half < 0)
      half = debug_get_bool_option("KK_IMPLICIT_HALF", false) &&
             debug_get_bool_option("KK_IMPLICIT_SINGLE", false);
   if (half && image->vk.samples == VK_SAMPLE_COUNT_1_BIT &&
       image->vk.extent.width == DIV_ROUND_UP(kk_implicit.view_width, 2) &&
       image->vk.extent.height == DIV_ROUND_UP(kk_implicit.view_height, 2))
      return 1;
   /* KK_IMPLICIT_SCALED=1 (single-sample route): the eye layout scaled down uniformly,
    * 40-100% (kk_scaled_min), within 0.8%, not powers of two, as a render scale or dynamic
    * resolution makes it (No Man's Sky renders its scene at about 59% and upscales). Each
    * size gets its own map of the profile. */
   static int scaled = -1;
   if (scaled < 0)
      scaled = debug_get_bool_option("KK_IMPLICIT_SCALED", false) &&
               debug_get_bool_option("KK_IMPLICIT_SINGLE", false);
   /* Power-of-two sizes are textures, shadow maps or probes, not scaled eyes; with a
    * nearly square eye a square texture would otherwise match. */
   const bool pot = util_is_power_of_two_nonzero(image->vk.extent.width) &&
                    util_is_power_of_two_nonzero(image->vk.extent.height);
   if (scaled && image->vk.samples == VK_SAMPLE_COUNT_1_BIT && !pot) {
      for (uint32_t eyes = 1; eyes <= 2; eyes++) {
         const float sx = (float)image->vk.extent.width / (eyes * kk_implicit.view_width);
         const float sy = (float)image->vk.extent.height / kk_implicit.view_height;
         if (sx >= kk_scaled_min && sx < 1.0f && fabsf(sx - sy) <= 0.008f)
            return eyes;
      }
   }
   if (image->vk.extent.height != kk_implicit.view_height)
      return 0;
   return image->vk.extent.width == kk_implicit.view_width ? 1
          : image->vk.extent.width == 2 * kk_implicit.view_width ? 2 : 0;
}

static void kk_implicit_poll_runtime_locked(void);

bool
kk_implicit_fdm_match(struct kk_image *image)
{
   if (image->implicit_fdm || !image->implicit_candidate || image->implicit_poison)
      return image->implicit_fdm;
   simple_mtx_lock(&kk_fdm_lock);
   /* The eye size may only be known from a runtime profile not read yet. */
   if (kk_implicit.automatic && !kk_implicit_eyes(image))
      kk_implicit_poll_runtime_locked();
   if (kk_implicit_eyes(image) && kk_implicit.pending.serial) {
      image->implicit_fdm = true;
      mesa_logi("kk-implicit: foveating %ux%u s%u image %p (%u eye%s)", image->vk.extent.width,
                image->vk.extent.height, image->vk.samples, (void *)image, kk_implicit_eyes(image),
                kk_implicit_eyes(image) == 2 ? "s side by side" : "");
   }
   simple_mtx_unlock(&kk_fdm_lock);
   return image->implicit_fdm;
}

static void
kk_implicit_fill(struct kk_image *density, uint32_t width, uint32_t height, uint32_t eyes)
{
   const struct kk_implicit_profile *p = &kk_implicit.latched;
   const uint32_t w = density->fdm.width, h = density->fdm.height;
   const float eye_width = (float)width / eyes;
   for (uint32_t y = 0; y < h; y++) {
      float py = MIN2((y + 0.5f) * 32.0f, height - 0.5f);
      uint32_t vi = MIN2((uint32_t)(py / height * p->count), p->count - 1);
      for (uint32_t x = 0; x < w; x++) {
         float px = MIN2((x + 0.5f) * 32.0f, width - 0.5f);
         uint32_t eye = MIN2((uint32_t)(px / eye_width), eyes - 1);
         uint32_t hi = MIN2((uint32_t)((px - eye * eye_width) / eye_width * p->count), p->count - 1);
         uint8_t *texel = density->fdm.cpu + ((size_t)y * w + x) * 2;
         /* A one-eye target may hold either eye (Talos renders each eye to its own image
          * of the same size), and the two eyes' fields of view are not mirror images:
          * take the higher of the two eyes' rates, so neither eye's sharp region lands
          * off its gaze. */
         float hrate = p->horizontal[eye][hi];
         if (eyes == 1)
            hrate = MAX2(p->horizontal[0][hi], p->horizontal[1][hi]);
         texel[0] = (uint8_t)lroundf(255.0f * hrate);
         texel[1] = (uint8_t)lroundf(255.0f * p->vertical[vi]);
      }
   }
   density->fdm.version++;
}

static uint64_t kk_implicit_last_present_ns;
static bool kk_implicit_frame_boundary;

/* Called at every present (kk_queue.c), and by the OpenXR layer when the application
 * releases its eye image (kk_implicit_fdm_frame_boundary): the next pass may take a new
 * profile. */
void
kk_implicit_fdm_present(void)
{
   kk_implicit_last_present_ns = os_time_get_nano();
   kk_implicit_frame_boundary = true;
}

PUBLIC void
kk_implicit_fdm_frame_boundary(void)
{
   /* KK_IMPLICIT_BOUNDARY=present (diagnostic): only the application's own presents end
    * a frame, not the OpenXR layer's swapchain releases. */
   static int present_only = -1;
   if (present_only < 0) {
      const char *v = getenv("KK_IMPLICIT_BOUNDARY");
      present_only = v && !strcmp(v, "present");
   }
   if (!present_only)
      kk_implicit_fdm_present();
}

/* The application's finished frames as the OpenXR runtime counts them (threshold-foveation
 * API level 5), or 0 when it doesn't count them or KK_IMPLICIT_BOUNDARY=present asks
 * for presents alone (diagnostic). */
static uint64_t
kk_implicit_runtime_frames(void)
{
#ifdef THRESHOLD_SHARED_FOVEATION
   typedef uint64_t (*frame_count_fn)(void);
   static frame_count_fn frame_count;
   static int looked;
   if (!looked) {
      const char *v = getenv("KK_IMPLICIT_BOUNDARY");
      if (!v || strcmp(v, "present"))
         frame_count = (frame_count_fn)dlsym(RTLD_DEFAULT, "tf_process_frame_count");
      looked = 1;
   }
   const uint64_t frames = frame_count ? frame_count() : 0;
   static bool reported;
   if (frames && !reported) {
      reported = true;
      mesa_logi("kk-implicit: frame boundaries from the OpenXR runtime");
   }
   return frames;
#else
   return 0;
#endif
}

/* KK_IMPLICIT_NOMAP=1, or a file named after the capture trigger file plus ".nomap" when
 * the process starts: passes render without maps. Read once, since images drawn under
 * maps keep them. */
static bool
kk_implicit_nomap(void)
{
   static int nomap = -1;
   if (nomap < 0) {
      nomap = debug_get_bool_option("KK_IMPLICIT_NOMAP", false);
      const char *trigger = getenv("KK_CAPTURE_TRIGGER_FILE");
      char path[1024];
      if (!nomap && trigger) {
         snprintf(path, sizeof(path), "%s.nomap", trigger);
         nomap = access(path, F_OK) == 0;
         if (nomap)
            mesa_logi("kk-implicit: maps off (%s)", path);
      }
      /* Or a "nomap" file in Monado's session recording folder. */
      const char *session = getenv("XRT_SESSION_RECORD_DIR");
      if (!nomap && session) {
         snprintf(path, sizeof(path), "%s/nomap", session);
         nomap = access(path, F_OK) == 0;
         if (nomap)
            mesa_logi("kk-implicit: maps off (%s)", path);
      }
   }
   return nomap;
}

/* The driver-made density map for a target of this size. `new_frame`: the pass does
 * not load the target, so a new profile cannot leave it in two layouts. Called with
 * kk_fdm_lock held. */
static struct kk_image *
kk_implicit_density(struct kk_device *dev, const struct kk_image *image, bool new_frame)
{
   if (kk_implicit.automatic && (new_frame || !kk_implicit.pending.serial))
      kk_implicit_poll_runtime_locked();
   const uint32_t eyes = kk_implicit_eyes(image);
   if (!eyes || !kk_implicit.pending.serial)
      return NULL;
   /* A new profile is taken only at the first pass after a present that does not load its
    * target. Taken at any such pass, it changed the map in the middle of a frame: later
    * passes loading targets drawn earlier in the frame rendered them in the new layout
    * over content in the old one (NMS with a moving gaze: hard-edged rectangles). That
    * first pass ends the chance whether or not the profile changed: a profile arriving
    * later in the frame (a runtime publishes one at each swapchain acquire) waits for the
    * next present. An application that never presents keeps the old rule. */
   /* A runtime that counts the application's frames says where they end: the first such
    * pass after the count changes starts a frame, whatever the application presents. */
   static uint64_t frames_seen;
   const uint64_t frames = kk_implicit_runtime_frames();
   bool frame_start;
   if (frames) {
      frame_start = new_frame && frames != frames_seen;
      if (frame_start)
         frames_seen = frames;
   } else {
      const bool presents = kk_implicit_last_present_ns &&
                            os_time_get_nano() - kk_implicit_last_present_ns < 200000000ull;
      frame_start = new_frame && (presents ? kk_implicit_frame_boundary : true);
      if (frame_start)
         kk_implicit_frame_boundary = false;
   }
   if (kk_implicit.latched.serial != kk_implicit.pending.serial &&
       (frame_start || !kk_implicit.latched.serial)) {
      kk_implicit.latched = kk_implicit.pending;
      for (uint32_t i = 0; i < kk_implicit.target_count; i++)
         kk_implicit_fill(kk_implicit.targets[i].density, kk_implicit.targets[i].width,
                          kk_implicit.targets[i].height, kk_implicit.targets[i].eyes);
   }
   for (uint32_t i = 0; i < kk_implicit.target_count; i++) {
      if (kk_implicit.targets[i].width == image->vk.extent.width &&
          kk_implicit.targets[i].height == image->vk.extent.height)
         return kk_implicit.targets[i].density;
   }
   if (kk_implicit.target_count == ARRAY_SIZE(kk_implicit.targets))
      return NULL;
   struct kk_image *density = calloc(1, sizeof(*density));
   if (density == NULL)
      return NULL;
   vk_object_base_init(&dev->vk, &density->vk.base, VK_OBJECT_TYPE_IMAGE);
   const uint32_t w = DIV_ROUND_UP(image->vk.extent.width, 32), h = DIV_ROUND_UP(image->vk.extent.height, 32);
   density->vk.extent = (VkExtent3D){w, h, 1};
   density->fdm.width = w; density->fdm.height = h; density->fdm.layers = 1;
   density->fdm.cpu = malloc((size_t)w * h * 2);
   if (density->fdm.cpu == NULL) {
      free(density);
      return NULL;
   }
   kk_implicit_fill(density, image->vk.extent.width, image->vk.extent.height, eyes);
   kk_implicit.targets[kk_implicit.target_count].width = image->vk.extent.width;
   kk_implicit.targets[kk_implicit.target_count].height = image->vk.extent.height;
   kk_implicit.targets[kk_implicit.target_count].eyes = eyes;
   kk_implicit.targets[kk_implicit.target_count++].density = density;
   return density;
}

struct kk_image_view *
kk_implicit_resolve_view(struct kk_device *dev, struct kk_image *src,
                         const struct kk_image_view *dst)
{
   if (!src->implicit_fdm || dst == NULL ||
       (dst->vk.image->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT) ||
       debug_get_bool_option("KK_IMPLICIT_NATIVE_RESOLVE", false))
      return NULL;
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkDevice device = kk_device_to_handle(dev);
   simple_mtx_lock(&kk_fdm_lock);
   if (src->implicit_resolve_view == VK_NULL_HANDLE) {
      const VkImageCreateInfo image_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .flags = VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = dst->vk.view_format,
         .extent = src->vk.extent,
         .mipLevels = 1,
         .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      VkImage image = VK_NULL_HANDLE;
      VkDeviceMemory memory = VK_NULL_HANDLE;
      VkImageView view = VK_NULL_HANDLE;
      VkResult result = disp->CreateImage(device, &image_info, NULL, &image);
      if (result == VK_SUCCESS) {
         VkMemoryRequirements2 reqs = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
         const VkImageMemoryRequirementsInfo2 reqs_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = image};
         disp->GetImageMemoryRequirements2(device, &reqs_info, &reqs);
         const VkMemoryAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = reqs.memoryRequirements.size,
            .memoryTypeIndex = ffs(reqs.memoryRequirements.memoryTypeBits) - 1,
         };
         result = disp->AllocateMemory(device, &alloc, NULL, &memory);
      }
      if (result == VK_SUCCESS) {
         const VkBindImageMemoryInfo bind = {
            .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, .image = image, .memory = memory};
         result = disp->BindImageMemory2(device, 1, &bind);
      }
      if (result == VK_SUCCESS) {
         const VkImageViewCreateInfo view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = dst->vk.view_format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
         };
         result = disp->CreateImageView(device, &view_info, NULL, &view);
      }
      if (result != VK_SUCCESS) {
         mesa_logw("kk-implicit: hidden resolve image failed (%d)", result);
         if (image != VK_NULL_HANDLE)
            disp->DestroyImage(device, image, NULL);
         if (memory != VK_NULL_HANDLE)
            disp->FreeMemory(device, memory, NULL);
         simple_mtx_unlock(&kk_fdm_lock);
         return NULL;
      }
      src->implicit_resolve_image = image;
      src->implicit_resolve_memory = memory;
      src->implicit_resolve_view = view;
      mesa_logi("kk-implicit: hidden resolve image %ux%u format %d for %p", src->vk.extent.width,
                src->vk.extent.height, dst->vk.view_format, (void *)src);
   }
   simple_mtx_unlock(&kk_fdm_lock);
   VK_FROM_HANDLE(kk_image_view, hidden, src->implicit_resolve_view);
   return hidden->vk.view_format == dst->vk.view_format ? hidden : NULL;
}

void
kk_implicit_image_finish(struct kk_device *dev, struct kk_image *image)
{
   if (image->metal_imported)
      kk_implicit_forget_import(image);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkDevice device = kk_device_to_handle(dev);
   if (image->implicit_unwarp_image != VK_NULL_HANDLE) {
      disp->DestroyImage(device, image->implicit_unwarp_image, NULL);
      disp->FreeMemory(device, image->implicit_unwarp_memory, NULL);
      image->implicit_unwarp_image = VK_NULL_HANDLE;
   }
   if (image->implicit_resolve_view == VK_NULL_HANDLE)
      return;
   disp->DestroyImageView(device, image->implicit_resolve_view, NULL);
   disp->DestroyImage(device, image->implicit_resolve_image, NULL);
   disp->FreeMemory(device, image->implicit_resolve_memory, NULL);
   image->implicit_resolve_view = VK_NULL_HANDLE;
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
   kk_rate_registry_add(image);
   return VK_SUCCESS;
}

static void
kk_fdm_params_unref(struct kk_fdm_params *params)
{
   assert(params->refs > 0);
   if (--params->refs)
      return;
   if (params->map)
      mtl_release(params->map);
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
   if (image->rate_params) {
      kk_rate_registry_remove(image);
      kk_destroy_bo(dev, image->rate_params);
   }
   if (image->rate_ref) {
      simple_mtx_lock(&kk_fdm_lock);
      kk_fdm_params_unref(image->rate_ref);
      simple_mtx_unlock(&kk_fdm_lock);
      image->rate_ref = NULL;
   }
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
   params->map = mtl_retain(map);
   if (density->fdm.params)
      kk_fdm_params_unref(density->fdm.params);
   density->fdm.params = params;
   density->fdm.serial = ++kk_fdm_serial;
   params->serial = density->fdm.serial;
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
      /* Implicit foveation resolves into ordinary images after the pass, through the map. */
      if (!kk_fdm_subsampled(render->color_att[i].iview) ||
          (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE &&
           !kk_fdm_subsampled(render->color_att[i].resolve_iview) &&
           !kk_implicit_fdm_enabled()))
         return false;
   }
   return kk_fdm_depth_compatible(render, &render->depth_att) &&
          kk_fdm_depth_compatible(render, &render->stencil_att) &&
          (render->depth_att.resolve_mode == VK_RESOLVE_MODE_NONE ||
           kk_implicit_fdm_enabled() ||
           kk_fdm_subsampled(render->depth_att.resolve_iview)) &&
          (render->stencil_att.resolve_mode == VK_RESOLVE_MODE_NONE ||
           kk_implicit_fdm_enabled() ||
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


/* The driver's own compute-encoder writes of rate parameters (copies, clears) are read
 * by the next passes. With light barriers a render encoder starts with only the
 * accumulated barrier, so add one for them. */
static void
kk_fdm_order_for_next_encoder(struct kk_cmd_buffer *cmd)
{
   if (KK_EXPERIMENTAL(LIGHT_BARRIERS)) {
      cmd->state.gfx.pending_barrier_src |= MTL_STAGE_DISPATCH | MTL_STAGE_BLIT;
      cmd->state.gfx.pending_barrier_dst |=
         MTL_STAGE_VERTEX | MTL_STAGE_FRAGMENT | MTL_STAGE_DISPATCH | MTL_STAGE_BLIT;
   }
}

static void
kk_fdm_publish(struct kk_cmd_buffer *cmd, struct kk_image *density,
               struct kk_fdm_params *params, struct kk_image_view *iview)
{
   if (iview == NULL)
      return;
   struct kk_image *image = container_of(iview->vk.image, struct kk_image, vk);
   if (image->rate_params == NULL)
      return;
   uint32_t bytes = MIN2(density->fdm.params_bytes, KK_RATE_PARAMS_SIZE_B);
   /* Always written in GPU order. Whether the image "already has" these parameters is
    * known only at record time; rewriting them from the CPU then (even the same bytes)
    * lands before a still-queued copy of other parameters takes effect, so work recorded
    * in between reads the wrong map (Talos's right eye was unwarped through the map its
    * left eye had moved on to). */
   if (cmd->metal.render != NULL)
      cs_end(cmd);
   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   mtl_copy_from_buffer_to_buffer(encoder, params->bo->map, 0, image->rate_params->map, 0,
                                  bytes);
   /* Later work in this encoder (dispatches, the driver's blits) reads them too. */
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_BLIT, MTL_STAGE_DISPATCH | MTL_STAGE_BLIT);
   kk_fdm_order_for_next_encoder(cmd);
   if (image->rate_ref != params) {
      params->refs++;
      if (image->rate_ref)
         kk_fdm_params_unref(image->rate_ref);
      image->rate_ref = params;
   }
   image->rate_params_owner = density;
   image->rate_params_serial = params->serial;
}

static uint64_t
kk_metal_foveation_setup(struct kk_cmd_buffer *cmd,
                         mtl_render_pass_descriptor *descriptor,
                         VkExtent2D screen)
{
   struct kk_rendering_state *render = &cmd->state.gfx.render;
   if (!debug_get_bool_option("KK_METAL_FOVEATION", false))
      return 0;
   void *map = NULL, *mapped = NULL;
   for (uint32_t i = 0; i < render->color_att_count && !map; i++) {
      struct kk_image_view *views[] = {render->color_att[i].iview, render->color_att[i].resolve_iview};
      for (uint32_t j = 0; j < ARRAY_SIZE(views) && !map; j++)
         if (views[j]) {
            mapped = views[j]->planes[0].mtl_handle_render;
            map = mtl_texture_copy_rate_map(mapped);
         }
   }
   if (!map)
      return 0;
#ifdef THRESHOLD_SHARED_FOVEATION
   /* This path leaves the image warped for a consumer that reconstructs it with this map.
    * A runtime that publishes maps on its images for drivers to foveate their own
    * targets (MNDX_foveation_metal version 7) also publishes the profile, and expects
    * ordinary pixels in the image: such a map is not for rendering into it. */
   typedef int (*copy_profile_fn)(void *, struct tf_image_profile *);
   static copy_profile_fn copy_profile;
   static bool looked;
   if (!looked) {
      copy_profile = (copy_profile_fn)dlsym(RTLD_DEFAULT, "tf_metal_copy_texture_profile");
      looked = true;
   }
   struct tf_image_profile profile = {.struct_size = sizeof(profile)};
   if (copy_profile && copy_profile(mapped, &profile) == 0) {
      static bool reported;
      if (!reported) {
         reported = true;
         mesa_logi("kk-monado: the runtime's image maps are for implicit foveation; "
                   "rendering into its images at full rate");
      }
      mtl_release(map);
      return 0;
   }
#endif
   struct kk_image_view *view = render->color_att[0].iview;
   if (render->color_att_count != 1 || !view || render->samples != 1 ||
       render->view_mask || render->layer_count != 1 ||
       render->color_att[0].resolve_mode != VK_RESOLVE_MODE_NONE ||
       view->vk.base_mip_level || view->vk.base_array_layer || view->vk.layer_count != 1 ||
       view->vk.image->extent.width != screen.width ||
       view->vk.image->extent.height != screen.height) {
      mesa_loge("kk-monado: unsupported render pass targets a rate-mapped image");
      mtl_release(map);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }
   uint64_t bytes = mtl_rate_map_parameter_size(map);
   struct kk_fdm_params *params = calloc(1, sizeof(*params));
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   if (!params || bytes > KK_RATE_PARAMS_SIZE_B - KK_RATE_PARAMS_HEADER_B) {
      free(params);
      mtl_release(map);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return 0;
   }
   VkResult result = kk_alloc_bo(dev, &dev->vk.base, KK_RATE_PARAMS_SIZE_B, 0, &params->bo);
   if (result != VK_SUCCESS) {
      free(params);
      mtl_release(map);
      vk_command_buffer_set_error(&cmd->vk, result);
      return 0;
   }
   params->dev = dev;
   params->refs = 1;
   struct kk_rate_params_header *header = params->bo->cpu;
   memset(header, 0, KK_RATE_PARAMS_HEADER_B);
   mtl_rate_map_copy_parameters(map, params->bo->map, KK_RATE_PARAMS_HEADER_B);
   header->layers = 1;
   header->screen[0] = screen.width;
   header->screen[1] = screen.height;
   mtl_rate_map_physical_size(map, 0, &header->physical[0][0], &header->physical[0][1]);
   header->valid = 1;
   mtl_render_pass_descriptor_set_rate_map(descriptor, map);
   mtl_release(map);
   kk_fdm_cmd_ref(cmd, params);
   uint64_t address = params->bo->gpu;
   if (kk_fdm_debug()) {
      static unsigned logged;
      if (logged++ < 8)
         mesa_logi("kk-monado: rate mapped pass %ux%u physical %ux%u", screen.width,
                   screen.height, header->physical[0][0], header->physical[0][1]);
   }
   kk_fdm_params_unref(params);
   return address;
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

   const VkRenderingFragmentDensityMapAttachmentInfoEXT *fdm_info =
      vk_find_struct_const(pRenderingInfo->pNext,
                           RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO_EXT);
   bool explicit_fdm_present = fdm_info != NULL || gfx->fdm.carry;
   if (KK_EXPERIMENTAL(FDM_RATE)) {
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

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].iview)
         kk_implicit_trace(render->color_att[i].resolve_iview ? "render colour+resolve" : "render colour",
                           render->color_att[i].iview->vk.image,
                           render->color_att[i].resolve_iview ? render->color_att[i].resolve_iview->vk.image : NULL);
   }
   if (render->depth_att.iview)
      kk_implicit_trace("render depth", render->depth_att.iview->vk.image,
                        render->depth_att.resolve_iview ? render->depth_att.resolve_iview->vk.image : NULL);

   bool implicit_density = false;
   simple_mtx_lock(&kk_fdm_lock);
   if (!density && kk_implicit_fdm_enabled() && !gfx->in_custom_resolve && !kk_implicit_raw_active() &&
       !kk_implicit_nomap()) {
      const struct kk_image_view *first =
         render->color_att_count && render->color_att[0].iview ? render->color_att[0].iview
                                                               : render->depth_att.iview;
      struct kk_image *target = first ? container_of(first->vk.image, struct kk_image, vk) : NULL;
      /* Every pass on a foveated image, whatever its render area: an unmapped pass
       * would write the image in a second layout. */
      if (target && target->implicit_fdm) {
         const struct kk_attachment *att = render->color_att_count && render->color_att[0].iview
                                              ? &render->color_att[0] : &render->depth_att;
         density = kk_implicit_density(dev, target, att->load_op != VK_ATTACHMENT_LOAD_OP_LOAD);
         implicit_density = density != NULL;
      }
   }
   void *map = density ? kk_fdm_get_map(cmd, density, screen, render->area) : NULL;
   /* A pass that loads an attachment drawn under an older map renders in that map, so
    * the old contents and the new drawing share one layout; the image takes the new map
    * when a pass next rewrites it without loading (usually the next frame). */
   struct kk_fdm_params *use = map ? density->fdm.params : NULL;
   if (map && implicit_density) {
      struct kk_fdm_params *older = NULL;
      bool mixed = false;
      for (uint32_t i = 0; i <= render->color_att_count; i++) {
         const struct kk_attachment *att =
            i < render->color_att_count ? &render->color_att[i] : &render->depth_att;
         if (!att->iview || att->load_op != VK_ATTACHMENT_LOAD_OP_LOAD)
            continue;
         struct kk_image *img = container_of(att->iview->vk.image, struct kk_image, vk);
         if (!img->rate_ref || img->rate_ref == use)
            continue;
         if (older && older != img->rate_ref)
            mixed = true;
         else
            older = img->rate_ref;
      }
      if (older && older->map) {
         use = older;
         map = older->map;
      }
      if (mixed) {
         static uint32_t logged;
         if (logged++ < 8)
            mesa_logw("kk-implicit: a pass loads attachments drawn under different maps");
      }
   }
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
      kk_fdm_cmd_ref(cmd, use);
      params = use->bo->gpu;
      for (uint32_t i = 0; publish && i < render->color_att_count; i++) {
         kk_fdm_publish(cmd, density, use, render->color_att[i].iview);
         kk_fdm_publish(cmd, density, use, render->color_att[i].resolve_iview);
      }
      if (publish) {
         kk_fdm_publish(cmd, density, use, render->depth_att.iview);
         kk_fdm_publish(cmd, density, use, render->depth_att.resolve_iview);
         kk_fdm_publish(cmd, density, use, render->stencil_att.iview);
         kk_fdm_publish(cmd, density, use, render->stencil_att.resolve_iview);
      }
   }
#ifdef THRESHOLD_SHARED_FOVEATION
   if (tf_select_source(explicit_fdm_present, true, false) == TF_SOURCE_RUNTIME)
#else
   if (!explicit_fdm_present)
#endif
      params = kk_metal_foveation_setup(cmd, descriptor, screen);
   simple_mtx_unlock(&kk_fdm_lock);

   struct kk_descriptor_state *desc = &gfx->descriptors;
   if (desc->root.fdm_rate_params != params) {
      desc->root.fdm_rate_params = params;
      desc->root_dirty = true;
   }
}


/* Diagnostic (KK_IMPLICIT_TRACE=1): each distinct image operation an application
 * performs, once, to plan foveation of applications that do not ask for it. */
void
kk_implicit_trace(const char *what, const struct vk_image *a, const struct vk_image *b)
{
   for (uint32_t i = 0; i < ARRAY_SIZE(kk_watched_res_traced) && kk_watched_res_traced[i]; i++) {
      const struct vk_image *w = &kk_watched_res_traced[i]->vk;
      static uint32_t told;
      if ((a == w || b == w) && told < 64) {
         told++;
         mesa_logi("kk-res: %s %p fmt %d %ux%u -> %p fmt %d %ux%u", what, (const void *)a,
                   a ? a->format : 0, a ? a->extent.width : 0, a ? a->extent.height : 0,
                   (const void *)b, b ? b->format : 0, b ? b->extent.width : 0,
                   b ? b->extent.height : 0);
      }
   }
   static int enabled = -1;
   if (enabled < 0) {
      enabled = debug_get_bool_option("KK_IMPLICIT_TRACE", false);
      /* Or a file named after the capture trigger file plus ".trace". */
      const char *trigger = getenv("KK_CAPTURE_TRIGGER_FILE");
      char path[1024];
      if (!enabled && trigger) {
         snprintf(path, sizeof(path), "%s.trace", trigger);
         enabled = access(path, F_OK) == 0;
      }
      const char *session = getenv("XRT_SESSION_RECORD_DIR");
      if (!enabled && session) {
         snprintf(path, sizeof(path), "%s/trace", session);
         enabled = access(path, F_OK) == 0;
      }
   }
   if (!enabled)
      return;
   {
      /* KK_IMPLICIT_SEQ=N: also the 60 consecutive operations from the Nth, in order. */
      static int64_t first = -2;
      static uint32_t calls;
      if (first == -2)
         first = debug_get_num_option("KK_IMPLICIT_SEQ", -1);
      uint32_t n = p_atomic_inc_return(&calls);
      if (first >= 0 && n >= first && n < first + 400) {
         const struct kk_image *ka = a ? container_of(a, struct kk_image, vk) : NULL;
         mesa_logi("kk-seqop %u: %s %p s%u %ux%u map %llu latched %llu -> %p", n, what,
                   (const void *)a, a ? a->samples : 0, a ? a->extent.width : 0,
                   a ? a->extent.height : 0,
                   ka && ka->rate_ref ? (unsigned long long)ka->rate_ref->serial : 0ull,
                   (unsigned long long)kk_implicit.latched.serial, (const void *)b);
      }
   }
   static struct { const char *what; const void *a, *b; uint32_t n; } seen[512];
   static uint32_t count;
   simple_mtx_lock(&kk_fdm_lock);
   for (uint32_t i = 0; i < count; i++) {
      if (seen[i].what == what && seen[i].a == a && seen[i].b == b) {
         seen[i].n++;
         simple_mtx_unlock(&kk_fdm_lock);
         return;
      }
   }
   if (count < ARRAY_SIZE(seen)) {
      seen[count].what = what; seen[count].a = a; seen[count].b = b; seen[count].n = 1;
      count++;
      char sb[96] = "";
      if (b)
         snprintf(sb, sizeof(sb), " -> %p fmt %d s%u %ux%ux%u usage 0x%x", (const void *)b, b->format,
                  b->samples, b->extent.width, b->extent.height, b->array_layers, (unsigned)b->usage);
      mesa_logi("kk-implicit: %s %p fmt %d s%u %ux%ux%u usage 0x%x%s", what, (const void *)a,
                a ? a->format : 0, a ? a->samples : 0, a ? a->extent.width : 0, a ? a->extent.height : 0,
                a ? a->array_layers : 0, a ? (unsigned)a->usage : 0, sb);
   }
   simple_mtx_unlock(&kk_fdm_lock);
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
   kk_fdm_order_for_next_encoder(cmd);
   image->rate_params_owner = NULL;
   image->rate_params_serial = 0;
   if (image->rate_ref) {
      simple_mtx_lock(&kk_fdm_lock);
      kk_fdm_params_unref(image->rate_ref);
      simple_mtx_unlock(&kk_fdm_lock);
      image->rate_ref = NULL;
   }
}

void
kk_fdm_carry_to_next_rendering(struct kk_cmd_buffer *cmd)
{
   cmd->state.gfx.fdm.carry = cmd->state.gfx.fdm.density != NULL;
}

VkImage
kk_implicit_unwarp_for_copy(struct kk_cmd_buffer *cmd, struct kk_image *src,
                            VkImageLayout src_layout, const struct kk_image *dst)
{
   if (!src->implicit_fdm || src->vk.samples != VK_SAMPLE_COUNT_1_BIT)
      return VK_NULL_HANDLE;
   /* Between two foveated images of one size the layouts agree: copy as stored. */
   if (dst && dst->implicit_fdm && dst->vk.extent.width == src->vk.extent.width &&
       dst->vk.extent.height == src->vk.extent.height)
      return VK_NULL_HANDLE;
   if (vk_format_is_depth_or_stencil(src->vk.format)) {
      if (!src->implicit_unwarp_logged) {
         src->implicit_unwarp_logged = true;
         mesa_logw("kk-implicit: raw copy of foveated depth image %p is not unwarped", (void *)src);
      }
      return VK_NULL_HANDLE;
   }
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkDevice device = kk_device_to_handle(dev);
   simple_mtx_lock(&kk_fdm_lock);
   if (src->implicit_unwarp_image == VK_NULL_HANDLE) {
      const VkImageCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = src->vk.format,
         .extent = src->vk.extent,
         .mipLevels = 1,
         .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      VkImage image = VK_NULL_HANDLE;
      VkDeviceMemory memory = VK_NULL_HANDLE;
      kk_implicit_internal_image = true;
      VkResult result = disp->CreateImage(device, &info, NULL, &image);
      kk_implicit_internal_image = false;
      if (result == VK_SUCCESS) {
         VkMemoryRequirements2 reqs = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
         const VkImageMemoryRequirementsInfo2 reqs_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = image};
         disp->GetImageMemoryRequirements2(device, &reqs_info, &reqs);
         const VkMemoryAllocateInfo alloc = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = reqs.memoryRequirements.size,
            .memoryTypeIndex = ffs(reqs.memoryRequirements.memoryTypeBits) - 1,
         };
         result = disp->AllocateMemory(device, &alloc, NULL, &memory);
      }
      if (result == VK_SUCCESS) {
         const VkBindImageMemoryInfo bind = {
            .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, .image = image, .memory = memory};
         result = disp->BindImageMemory2(device, 1, &bind);
      }
      if (result != VK_SUCCESS) {
         if (image != VK_NULL_HANDLE)
            disp->DestroyImage(device, image, NULL);
         if (memory != VK_NULL_HANDLE)
            disp->FreeMemory(device, memory, NULL);
         simple_mtx_unlock(&kk_fdm_lock);
         mesa_logw("kk-implicit: unwarp image failed (%d)", result);
         return VK_NULL_HANDLE;
      }
      src->implicit_unwarp_image = image;
      src->implicit_unwarp_memory = memory;
      mesa_logi("kk-implicit: unwarping copies of %ux%u format %d image %p", src->vk.extent.width,
                src->vk.extent.height, src->vk.format, (void *)src);
   }
   VkImage unwarp = src->implicit_unwarp_image;
   simple_mtx_unlock(&kk_fdm_lock);
   /* The driver's blit samples `src` through its rate map (every sampled read is
    * translated in implicit mode) and renders an ordinary image. */
   const VkImageBlit2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0}, {(int32_t)src->vk.extent.width, (int32_t)src->vk.extent.height, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0}, {(int32_t)src->vk.extent.width, (int32_t)src->vk.extent.height, 1}},
   };
   const VkBlitImageInfo2 blit = {
      .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage = kk_image_to_handle(src),
      .srcImageLayout = src_layout,
      .dstImage = unwarp,
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &region,
      .filter = VK_FILTER_NEAREST,
   };
   kk_CmdBlitImage2(kk_cmd_buffer_to_handle(cmd), &blit);
   /* Diagnostic: which command buffer each unwarp of a source lands in, first 60. */
   if (getenv("XRT_SESSION_RECORD_DIR")) {
      static struct { const void *src, *cmd, *dst; } seen[256];
      static uint32_t count;
      bool known = false;
      for (uint32_t i = 0; i < count; i++)
         known |= seen[i].src == src && seen[i].cmd == cmd && seen[i].dst == dst;
      if (!known && count < ARRAY_SIZE(seen))
         seen[count++] = (__typeof__(seen[0])){src, cmd, dst};
      if (!known && count < ARRAY_SIZE(seen))
         mesa_logi("kk-unwarp: src %p fmt %d into %p, command buffer %p, dst %p, at %llu ms",
                   (void *)src, src->vk.format, (void *)unwarp, (void *)cmd, (const void *)dst,
                   (unsigned long long)(os_time_get_nano() / 1000000ull));
   }
   return unwarp;
}

bool
kk_implicit_storage_enabled(void)
{
   /* KK_IMPLICIT_STORAGE=1: storage images can be foveated; their loads, stores and
    * atomics are translated (kk_nir_lower_descriptors.c). */
   static int storage = -1;
   if (storage < 0)
      storage = debug_get_bool_option("KK_IMPLICIT_STORAGE", false);
   return storage;
}

static _Thread_local bool kk_implicit_raw;
bool kk_implicit_raw_active(void) { return kk_implicit_raw; }
void kk_implicit_set_raw(bool raw) { kk_implicit_raw = raw; }

static const char *
kk_implicit_why_not(const struct kk_image *image, const struct kk_image_view *view)
{
   if (image->implicit_poison)
      return "shares a pass with an ordinary attachment";
   if (!image->implicit_candidate)
      return image->vk.array_layers != 1 ? "array" :
             ((image->vk.usage & VK_IMAGE_USAGE_STORAGE_BIT) && !kk_implicit_storage_enabled())
                ? "storage usage" :
             !(image->vk.usage & (VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) ? "not sampled" :
             "not a candidate";
   if (!kk_implicit_eyes(image))
      return "not eye sized";
   if (view->vk.base_mip_level || view->vk.base_array_layer)
      return "renders a mip level or layer";
   return NULL;
}

void
kk_implicit_match_pass(struct kk_cmd_buffer *cmd, struct kk_rendering_state *render)
{
   static int single = -1;
   if (single < 0)
      single = debug_get_bool_option("KK_IMPLICIT_SINGLE", false);
   struct kk_image_view *views[KK_MAX_RTS + 2];
   uint32_t n = 0;
   for (uint32_t i = 0; i < render->color_att_count; i++)
      if (render->color_att[i].iview)
         views[n++] = render->color_att[i].iview;
   if (render->depth_att.iview)
      views[n++] = render->depth_att.iview;
   if (render->stencil_att.iview && render->stencil_att.iview != render->depth_att.iview)
      views[n++] = render->stencil_att.iview;
   if (!single) {
      /* The multisampled route: each attachment on its own (resolves are redirected). */
      for (uint32_t i = 0; i < n; i++)
         kk_implicit_fdm_match(container_of(views[i]->vk.image, struct kk_image, vk));
      return;
   }
   if (kk_implicit_raw)
      return;
   /* All attachments together or none: an ordinary pass that writes or tests against a
    * foveated image would leave it in two layouts. */
   const char *why = NULL;
   const struct kk_image *blocker = NULL;
   bool any = false;
   simple_mtx_lock(&kk_fdm_lock);
   /* The first profile may still be waiting on the runtime's images: read it here, since
    * this route matches no attachment until a profile exists. */
   if (kk_implicit.automatic && !kk_implicit.pending.serial)
      kk_implicit_poll_runtime_locked();
   const bool have_profile = kk_implicit.pending.serial != 0;
   simple_mtx_unlock(&kk_fdm_lock);
   for (uint32_t i = 0; i < n; i++) {
      const struct kk_image *image = container_of(views[i]->vk.image, struct kk_image, vk);
      any |= image->implicit_candidate || image->implicit_fdm;
      const char *w = kk_implicit_why_not(image, views[i]);
      if (w && !why) {
         why = w;
         blocker = image;
      }
   }
   if (!any && have_profile) {
      /* Diagnostic: eye-sized attachments that were never candidates, once each. */
      for (uint32_t i = 0; i < n; i++) {
         struct kk_image *image = container_of(views[i]->vk.image, struct kk_image, vk);
         if (!kk_implicit_eyes(image) || image->implicit_unwarp_logged)
            continue;
         image->implicit_unwarp_logged = true;
         mesa_logi("kk-implicit: eye-sized image %p fmt %d is not a candidate: usage 0x%x flags 0x%x "
                   "mips %u layers %u samples %u", (void *)image, image->vk.format,
                   (unsigned)image->vk.usage, (unsigned)image->vk.create_flags,
                   image->vk.mip_levels, image->vk.array_layers, (unsigned)image->vk.samples);
      }
   }
   if (!any || !have_profile)
      return;
   if (!why) {
      for (uint32_t i = 0; i < n; i++)
         kk_implicit_fdm_match(container_of(views[i]->vk.image, struct kk_image, vk));
      return;
   }
   for (uint32_t i = 0; i < n; i++) {
      struct kk_image *image = container_of(views[i]->vk.image, struct kk_image, vk);
      if (!image->implicit_candidate || image->implicit_poison)
         continue;
      image->implicit_poison = true;
      mesa_logi("kk-implicit: not foveating %ux%u fmt %d image %p: pass with %ux%u fmt %d "
                "image %p (%s)%s", image->vk.extent.width, image->vk.extent.height,
                image->vk.format, (void *)image, blocker->vk.extent.width,
                blocker->vk.extent.height, blocker->vk.format, (void *)blocker, why,
                image->implicit_fdm ? "; was foveated, now ordinary" : "");
      if (image->implicit_fdm) {
         image->implicit_fdm = false;
         kk_fdm_clear_image_mapping(cmd, image);
      }
   }
}

bool
kk_implicit_compact_enabled(void)
{
   static int compact = -1;
   if (compact < 0)
      compact = debug_get_bool_option("KK_IMPLICIT_COMPACT", false);
   return compact;
}

static void
kk_compact_log(const struct kk_shader *cs, const char *fmt, ...)
{
   static uint32_t logged;
   if (!debug_get_bool_option("KK_IMPLICIT_COMPACT_LOG", false) || logged >= 200)
      return;
   /* Once per shader and outcome. */
   static struct { const void *cs; char what[96]; } seen[256];
   char buf[96];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   for (uint32_t i = 0; i < MIN2(logged, ARRAY_SIZE(seen)); i++)
      if (seen[i].cs == cs && !strcmp(seen[i].what, buf))
         return;
   if (logged < ARRAY_SIZE(seen)) {
      seen[logged].cs = cs;
      snprintf(seen[logged].what, sizeof(seen[logged].what), "%s", buf);
   }
   logged++;
   mesa_logi("kk: compact dispatch, shader %p (%ux%u): %s", (const void *)cs,
             (unsigned)cs->info.cs.local_size.x, (unsigned)cs->info.cs.local_size.y, buf);
}

/* KK_IMPLICIT_COMPACT_FILE=path (diagnostic): read at a present at most once a second,
 * the file's words switch behaviour while the application runs, for A/B comparisons in
 * one scene: "off", "map" (map compute-only targets, dispatch at full size) or
 * "compact" (the default) for compaction; "lb"/"nolb" turn light barriers on or off;
 * "quarter" lets scaled targets from 25% match (default 40%; images already judged
 * not eye-sized stay ordinary). At 25% NMS showed a hard horizontal lighting edge. */
static int kk_compact_runtime = 2;

void
kk_fdm_runtime_poll(void)
{
   static const char *path;
   static int init;
   static uint64_t last_ns;
   if (!init) {
      path = getenv("KK_IMPLICIT_COMPACT_FILE");
      init = 1;
   }
   if (!path)
      return;
   uint64_t now = os_time_get_nano();
   if (now - last_ns < 1000000000ull)
      return;
   last_ns = now;
   char text[256] = {0};
   FILE *f = fopen(path, "r");
   if (!f)
      return;
   size_t n = fread(text, 1, sizeof(text) - 1, f);
   fclose(f);
   text[n] = 0;
   const int mode = strstr(text, "off") ? 0 : strstr(text, "map") ? 1 : 2;
   const bool lb = strstr(text, "nolb") ? false : strstr(text, "lb") ? true
                 : KK_EXPERIMENTAL(LIGHT_BARRIERS);
   const float min = strstr(text, "quarter") && !strstr(text, "noquarter") ? 0.25f : 0.4f;
   if (mode != kk_compact_runtime || lb != !!KK_EXPERIMENTAL(LIGHT_BARRIERS) ||
       min != kk_scaled_min)
      mesa_logi("kk: runtime mode: compaction %s, light barriers %s, scaled from %.0f%%",
                mode == 0 ? "off" : mode == 1 ? "map" : "compact", lb ? "on" : "off",
                min * 100.0f);
   kk_compact_runtime = mode;
   kk_scaled_min = min;
   if (lb)
      kk_mesa_experimental_flags |= KK_EXPERIMENTAL_LIGHT_BARRIERS;
   else
      kk_mesa_experimental_flags &= ~KK_EXPERIMENTAL_LIGHT_BARRIERS;
}

static int
kk_compact_mode(void)
{
   return kk_compact_runtime;
}

uint64_t
kk_implicit_compact_dispatch(struct kk_cmd_buffer *cmd, const struct kk_shader *cs,
                             uint32_t *group_x, uint32_t *group_y, uint32_t group_z)
{
   if (!cs || !cs->info.cs.compact_count || !kk_implicit_compact_enabled() ||
       group_z * cs->info.cs.local_size.z != 1)
      return 0;
   const int mode = kk_compact_mode();
   if (mode == 0)
      return 0;
   const struct kk_descriptor_state *desc = &cmd->state.cs.descriptors;
   uint64_t params[4];
   bool reads[4];
   uint32_t count = 0;
   for (uint32_t i = 0; i < cs->info.cs.compact_count; i++) {
      const uint32_t set = cs->info.cs.compact[i].set;
      const uint32_t binding = cs->info.cs.compact[i].binding;
      const uint32_t index = cs->info.cs.compact[i].index;
      const uint8_t *base = NULL;
      const struct kk_descriptor_set_layout *layout = NULL;
      if (set < KK_MAX_SETS && desc->sets[set]) {
         base = desc->sets[set]->mapped_ptr;
         layout = desc->sets[set]->layout;
      } else if (set < KK_MAX_SETS && desc->push[set]) {
         base = desc->push[set]->data;
         layout = desc->push[set]->layout;
      }
      if (!base || !layout || binding >= layout->binding_count)
         return 0;
      const struct kk_descriptor_set_binding_layout *bl = &layout->binding[binding];
      if (bl->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || index >= bl->array_size)
         return 0;
      const struct kk_storage_image_descriptor *d =
         (const void *)(base + bl->offset + index * bl->stride);
      if (!d->rate_params) {
         kk_compact_log(cs, "writes an ordinary image (set %u binding %u)", set, binding);
         return 0;
      }
      uint32_t j;
      for (j = 0; j < count && params[j] != d->rate_params; j++)
         ;
      if (j == count) {
         params[count] = d->rate_params;
         reads[count++] = false;
      }
      reads[j] |= cs->info.cs.compact[i].reads;
   }

   /* Every written image must hold the same map (the same density map and serial),
    * so one invocation per physical texel covers each of them. */
   uint32_t screen[2] = {0}, physical[2] = {0};
   const struct kk_image *shared_density = NULL;
   uint64_t shared_serial = 0;
   const struct kk_image *image = NULL;
   simple_mtx_lock(&kk_rate_registry_lock);
   for (uint32_t i = 0; i < count; i++) {
      struct kk_image *img = kk_rate_registry_find_locked(params[i]);
      image = img;
      const struct kk_image *density = img ? img->rate_params_owner : NULL;
      /* A foveated image with no map yet, which the dispatch rewrites whole without
       * reading: give it the profile's map here, as a pass that does not load its
       * target would. Its readers translate through the published parameters. */
      if (img && !density && !reads[i] && kk_implicit_fdm_match(img) &&
          *group_x == DIV_ROUND_UP(img->vk.extent.width, cs->info.cs.local_size.x) &&
          *group_y == DIV_ROUND_UP(img->vk.extent.height, cs->info.cs.local_size.y)) {
         struct kk_device *dev = kk_cmd_buffer_device(cmd);
         simple_mtx_lock(&kk_fdm_lock);
         struct kk_image *d = kk_implicit_density(dev, img, true);
         const VkExtent2D extent = {img->vk.extent.width, img->vk.extent.height};
         const VkRect2D area = {{0, 0}, extent};
         void *map = d ? kk_fdm_get_map(cmd, d, extent, area) : NULL;
         if (map && d->fdm.params) {
            kk_fdm_cmd_ref(cmd, d->fdm.params);
            mtl_compute_encoder *encoder = cs_get_compute(cmd);
            uint32_t bytes = MIN2(d->fdm.params_bytes, KK_RATE_PARAMS_SIZE_B);
            mtl_copy_from_buffer_to_buffer(encoder, d->fdm.params->bo->map, 0,
                                           img->rate_params->map, 0, bytes);
            mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_BLIT, MTL_STAGE_DISPATCH);
            kk_fdm_order_for_next_encoder(cmd);
            img->rate_params_owner = d;
            img->rate_params_serial = d->fdm.serial;
            if (img->rate_ref != d->fdm.params) {
               d->fdm.params->refs++;
               if (img->rate_ref)
                  kk_fdm_params_unref(img->rate_ref);
               img->rate_ref = d->fdm.params;
            }
            density = d;
            kk_compact_log(cs, "maps %ux%u", extent.width, extent.height);
         }
         simple_mtx_unlock(&kk_fdm_lock);
      }
      const char *why = NULL;
      if (!img)
         why = "unknown";
      else if (!density)
         why = "has no map yet";
      else if (i && (density != shared_density || img->rate_params_serial != shared_serial))
         why = "has a different map from another written image";
      else {
         simple_mtx_lock(&kk_fdm_lock);
         if (density->fdm.params && img->rate_params_serial == density->fdm.serial) {
            const struct kk_rate_params_header *h = density->fdm.params->bo->cpu;
            if (h->valid && h->layers == 1) {
               screen[0] = h->screen[0];
               screen[1] = h->screen[1];
               physical[0] = h->physical[0][0];
               physical[1] = h->physical[0][1];
               shared_density = density;
               shared_serial = density->fdm.serial;
            } else {
               why = "has a layered map";
            }
         } else {
            why = "has an old map";
         }
         simple_mtx_unlock(&kk_fdm_lock);
      }
      if (why) {
         simple_mtx_unlock(&kk_rate_registry_lock);
         kk_compact_log(cs, "image %ux%u %s", img ? img->vk.extent.width : 0,
                        img ? img->vk.extent.height : 0, why);
         return 0;
      }
   }
   simple_mtx_unlock(&kk_rate_registry_lock);
   if (count > 1)
      kk_compact_log(cs, "writes %u foveated images with one map", count);
   /* A shader that also reads what it writes (loads or atomics) is often temporal
    * accumulation; at one invocation per physical texel NMS's showed ghosting and
    * banding. KK_IMPLICIT_COMPACT_READS=1 compacts it anyway (diagnostic). */
   bool any_reads = false;
   for (uint32_t i = 0; i < count; i++)
      any_reads |= reads[i];
   if (any_reads && !debug_get_bool_option("KK_IMPLICIT_COMPACT_READS", false)) {
      kk_compact_log(cs, "reads an image it writes (%s, %u texture samples, %u images)",
                     cs->info.cs.compact_own_texel ? "own texel only" : "other texels too",
                     cs->info.cs.compact_samples, count);
      return 0;
   }

   const uint32_t lx = cs->info.cs.local_size.x, ly = cs->info.cs.local_size.y;
   if (*group_x != DIV_ROUND_UP(screen[0], lx) || *group_y != DIV_ROUND_UP(screen[1], ly)) {
      kk_compact_log(cs, "grid %ux%u does not cover %ux%u (image %ux%u)", *group_x * lx,
                     *group_y * ly, screen[0], screen[1], image->vk.extent.width,
                     image->vk.extent.height);
      return 0;
   }
   if (mode == 1)
      return 0;
   kk_compact_log(cs, "%ux%u -> %ux%u", screen[0], screen[1], physical[0], physical[1]);
   *group_x = DIV_ROUND_UP(physical[0], lx);
   *group_y = DIV_ROUND_UP(physical[1], ly);
   return params[0];
}

/* KK_FAKE_VRS logging. */
#define KK_VRS_KEYS 256
static struct {
   simple_mtx_t lock;
   struct { char key[160]; uint64_t count; uint32_t w, h; } e[KK_VRS_KEYS];
   uint32_t n;
   uint64_t frames, last_ns;
   struct kk_vrs_capture { struct kk_bo *bo; struct kk_device *dev; uint32_t w, h;
                           uint64_t at_ns; char name[64]; } cap[16];
   uint32_t cap_n, cap_done;
   struct { uint32_t w, h, seen, taken; } cap_sizes[8];
   uint32_t cap_size_n;
} kk_vrs = {.lock = SIMPLE_MTX_INITIALIZER};

static bool
kk_vrs_enabled(void)
{
   static int on = -1;
   if (on < 0)
      on = debug_get_bool_option("KK_FAKE_VRS", false);
   return on;
}

void
kk_vrs_note_draw(struct kk_cmd_buffer *cmd)
{
   if (!kk_vrs_enabled())
      return;
   const struct vk_fragment_shading_rate_state *fsr = &cmd->vk.dynamic_graphics_state.fsr;
   if (!vk_fragment_shading_rate_is_disabled(fsr))
      cmd->state.gfx.render.vrs_note.pipeline_rate = true;
}

static void
kk_vrs_capture(struct kk_cmd_buffer *cmd, struct kk_image_view *iview)
{
   const char *dir = getenv("KK_VRS_CAPTURE_DIR");
   if (!dir || kk_vrs.cap_n >= ARRAY_SIZE(kk_vrs.cap))
      return;
   struct kk_image *image = container_of(iview->vk.image, struct kk_image, vk);
   const uint32_t w = image->vk.extent.width, h = image->vk.extent.height;
   uint32_t i;
   for (i = 0; i < kk_vrs.cap_size_n; i++)
      if (kk_vrs.cap_sizes[i].w == w && kk_vrs.cap_sizes[i].h == h)
         break;
   if (i == kk_vrs.cap_size_n) {
      if (i == ARRAY_SIZE(kk_vrs.cap_sizes))
         return;
      kk_vrs.cap_sizes[kk_vrs.cap_size_n++] = (__typeof__(kk_vrs.cap_sizes[0])){w, h, 0, 0};
   }
   /* Two per size: early in play and later. */
   const uint32_t seen = ++kk_vrs.cap_sizes[i].seen;
   if (seen != 2000 && seen != 20000)
      return;
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_bo *bo = NULL;
   if (kk_alloc_bo(dev, &dev->vk.base, (uint64_t)w * h * image->vk.array_layers, 0, &bo) !=
       VK_SUCCESS)
      return;
   for (uint32_t l = 0; l < image->vk.array_layers; l++) {
      struct mtl_buffer_image_copy copy = {
         .image_size = {w, h, 1},
         .buffer = bo->map,
         .image = image->planes[0].mtl_handle,
         .buffer_offset_B = (size_t)l * w * h,
         .buffer_stride_B = w,
         .buffer_2d_image_size_B = (size_t)w * h,
         .image_slice = l,
      };
      mtl_copy_from_texture_to_buffer(cs_get_compute(cmd), &copy);
   }
   struct kk_vrs_capture *c = &kk_vrs.cap[kk_vrs.cap_n++];
   *c = (struct kk_vrs_capture){bo, dev, w, h * image->vk.array_layers,
                                os_time_get_nano(), ""};
   snprintf(c->name, sizeof(c->name), "vrs-%ux%u-%u.pgm", w, h, ++kk_vrs.cap_sizes[i].taken);
}

void
kk_vrs_end(struct kk_cmd_buffer *cmd)
{
   if (!kk_vrs_enabled())
      return;
   struct kk_rendering_state *r = &cmd->state.gfx.render;
   kk_vrs_note_draw(cmd);
   const struct kk_image_view *c = r->color_att_count ? r->color_att[0].iview : NULL;
   const struct kk_image_view *d = r->depth_att.iview;
   const struct kk_image_view *v = c ? c : d;
   const struct kk_image_view *rate = r->fsr_att.iview;
   char key[160];
   snprintf(key, sizeof(key), "%-13s %5ux%-5u c%d d%d s%u x%u%s | rate %ux%u texel %ux%u",
            rate && r->vrs_note.pipeline_rate ? "VRS+pipeline"
            : rate                            ? "VRS attachment"
            : r->vrs_note.pipeline_rate       ? "pipeline rate"
                                              : "-",
            v ? v->vk.image->extent.width : 0, v ? v->vk.image->extent.height : 0,
            c ? (int)c->vk.view_format : 0, d ? (int)d->vk.view_format : 0,
            v ? (unsigned)v->vk.image->samples : 0, r->color_att_count,
            v && v->vk.image->array_layers > 1 ? " layered" : "",
            rate ? rate->vk.image->extent.width : 0, rate ? rate->vk.image->extent.height : 0,
            r->vrs_note.texel.width, r->vrs_note.texel.height);
   simple_mtx_lock(&kk_vrs.lock);
   uint32_t i;
   for (i = 0; i < kk_vrs.n && strcmp(kk_vrs.e[i].key, key); i++)
      ;
   if (i == kk_vrs.n && i < KK_VRS_KEYS) {
      snprintf(kk_vrs.e[i].key, sizeof(kk_vrs.e[i].key), "%s", key);
      kk_vrs.e[i].w = v ? v->vk.image->extent.width : 0;
      kk_vrs.e[i].h = v ? v->vk.image->extent.height : 0;
      kk_vrs.n++;
   }
   if (i < kk_vrs.n)
      kk_vrs.e[i].count++;
   if (rate)
      kk_vrs_capture(cmd, (struct kk_image_view *)rate);
   simple_mtx_unlock(&kk_vrs.lock);
   r->vrs_note.pipeline_rate = false;
   r->vrs_note.texel = (VkExtent2D){0, 0};
}

void
kk_vrs_frame(void)
{
   if (!kk_vrs_enabled())
      return;
   simple_mtx_lock(&kk_vrs.lock);
   kk_vrs.frames++;
   const uint64_t now = os_time_get_nano();
   /* Captured rate images, two seconds after recording (their command buffers have
    * completed by then), as PGM with the rate code scaled for viewing (and .bin raw). */
   const char *dir = getenv("KK_VRS_CAPTURE_DIR");
   while (dir && kk_vrs.cap_done < kk_vrs.cap_n &&
          now - kk_vrs.cap[kk_vrs.cap_done].at_ns > 2000000000ull) {
      struct kk_vrs_capture *c = &kk_vrs.cap[kk_vrs.cap_done++];
      char path[512];
      snprintf(path, sizeof(path), "%s/%s", dir, c->name);
      FILE *f = fopen(path, "wb");
      if (f) {
         fprintf(f, "P5\n%u %u\n255\n", c->w, c->h);
         const uint8_t *src = c->bo->cpu;
         for (uint64_t p = 0; p < (uint64_t)c->w * c->h; p++)
            fputc(MIN2(src[p] * 25, 255), f);
         fclose(f);
      }
      snprintf(path, sizeof(path), "%s/%s.bin", dir, c->name);
      f = fopen(path, "wb");
      if (f) {
         fwrite(c->bo->cpu, 1, (size_t)c->w * c->h, f);
         fclose(f);
      }
      mesa_logi("kk-vrs: saved %s", c->name);
      kk_destroy_bo(c->dev, c->bo);
   }
   if (!kk_vrs.last_ns)
      kk_vrs.last_ns = now;
   if (now - kk_vrs.last_ns > 10000000000ull && kk_vrs.frames) {
      mesa_logi("kk-vrs: passes per frame over %llu frames (largest first):",
                (unsigned long long)kk_vrs.frames);
      uint32_t order[KK_VRS_KEYS];
      for (uint32_t i = 0; i < kk_vrs.n; i++)
         order[i] = i;
      for (uint32_t i = 0; i < kk_vrs.n; i++)
         for (uint32_t j = i + 1; j < kk_vrs.n; j++)
            if ((uint64_t)kk_vrs.e[order[j]].w * kk_vrs.e[order[j]].h >
                (uint64_t)kk_vrs.e[order[i]].w * kk_vrs.e[order[i]].h) {
               uint32_t x = order[i]; order[i] = order[j]; order[j] = x;
            }
      for (uint32_t i = 0; i < kk_vrs.n; i++)
         if (kk_vrs.e[order[i]].count)
            mesa_logi("kk-vrs: %6.1f  %s", (double)kk_vrs.e[order[i]].count / kk_vrs.frames,
                      kk_vrs.e[order[i]].key);
      for (uint32_t i = 0; i < kk_vrs.n; i++)
         kk_vrs.e[i].count = 0;
      kk_vrs.frames = 0;
      kk_vrs.last_ns = now;
   }
   simple_mtx_unlock(&kk_vrs.lock);
}

void
kk_implicit_copy_map(struct kk_cmd_buffer *cmd, struct kk_image *src, struct kk_image *dst)
{
   if (!src->rate_ref || !dst->rate_params || !dst->implicit_fdm || src == dst)
      return;
   simple_mtx_lock(&kk_fdm_lock);
   struct kk_fdm_params *params = src->rate_ref;
   /* The destination now holds the source's texels, laid out in the source's map. */
   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_BLIT, MTL_STAGE_BLIT);
   mtl_copy_from_buffer_to_buffer(encoder, params->bo->map, 0, dst->rate_params->map, 0,
                                  KK_RATE_PARAMS_SIZE_B);
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_BLIT, MTL_STAGE_DISPATCH | MTL_STAGE_BLIT);
   kk_fdm_order_for_next_encoder(cmd);
   kk_fdm_cmd_ref(cmd, params);
   if (dst->rate_ref != params) {
      params->refs++;
      if (dst->rate_ref)
         kk_fdm_params_unref(dst->rate_ref);
      dst->rate_ref = params;
   }
   dst->rate_params_owner = src->rate_params_owner;
   dst->rate_params_serial = params->serial;
   simple_mtx_unlock(&kk_fdm_lock);
}

/*
 * Diagnostic: for draws whose fragment shader's MSL hash is listed in a file named after
 * the capture trigger file plus ".watch" (hex hashes, as the MSL dump names them), log
 * the pass's target and every sampled image the bound sets give the shader: whether it
 * is foveated and which map its parameters carry. Shows a texture read in a layout its
 * parameters do not describe. At most 400 lines.
 */
static void kk_dump_note(struct kk_image *image, uint32_t mark);
static void kk_dump_arm(uint32_t mark);

void
kk_implicit_watch_draw(struct kk_cmd_buffer *cmd, uint32_t fs_hash)
{
   static uint32_t watched[16], watched_count;
   static int loaded;
   static uint32_t lines;
   static bool watch_all;
   if (!loaded) {
      loaded = 1;
      const char *trigger = getenv("KK_CAPTURE_TRIGGER_FILE");
      char path[1024];
      if (trigger) {
         snprintf(path, sizeof(path), "%s.watch", trigger);
         FILE *f = fopen(path, "r");
         if (f) {
            char word[64];
            while (watched_count < ARRAY_SIZE(watched) && fscanf(f, " %63s", word) == 1) {
               if (!strcmp(word, "all"))
                  watch_all = true;
               else
                  watched[watched_count++] = (uint32_t)strtoul(word, NULL, 16);
            }
            fclose(f);
            mesa_logi("kk-watch: %u shader(s) from %s", watched_count, path);
         }
      }
   }
   /* Each distinct fragment shader drawn, once (at most 256). */
   static uint32_t seen[256], seen_count;
   bool known = false;
   for (uint32_t i = 0; i < seen_count; i++)
      known |= seen[i] == fs_hash;
   if (!known && seen_count < ARRAY_SIZE(seen)) {
      seen[seen_count++] = fs_hash;
      mesa_logi("kk-watch: drawn FS-%08x", fs_hash);
   }
   bool hit = false;
   for (uint32_t i = 0; i < watched_count; i++)
      hit |= watched[i] == fs_hash;

   struct kk_rendering_state *render = &cmd->state.gfx.render;
   const struct kk_image_view *tv =
      render->color_att_count && render->color_att[0].iview ? render->color_att[0].iview
                                                            : render->depth_att.iview;
   const struct kk_image *target = tv ? container_of(tv->vk.image, struct kk_image, vk) : NULL;

   /* Session marks: XRT_SESSION_RECORD_DIR is Monado's recording folder, where each
    * mark (Enter in the recorder, or both thumbsticks held) saves mark_<n>.ppm. For
    * half a second after a new one, every distinct shader and target pair is logged
    * with a "kk-mark <n>:" prefix. */
   static const char *mark_dir;
   static int mark_looked;
   static uint32_t next_mark = 1, mark_now, mark_lines;
   static uint64_t mark_until_ns, mark_checked_ns;
   static struct { uint32_t fs; const void *target; } mark_pairs[512];
   static uint32_t mark_pair_count;
   if (!mark_looked) {
      mark_looked = 1;
      mark_dir = getenv("XRT_SESSION_RECORD_DIR");
      if (mark_dir)
         mesa_logi("kk-mark: watching %s for marks", mark_dir);
   }
   char prefix[32] = "kk-watch:";
   if (mark_dir) {
      const uint64_t now = os_time_get_nano();
      if (now - mark_checked_ns > 100000000ull) {
         mark_checked_ns = now;
         char path[1024];
         snprintf(path, sizeof(path), "%s/mark_%u.ppm", mark_dir, next_mark);
         if (access(path, F_OK) == 0) {
            mark_now = next_mark++;
            mark_until_ns = now + 500000000ull;
            mark_pair_count = 0;
            mark_lines = 0;
            mesa_logi("kk-mark %u: begin (%s)", mark_now, path);
         }
      }
      if (mark_now && now >= mark_until_ns)
         kk_dump_arm(mark_now);
      if (mark_now && now < mark_until_ns && !hit) {
         bool known_pair = false;
         for (uint32_t i = 0; i < mark_pair_count; i++)
            known_pair |= mark_pairs[i].fs == fs_hash && mark_pairs[i].target == target;
         if (!known_pair && mark_pair_count < ARRAY_SIZE(mark_pairs) && mark_lines < 3000) {
            mark_pairs[mark_pair_count].fs = fs_hash;
            mark_pairs[mark_pair_count++].target = target;
            snprintf(prefix, sizeof(prefix), "kk-mark %u:", mark_now);
            kk_dump_note((struct kk_image *)target, mark_now);
            hit = true;
            lines--; /* mark lines do not count against the watch limit */
            mark_lines += 8;
         }
      }
   }
   /* "all": every distinct shader and target pair, once. */
   if (!hit && watch_all) {
      static struct { uint32_t fs; const void *target; } pairs[1024];
      static uint32_t pair_count;
      bool known_pair = false;
      for (uint32_t i = 0; i < pair_count; i++)
         known_pair |= pairs[i].fs == fs_hash && pairs[i].target == target;
      if (!known_pair && pair_count < ARRAY_SIZE(pairs)) {
         pairs[pair_count].fs = fs_hash;
         pairs[pair_count++].target = target;
         hit = true;
      }
   }
   if (!hit || (lines >= 6000 && strncmp(prefix, "kk-mark", 7)))
      return;
   simple_mtx_lock(&kk_fdm_lock);
   const uint64_t latched = kk_implicit.latched.serial;
   simple_mtx_unlock(&kk_fdm_lock);
   lines++;
   mesa_logi("%s FS-%08x draw into %p %ux%u fmt %d fdm=%d map=%llu latched=%llu",
             prefix, fs_hash, (const void *)target, target ? target->vk.extent.width : 0,
             target ? target->vk.extent.height : 0, target ? (int)target->vk.format : 0,
             target ? target->implicit_fdm : 0,
             target && target->rate_ref ? (unsigned long long)target->rate_ref->serial : 0ull,
             (unsigned long long)latched);

   struct kk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   for (uint32_t s = 0; s < KK_MAX_SETS; s++) {
      /* An allocated set, else a push set. */
      const struct kk_descriptor_set *set = desc->sets[s];
      const struct kk_push_descriptor_set *push = desc->push[s];
      const struct kk_descriptor_set_layout *layout;
      const void *mapped;
      uint32_t size;
      if (set && set->layout && set->mapped_ptr) {
         layout = set->layout;
         mapped = set->mapped_ptr;
         size = set->size;
      } else if (push && push->layout) {
         layout = push->layout;
         mapped = push->data;
         size = sizeof(push->data);
      } else {
         continue;
      }
      for (uint32_t b = 0; b < layout->binding_count; b++) {
         const struct kk_descriptor_set_binding_layout *bl = &layout->binding[b];
         if (bl->type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
             bl->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            continue;
         for (uint32_t e = 0; e < MIN2(bl->array_size, 4); e++) {
            const uint32_t at = bl->offset + e * bl->stride;
            if (at + sizeof(struct kk_sampled_image_descriptor) > size)
               break;
            const struct kk_sampled_image_descriptor *d =
               (const void *)((const uint8_t *)mapped + at);
            if (!d->image_gpu_resource_id)
               continue;
            if (!d->rate_params) {
               simple_mtx_lock(&kk_rate_registry_lock);
               const struct kk_image *known = kk_view_id_find_locked(d->image_gpu_resource_id);
               if (known && !strncmp(prefix, "kk-mark", 7))
                  kk_dump_note((struct kk_image *)known, mark_now);
               if (known) {
                  lines++;
                  mesa_logi("%s   set %u binding %u[%u] @%u: %p %ux%u fmt %d s%u fdm=%d map=%llu "
                            "NO MAP IN DESCRIPTOR%s",
                            prefix, s, b, e, at, (const void *)known, known->vk.extent.width,
                            known->vk.extent.height, (int)known->vk.format, known->vk.samples,
                            known->implicit_fdm,
                            known->rate_ref ? (unsigned long long)known->rate_ref->serial : 0ull,
                            known->implicit_fdm ? " (foveated image read raw)" : "");
                  simple_mtx_unlock(&kk_rate_registry_lock);
                  continue;
               }
               const __typeof__(kk_names[0]) *named = NULL;
               for (uint32_t k = 0; k < kk_name_count; k++)
                  if (kk_names[k].id == d->image_gpu_resource_id)
                     named = &kk_names[k];
               if (named) {
                  lines++;
                  mesa_logi("%s   set %u binding %u[%u] @%u: ordinary %p %ux%u fmt %d mips %u%s "
                            "(resource %llx)", prefix, s, b, e, at, named->image, named->w,
                            named->h, named->format, named->levels,
                            named->candidate ? " candidate" : "",
                            (unsigned long long)d->image_gpu_resource_id);
                  simple_mtx_unlock(&kk_rate_registry_lock);
                  continue;
               }
               simple_mtx_unlock(&kk_rate_registry_lock);
               lines++;
               mesa_logi("%s   set %u%s binding %u[%u] @%u: no map (resource %llx)", prefix, s,
                         mapped == (const void *)set ? "" : (set ? "" : " push"), b, e, at,
                         (unsigned long long)d->image_gpu_resource_id);
               continue;
            }
            simple_mtx_lock(&kk_rate_registry_lock);
            const struct kk_image *img = kk_rate_registry_find_locked(d->rate_params);
            if (img && !strncmp(prefix, "kk-mark", 7))
               kk_dump_note((struct kk_image *)img, mark_now);
            lines++;
            if (img)
               mesa_logi("%s   set %u binding %u[%u] @%u: %p %ux%u fmt %d s%u fdm=%d "
                         "map=%llu params_serial=%llu%s",
                         prefix, s, b, e, at, (const void *)img, img->vk.extent.width,
                         img->vk.extent.height, (int)img->vk.format, img->vk.samples,
                         img->implicit_fdm,
                         img->rate_ref ? (unsigned long long)img->rate_ref->serial : 0ull,
                         (unsigned long long)img->rate_params_serial,
                         img == target ? " (the target)" : "");
            else
               mesa_logi("%s   set %u binding %u[%u] @%u: params of an unknown image", prefix, s,
                         b, e, at);
            simple_mtx_unlock(&kk_rate_registry_lock);
         }
      }
   }
}

/*
 * Diagnostic: image dumps at Monado session marks. The images a mark window logs (the
 * targets drawn into and the foveatable images sampled) are copied out at the start of
 * the next primary command buffer after the window, through the unwarp when foveated
 * (so they are in the application's layout), level 0 and up to three more mip levels
 * as stored; two seconds later, when that work has completed, they are written to
 * <XRT_SESSION_RECORD_DIR>/dumps as PPM (with .bin raw data).
 */
#define KK_DUMP_IMAGES 24
#define KK_DUMP_CAPS 96
static struct {
   simple_mtx_t lock;
   struct kk_image *images[KK_DUMP_IMAGES];
   uint32_t image_count, mark, armed_mark, copy_capture_mark;
   bool pending;
   struct {
      struct kk_bo *bo;
      struct kk_device *dev;
      uint32_t w, h, bpp;
      VkFormat format;
      uint64_t at_ns;
      char name[160];
   } caps[KK_DUMP_CAPS];
   uint32_t cap_count;
} kk_dump = {.lock = SIMPLE_MTX_INITIALIZER};
static volatile bool kk_dump_busy;

static bool kk_watched_res_image(const struct kk_image *image);

static void
kk_dump_note(struct kk_image *image, uint32_t mark)
{
   if (!image || (!image->rate_params && !kk_watched_res_image(image)))
      return;
   simple_mtx_lock(&kk_dump.lock);
   if (kk_dump.mark != mark) {
      kk_dump.mark = mark;
      kk_dump.image_count = 0;
   }
   bool known = false;
   for (uint32_t i = 0; i < kk_dump.image_count; i++)
      known |= kk_dump.images[i] == image;
   if (!known && kk_dump.image_count < KK_DUMP_IMAGES)
      kk_dump.images[kk_dump.image_count++] = image;
   simple_mtx_unlock(&kk_dump.lock);
}

static void
kk_dump_arm(uint32_t mark)
{
   simple_mtx_lock(&kk_dump.lock);
   if (kk_dump.armed_mark != mark && kk_dump.mark == mark && kk_dump.image_count) {
      kk_dump.armed_mark = mark;
      kk_dump.pending = true;
      kk_dump.copy_capture_mark = mark;
      kk_dump_busy = true;
      mesa_logi("kk-dump: mark %u armed with %u image(s)", mark, kk_dump.image_count);
   }
   simple_mtx_unlock(&kk_dump.lock);
}

static float
kk_dump_tone(float v)
{
   if (!(v > 0.0f))
      return 0.0f;
   v = v / (1.0f + v);
   return powf(v, 1.0f / 2.2f);
}

static void
kk_dump_write(const char *dir, const char *name, const uint8_t *data, uint32_t w, uint32_t h,
              uint32_t bpp, VkFormat format)
{
   char path[1200];
   snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
   FILE *f = fopen(path, "wb");
   if (f) {
      fwrite(data, 1, (size_t)w * h * bpp, f);
      fclose(f);
   }
   uint8_t *rgb = malloc((size_t)w * h * 3);
   if (!rgb)
      return;
   bool ok = true;
   if (format == VK_FORMAT_R16_SFLOAT || format == VK_FORMAT_R32_SFLOAT) {
      float lo = INFINITY, hi = -INFINITY;
      for (size_t i = 0; i < (size_t)w * h; i++) {
         float v = format == VK_FORMAT_R16_SFLOAT ? _mesa_half_to_float(((const uint16_t *)data)[i])
                                                  : ((const float *)data)[i];
         if (isfinite(v)) {
            lo = MIN2(lo, v);
            hi = MAX2(hi, v);
         }
      }
      const float span = hi > lo ? hi - lo : 1.0f;
      for (size_t i = 0; i < (size_t)w * h; i++) {
         float v = format == VK_FORMAT_R16_SFLOAT ? _mesa_half_to_float(((const uint16_t *)data)[i])
                                                  : ((const float *)data)[i];
         uint8_t g = isfinite(v) ? (uint8_t)CLAMP(sqrtf((v - lo) / span) * 255.0f, 0, 255) : 255;
         rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = g;
      }
   } else if (format == VK_FORMAT_R16G16B16A16_SFLOAT) {
      for (size_t i = 0; i < (size_t)w * h; i++)
         for (int c = 0; c < 3; c++)
            rgb[i * 3 + c] = (uint8_t)(kk_dump_tone(_mesa_half_to_float(((const uint16_t *)data)[i * 4 + c])) * 255.0f);
   } else if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
      for (size_t i = 0; i < (size_t)w * h; i++) {
         rgb[i * 3] = data[i * 4 + 2];
         rgb[i * 3 + 1] = data[i * 4 + 1];
         rgb[i * 3 + 2] = data[i * 4];
      }
   } else if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB) {
      for (size_t i = 0; i < (size_t)w * h; i++)
         memcpy(&rgb[i * 3], &data[i * 4], 3);
   } else {
      ok = false;
   }
   if (ok) {
      snprintf(path, sizeof(path), "%s/%s.ppm", dir, name);
      f = fopen(path, "wb");
      if (f) {
         fprintf(f, "P6\n%u %u\n255\n", w, h);
         fwrite(rgb, 1, (size_t)w * h * 3, f);
         fclose(f);
      }
   }
   free(rgb);
}

static void
kk_dump_add_copy(struct kk_cmd_buffer *cmd, struct kk_image *image, mtl_texture *texture,
                 uint32_t level, bool unwarped, uint32_t mark)
{
   simple_mtx_lock(&kk_dump.lock);
   const bool full = kk_dump.cap_count >= KK_DUMP_CAPS;
   simple_mtx_unlock(&kk_dump.lock);
   if (full)
      return;
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   const uint32_t w = u_minify(image->vk.extent.width, level);
   const uint32_t h = u_minify(image->vk.extent.height, level);
   const uint32_t bpp = vk_format_get_blocksize(image->vk.format);
   struct kk_bo *bo = NULL;
   if (kk_alloc_bo(dev, &dev->vk.base, (uint64_t)w * h * bpp, 0, &bo) != VK_SUCCESS)
      return;
   struct mtl_buffer_image_copy copy = {
      .image_size = {w, h, 1},
      .buffer = bo->map,
      .image = texture,
      .buffer_stride_B = (size_t)w * bpp,
      .buffer_2d_image_size_B = (size_t)w * h * bpp,
      .image_level = unwarped ? 0 : level,
   };
   mtl_copy_from_texture_to_buffer(cs_get_compute(cmd), &copy);
   simple_mtx_lock(&kk_dump.lock);
   if (kk_dump.cap_count >= KK_DUMP_CAPS) {
      simple_mtx_unlock(&kk_dump.lock);
      kk_destroy_bo(dev, bo);
      return;
   }
   __typeof__(kk_dump.caps[0]) *c = &kk_dump.caps[kk_dump.cap_count++];
   c->bo = bo;
   c->dev = dev;
   c->w = w;
   c->h = h;
   c->bpp = bpp;
   c->format = image->vk.format;
   c->at_ns = os_time_get_nano();
   snprintf(c->name, sizeof(c->name), "mark%u_%p_%ux%u_fmt%d_level%u%s%s", mark, (void *)image,
            image->vk.extent.width, image->vk.extent.height, image->vk.format, level,
            image->implicit_fdm ? "_foveated" : "", unwarped ? "_unwarped" : "");
   simple_mtx_unlock(&kk_dump.lock);
}

void
kk_implicit_dump_begin(struct kk_cmd_buffer *cmd)
{
   if (!kk_dump_busy || cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return;
   const char *session = getenv("XRT_SESSION_RECORD_DIR");
   if (!session)
      return;
   /* Write the copies whose work has had two seconds to complete. The dump lock is
    * never held with the rate registry's (the draw-time log takes them the other way). */
   const uint64_t now = os_time_get_nano();
   char dir[1100];
   snprintf(dir, sizeof(dir), "%s/dumps", session);
   for (;;) {
      __typeof__(kk_dump.caps[0]) done = {0};
      simple_mtx_lock(&kk_dump.lock);
      for (uint32_t i = 0; i < kk_dump.cap_count; i++) {
         if (now - kk_dump.caps[i].at_ns >= 2000000000ull) {
            done = kk_dump.caps[i];
            kk_dump.caps[i] = kk_dump.caps[--kk_dump.cap_count];
            break;
         }
      }
      simple_mtx_unlock(&kk_dump.lock);
      if (!done.bo)
         break;
      mkdir(dir, 0755);
      kk_dump_write(dir, done.name, done.bo->cpu, done.w, done.h, done.bpp, done.format);
      kk_destroy_bo(done.dev, done.bo);
      mesa_logi("kk-dump: wrote %s", done.name);
   }

   struct kk_image *images[KK_DUMP_IMAGES];
   uint32_t count = 0, mark = 0;
   simple_mtx_lock(&kk_dump.lock);
   if (kk_dump.pending) {
      kk_dump.pending = false;
      count = kk_dump.image_count;
      mark = kk_dump.armed_mark;
      memcpy(images, kk_dump.images, count * sizeof(images[0]));
   }
   simple_mtx_unlock(&kk_dump.lock);

   if (count) {
      /* Earlier work on these images first. */
      const VkMemoryBarrier2 all = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      };
      const VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .memoryBarrierCount = 1, .pMemoryBarriers = &all};
      kk_CmdPipelineBarrier2(kk_cmd_buffer_to_handle(cmd), &dep);
      for (uint32_t i = 0; i < count; i++) {
         struct kk_image *image = images[i];
         /* Still alive? (Foveatable images stay in the rate registry while they live.) */
         bool alive = false;
         simple_mtx_lock(&kk_rate_registry_lock);
         for (uint32_t j = 0; j < kk_rate_registry_count; j++)
            alive |= kk_rate_registry[j] == image;
         for (uint32_t j = 0; j < kk_watched_res.view_count; j++)
            alive |= kk_watched_res.views[j].image == image;
         simple_mtx_unlock(&kk_rate_registry_lock);
         if (!alive || vk_format_is_depth_or_stencil(image->vk.format) ||
             image->vk.samples != VK_SAMPLE_COUNT_1_BIT || image->vk.array_layers != 1)
            continue;
         VkImage unwarp = image->implicit_fdm
                             ? kk_implicit_unwarp_for_copy(cmd, image, VK_IMAGE_LAYOUT_GENERAL, NULL)
                             : VK_NULL_HANDLE;
         if (cmd->metal.render != NULL)
            cs_end(cmd);
         mtl_barrier_after_encoder_stages(cs_get_compute(cmd), MTL_STAGE_ALL,
                                          MTL_STAGE_BLIT | MTL_STAGE_DISPATCH);
         if (unwarp != VK_NULL_HANDLE)
            kk_dump_add_copy(cmd, image, kk_image_from_handle(unwarp)->planes[0].mtl_handle, 0, true,
                             mark);
         for (uint32_t level = 0; level < MIN2(image->vk.mip_levels, 4); level++)
            kk_dump_add_copy(cmd, image, image->planes[0].mtl_handle, level, false, mark);
      }
      kk_fdm_order_for_next_encoder(cmd);
      kk_CmdPipelineBarrier2(kk_cmd_buffer_to_handle(cmd), &dep);
      mesa_logi("kk-dump: mark %u copies recorded", mark);
   }
   simple_mtx_lock(&kk_dump.lock);
   kk_dump_busy = kk_dump.pending || kk_dump.cap_count;
   simple_mtx_unlock(&kk_dump.lock);
}

static bool
kk_watched_res_image(const struct kk_image *image)
{
   bool found = false;
   simple_mtx_lock(&kk_rate_registry_lock);
   for (uint32_t i = 0; i < kk_watched_res.view_count; i++)
      found |= kk_watched_res.views[i].image == image;
   simple_mtx_unlock(&kk_rate_registry_lock);
   return found;
}

/* Diagnostic (session recordings): each distinct transfer out of a foveated image, with
 * the path it took, once. */
void
kk_implicit_fov_op(const char *what, const struct kk_image *src, const struct kk_image *dst,
                   uint32_t level, const char *path)
{
   if (!src || !src->implicit_fdm || !getenv("XRT_SESSION_RECORD_DIR"))
      return;
   static struct { const char *what, *path; const void *src, *dst; uint32_t level; } seen[256];
   static uint32_t count;
   simple_mtx_lock(&kk_fdm_lock);
   for (uint32_t i = 0; i < count; i++) {
      if (seen[i].what == what && seen[i].path == path && seen[i].src == src &&
          seen[i].dst == dst && seen[i].level == level) {
         simple_mtx_unlock(&kk_fdm_lock);
         return;
      }
   }
   if (count < ARRAY_SIZE(seen)) {
      seen[count++] = (__typeof__(seen[0])){what, path, src, dst, level};
      mesa_logi("kk-fovop: %s %p %ux%u fmt %d level %u -> %p %ux%u fmt %d fdm=%d: %s", what,
                (const void *)src, src->vk.extent.width, src->vk.extent.height, src->vk.format,
                level, (const void *)dst, dst ? dst->vk.extent.width : 0,
                dst ? dst->vk.extent.height : 0, dst ? dst->vk.format : 0,
                dst ? dst->implicit_fdm : 0, path);
   }
   simple_mtx_unlock(&kk_fdm_lock);
}

/* Diagnostic: right after an armed mark, the next copy out of a foveated image is
 * captured as its destination holds it then (what later passes read), as
 * mark<n>_..._aftercopy. Called with no render pass open. */
void
kk_implicit_dump_after_copy(struct kk_cmd_buffer *cmd, struct kk_image *src, struct kk_image *dst)
{
   if (!src->implicit_fdm || dst->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       vk_format_is_depth_or_stencil(dst->vk.format) || dst->vk.array_layers != 1)
      return;
   simple_mtx_lock(&kk_dump.lock);
   uint32_t mark = kk_dump.copy_capture_mark;
   bool logged = false;
   for (uint32_t i = 0; i < kk_dump.image_count; i++)
      logged |= kk_dump.images[i] == dst;
   if (!logged)
      mark = 0;
   if (mark)
      kk_dump.copy_capture_mark = 0;
   simple_mtx_unlock(&kk_dump.lock);
   if (!mark)
      return;
   if (cmd->metal.render != NULL)
      cs_end(cmd);
   mtl_barrier_after_encoder_stages(cs_get_compute(cmd), MTL_STAGE_ALL,
                                    MTL_STAGE_BLIT | MTL_STAGE_DISPATCH);
   kk_dump_add_copy(cmd, dst, dst->planes[0].mtl_handle, 0, false, mark);
   kk_fdm_order_for_next_encoder(cmd);
   simple_mtx_lock(&kk_dump.lock);
   if (kk_dump.cap_count) {
      char *name = kk_dump.caps[kk_dump.cap_count - 1].name;
      strncat(name, "_aftercopy", sizeof(kk_dump.caps[0].name) - strlen(name) - 1);
   }
   kk_dump_busy = true;
   simple_mtx_unlock(&kk_dump.lock);
   mesa_logi("kk-dump: mark %u captured %p right after a copy from %p", mark, (void *)dst,
             (void *)src);
}
