/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_private.h"
#include "kk_fdm.h"

#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_debug.h"
#include "kk_format.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_rate_map.h"

#include "kk_entrypoints.h"
#include "util/u_debug.h"
#include "vk_format.h"

DEBUG_GET_ONCE_BOOL_OPTION(kk_native_full_resolve, "KK_NATIVE_FULL_RESOLVE", false);

static VkResult
kk_cmd_bind_map_buffer(struct vk_command_buffer *vk_cmd,
                       struct vk_meta_device *meta, VkBuffer _buffer,
                       void **map_out)
{
   struct kk_cmd_buffer *cmd = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   VK_FROM_HANDLE(kk_buffer, buffer, _buffer);

   assert(buffer->vk.size < UINT_MAX);
   struct kk_ptr buf = kk_pool_alloc(cmd, buffer->vk.size, 16u);
   if (unlikely(!buf.gpu))
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   /* Need to retain since VkBuffers release the metal.handle too */
   mtl_retain(buf.buffer);
   buffer->metal.handle = buf.buffer;
   buffer->vk.device_address = buf.gpu;
   *map_out = buf.cpu;

   return VK_SUCCESS;
}

VkResult
kk_device_init_meta(struct kk_device *dev)
{
   VkResult result = vk_meta_device_init(&dev->vk, &dev->meta);
   if (result != VK_SUCCESS)
      return result;

   dev->meta.use_gs_for_layer = false;
   dev->meta.use_stencil_export = true;
   dev->meta.use_rect_list_pipeline = true;
   dev->meta.cmd_bind_map_buffer = kk_cmd_bind_map_buffer;
   dev->meta.max_bind_map_buffer_size_B = 64 * 1024;

   for (unsigned i = 0; i < VK_META_BUFFER_CHUNK_SIZE_COUNT; ++i) {
      dev->meta.buffer_access.optimal_wg_size[i] = 64;
   }

   return VK_SUCCESS;
}

void
kk_device_finish_meta(struct kk_device *dev)
{
   vk_meta_device_finish(&dev->vk, &dev->meta);
}

struct kk_meta_save {
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
   struct vk_dynamic_graphics_state dynamic;
   struct kk_shader *shaders[MESA_SHADER_STAGES];
   struct {
      union {
         struct {
            mtl_render_pipeline_state *ps;
            mtl_depth_stencil_state *ds;
            enum mtl_visibility_result_mode occlusion;
            bool is_ds_dynamic;
         } gfx;
      };
   } pipeline;
   struct kk_conditional_rendering_state cond_render;
   struct kk_descriptor_set *desc0;
   struct kk_push_descriptor_set *push_desc0;
   /* The application's pushed descriptors: meta pushes into the same object, and an
    * application need not push again after an operation it did not ask for. */
   struct kk_push_descriptor_set *push_desc0_contents;
   struct kk_addr_range vb0;
   struct kk_buffer_address desc0_set_addr;
   bool has_push_desc0;
   uint8_t push[KK_MAX_PUSH_SIZE];
};

static void
kk_meta_begin(struct kk_cmd_buffer *cmd, struct kk_meta_save *save,
              VkPipelineBindPoint bind_point, const char *debug_label)
{
   kk_shader_set_sync_compiles(true);
   struct kk_descriptor_state *desc = kk_get_descriptors_state(cmd, bind_point);

   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      save->dynamic = cmd->vk.dynamic_graphics_state;
      save->_dynamic_vi = cmd->state.gfx._dynamic_vi;
      save->_dynamic_sl = cmd->state.gfx._dynamic_sl;

      for (uint32_t stage = 0u; stage < MESA_SHADER_COMPUTE; ++stage) {
         save->shaders[stage] = cmd->state.shaders[stage];
      }

      save->pipeline.gfx.ds = cmd->state.gfx.depth_stencil_state;
      save->pipeline.gfx.occlusion = cmd->state.gfx.occlusion.mode;
      save->pipeline.gfx.is_ds_dynamic =
         cmd->state.gfx.is_depth_stencil_dynamic;

      cmd->state.gfx.is_depth_stencil_dynamic = false;
      cmd->state.gfx.depth_stencil_state = NULL;
      cmd->state.gfx.occlusion.mode = MTL_VISIBILITY_RESULT_MODE_DISABLED;
      cmd->state.gfx.dirty |= KK_DIRTY_OCCLUSION;
      desc->root_dirty = true;
   } else {
      save->shaders[MESA_SHADER_COMPUTE] =
         cmd->state.shaders[MESA_SHADER_COMPUTE];
   }

   save->cond_render = cmd->state.cond_render;
   cmd->state.cond_render.enabled = false;

   save->vb0 = cmd->state.gfx.vb.addr_range[0];

   save->desc0 = desc->sets[0];
   save->has_push_desc0 = desc->push[0];
   save->push_desc0_contents = NULL;
   if (save->has_push_desc0) {
      save->push_desc0 = desc->push[0];
      save->push_desc0_contents = malloc(sizeof(*desc->push[0]));
      if (save->push_desc0_contents)
         memcpy(save->push_desc0_contents, desc->push[0], sizeof(*desc->push[0]));
   }

   static_assert(sizeof(save->push) == sizeof(desc->root.push),
                 "Size mismatch for push in meta_save");
   memcpy(save->push, desc->root.push, sizeof(save->push));

   if (kk_device_physical(kk_cmd_buffer_device(cmd))
          ->settings.gpu_capture_enabled) {
      VkDebugUtilsLabelEXT label_info = {.pLabelName = debug_label};
      kk_CmdBeginDebugUtilsLabelEXT(kk_cmd_buffer_to_handle(cmd), &label_info);
   }
}

static void
kk_meta_end_impl(struct kk_cmd_buffer *cmd, struct kk_meta_save *save,
            VkPipelineBindPoint bind_point)
{
   struct kk_descriptor_state *desc = kk_get_descriptors_state(cmd, bind_point);
   desc->root_dirty = true;

   if (kk_device_physical(kk_cmd_buffer_device(cmd))
          ->settings.gpu_capture_enabled) {
      kk_CmdEndDebugUtilsLabelEXT(kk_cmd_buffer_to_handle(cmd));
   }

   if (save->desc0) {
      desc->sets[0] = save->desc0;
      desc->root.sets[0] = save->desc0->addr;
      desc->set_sizes[0] = save->desc0->size;
      desc->push_dirty &= ~BITFIELD_BIT(0);
   } else if (save->has_push_desc0) {
      desc->push[0] = save->push_desc0;
      if (save->push_desc0_contents)
         memcpy(desc->push[0], save->push_desc0_contents, sizeof(*desc->push[0]));
      desc->sets[0] = NULL;
      desc->push_dirty |= BITFIELD_BIT(0);
   }
   free(save->push_desc0_contents);
   save->push_desc0_contents = NULL;

   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      /* Restore the dynamic state */
      assert(save->dynamic.vi == &cmd->state.gfx._dynamic_vi);
      assert(save->dynamic.ms.sample_locations == &cmd->state.gfx._dynamic_sl);
      cmd->vk.dynamic_graphics_state = save->dynamic;
      cmd->state.gfx._dynamic_vi = save->_dynamic_vi;
      cmd->state.gfx._dynamic_sl = save->_dynamic_sl;
      memcpy(cmd->vk.dynamic_graphics_state.dirty,
             cmd->vk.dynamic_graphics_state.set,
             sizeof(cmd->vk.dynamic_graphics_state.set));

      for (uint32_t stage = 0u; stage < MESA_SHADER_COMPUTE; ++stage) {
         cmd->state.shaders[stage] = save->shaders[stage];
      }
      cmd->state.dirty_shaders |= BITFIELD_MASK(MESA_SHADER_COMPUTE);

      if (cmd->state.gfx.is_depth_stencil_dynamic)
         mtl_release(cmd->state.gfx.depth_stencil_state);
      cmd->state.gfx.depth_stencil_state = save->pipeline.gfx.ds;
      cmd->state.gfx.is_depth_stencil_dynamic =
         save->pipeline.gfx.is_ds_dynamic;

      cmd->state.gfx.vb.addr_range[0] = save->vb0;
      cmd->state.gfx.dirty |= KK_DIRTY_VB;

      cmd->state.gfx.occlusion.mode = save->pipeline.gfx.occlusion;
      cmd->state.gfx.dirty |= KK_DIRTY_OCCLUSION;

      desc->root_dirty = true;
   } else {
      kk_cmd_bind_compute_shader(cmd, save->shaders[MESA_SHADER_COMPUTE]);
   }

   cmd->state.cond_render = save->cond_render;

   memcpy(desc->root.push, save->push, sizeof(save->push));
}

static void
kk_meta_end(struct kk_cmd_buffer *cmd, struct kk_meta_save *save,
            VkPipelineBindPoint bind_point)
{
   kk_meta_end_impl(cmd, save, bind_point);
   kk_shader_set_sync_compiles(false);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdFillMemoryKHR(VkCommandBuffer commandBuffer,
                    const VkDeviceAddressRangeKHR *dstRange,
                    VkAddressCommandFlagsKHR dstFlags, uint32_t data)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   struct kk_meta_save save;
   kk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE,
                 "meta:vkCmdFillMemoryKHR");
   vk_meta_fill_memory(&cmd->vk, &dev->meta, dstRange, dstFlags, data);
   kk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdUpdateMemoryKHR(VkCommandBuffer commandBuffer,
                      const VkDeviceAddressRangeKHR *pDstRange,
                      VkAddressCommandFlagsKHR dstFlags, VkDeviceSize dataSize,
                      const void *pData)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   struct kk_meta_save save;
   kk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE,
                 "meta:vkCmdUpdateMemoryKHR");
   vk_meta_update_memory(&cmd->vk, &dev->meta, pDstRange, dstFlags, dataSize,
                         pData);
   kk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBlitImage2(VkCommandBuffer commandBuffer,
                 const VkBlitImageInfo2 *pBlitImageInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   struct kk_meta_save save;
   kk_implicit_trace("blit", vk_image_from_handle(pBlitImageInfo->srcImage), vk_image_from_handle(pBlitImageInfo->dstImage));
   kk_fdm_note_use(vk_image_from_handle(pBlitImageInfo->srcImage), "blit source");
   kk_fdm_note_use(vk_image_from_handle(pBlitImageInfo->dstImage), "blit destination");
   /* A foveated image blitted into itself (mip generation) keeps its warped layout. */
   const struct kk_image *blit_src =
      container_of(vk_image_from_handle(pBlitImageInfo->srcImage), struct kk_image, vk);
   const bool raw = blit_src->implicit_fdm && pBlitImageInfo->srcImage == pBlitImageInfo->dstImage;
   for (uint32_t r = 0; r < pBlitImageInfo->regionCount; r++)
      kk_implicit_fov_op("blit", blit_src,
                         container_of(vk_image_from_handle(pBlitImageInfo->dstImage), struct kk_image, vk),
                         pBlitImageInfo->pRegions[r].srcSubresource.mipLevel,
                         raw ? "raw into itself" : "translated");
   kk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 "meta:vkCmdBlitImage2");
   if (raw)
      kk_implicit_set_raw(true);
   vk_meta_blit_image2(&cmd->vk, &dev->meta, pBlitImageInfo);
   if (raw)
      kk_implicit_set_raw(false);
   kk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

static bool
kk_native_full_resolve(struct kk_cmd_buffer *cmd, struct kk_image *src,
                       struct kk_image *dst, const VkResolveImageInfo2 *info)
{
   if (!debug_get_option_kk_native_full_resolve() || info->regionCount != 1 ||
       !src->implicit_fdm || !dst->implicit_fdm || !dst->rate_params ||
       !src->rate_ref || !src->rate_ref->map || kk_implicit_raw_active() ||
       info->pNext || src->metal_imported || dst->metal_imported ||
       src->wsi_direct || dst->wsi_direct ||
       src->vk.image_type != VK_IMAGE_TYPE_2D || dst->vk.image_type != VK_IMAGE_TYPE_2D ||
       src->vk.array_layers != 1 || dst->vk.array_layers != 1 ||
       src->vk.samples <= VK_SAMPLE_COUNT_1_BIT || dst->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       src->vk.format != dst->vk.format || !vk_format_is_color(src->vk.format) ||
       vk_format_is_int(src->vk.format) || vk_format_is_srgb(src->vk.format) ||
       !(src->vk.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
       !(dst->vk.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
       !kk_get_va_format(vk_format_to_pipe_format(src->vk.format))->resolve)
      return false;

   const VkImageResolve2 *region = info->pRegions;
   if (region->srcSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
       region->dstSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
       region->srcSubresource.mipLevel || region->dstSubresource.mipLevel ||
       region->srcOffset.x || region->srcOffset.y || region->srcOffset.z ||
       region->dstOffset.x || region->dstOffset.y || region->dstOffset.z ||
       region->extent.width != src->vk.extent.width ||
       region->extent.height != src->vk.extent.height ||
       region->extent.width != dst->vk.extent.width ||
       region->extent.height != dst->vk.extent.height)
      return false;

   mtl_render_pass_descriptor *pass = mtl_new_render_pass_descriptor();
   mtl_render_pass_attachment_descriptor *color =
      mtl_render_pass_descriptor_get_color_attachment(pass, 0);
   mtl_render_pass_attachment_descriptor_set_texture(color, src->planes[0].mtl_handle);
   mtl_render_pass_attachment_descriptor_set_resolve_texture(color, dst->planes[0].mtl_handle);
   mtl_render_pass_attachment_descriptor_set_load_action(color, MTL_LOAD_ACTION_LOAD);
   mtl_render_pass_attachment_descriptor_set_store_action(color, MTL_STORE_ACTION_UNKNOWN);
   mtl_render_pass_descriptor_set_render_target_width(pass, region->extent.width);
   mtl_render_pass_descriptor_set_render_target_height(pass, region->extent.height);
   mtl_render_pass_descriptor_set_render_target_array_length(pass, 1);
   mtl_render_pass_descriptor_set_rate_map(pass, src->rate_ref->map);

   struct kk_graphics_state *gfx = &cmd->state.gfx;
   assert(!gfx->render_pass_descriptor);
   gfx->render_pass_descriptor = pass;
   gfx->render.samples = src->vk.samples;
   gfx->restart_barrier |= KK_EXPERIMENTAL(LIGHT_BARRIERS);

   cs_start_render(cmd);
   mtl_render_set_color_store_action(cmd->metal.render,
                                    MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE, 0);
   if (KK_EXPERIMENTAL(LIGHT_BARRIERS))
      mtl_barrier_after_stages(cmd->metal.render, MTL_STAGE_ALL, MTL_STAGE_ALL);
   cs_end(cmd);

   gfx->render_pass_descriptor = NULL;
   memset(&gfx->render, 0, sizeof(gfx->render));
   mtl_release(pass);
   kk_implicit_copy_map(cmd, src, dst);
   return true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdResolveImage2(VkCommandBuffer commandBuffer,
                    const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   VK_FROM_HANDLE(kk_image, src, pResolveImageInfo->srcImage);
   VK_FROM_HANDLE(kk_image, dst, pResolveImageInfo->dstImage);
   kk_implicit_trace("resolve", &src->vk, &dst->vk);
   if (src->implicit_fdm && debug_get_bool_option("KK_IMPLICIT_SKIP_EXPLICIT", false))
      return;

   kk_fdm_note_use(&src->vk, "resolve source");
   kk_fdm_note_use(&dst->vk, "resolve destination");
   const bool native = kk_native_full_resolve(cmd, src, dst, pResolveImageInfo);
   kk_implicit_fov_op("resolve", src, dst, pResolveImageInfo->pRegions[0].srcSubresource.mipLevel,
                     native ? "native full-image resolve" : "resolve");
   if (native) {
      static uint32_t resolved;
      if (p_atomic_inc_return(&resolved) == 1)
         mesa_logi("kk: Metal native full-image resolve in use (%ux%u, %ux MSAA)",
                   dst->vk.extent.width, dst->vk.extent.height, (unsigned)src->vk.samples);
      return;
   }

   struct kk_meta_save save;
   kk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 "meta:vkCmdResolveImage2");
   vk_meta_resolve_image2(&cmd->vk, &dev->meta, pResolveImageInfo);
   kk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

static void
kk_meta_init_render(struct kk_cmd_buffer *cmd,
                    struct vk_meta_rendering_info *info)
{
   const struct kk_rendering_state *render = &cmd->state.gfx.render;

   *info = (struct vk_meta_rendering_info){
      .samples = MAX2(render->samples, 1),
      .view_mask = render->view_mask,
      .color_attachment_count = render->color_att_count,
      .depth_attachment_format = render->depth_att.vk_format,
      .stencil_attachment_format = render->stencil_att.vk_format,
   };
   for (uint32_t a = 0; a < render->color_att_count; a++) {
      info->color_attachment_formats[a] = render->color_att[a].vk_format;
      info->color_attachment_write_masks[a] =
         VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                       const VkClearAttachment *pAttachments,
                       uint32_t rectCount, const VkClearRect *pRects)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   struct vk_meta_rendering_info render_info;
   kk_meta_init_render(cmd, &render_info);

   uint32_t view_mask = cmd->state.gfx.render.view_mask;
   uint32_t layer_ids[KK_MAX_MULTIVIEW_VIEW_COUNT] = {};
   mtl_set_vertex_amplification_count(cs_get_render(cmd), layer_ids, 1u);

   /* Preserve conditional rendering state for clearing attachments */
   struct kk_conditional_rendering_state cond_render = cmd->state.cond_render;

   struct kk_meta_save save;
   kk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 "meta:vkCmdClearAttachments");
   cmd->state.cond_render = cond_render;
   vk_meta_clear_attachments(&cmd->vk, &dev->meta, &render_info,
                             attachmentCount, pAttachments, rectCount, pRects);
   kk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);

   uint32_t count = 0u;
   u_foreach_bit(id, view_mask)
      layer_ids[count++] = id;
   if (view_mask == 0u) {
      layer_ids[count++] = 0;
   }
   mtl_set_vertex_amplification_count(cs_get_render(cmd), layer_ids, count);
}

void
kk_meta_resolve_rendering(struct kk_cmd_buffer *cmd,
                          const VkRenderingInfo *pRenderingInfo)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   struct kk_meta_save save;
   kk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS,
                 "meta:resolve_rendering");
   vk_meta_resolve_rendering(&cmd->vk, &dev->meta, pRenderingInfo);
   kk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
}
