/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_entrypoints.h"
#include "kk_debug.h"
#include "kk_fdm.h"
#include "kosmickrisp/bridge/mtl_tile.h"

#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_format.h"
#include "kk_image_view.h"
#include "kk_physical_device.h"
#include "kk_query_pool.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "kosmickrisp/libkk/kk_tessellator.h"

#include "poly/geometry.h"
#include "poly/tessellator.h"

#include "vulkan/runtime/vk_render_pass.h"
#include "vulkan/util/vk_format.h"

#include "util/log.h"
#include "util/u_debug.h"
#include "util/os_time.h"
#include "util/format/u_format.h"
#include "c11/threads.h"
#include "kk_bo.h"
#include "kk_image.h"
#include "kosmickrisp/libkk/kk_query.h"

DEBUG_GET_ONCE_BOOL_OPTION(kk_target_submit_log, "KK_TARGET_SUBMIT_LOG", false);
DEBUG_GET_ONCE_BOOL_OPTION(kk_target_draw_log, "KK_TARGET_DRAW_LOG", false);
DEBUG_GET_ONCE_BOOL_OPTION(kk_descriptor_diag, "KK_DESCRIPTOR_DIAG", false);

static void
kk_cmd_buffer_dirty_render_pass(struct kk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
}

static void
kk_attachment_init(struct kk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct kk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(kk_image_view, iview, info->imageView);
   *att = (struct kk_attachment){
      .flags = vk_get_rendering_attachment_flags(info),
      .vk_format = iview->vk.format,
      .iview = iview,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(kk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }

   att->store_op = info->storeOp;
   att->load_op = info->loadOp;
   att->clear_value = info->clearValue;
}

VKAPI_ATTR void VKAPI_CALL
kk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   *pGranularity = (VkExtent2D){
      .width = pdev->info.rendering_tile_width,
      .height = pdev->info.rendering_tile_height,
   };
}

static void
kk_merge_render_iview(VkExtent2D *extent, struct kk_image_view *iview)
{
   if (iview) {
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      extent->width = MAX2(extent->width, width);
      extent->height = MAX2(extent->height, height);
   }
}

static void
kk_clear_common_attachment_description(
   mtl_render_pass_attachment_descriptor *descriptor)
{
   mtl_render_pass_attachment_descriptor_set_texture(descriptor, NULL);
   mtl_render_pass_attachment_descriptor_set_load_action(
      descriptor, MTL_LOAD_ACTION_DONT_CARE);
   mtl_render_pass_attachment_descriptor_set_store_action(
      descriptor, MTL_STORE_ACTION_UNKNOWN);
   mtl_render_pass_attachment_descriptor_set_resolve_texture(descriptor, NULL);
}

static void
kk_fill_common_attachment_description(
   mtl_render_pass_attachment_descriptor *descriptor,
   const struct kk_attachment *info, bool force_attachment_load,
   bool attach_resolve_texture)
{
   const struct kk_image_view *iview = info->iview;
   assert(iview->plane_count ==
          1); /* TODO_KOSMICKRISP Handle multiplanar images? */
   const struct kk_image *image =
      container_of(iview->vk.image, struct kk_image, vk);
   mtl_render_pass_attachment_descriptor_set_texture(
      descriptor, image->wsi_direct ? image->planes[0].mtl_handle
                                    : iview->planes[0].mtl_handle_render);
   /* If a view is being used e.g. for format change, it already points to the
    * required slice and level */
   if (!iview->planes[0].render_is_view) {
      mtl_render_pass_attachment_descriptor_set_level(descriptor,
                                                      iview->vk.base_mip_level);
      mtl_render_pass_attachment_descriptor_set_slice(
         descriptor, iview->vk.base_array_layer);
   }

   /* STORE_OP_NONE behaves like DONT_CARE if there are writes
    * (CLEAR or DONT_CARE are writes).
    * If there are no writes, we need to preserve the contents. */
   force_attachment_load |= (info->load_op == VK_ATTACHMENT_LOAD_OP_LOAD ||
                             info->load_op == VK_ATTACHMENT_LOAD_OP_NONE) &&
                            info->store_op == VK_ATTACHMENT_STORE_OP_NONE;

   enum mtl_load_action load_action =
      force_attachment_load
         ? MTL_LOAD_ACTION_LOAD
         : vk_attachment_load_op_to_mtl_load_action(info->load_op);

   mtl_render_pass_attachment_descriptor_set_load_action(descriptor,
                                                         load_action);
   mtl_render_pass_attachment_descriptor_set_store_action(
      descriptor, MTL_STORE_ACTION_UNKNOWN);

   if (attach_resolve_texture) {
      const struct kk_image *resolve_image =
         container_of(info->resolve_iview->vk.image, struct kk_image, vk);
      mtl_render_pass_attachment_descriptor_set_resolve_texture(
         descriptor, resolve_image->wsi_direct
                        ? resolve_image->planes[0].mtl_handle
                        : info->resolve_iview->planes[0].mtl_handle_render);

      if (!info->resolve_iview->planes[0].render_is_view) {
         mtl_render_pass_attachment_descriptor_set_resolve_level(
            descriptor, info->resolve_iview->vk.base_mip_level);
         mtl_render_pass_attachment_descriptor_set_resolve_slice(
            descriptor, info->resolve_iview->vk.base_array_layer);
      }
   }
}

static struct mtl_clear_color
vk_clear_color_value_to_mtl_clear_color(union VkClearColorValue color,
                                        enum pipe_format format)
{
   struct mtl_clear_color value;
   if (util_format_is_pure_sint(format)) {
      value.red = color.int32[0];
      value.green = color.int32[1];
      value.blue = color.int32[2];
      value.alpha = color.int32[3];
   } else if (util_format_is_pure_uint(format)) {
      value.red = color.uint32[0];
      value.green = color.uint32[1];
      value.blue = color.uint32[2];
      value.alpha = color.uint32[3];
   } else {
      value.red = color.float32[0];
      value.green = color.float32[1];
      value.blue = color.float32[2];
      value.alpha = color.float32[3];
   }

   /* Apply swizzle to color since Metal does not allow swizzle for renderable
    * textures, but we need to support that for formats like
    * VK_FORMAT_B4G4R4A4_UNORM_PACK16 */
   const struct kk_va_format *supported_format = kk_get_va_format(format);
   struct mtl_clear_color swizzled_color;
   for (uint32_t i = 0u; i < 4; ++i)
      swizzled_color.channel[supported_format->unswizzle.channels[i]] =
         value.channel[i];

   return swizzled_color;
}

static void
kk_set_color_attachments(mtl_render_pass_descriptor *pass_descriptor,
                         struct kk_rendering_state *render,
                         const struct vk_dynamic_graphics_state *dyn,
                         bool force_attachment_load,
                         bool rendering_to_whole_framebuffer)
{
   for (uint32_t i = 0; i < ARRAY_SIZE(render->color_att); i++) {
      mtl_render_pass_attachment_descriptor *attachment_descriptor =
         mtl_render_pass_descriptor_get_color_attachment(pass_descriptor, i);
      kk_clear_common_attachment_description(attachment_descriptor);
   }
   for (uint32_t i = 0; i < ARRAY_SIZE(render->color_att); i++) {
      struct kk_attachment *color_att = &render->color_att[i];
      const struct kk_image_view *iview = color_att->iview;
      uint8_t logical_index = dyn->cal.color_map[i];
      render->color_map[i] = logical_index;

      if (!iview || logical_index == MESA_VK_ATTACHMENT_UNUSED) {
         continue;
      }
      mtl_render_pass_attachment_descriptor *attachment_descriptor =
         mtl_render_pass_descriptor_get_color_attachment(pass_descriptor,
                                                         logical_index);

      assert(iview->plane_count ==
             1); /* TODO_KOSMICKRISP Handle multiplanar images? */
      const struct kk_image *image =
         container_of(iview->vk.image, struct kk_image, vk);
      render->samples = MAX2(render->samples, image->vk.samples);

      bool renderpass_resolve = kk_attachment_do_renderpass_resolve(
         &render->color_att[i], rendering_to_whole_framebuffer,
         VK_IMAGE_ASPECT_COLOR_BIT);

      kk_fill_common_attachment_description(attachment_descriptor, color_att,
                                            force_attachment_load,
                                            renderpass_resolve);

      struct mtl_clear_color clear_color =
         vk_clear_color_value_to_mtl_clear_color(color_att->clear_value.color,
                                                 iview->planes[0].format);
      mtl_render_pass_attachment_descriptor_set_clear_color(
         attachment_descriptor, clear_color);
   }
}

static bool
kk_can_clear_subsampled_attachments(const struct kk_rendering_state *render)
{
   if (!KK_EXPERIMENTAL(TILE_RESOLVE) ||
       !(render->flags & VK_RENDERING_CUSTOM_RESOLVE_BIT_EXT) ||
       (render->flags & VK_RENDERING_RESUMING_BIT) ||
       render->area.offset.x || render->area.offset.y)
      return false;

   bool any = false;
   for (uint32_t i = 0; i < render->color_att_count + 2; i++) {
      const struct kk_attachment *attachment =
         i < render->color_att_count ? &render->color_att[i]
         : i == render->color_att_count ? &render->depth_att
                                         : &render->stencil_att;
      if (!attachment->iview)
         continue;
      const struct vk_image_view *view = &attachment->iview->vk;
      bool all_layers = render->view_mask
                           ? render->view_mask == BITFIELD64_MASK(view->layer_count)
                           : render->layer_count == view->layer_count;
      if (!(view->image->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT) ||
          view->image->samples == VK_SAMPLE_COUNT_1_BIT || !all_layers ||
          (attachment->load_op != VK_ATTACHMENT_LOAD_OP_CLEAR &&
           attachment->load_op != VK_ATTACHMENT_LOAD_OP_DONT_CARE) ||
          attachment->store_op != VK_ATTACHMENT_STORE_OP_DONT_CARE)
         return false;
      any = true;
   }
   return any;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct kk_rendering_state *render = &cmd->state.gfx.render;

   memset(render, 0, sizeof(*render));

   render->flags = pRenderingInfo->flags;
   if (render->flags & VK_RENDERING_RESUMING_BIT)
      cmd->state.gfx.restart_barrier = true;
   render->area = pRenderingInfo->renderArea;
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->samples = 0;
   render->color_att_count = pRenderingInfo->colorAttachmentCount;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   VkExtent2D framebuffer_extent = {.width = 0u, .height = 0u};
   bool does_any_attachment_clear = false;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      kk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
      /* The multisampled route needs the match before choosing the hidden resolve. */
      if (render->color_att[i].iview && kk_implicit_fdm_enabled() &&
          !debug_get_bool_option("KK_IMPLICIT_SINGLE", false))
         kk_implicit_fdm_match(container_of(render->color_att[i].iview->vk.image, struct kk_image, vk));
      if (render->color_att[i].iview && render->color_att[i].resolve_iview &&
          render->color_att[i].resolve_mode == VK_RESOLVE_MODE_AVERAGE_BIT) {
         struct kk_image_view *hidden = kk_implicit_resolve_view(
            dev, container_of(render->color_att[i].iview->vk.image, struct kk_image, vk),
            render->color_att[i].resolve_iview);
         if (hidden) {
            render->color_att[i].implicit_dst = render->color_att[i].resolve_iview;
            render->color_att[i].resolve_iview = hidden;
         }
      }
      struct kk_image_view *views[] = {
         render->color_att[i].iview, render->color_att[i].resolve_iview,
      };
      for (unsigned v = 0; v < ARRAY_SIZE(views); v++) {
         if (!views[v])
            continue;
         struct kk_image *image =
            container_of(views[v]->vk.image, struct kk_image, vk);
         if (image->wsi_direct)
            util_dynarray_append(&cmd->wsi_direct_images, image);
      }
      kk_merge_render_iview(&framebuffer_extent, render->color_att[i].iview);
      does_any_attachment_clear |=
         (pRenderingInfo->pColorAttachments[i].loadOp ==
          VK_ATTACHMENT_LOAD_OP_CLEAR);
   }
   if (pRenderingInfo->pDepthAttachment)
      does_any_attachment_clear |= (pRenderingInfo->pDepthAttachment->loadOp ==
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
   if (pRenderingInfo->pStencilAttachment)
      does_any_attachment_clear |=
         (pRenderingInfo->pStencilAttachment->loadOp ==
          VK_ATTACHMENT_LOAD_OP_CLEAR);

   kk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   kk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);
   if (kk_implicit_fdm_enabled())
      kk_implicit_match_pass(cmd, render);
   kk_merge_render_iview(&framebuffer_extent,
                         render->depth_att.iview ?: render->stencil_att.iview);

   const VkRenderingFragmentShadingRateAttachmentInfoKHR *fsr_att_info =
      vk_find_struct_const(pRenderingInfo->pNext,
                           RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR);
   if (fsr_att_info != NULL && fsr_att_info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_image_view, iview, fsr_att_info->imageView);
      render->vrs_note.texel = fsr_att_info->shadingRateAttachmentTexelSize;
      render->fsr_att = (struct kk_attachment){
         .vk_format = iview->vk.format,
         .iview = iview,
         .store_op = VK_ATTACHMENT_STORE_OP_NONE,
      };
   }

   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);

   kk_cmd_buffer_dirty_render_pass(cmd);
   mtl_render_pass_descriptor *pass_descriptor =
      mtl_new_render_pass_descriptor();

   /* Framebufferless rendering, need to set pass_descriptors
    * renderTargetWidth/Height to non-0 values and defaultRasterSampleCount.
    * However, since the sample count will only be known at first pipeline bind,
    * we need to delay the start of the pass until then since Metal will ignore
    * bound pipeline's sample count. */
   bool no_framebuffer =
      framebuffer_extent.width == 0u && framebuffer_extent.height == 0u;
   if (no_framebuffer) {
      framebuffer_extent.width = render->area.extent.width;
      framebuffer_extent.height = render->area.extent.height;
      mtl_render_pass_descriptor_set_render_target_width(
         pass_descriptor, framebuffer_extent.width);
      mtl_render_pass_descriptor_set_render_target_height(
         pass_descriptor, framebuffer_extent.height);
      /* Applications may set maximal extents for attachment-less rendering, for
       * example DXVK. This can cause device loss from exceeding memory limits.
       * We can get around this by ignoring the layer count, without any known
       * impact on the resulting fragment invocations and side effects. */
      mtl_render_pass_descriptor_set_render_target_array_length(pass_descriptor,
                                                                0u);

      mtl_render_pass_descriptor_set_default_raster_sample_count(
         pass_descriptor, 1u);
   } else {
      mtl_render_pass_descriptor_set_render_target_width(
         pass_descriptor, render->area.extent.width + render->area.offset.x);
      mtl_render_pass_descriptor_set_render_target_height(
         pass_descriptor, render->area.extent.height + render->area.offset.y);

      /* Render targets are always arrays */
      mtl_render_pass_descriptor_set_render_target_array_length(
         pass_descriptor, layer_count ? layer_count : 1u);
   }

   /* Check if we are rendering to the whole framebuffer. Required to understand
    * if we need to load to avoid clearing all attachment when loading.
    */
   uint32_t view_layers = render->layer_count;
   if (KK_EXPERIMENTAL(TILE_RESOLVE) && render->view_mask) {
      view_layers = UINT32_MAX;
      for (uint32_t i = 0; i < render->color_att_count; i++)
         if (render->color_att[i].iview)
            view_layers = MIN2(view_layers, render->color_att[i].iview->vk.layer_count);
      if (render->depth_att.iview)
         view_layers = MIN2(view_layers, render->depth_att.iview->vk.layer_count);
      if (render->stencil_att.iview)
         view_layers = MIN2(view_layers, render->stencil_att.iview->vk.layer_count);
      if (view_layers == UINT32_MAX)
         view_layers = render->layer_count;
   }
   bool is_whole_framebuffer =
      framebuffer_extent.width == render->area.extent.width &&
      framebuffer_extent.height == render->area.extent.height &&
      render->area.offset.x == 0u && render->area.offset.y == 0u &&
      (render->view_mask == 0u ||
       render->view_mask == BITFIELD64_MASK(view_layers));

   render->rendering_to_whole_framebuffer = is_whole_framebuffer;

   /* renderTargetWidth/Height doesn't seem to guarantee
    * that the attachments won't get touched outside the area
    * so just checking for offset = 0 doesn't cut it. */
   render->stitch = KK_EXPERIMENTAL(STITCH_PASSES) && !cmd->no_stitch &&
                    (render->flags & (VK_RENDERING_RESUMING_BIT |
                                      VK_RENDERING_SUSPENDING_BIT));

   /* A stitched pass is followed by a queue wait, which also orders its
    * depth/stencil resolves. The render target size is the area, so with no
    * offset the resolve covers exactly the area. */
   bool ds_in_pass = KK_EXPERIMENTAL(DS_RESOLVE) && render->stitch &&
                     render->area.offset.x == 0 && render->area.offset.y == 0;
   render->depth_att.resolve_in_pass = ds_in_pass;
   render->stencil_att.resolve_in_pass = ds_in_pass;
   bool force_attachment_load =
      (!is_whole_framebuffer && !kk_can_clear_subsampled_attachments(render)) ||
      ((render->flags & VK_RENDERING_RESUMING_BIT) && !render->stitch);

   render->force_attachment_store =
      !is_whole_framebuffer || (render->flags & VK_RENDERING_SUSPENDING_BIT);

   kk_set_color_attachments(pass_descriptor, render, dyn, force_attachment_load,
                            is_whole_framebuffer);

   if (render->depth_att.iview) {
      const struct kk_image_view *iview = render->depth_att.iview;
      const struct kk_image *image =
         container_of(iview->vk.image, struct kk_image, vk);
      render->samples = image->vk.samples;

      bool renderpass_resolve = kk_attachment_do_renderpass_resolve(
         &render->depth_att, is_whole_framebuffer, VK_IMAGE_ASPECT_DEPTH_BIT);

      mtl_render_pass_attachment_descriptor *attachment_descriptor =
         mtl_render_pass_descriptor_get_depth_attachment(pass_descriptor);
      kk_fill_common_attachment_description(
         attachment_descriptor, &render->depth_att, force_attachment_load,
         renderpass_resolve);

      /* clearValue.depthStencil.depth could have invalid values such as NaN
       * which will trigger a Metal validation error. Ensure we only use this
       * value if the attachment is actually cleared. */
      if (pRenderingInfo->pDepthAttachment->loadOp ==
          VK_ATTACHMENT_LOAD_OP_CLEAR)
         mtl_render_pass_attachment_descriptor_set_clear_depth(
            attachment_descriptor,
            pRenderingInfo->pDepthAttachment->clearValue.depthStencil.depth);

      if (renderpass_resolve)
         mtl_render_pass_attachment_descriptor_set_depth_resolve_filter(
            attachment_descriptor, vk_resolve_mode_to_mtl_depth_resolve_filter(
                                      render->depth_att.resolve_mode));
   }
   if (render->stencil_att.iview) {
      const struct kk_image_view *iview = render->stencil_att.iview;
      const struct kk_image *image =
         container_of(iview->vk.image, struct kk_image, vk);
      render->samples = image->vk.samples;

      bool renderpass_resolve = kk_attachment_do_renderpass_resolve(
         &render->stencil_att, is_whole_framebuffer,
         VK_IMAGE_ASPECT_STENCIL_BIT);

      mtl_render_pass_attachment_descriptor *attachment_descriptor =
         mtl_render_pass_descriptor_get_stencil_attachment(pass_descriptor);
      kk_fill_common_attachment_description(
         attachment_descriptor, &render->stencil_att, force_attachment_load,
         renderpass_resolve);
      mtl_render_pass_attachment_descriptor_set_clear_stencil(
         attachment_descriptor,
         pRenderingInfo->pStencilAttachment->clearValue.depthStencil.stencil);

      if (renderpass_resolve)
         mtl_render_pass_attachment_descriptor_set_stencil_resolve_filter(
            attachment_descriptor,
            vk_resolve_mode_to_mtl_stencil_resolve_filter(
               render->stencil_att.resolve_mode));
   }

   /* Set global visibility buffer */
   mtl_render_pass_descriptor_set_visibility_buffer(
      pass_descriptor, dev->occlusion_queries.bo->map);

   // TODO_KOSMICKRISP Fragment shading rate support goes here if Metal supports
   // it

   kk_fdm_setup_rendering(cmd, pRenderingInfo, pass_descriptor,
                          framebuffer_extent);
   kk_seq_log(cmd, "begin rendering%s flags 0x%x samples %u area %ux%u colors %u "
                   "resolve %d/%d depth %s resolve %d stencil resolve %d views 0x%x",
              cmd->state.gfx.in_custom_resolve ? " (custom resolve stage)" : "",
              render->flags, render->samples, render->area.extent.width,
              render->area.extent.height, render->color_att_count,
              render->color_att_count ? (int)render->color_att[0].resolve_mode : -1,
              render->color_att_count > 1 ? (int)render->color_att[1].resolve_mode : -1,
              render->depth_att.iview ? "yes" : "no", (int)render->depth_att.resolve_mode,
              (int)render->stencil_att.resolve_mode, render->view_mask);
   kk_seq_log(cmd, "  colour load %d store %d, depth load %d store %d, force store %d",
              render->color_att_count ? (int)render->color_att[0].load_op : -1,
              render->color_att_count ? (int)render->color_att[0].store_op : -1,
              (int)render->depth_att.load_op, (int)render->depth_att.store_op,
              (int)render->force_attachment_store);

   if (!cmd->state.gfx.in_custom_resolve) {
      struct kk_graphics_state *gfx = &cmd->state.gfx;
      gfx->tile_resolve.possible = false;
      gfx->tile_resolve.active = false;
      gfx->tile_resolve.pending_split = false;
      if (KK_EXPERIMENTAL(TILE_RESOLVE) &&
          (render->flags & VK_RENDERING_CUSTOM_RESOLVE_BIT_EXT) &&
          render->samples > 1 && render->color_att_count) {
         uint32_t px = 32;
         while (px > 8 && render->color_att_count * 16u * render->samples * px * px >
                             32768u)
            px /= 2;
         gfx->tile_resolve.possible = !(render->flags & VK_RENDERING_SUSPENDING_BIT);
         gfx->tile_resolve.tile_px = px;
         mtl_render_pass_descriptor_set_tile_size(pass_descriptor, px, px);
      }
   }

   /* Rendering with no attachments requires pushing the start of the render
    * pass to first pipeline binding to know sample count. */
   cmd->state.gfx.render_pass_descriptor = pass_descriptor;
   if (!no_framebuffer)
      cs_start_render(cmd);

   /* Store descriptor in case we need to restart the pass at pipeline barrier,
    * but force loads */
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      const struct kk_image_view *iview = render->color_att[i].iview;
      if (!iview)
         continue;
      mtl_render_pass_attachment_descriptor *attachment_descriptor =
         mtl_render_pass_descriptor_get_color_attachment(pass_descriptor, i);
      mtl_render_pass_attachment_descriptor_set_load_action(
         attachment_descriptor, MTL_LOAD_ACTION_LOAD);
   }
   if (render->depth_att.iview) {
      mtl_render_pass_attachment_descriptor *attachment_descriptor =
         mtl_render_pass_descriptor_get_depth_attachment(pass_descriptor);
      mtl_render_pass_attachment_descriptor_set_load_action(
         attachment_descriptor, MTL_LOAD_ACTION_LOAD);
   }
   if (render->stencil_att.iview) {
      mtl_render_pass_attachment_descriptor *attachment_descriptor =
         mtl_render_pass_descriptor_get_stencil_attachment(pass_descriptor);
      mtl_render_pass_attachment_descriptor_set_load_action(
         attachment_descriptor, MTL_LOAD_ACTION_LOAD);
   }
   cmd->state.gfx.need_to_start_render_pass = no_framebuffer;

   kk_cmd_buffer_dirty_all_gfx(cmd);

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   /* Clear attachments if we forced a load and there's a clear */
   if (!force_attachment_load || !does_any_attachment_clear)
      return;

   uint32_t clear_count = 0;
   VkClearAttachment clear_att[KK_MAX_RTS + 1];
   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att_info =
         &pRenderingInfo->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE ||
          att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      clear_att[clear_count++] = (VkClearAttachment){
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = att_info->clearValue,
      };
   }

   clear_att[clear_count] = (VkClearAttachment){
      .aspectMask = 0,
   };
   if (pRenderingInfo->pDepthAttachment != NULL &&
       pRenderingInfo->pDepthAttachment->imageView != VK_NULL_HANDLE &&
       pRenderingInfo->pDepthAttachment->loadOp ==
          VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      clear_att[clear_count].clearValue.depthStencil.depth =
         pRenderingInfo->pDepthAttachment->clearValue.depthStencil.depth;
   }
   if (pRenderingInfo->pStencilAttachment != NULL &&
       pRenderingInfo->pStencilAttachment->imageView != VK_NULL_HANDLE &&
       pRenderingInfo->pStencilAttachment->loadOp ==
          VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      clear_att[clear_count].clearValue.depthStencil.stencil =
         pRenderingInfo->pStencilAttachment->clearValue.depthStencil.stencil;
   }
   if (clear_att[clear_count].aspectMask != 0)
      clear_count++;

   if (clear_count > 0) {
      const VkClearRect clear_rect = {
         .rect = render->area,
         .baseArrayLayer = 0,
         .layerCount = render->view_mask ? 1 : render->layer_count,
      };

      kk_CmdClearAttachments(kk_cmd_buffer_to_handle(cmd), clear_count,
                             clear_att, 1, &clear_rect);
   }
}


enum kk_custom_resolve_debug_mode {
   KK_CR_DEBUG_NONE,
   KK_CR_DEBUG_LOG,
   KK_CR_DEBUG_MAGENTA,
   KK_CR_DEBUG_AVERAGE,
};

static enum kk_custom_resolve_debug_mode
kk_custom_resolve_debug_mode(void)
{
   static int mode = -1;
   if (mode < 0) {
      const char *s = debug_get_option("KK_CUSTOM_RESOLVE_DEBUG", "");
      mode = !strcmp(s, "log")       ? KK_CR_DEBUG_LOG
             : !strcmp(s, "magenta") ? KK_CR_DEBUG_MAGENTA
             : !strcmp(s, "average") ? KK_CR_DEBUG_AVERAGE
                                     : KK_CR_DEBUG_NONE;
   }
   return mode;
}

static void
kk_custom_resolve_debug_log(const struct kk_rendering_state *render)
{
   static uint32_t logged;
   if (logged >= 8)
      return;
   logged++;
   mesa_logi("kk-cr: flags=0x%x area=%d,%d/%ux%u layers=%u view_mask=0x%x colors=%u",
             render->flags, render->area.offset.x, render->area.offset.y,
             render->area.extent.width, render->area.extent.height,
             render->layer_count, render->view_mask, render->color_att_count);
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      const struct kk_attachment *a = &render->color_att[i];
      if (!a->iview)
         continue;
      const struct vk_image *im = a->iview->vk.image;
      const struct vk_image *rim = a->resolve_iview ? a->resolve_iview->vk.image : NULL;
      mesa_logi("kk-cr: color%u mode=0x%x load=%d store=%d fmt=%d %ux%u layers=%u "
                "samples=%u usage=0x%x -> resolve fmt=%d %ux%u layers=%u usage=0x%x",
                i, a->resolve_mode, a->load_op, a->store_op, a->vk_format,
                im->extent.width, im->extent.height, im->array_layers,
                im->samples, (unsigned)im->usage, rim ? rim->format : 0,
                rim ? rim->extent.width : 0, rim ? rim->extent.height : 0,
                rim ? rim->array_layers : 0, rim ? (unsigned)rim->usage : 0);
   }
   mesa_logi("kk-cr: depth mode=0x%x stencil mode=0x%x whole=%d force_store=%d",
             render->depth_att.resolve_mode, render->stencil_att.resolve_mode,
             render->rendering_to_whole_framebuffer,
             render->force_attachment_store);
}

struct kk_dump {
   struct kk_bo *bo;
   char path[512];
   uint32_t width, height, layers, bytes_per_pixel, format;
};

static int
kk_dump_thread(void *data)
{
   struct kk_dump *dump = data;
   uint64_t size = (uint64_t)dump->width * dump->height * dump->bytes_per_pixel *
                   dump->layers;
   volatile uint32_t *marker = (uint32_t *)((char *)dump->bo->cpu + size);
   for (int i = 0; i < 3000 && *marker != 1; i++)
      os_time_sleep(10000);
   if (*marker != 1) {
      mesa_logw("kk-dump: %s never completed", dump->path);
      free(dump);
      return 0;
   }
   FILE *f = fopen(dump->path, "wb");
   if (f) {
      fprintf(f, "KKDUMP %u %u %u %u %u\n", dump->width, dump->height,
              dump->layers, dump->bytes_per_pixel, dump->format);
      fwrite(dump->bo->cpu, 1, size, f);
      fclose(f);
      mesa_logi("kk-dump: wrote %s", dump->path);
   }
   free(dump);
   return 0;
}

static void
kk_dump_view(struct kk_cmd_buffer *cmd, const char *dir, const char *name,
             uint32_t index, const struct kk_image_view *view, VkRect2D area,
             bool depth)
{
   struct kk_image *image = container_of(view->vk.image, struct kk_image, vk);
   const struct kk_image_plane *plane = &image->planes[0];
   uint32_t width = area.extent.width, height = area.extent.height;
   enum pipe_format format = plane->layout.format.pipe;
   if (depth)
      format = util_format_get_depth_only(format);
   uint32_t bpp = util_format_get_blocksize(format);
   uint32_t layers = view->vk.layer_count;
   uint64_t size = (uint64_t)width * height * bpp * layers;

   struct kk_dump *dump = calloc(1, sizeof(*dump));
   if (!dump)
      return;
   if (kk_alloc_bo(kk_cmd_buffer_device(cmd), &cmd->vk.base, size + 4, 16,
                   &dump->bo) != VK_SUCCESS) {
      free(dump);
      return;
   }
   memset((char *)dump->bo->cpu + size, 0, 4);
   snprintf(dump->path, sizeof(dump->path), "%s/%s-%05u.kkdump", dir, name, index);
   dump->width = width;
   dump->height = height;
   dump->layers = layers;
   dump->bytes_per_pixel = bpp;
   dump->format = depth ? VK_FORMAT_D32_SFLOAT : view->vk.format;

   mtl_compute_encoder *enc = cs_get_compute(cmd);
   for (uint32_t l = 0; l < layers; l++) {
      struct mtl_buffer_image_copy copy = {
         .image_size = {.x = width, .y = height, .z = 1},
         .image_origin = {.x = area.offset.x, .y = area.offset.y, .z = 0},
         .buffer = dump->bo->map,
         .image = plane->mtl_handle,
         .buffer_offset_B = (size_t)width * height * bpp * l,
         .buffer_stride_B = (size_t)width * bpp,
         .buffer_2d_image_size_B = (size_t)width * height * bpp,
         .image_slice = view->vk.base_array_layer + l,
         .image_level = view->vk.base_mip_level,
         .options = depth ? MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL
                          : MTL_BLIT_OPTION_NONE,
      };
      mtl_copy_from_texture_to_buffer(enc, &copy);
   }
   mtl_barrier_after_encoder_stages(enc, MTL_STAGE_BLIT, MTL_STAGE_DISPATCH);
   libkk_write_u32(cmd, kk_grid_1d(1), true, dump->bo->gpu + size, 1);

   thrd_t thread;
   if (thrd_create(&thread, kk_dump_thread, dump) == thrd_success)
      thrd_detach(thread);
}

/* KK_DUMP_PASSES_DIR writes colour attachment 0 of every multiview pass
 * from pass KK_DUMP_PASSES_FIRST for KK_DUMP_PASSES_COUNT passes. */
static void
kk_dump_pass(struct kk_cmd_buffer *cmd, const struct kk_image_view *view,
             VkRect2D area)
{
   static const char *dir;
   static int64_t first = -1, count;
   static uint32_t seen;
   if (first < 0) {
      dir = debug_get_option("KK_DUMP_PASSES_DIR", NULL);
      first = debug_get_num_option("KK_DUMP_PASSES_FIRST", 6000);
      count = debug_get_num_option("KK_DUMP_PASSES_COUNT", 40);
   }
   if (!dir || !view || view->vk.image->samples > 1)
      return;
   uint32_t index = p_atomic_inc_return(&seen) - 1;
   if (index < first || index - first >= count)
      return;
   char name[64];
   snprintf(name, sizeof(name), "pass-f%d-%ux%u%s", view->vk.format,
            area.extent.width, area.extent.height,
            (view->vk.image->create_flags & VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT)
               ? "-sub"
               : "");
   kk_dump_view(cmd, dir, name, index, view, area, false);
}

static void
kk_dump_custom_resolve(struct kk_cmd_buffer *cmd)
{
   static const char *dir;
   static int64_t first = -1, step, count;
   static uint32_t seen;
   if (first < 0) {
      dir = debug_get_option("KK_DUMP_RESOLVE_DIR", NULL);
      first = debug_get_num_option("KK_DUMP_RESOLVE_FIRST", 600);
      step = MAX2(debug_get_num_option("KK_DUMP_RESOLVE_STEP", 600), 1);
      count = debug_get_num_option("KK_DUMP_RESOLVE_COUNT", 4);
   }
   const struct kk_image_view *view = cmd->state.gfx.dump.view;
   const struct kk_image_view *depth_view = cmd->state.gfx.dump.depth_view;
   cmd->state.gfx.dump.view = NULL;
   cmd->state.gfx.dump.depth_view = NULL;
   if (!dir || !view)
      return;

   uint32_t index = p_atomic_inc_return(&seen) - 1;
   if (index < first || (index - first) % step || (index - first) / step >= count)
      return;

   kk_dump_view(cmd, dir, "resolve", index, view, cmd->state.gfx.dump.area, false);
   if (depth_view)
      kk_dump_view(cmd, dir, "depth", index, depth_view, cmd->state.gfx.dump.area,
                   true);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdEndRendering2KHR(VkCommandBuffer commandBuffer,
                       UNUSED const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_rendering_state *render = &cmd->state.gfx.render;
   bool need_meta_resolve = false;
   kk_seq_log(cmd, "end rendering");
   kk_vrs_end(cmd);
   cmd->state.gfx.in_custom_resolve = false;
   cmd->state.gfx.tile_resolve.possible = false;
   cmd->state.gfx.tile_resolve.active = false;
   cmd->state.gfx.tile_resolve.pending_split = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[KK_MAX_RTS];
   VkRenderingAttachmentFlagsInfoKHR vk_color_att_flags[KK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      bool attachment_meta_resolve =
         render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE &&
         render->color_att[i].resolve_mode != VK_RESOLVE_MODE_CUSTOM_BIT_EXT &&
         !kk_attachment_do_renderpass_resolve(
            &render->color_att[i], render->rendering_to_whole_framebuffer,
            VK_IMAGE_ASPECT_COLOR_BIT);
      need_meta_resolve |= attachment_meta_resolve;

      vk_color_att_flags[i] = (VkRenderingAttachmentFlagsInfoKHR){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
         .flags = render->color_att[i].flags,
      };
      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .pNext = &vk_color_att_flags[i],
         .imageView = kk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = attachment_meta_resolve
                           ? render->color_att[i].resolve_mode
                           : VK_RESOLVE_MODE_NONE,
         .resolveImageView =
            attachment_meta_resolve
               ? kk_image_view_to_handle(render->color_att[i].resolve_iview)
               : VK_NULL_HANDLE,
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   bool depth_meta_resolve =
      render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE &&
      render->depth_att.resolve_mode != VK_RESOLVE_MODE_CUSTOM_BIT_EXT &&
      !kk_attachment_do_renderpass_resolve(
         &render->depth_att, render->rendering_to_whole_framebuffer,
         VK_IMAGE_ASPECT_DEPTH_BIT);
   need_meta_resolve |= depth_meta_resolve;

   const VkRenderingAttachmentFlagsInfoKHR vk_depth_att_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->depth_att.flags,
   };
   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_depth_att_flags,
      .imageView = kk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = depth_meta_resolve ? render->depth_att.resolve_mode
                                        : VK_RESOLVE_MODE_NONE,
      .resolveImageView =
         depth_meta_resolve
            ? kk_image_view_to_handle(render->depth_att.resolve_iview)
            : VK_NULL_HANDLE,
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };

   bool stencil_meta_resolve =
      render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE &&
      render->stencil_att.resolve_mode != VK_RESOLVE_MODE_CUSTOM_BIT_EXT &&
      !kk_attachment_do_renderpass_resolve(
         &render->stencil_att, render->rendering_to_whole_framebuffer,
         VK_IMAGE_ASPECT_STENCIL_BIT);
   need_meta_resolve |= stencil_meta_resolve;

   const VkRenderingAttachmentFlagsInfoKHR vk_stencil_att_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->stencil_att.flags,
   };
   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_stencil_att_flags,
      .imageView = kk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = stencil_meta_resolve ? render->stencil_att.resolve_mode
                                          : VK_RESOLVE_MODE_NONE,
      .resolveImageView =
         stencil_meta_resolve
            ? kk_image_view_to_handle(render->stencil_att.resolve_iview)
            : VK_NULL_HANDLE,
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   const VkRenderingFlags pass_flags_for_log = render->flags;
   const int had_encoder_for_log = cmd->metal.render != NULL;
   struct kk_image_view *implicit_src[KK_MAX_RTS] = {NULL}, *implicit_dst[KK_MAX_RTS] = {NULL};
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].implicit_dst && !(render->flags & VK_RENDERING_SUSPENDING_BIT)) {
         implicit_src[i] = render->color_att[i].resolve_iview;
         implicit_dst[i] = render->color_att[i].implicit_dst;
      }
   }
   const VkRect2D implicit_area = render->area;
   kk_apply_attachment_store_ops(cmd, false);

   /* Clean up previous encoder */
   cs_end(cmd);
   mtl_release(cmd->state.gfx.render_pass_descriptor);
   cmd->state.gfx.render_pass_descriptor = NULL;

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_meta_resolve = false;

   const struct kk_image_view *pass_view =
      render->color_att_count && render->view_mask &&
            !(render->flags & VK_RENDERING_SUSPENDING_BIT)
         ? render->color_att[0].iview
         : NULL;
   VkRect2D pass_area = render->area;

   bool full_rate_depth_resolve = render->fdm_discard_depth &&
      !(render->flags & VK_RENDERING_SUSPENDING_BIT);
   struct kk_image *discard_depth = NULL;
   struct kk_image *discard_stencil = NULL;
   struct kk_image *resolved_depth = NULL;
   struct kk_image *resolved_stencil = NULL;
   if (full_rate_depth_resolve) {
      if (render->depth_att.resolve_iview)
         resolved_depth = container_of(render->depth_att.resolve_iview->vk.image,
                                      struct kk_image, vk);
      if (render->stencil_att.resolve_iview)
         resolved_stencil = container_of(render->stencil_att.resolve_iview->vk.image,
                                        struct kk_image, vk);
      if (render->depth_att.iview &&
          !(render->depth_att.iview->vk.image->create_flags &
            VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT))
         discard_depth = container_of(render->depth_att.iview->vk.image,
                                       struct kk_image, vk);
      if (render->stencil_att.iview &&
          !(render->stencil_att.iview->vk.image->create_flags &
            VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT))
         discard_stencil = container_of(render->stencil_att.iview->vk.image,
                                         struct kk_image, vk);
   }

   memset(render, 0, sizeof(*render));

   if (kk_implicit_fdm_enabled() && vk_render.colorAttachmentCount &&
       vk_color_att[0].resolveImageView != VK_NULL_HANDLE) {
      static uint32_t logged;
      if (logged++ < 6)
         mesa_logi("kk-implicit: end rendering, meta resolve %d mode %d area %d,%d %ux%u layers %u views 0x%x flags 0x%x encoder %d",
                   need_meta_resolve, (int)vk_color_att[0].resolveMode, vk_render.renderArea.offset.x,
                   vk_render.renderArea.offset.y, vk_render.renderArea.extent.width,
                   vk_render.renderArea.extent.height, vk_render.layerCount, vk_render.viewMask,
                   (unsigned)pass_flags_for_log, had_encoder_for_log);
   }
   /* Implicit foveation: the pass resolved into the hidden foveated image; a sampled
    * blit reads it through the rate map into the application's ordinary target. */
   for (uint32_t i = 0; i < KK_MAX_RTS; i++) {
      if (!implicit_dst[i])
         continue;
      const VkOffset3D lo = {implicit_area.offset.x, implicit_area.offset.y, 0};
      const VkOffset3D hi = {implicit_area.offset.x + (int32_t)implicit_area.extent.width,
                             implicit_area.offset.y + (int32_t)implicit_area.extent.height, 1};
      const VkImageBlit2 region = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
         .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .srcOffsets = {lo, hi},
         .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, implicit_dst[i]->vk.base_mip_level,
                            implicit_dst[i]->vk.base_array_layer, 1},
         .dstOffsets = {lo, hi},
      };
      const VkBlitImageInfo2 blit = {
         .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
         .srcImage = vk_image_to_handle(implicit_src[i]->vk.image),
         .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .dstImage = vk_image_to_handle(implicit_dst[i]->vk.image),
         .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .regionCount = 1,
         .pRegions = &region,
         .filter = VK_FILTER_LINEAR,
      };
      if (debug_get_bool_option("KK_IMPLICIT_SKIP_BLIT", false))
         continue;
      if (debug_get_bool_option("KK_IMPLICIT_CLEAR_TEST", false)) {
         const VkClearColorValue magenta = {.float32 = {1.0f, 0.0f, 1.0f, 1.0f}};
         const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
         kk_CmdClearColorImage(kk_cmd_buffer_to_handle(cmd), blit.dstImage, VK_IMAGE_LAYOUT_GENERAL,
                               &magenta, 1, &range);
         static uint32_t logged;
         if (logged++ < 4)
            mesa_logi("kk-implicit: clear test on %p (pass colour %p)", (void *)implicit_dst[i]->vk.image,
                      (void *)implicit_src[i]->vk.image);
         continue;
      }
      kk_CmdBlitImage2(kk_cmd_buffer_to_handle(cmd), &blit);
   }

   if (need_meta_resolve) {
      if (full_rate_depth_resolve)
         cmd->state.gfx.fdm.carry = false;
      kk_meta_resolve_rendering(cmd, &vk_render);
   }
   kk_fdm_clear_image_mapping(cmd, resolved_depth);
   if (resolved_stencil != resolved_depth)
      kk_fdm_clear_image_mapping(cmd, resolved_stencil);
   kk_fdm_clear_image_mapping(cmd, discard_depth);
   if (discard_stencil != discard_depth)
      kk_fdm_clear_image_mapping(cmd, discard_stencil);

   if (cmd->state.gfx.custom_resolve_debug.active) {
      cmd->state.gfx.custom_resolve_debug.active = false;
      const VkRenderingAttachmentInfo avg_att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView =
            kk_image_view_to_handle(cmd->state.gfx.custom_resolve_debug.ms_iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT,
         .resolveImageView = kk_image_view_to_handle(
            cmd->state.gfx.custom_resolve_debug.resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      const VkRenderingInfo avg_render = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = cmd->state.gfx.custom_resolve_debug.area,
         .layerCount = cmd->state.gfx.custom_resolve_debug.layer_count,
         .viewMask = cmd->state.gfx.custom_resolve_debug.view_mask,
         .colorAttachmentCount = 1,
         .pColorAttachments = &avg_att,
      };
      kk_meta_resolve_rendering(cmd, &avg_render);
   }

   kk_dump_custom_resolve(cmd);
   kk_dump_pass(cmd, pass_view, pass_area);
}

static void
kk_begin_custom_resolve_legacy(struct kk_cmd_buffer *cmd)
{
   VkCommandBuffer commandBuffer = kk_cmd_buffer_to_handle(cmd);
   struct kk_rendering_state *render = &cmd->state.gfx.render;

   /* The resolve shader may read the multisampled depth and stencil, so they
    * have to be stored, and are then resolved after the pass as usual. */
   render->depth_att.resolve_in_pass = false;
   render->stencil_att.resolve_in_pass = false;

   VkRenderingAttachmentInfo color_atts[KK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      color_atts[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      };

      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_CUSTOM_BIT_EXT)
         continue;

      struct kk_image_view *iview = render->color_att[i].resolve_iview;

      color_atts[i].imageView = kk_image_view_to_handle(iview);
      color_atts[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
   }

   VkRenderingAttachmentInfo depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
   };
   if (render->depth_att.resolve_mode == VK_RESOLVE_MODE_CUSTOM_BIT_EXT) {
      struct kk_image_view *iview = render->depth_att.resolve_iview;

      depth_att.imageView = kk_image_view_to_handle(iview);
      depth_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
   }

   VkRenderingAttachmentInfo stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
   };
   if (render->stencil_att.resolve_mode == VK_RESOLVE_MODE_CUSTOM_BIT_EXT) {
      struct kk_image_view *iview = render->stencil_att.resolve_iview;

      stencil_att.imageView = kk_image_view_to_handle(iview);
      stencil_att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
   }

   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_LOCAL_READ_CONCURRENT_ACCESS_CONTROL_BIT_KHR,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = color_atts,
      .pDepthAttachment = &depth_att,
      .pStencilAttachment = &stencil_att,
   };

   const VkRenderingEndInfoKHR end_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_END_INFO_KHR,
   };

   uint64_t resolve_inputs[KK_MAX_RTS + 2] = {0};
   for (uint32_t i = 0; i < render->color_att_count && i < KK_MAX_RTS; i++) {
      if (render->color_att[i].iview)
         resolve_inputs[i] =
            render->color_att[i].iview->planes[0].input_gpu_resource_id;
   }
   if (render->depth_att.iview)
      resolve_inputs[KK_MAX_RTS] =
         render->depth_att.iview->planes[0].input_gpu_resource_id;
   if (render->stencil_att.iview)
      resolve_inputs[KK_MAX_RTS + 1] =
         render->stencil_att.iview->planes[0].input_gpu_resource_id;

   enum kk_custom_resolve_debug_mode debug = kk_custom_resolve_debug_mode();
   if (debug != KK_CR_DEBUG_NONE)
      kk_custom_resolve_debug_log(render);
   struct kk_image_view *debug_ms = NULL, *debug_resolve = NULL;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode == VK_RESOLVE_MODE_CUSTOM_BIT_EXT) {
         debug_ms = render->color_att[i].iview;
         debug_resolve = render->color_att[i].resolve_iview;
         break;
      }
   }

   kk_seq_log(cmd, "begin custom resolve");
   kk_CmdEndRendering2KHR(commandBuffer, &end_info);
   cmd->state.gfx.in_custom_resolve = true;
   kk_fdm_carry_to_next_rendering(cmd);
   cmd->state.gfx.restart_barrier = true;
   kk_CmdBeginRendering(commandBuffer, &rendering_info);

   struct kk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   memcpy(desc->root.custom_resolve_inputs, resolve_inputs,
          sizeof(resolve_inputs));
   desc->root_dirty = true;

   if (debug == KK_CR_DEBUG_MAGENTA && debug_resolve) {
      const VkClearAttachment clear = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = 0,
         .clearValue = {.color = {.float32 = {1.0f, 0.0f, 1.0f, 1.0f}}},
      };
      const VkClearRect rect = {
         .rect = rendering_info.renderArea,
         .baseArrayLayer = 0,
         .layerCount = rendering_info.viewMask ? 1 : rendering_info.layerCount,
      };
      kk_CmdClearAttachments(commandBuffer, 1, &clear, 1, &rect);
   } else if (debug == KK_CR_DEBUG_AVERAGE && debug_ms && debug_resolve) {
      cmd->state.gfx.custom_resolve_debug.ms_iview = debug_ms;
      cmd->state.gfx.custom_resolve_debug.resolve_iview = debug_resolve;
      cmd->state.gfx.custom_resolve_debug.area = rendering_info.renderArea;
      cmd->state.gfx.custom_resolve_debug.layer_count = rendering_info.layerCount;
      cmd->state.gfx.custom_resolve_debug.view_mask = rendering_info.viewMask;
      cmd->state.gfx.custom_resolve_debug.active = true;
   }
}

static bool
kk_tile_resolve_can_start(struct kk_cmd_buffer *cmd)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_rendering_state *render = &gfx->render;
   if (!gfx->tile_resolve.possible ||
       kk_custom_resolve_debug_mode() != KK_CR_DEBUG_NONE ||
       render->depth_att.resolve_mode == VK_RESOLVE_MODE_CUSTOM_BIT_EXT ||
       render->stencil_att.resolve_mode == VK_RESOLVE_MODE_CUSTOM_BIT_EXT)
      return false;
   bool any = false;
   for (uint32_t i = 0; i < render->color_att_count && i < KK_MAX_RTS; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_CUSTOM_BIT_EXT)
         continue;
      const struct kk_image_view *iview = render->color_att[i].resolve_iview;
      if (!iview || !iview->planes[0].storage_gpu_resource_id)
         return false;
      any = true;
   }
   return any;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBeginCustomResolveEXT(
   VkCommandBuffer commandBuffer,
   UNUSED const VkBeginCustomResolveInfoEXT *pBeginCustomResolveInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_graphics_state *gfx = &cmd->state.gfx;

   gfx->dump.view = gfx->render.color_att_count &&
                          gfx->render.color_att[0].resolve_mode ==
                             VK_RESOLVE_MODE_CUSTOM_BIT_EXT
                       ? gfx->render.color_att[0].resolve_iview
                       : NULL;
   gfx->dump.area = gfx->render.area;
   gfx->dump.depth_view = gfx->render.depth_att.resolve_mode != VK_RESOLVE_MODE_NONE &&
                                gfx->render.depth_att.resolve_mode !=
                                   VK_RESOLVE_MODE_CUSTOM_BIT_EXT
                             ? gfx->render.depth_att.resolve_iview
                             : NULL;

   if (!kk_tile_resolve_can_start(cmd)) {
      gfx->tile_resolve.possible = false;
      kk_begin_custom_resolve_legacy(cmd);
      return;
   }

   kk_seq_log(cmd, "begin custom resolve in tile memory");
   struct kk_rendering_state *render = &gfx->render;
   struct kk_descriptor_state *desc = &gfx->descriptors;
   for (uint32_t i = 0; i < KK_MAX_RTS; i++) {
      const struct kk_image_view *iview =
         i < render->color_att_count &&
               render->color_att[i].resolve_mode == VK_RESOLVE_MODE_CUSTOM_BIT_EXT
            ? render->color_att[i].resolve_iview
            : NULL;
      desc->root.custom_resolve_outputs[i] =
         iview ? iview->planes[0].storage_gpu_resource_id : 0;
   }
   desc->root_dirty = true;
   gfx->tile_resolve.active = true;
   gfx->tile_resolve.pending_split = false;
}

static bool
kk_draw_prologue(struct kk_cmd_buffer *cmd)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;

   struct kk_shader *pipeline_vs = cmd->state.shaders[MESA_SHADER_VERTEX];
   if (pipeline_vs && !kk_shader_pipeline_ready(pipeline_vs)) {
      kk_encoder_stat("draw skipped, pipeline still compiling", false);
      return true;
   }

   const struct kk_shader *fs = cmd->state.shaders[MESA_SHADER_FRAGMENT];
   bool fetch_only = KK_EXPERIMENTAL(MS_INPUT_FETCH) &&
                     (!fs || !fs->info.fs.reads_ms_texture);
   if (gfx->tile_resolve.pending_split && !gfx->tile_resolve.active &&
       fetch_only)
      kk_seq_log(cmd, "deferred barrier kept, draw reads no multisampled texture");
   if (gfx->tile_resolve.pending_split && !gfx->tile_resolve.active &&
       !fetch_only) {
      gfx->tile_resolve.pending_split = false;
      if (cmd->metal.render) {
         kk_seq_log(cmd, "deferred barrier split");
         kk_apply_attachment_store_ops(cmd, true);
         cs_end(cmd);
         gfx->restart_barrier = true;
         cs_start_render(cmd);
      }
   }

   if (!gfx->tile_resolve.active)
      return false;

   struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
   mtl_render_pipeline_state *tile =
      vs ? kk_shader_get_tile_pipeline(kk_cmd_buffer_device(cmd), vs,
                                       gfx->render.samples)
         : NULL;
   if (!tile) {
      kk_seq_log(cmd, "no tile variant, custom resolve falls back");
      gfx->tile_resolve.active = false;
      gfx->tile_resolve.possible = false;
      kk_begin_custom_resolve_legacy(cmd);
      return false;
   }

   struct kk_descriptor_state *desc = &gfx->descriptors;
   mtl_render_encoder *enc = cs_get_render(cmd);
   if (desc->push_dirty)
      kk_cmd_buffer_flush_push_descriptors(cmd, desc);
   if (desc->root_dirty)
      kk_upload_descriptor_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   if (desc->root.addr)
      kk_cmd_bind_root_to_argument_table(cmd, desc->root.addr);
   mtl_render_set_pipeline_state(enc, tile);
   mtl_dispatch_threads_per_tile(enc, gfx->tile_resolve.tile_px,
                                 gfx->tile_resolve.tile_px);
   kk_seq_log(cmd, "tile resolve dispatch %ux%u", gfx->tile_resolve.tile_px,
              gfx->tile_resolve.tile_px);
   kk_cmd_buffer_dirty_all_gfx(cmd);
   return true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBindIndexBuffer3KHR(VkCommandBuffer commandBuffer,
                          const VkBindIndexBuffer3InfoKHR *pInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   cmd->state.gfx.index.gpu.addr = pInfo->addressRange.address;
   cmd->state.gfx.index.gpu.range = pInfo->addressRange.size;
   cmd->state.gfx.index.bytes_per_index =
      vk_index_type_to_bytes(pInfo->indexType);
   vk_cmd_set_index_buffer_type(&cmd->vk, pInfo->indexType);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBindVertexBuffers3KHR(VkCommandBuffer commandBuffer,
                            uint32_t firstBinding, uint32_t bindingCount,
                            const VkBindVertexBuffer3InfoKHR *pBindingInfos)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   vk_cmd_set_vertex_binding_strides2(&cmd->vk, firstBinding, bindingCount,
                                      pBindingInfos);

   for (uint32_t i = 0; i < bindingCount; i++) {
      const VkStridedDeviceAddressRangeKHR range =
         pBindingInfos[i].addressRange;
      struct kk_addr_range *vb_range =
         &cmd->state.gfx.vb.addr_range[firstBinding + i];
      vb_range->addr = range.address;
      vb_range->range = range.size;
      cmd->state.gfx.dirty |= KK_DIRTY_VB;
   }
}

static void
kk_flush_vp_state(struct kk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   kk_vrs_note_draw(cmd);

   /* We always need at least 1 viewport for the hardware. With rasterizer
    * discard the app may not supply any, but we can just program garbage.
    */
   unsigned count = MAX2(dyn->vp.scissor_count, 1);

   /* Need to clamp scissor rectangles to render area, otherwise Metal doesn't
    * like it */
   struct mtl_scissor_rect rects[KK_MAX_VIEWPORTS] = {0};
   VkOffset2D origin = cmd->state.gfx.render.area.offset;
   VkOffset2D end = {.x = origin.x + cmd->state.gfx.render.area.extent.width,
                     .y = origin.y + cmd->state.gfx.render.area.extent.height};
   for (uint32_t i = 0; i < dyn->vp.scissor_count; i++) {
      const VkRect2D *rect = &dyn->vp.scissors[i];

      size_t x0 = CLAMP(rect->offset.x, origin.x, end.x);
      size_t x1 = CLAMP(rect->offset.x + rect->extent.width, origin.x, end.x);
      size_t y0 = CLAMP(rect->offset.y, origin.y, end.y);
      size_t y1 = CLAMP(rect->offset.y + rect->extent.height, origin.y, end.y);
      size_t minx = MIN2(x0, x1);
      size_t miny = MIN2(y0, y1);
      size_t maxx = MAX2(x0, x1);
      size_t maxy = MAX2(y0, y1);
      rects[i].x = minx;
      rects[i].y = miny;
      rects[i].width = maxx - minx;
      rects[i].height = maxy - miny;
   }

   mtl_render_encoder *encoder = cs_get_render(cmd);
   mtl_set_scissor_rects(encoder, rects, count);

   count = MAX2(dyn->vp.viewport_count, 1);

   struct mtl_viewport viewports[KK_MAX_VIEWPORTS] = {0};

   /* NDC in Metal is pointing downwards. Vulkan is pointing upwards. Account
    * for that here */
   for (uint32_t i = 0; i < dyn->vp.viewport_count; i++) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      viewports[i].originX = vp->x;
      viewports[i].originY = vp->y + vp->height;
      viewports[i].width = vp->width;
      viewports[i].height = -vp->height;

      if (dyn->rs.depth_clamp_enable ||
          vk_rasterization_state_depth_clip_enable(&dyn->rs)) {
         viewports[i].znear = vp->minDepth;
         viewports[i].zfar = vp->maxDepth;
      } else {
         /* When clamp and clip are disabled, we set the viewport Z range to
          * [0, 1] to disable hardware clamping. The viewport transform will be
          * applied in shader */
         viewports[i].znear = 0.f;
         viewports[i].zfar = 1.f;
      }
   }

   mtl_set_viewports(encoder, viewports, count);
}

static inline uint32_t
kk_calculate_vbo_clamp(uint64_t vbuf, uint64_t sink, enum pipe_format format,
                       uint32_t size_B, uint32_t stride_B, uint32_t offset_B,
                       uint64_t *vbuf_out)
{
   unsigned elsize_B = util_format_get_blocksize(format);
   unsigned subtracted_B = offset_B + elsize_B;

   /* If at least one index is valid, determine the max. Otherwise, direct reads
    * to zero.
    */
   if (size_B >= subtracted_B) {
      *vbuf_out = vbuf + offset_B;

      /* If stride is zero, do not clamp, everything is valid. */
      if (stride_B)
         return ((size_B - subtracted_B) / stride_B);
      else
         return UINT32_MAX;
   } else {
      *vbuf_out = sink;
      return 0;
   }
}

/* The GS output primitive if one is bound, otherwise the input topology */
static enum mesa_prim
kk_rast_prim(struct kk_cmd_buffer *cmd)
{
   struct kk_shader *gs = cmd->state.shaders[MESA_SHADER_GEOMETRY];
   if (gs)
      return gs->info.gs.mode;

   return vk_topology_to_mesa(
      cmd->vk.dynamic_graphics_state.ia.primitive_topology);
}

static void
set_empty_scissor(mtl_render_encoder *enc)
{
   struct mtl_scissor_rect rect = {.x = 0u, .y = 0u, .width = 0u, .height = 0u};
   mtl_set_scissor_rects(enc, &rect, 1);
}

#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)
#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.dirty_shaders & BITFIELD_BIT(MESA_SHADER_##bit))

static bool
kk_flush_sample_locations(struct kk_cmd_buffer *cmd)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_rendering_state *render = &gfx->render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   bool needs_update = false;

   /* Determine if the user-provided custom sample locations have changed */
   if (IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) || IS_DIRTY(MS_SAMPLE_LOCATIONS)) {
      needs_update |=
         render->sample_locations_enable != dyn->ms.sample_locations_enable;
      render->sample_locations_enable = dyn->ms.sample_locations_enable;

      if (render->sample_locations_enable) {
         struct vk_sample_locations_state *sample_locations =
            dyn->ms.sample_locations;

         uint32_t count = sample_locations->per_pixel *
                          sample_locations->grid_size.width *
                          sample_locations->grid_size.height;
         needs_update |= render->sample_locations_count != count;
         needs_update |= memcmp(render->sample_locations,
                                dyn->ms.sample_locations->locations,
                                count * sizeof(VkSampleLocationEXT)) != 0;

         render->sample_locations_count = count;
         typed_memcpy(render->sample_locations,
                      dyn->ms.sample_locations->locations, count);
      }
   }

   /* Determine if we have switched to or from multisampled bresenham lines */
   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || IS_DIRTY(RS_LINE_MODE)) {
      bool was_ms_bresenham_lines = render->ms_bresenham_lines;
      render->ms_bresenham_lines =
         u_reduced_prim(dyn->ia.primitive_topology) == MESA_PRIM_LINES &&
         dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM &&
         render->samples > 1;

      needs_update |= was_ms_bresenham_lines != render->ms_bresenham_lines;
   }

   if (needs_update) {
      if (render->sample_locations_enable) {
         /* Prioritize user-provided sample locations */
         struct mtl_sample_position sample_positions[KK_MAX_SAMPLES];
         for (uint32_t i = 0; i < KK_MAX_SAMPLES; i++) {
            VkSampleLocationEXT sl = render->sample_locations[i];
            sample_positions[i] = (struct mtl_sample_position){
               /* Metal asserts if values are out of range. Vulkan spec says
                * values are clamped, and dynamic state CTS tests hit this */
               .x = CLAMP(sl.x, KK_MIN_SAMPLE_LOCATION, KK_MAX_SAMPLE_LOCATION),
               .y = CLAMP(sl.y, KK_MIN_SAMPLE_LOCATION, KK_MAX_SAMPLE_LOCATION),
            };
         }

         mtl_render_pass_descriptor_set_sample_positions(
            gfx->render_pass_descriptor, sample_positions,
            render->sample_locations_count);
      } else if (render->ms_bresenham_lines) {
         /* For default sample locations with bresenham lines, set all to center
          * to provide correct rasterization */
         static const struct mtl_sample_position center = {0.5f, 0.5f};
         static const struct mtl_sample_position center_all[KK_MAX_SAMPLES] = {
            center, center, center, center, center, center, center, center};
         mtl_render_pass_descriptor_set_sample_positions(
            gfx->render_pass_descriptor, center_all, gfx->render.samples);
      } else {
         /* If custom sample locations are not needed, reset them */
         mtl_render_pass_descriptor_set_sample_positions(
            gfx->render_pass_descriptor, NULL, 0);
      }
   }

   return needs_update;
}

static void
kk_flush_render_pass(struct kk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct kk_rendering_state *render = &cmd->state.gfx.render;
   bool needs_restart = kk_flush_sample_locations(cmd);
   bool color_attachment_map_changed = false;

   if (IS_DIRTY(COLOR_ATTACHMENT_MAP)) {
      if (memcmp(render->color_map, dyn->cal.color_map,
                 render->color_att_count * sizeof(render->color_map[0])) != 0) {
         assert(cmd->state.gfx.render_pass_descriptor);

         /* Apply the store ops before the new color map is stored. */
         kk_apply_attachment_store_ops(cmd, true);

         kk_set_color_attachments(
            cmd->state.gfx.render_pass_descriptor, render, dyn, true,
            cmd->state.gfx.render.rendering_to_whole_framebuffer);
         needs_restart = true;
         color_attachment_map_changed = true;
      }
   }
   /* If render pass state changes and the pass is currently active, end the
    * current encoder and prepare to restart it */
   bool active_render = cmd->metal.render != NULL;
   if (needs_restart && active_render) {
      if (!color_attachment_map_changed) {
         /* Sample locations changed and color attachment map didn't. */
         kk_apply_attachment_store_ops(cmd, true);
      }

      cs_end(cmd);
      kk_cmd_buffer_dirty_all_gfx(cmd);
      cmd->state.gfx.need_to_start_render_pass = true;
      cmd->state.gfx.restart_barrier = true;
   }
}

static void
kk_flush_pipeline(struct kk_cmd_buffer *cmd)
{
   struct kk_device *device = kk_cmd_buffer_device(cmd);
   mtl_render_encoder *enc = cs_get_render(cmd);
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* Depth/stencil state may be dynamic, handle it as part of the pipeline. */
   if (cmd->state.gfx.is_depth_stencil_dynamic &&
       (IS_DIRTY(DS_DEPTH_TEST_ENABLE) | IS_DIRTY(DS_DEPTH_WRITE_ENABLE) |
        IS_DIRTY(DS_DEPTH_COMPARE_OP) | IS_DIRTY(DS_STENCIL_TEST_ENABLE) |
        IS_DIRTY(DS_STENCIL_OP) | IS_DIRTY(DS_STENCIL_COMPARE_MASK) |
        IS_DIRTY(DS_STENCIL_WRITE_MASK))) {
      kk_cmd_release_dynamic_ds_state(cmd);

      bool has_depth = dyn->rp.attachments & MESA_VK_RP_ATTACHMENT_DEPTH_BIT;
      bool has_stencil =
         dyn->rp.attachments & MESA_VK_RP_ATTACHMENT_STENCIL_BIT;
      gfx->depth_stencil_state = kk_compile_depth_stencil_state(
         device, &dyn->ds, has_depth, has_stencil);
      mtl_set_depth_stencil_state(enc, gfx->depth_stencil_state);
   }

   struct kk_shader *pipeline_vs = cmd->state.shaders[MESA_SHADER_VERTEX];
   bool masks_dirty = pipeline_vs && pipeline_vs->info.vs.dynamic_write_masks &&
                      (IS_DIRTY(CB_WRITE_MASKS) | IS_DIRTY(CB_COLOR_WRITE_ENABLES));
   if (IS_SHADER_DIRTY(VERTEX) || masks_dirty) {
      struct kk_shader *vs = pipeline_vs;
      mtl_render_pipeline_state *pipeline = vs->pipeline.gfx.render;
      if (vs->info.vs.dynamic_write_masks) {
         uint32_t masks = 0, defaults = 0;
         const struct kk_rendering_state *render = &gfx->render;
         for (uint32_t i = 0; i < render->color_att_count && i < MAX_DRAW_BUFFERS; i++) {
            uint8_t logical = render->color_map[i];
            if (logical >= MAX_DRAW_BUFFERS ||
                vs->info.vs.rt_formats[logical] == MTL_PIXEL_FORMAT_INVALID)
               continue;
            uint32_t mask = dyn->cb.attachments[i].write_mask & 0xf;
            if (!(dyn->cb.color_write_enables & BITFIELD_BIT(i)))
               mask = 0;
            masks |= mask << (4 * logical);
            defaults |= 0xfu << (4 * logical);
         }
         if (masks != defaults) {
            mtl_render_pipeline_state *masked =
               kk_shader_get_masked_pipeline(kk_cmd_buffer_device(cmd), vs, masks);
            if (masked)
               pipeline = masked;
         }
      }
      mtl_render_set_pipeline_state(enc, pipeline);
      if (gfx->depth_stencil_state)
         mtl_set_depth_stencil_state(enc, gfx->depth_stencil_state);
   }

   /* Merge tess info before GS construction since that depends on
    * gfx->tess.prim
    */
   if ((IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL)) &&
       cmd->state.shaders[MESA_SHADER_TESS_CTRL]) {
      struct kk_shader *tesc = cmd->state.shaders[MESA_SHADER_TESS_CTRL];
      struct kk_shader *tese = cmd->state.shaders[MESA_SHADER_TESS_EVAL];

      gfx->tess.info =
         kk_tess_info_merge(tese->info.tess.info, tesc->info.tess.info);

      /* Determine primitive based on the merged state */
      if (gfx->tess.info.points) {
         gfx->tess.prim = MESA_PRIM_POINTS;
      } else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES) {
         gfx->tess.prim = MESA_PRIM_LINES;
      } else {
         gfx->tess.prim = MESA_PRIM_TRIANGLES;
      }
   }
}

static void
kk_init_heap(const void *data)
{
   struct kk_cmd_buffer *cmd = (struct kk_cmd_buffer *)data;
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   size_t size = 128 * 1024 * 1024;
   kk_alloc_bo(dev, &dev->vk.base, size, 0, &dev->heap);

   struct poly_heap *map = (struct poly_heap *)dev->heap->cpu;

   /* TODO_KOSMICKRISP Self-contained until we have rodata at the device. */
   *map = (struct poly_heap){
      .base = dev->heap->gpu + sizeof(struct poly_heap),
      .size = size - sizeof(struct poly_heap),
   };
}

static uint64_t
kk_heap(struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   util_call_once_data(&dev->heap_init_once, kk_init_heap, cmd);

   /* We need to free all allocations after each command buffer execution */
   if (!cmd->uses_heap) {
      uint64_t addr = dev->heap->gpu;

      /* Zeroing the allocated index frees everything. Force compute encoder
       * since this path is only taken by tessellation, GS and unroll, which
       * already force a compute encoder break */
      mtl_compute_encoder *encoder = cs_get_compute(cmd);
      kk_cmd_write(cmd, (struct libkk_imm_write){
                           addr + offsetof(struct poly_heap, bottom), 0});

      /* Ensure heap is set to 0 before we allocate anything. */
      mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_DISPATCH,
                                       MTL_STAGE_DISPATCH);

      cmd->uses_heap = true;
   }

   return dev->heap->gpu;
}

/* GPU-generated indices are allocated from the heap, past its header */
static struct kk_addr_range
kk_heap_indices(struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   uint64_t heap = kk_heap(cmd);

   /* TODO_KOSMICKRISP Self-contained until we have rodata at the device. */
   return (struct kk_addr_range){
      .addr = heap + sizeof(struct poly_heap),
      .range = dev->heap->size_B - sizeof(struct poly_heap),
   };
}

enum kk_predicate_op : uint16_t {
   /* value > draw_id */
   KK_PREDICATE_GT_DRAW_ID,
   /* value == 0 */
   KK_PREDICATE_EQ_ZERO,
   /* value != 0 */
   KK_PREDICATE_NEQ_ZERO,
};

struct kk_draw_command {
   enum mesa_prim prim;
   /* Mask of stages that need per-draw data uploaded */
   uint32_t upload_mask;
   struct kk_addr_range index_buffer;
   uint32_t restart_index;
   uint8_t index_buffer_el_size_B;
   bool indirect;
   bool indexed;
   bool restart;
   bool flatshade_first;
   uint8_t pad_[3];
   uint32_t predicate_count;
   enum kk_predicate_op predicate_op[2];
   uint32_t draw_count;
   uint64_t predicate_addr[2];

   union {
      struct {
         uint64_t addr;
         uint32_t stride;
      } indirect_command;
      /* These arrays will be >1 when draw_count is >1 as this struct is
       * dynamically allocated. */
      VkDrawIndirectCommand draws[1];
      VkDrawIndexedIndirectCommand indexed_draws[1];
   };
};
static_assert(sizeof(struct kk_draw_command) == 88u, "Packed struct");

struct kk_draw_data {
   /* For non-indirect, 0 is vertex/index count, 1 instance count and 2 first
    * instance */
   struct kk_grid grid;
   struct {
      struct kk_addr_range gpu;
      uint32_t el_size_B;
   } index;
   uint32_t vertex_offset;
   enum mtl_primitive_type primitive_type;
};

static uint64_t
kk_upload_vertex_params(struct kk_cmd_buffer *cmd,
                        const struct kk_draw_data *data)
{
   /* Threadgroup size of indirect-local dispatches */
   const uint32_t wg_size[3] = {KK_POLY_WG_SIZE, 1, 1};

   struct poly_vertex_params params;
   poly_vertex_params_init(&params, 0, wg_size);

   /* XXX: We should deduplicate this logic */
   bool indirect = kk_grid_is_indirect(data->grid);

   if (!indirect)
      poly_vertex_params_set_draw(&params, data->grid.size.x,
                                  data->grid.size.y);

   if (data->index.el_size_B) {
      params.index_buffer = data->index.gpu.addr;
      params.index_buffer_range_el =
         data->index.gpu.range / data->index.el_size_B;
   }

   struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
   params.outputs = vs->info.outputs_written;

   if (!indirect) {
      uint32_t verts = data->grid.size.x, instances = data->grid.size.y;
      unsigned vb_size =
         poly_tcs_in_size(verts * instances, vs->info.outputs_written);

      /* Allocate if there are any outputs, or use the null sink to trap
       * reads if there aren't. Those reads are undefined but should not
       * fault. Affects:
       *
       *    dEQP-VK.pipeline.monolithic.no_position.explicit_declarations.basic.single_view.v0_g1
       */
      if (vb_size)
         params.output_buffer = kk_pool_alloc(cmd, vb_size, 4).gpu;
      else
         params.output_buffer = 0u;
   }

   cmd->state.gfx.per_draw_data.vertex_outputs = params.outputs;

   return kk_pool_upload(cmd, &params, sizeof(params), 8).gpu;
}

static void
kk_upload_tess_params(struct kk_cmd_buffer *cmd, struct poly_tess_params *out,
                      const struct kk_draw_data *draw)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_shader *tcs = cmd->state.shaders[MESA_SHADER_TESS_CTRL];

   enum poly_tess_partitioning partitioning =
      gfx->tess.info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : gfx->tess.info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   struct poly_tess_params args = {
      .heap = kk_heap(cmd),
      .tcs_stride_el = tcs->info.tess.tcs_output_stride / 4,
      .statistic = 0u,
      .input_patch_size = dyn->ts.patch_control_points,
      .output_patch_size = tcs->info.tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->info.tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->info.tess.tcs_per_vertex_outputs,
      .partitioning = partitioning,
      .points_mode = gfx->tess.info.points,
      .isolines = gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES,
   };

   if (!args.points_mode && gfx->tess.info.mode != TESS_PRIMITIVE_ISOLINES) {
      args.ccw = gfx->tess.info.ccw;
      args.ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   uint32_t draw_stride_el = 5;
   size_t draw_stride_B = draw_stride_el * sizeof(uint32_t);

   /* heap is allocated by kk_heap */
   /* TODO_KOSMICKRISP Self-contained until we have rodata at the device. */
   args.patch_coord_buffer = dev->heap->gpu + sizeof(struct poly_heap);

   if (!kk_grid_is_indirect(draw->grid)) {
      unsigned in_patches = draw->grid.size.x / args.input_patch_size;
      unsigned unrolled_patches = in_patches * draw->grid.size.y;

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * args.tcs_stride_el * sizeof(uint32_t);

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      /* Single API draw */
      uint32_t draw_offs = alloc;
      alloc += draw_stride_B;

      struct kk_ptr ptr = kk_pool_alloc(cmd, alloc, 4);
      gfx->tess.out_draws_addr = ptr.gpu + draw_offs;
      uint64_t addr = ptr.gpu;
      args.tcs_buffer = addr + tcs_out_offs;
      args.patches_per_instance = in_patches;
      args.coord_allocs = addr + patch_coord_offs;
      args.nr_patches = unrolled_patches;
      args.out_draws = addr + draw_offs;
      args.counts = addr + count_offs;
   } else {
      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint32_t grid_stride = sizeof(uint32_t) * 3;
      gfx->tess.indirect_ptr = kk_pool_alloc(cmd, grid_stride * 3, 4);

      struct kk_ptr ptr = kk_pool_alloc(cmd, draw_stride_B, 4);
      gfx->tess.out_draws_addr = ptr.gpu;
      args.out_draws = ptr.gpu;
   }

   memcpy(out, &args, sizeof(args));
}

static enum mesa_prim
kk_gs_in_prim(struct kk_cmd_buffer *cmd)
{
   if (cmd->state.shaders[MESA_SHADER_TESS_EVAL])
      return cmd->state.gfx.tess.prim;

   return vk_topology_to_mesa(
      cmd->vk.dynamic_graphics_state.ia.primitive_topology);
}

static uint64_t
kk_upload_geometry_params(struct kk_cmd_buffer *cmd,
                          const struct kk_draw_data *draw, bool restart)
{
   /* The tessellator output always feeds the GS indirectly */
   bool indirect = kk_grid_is_indirect(draw->grid) ||
                   cmd->state.shaders[MESA_SHADER_TESS_EVAL];
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_shader *gs = cmd->state.shaders[MESA_SHADER_GEOMETRY];
   enum poly_gs_shape shape = gs->info.gs.shape;
   uint32_t max_indices = gs->info.gs.max_indices;

   enum mesa_prim mode = kk_gs_in_prim(cmd);
   if (restart)
      mode = u_decomposed_prim(mode);

   const uint32_t wg_size[3] = {KK_POLY_WG_SIZE, 1, 1};

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, mode, wg_size);

   if (indirect) {
      /* The indirect setup kernel writes the draw and allocates from the
       * heap, including the index buffer */
      if (shape == POLY_GS_SHAPE_DYNAMIC_INDEXED)
         gfx->gs.index = kk_heap_indices(cmd);
   } else {
      poly_geometry_params_set_draw(&params, mode, shape, max_indices,
                                    draw->grid.size.x, draw->grid.size.y);

      gfx->gs.index_count = params.draw.index_count;
      gfx->gs.instance_count = params.draw.instance_count;

      if (shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         uint32_t size_B = gfx->gs.index_count * sizeof(uint32_t);
         params.output_index_buffer = kk_pool_alloc(cmd, size_B, 4).gpu;
         gfx->gs.index.addr = params.output_index_buffer;
         gfx->gs.index.range = size_B;
      }
   }

   if (shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      /* Metal has no 8-bit indices and restarts 16-bit ones at 0xFFFF */
      uint16_t indices[ARRAY_SIZE(gs->info.gs.topology)];
      for (uint32_t i = 0u; i < max_indices; ++i) {
         uint8_t index = gs->info.gs.topology[i];
         indices[i] = index == UINT8_MAX ? UINT16_MAX : index;
      }

      uint32_t size_B = max_indices * sizeof(uint16_t);
      gfx->gs.index.addr = kk_pool_upload(cmd, indices, size_B, 4).gpu;
      gfx->gs.index.range = size_B;
   }

   return kk_pool_upload(cmd, &params, sizeof(params), 8).gpu;
}

static void
kk_flush_dynamic_state(struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   mtl_render_encoder *enc = cs_get_render(cmd);

   if (IS_DIRTY(VI_BINDING_STRIDES)) {
      u_foreach_bit(ndx, dyn->vi->bindings_valid) {
         desc->root.draw.buffer_strides[ndx] = dyn->vi_binding_strides[ndx];
      }
      desc->root_dirty = true;
   }

   if (IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE)) {
      if (dyn->rs.rasterizer_discard_enable) {
         set_empty_scissor(enc);
      } else {
         /* Enforce setting the correct scissors */
         BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_VP_VIEWPORT_COUNT);
      }
   }

   if (IS_DIRTY(RS_CULL_MODE) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
       IS_SHADER_DIRTY(GEOMETRY)) {
      /* Only disable rendering if primitive type is culled. */
      gfx->is_cull_front_and_back =
         dyn->rs.cull_mode == VK_CULL_MODE_FRONT_AND_BACK &&
         u_reduced_prim(kk_rast_prim(cmd)) == MESA_PRIM_TRIANGLES;
      if (gfx->is_cull_front_and_back) {
         set_empty_scissor(enc);
      } else {
         mtl_set_cull_mode(enc,
                           vk_front_face_to_mtl_cull_mode(dyn->rs.cull_mode));
         /* Enforce setting the correct scissors */
         BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_VP_VIEWPORT_COUNT);
      }
   }

   /* We enable raster discard by setting scissor to size (0, 0) */
   if (!(dyn->rs.rasterizer_discard_enable || gfx->is_cull_front_and_back) &&
       (IS_DIRTY(VP_VIEWPORT_COUNT) || IS_DIRTY(VP_VIEWPORTS) ||
        IS_DIRTY(VP_SCISSOR_COUNT) || IS_DIRTY(VP_SCISSORS) ||
        IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE)))
      kk_flush_vp_state(cmd);

   if (IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE)) {
      desc->root.draw.clip_z_coeff =
         dyn->vp.depth_clip_negative_one_to_one ? 0.5f : 0.0f;
      desc->root_dirty = true;
   }

   if (IS_DIRTY(RS_FRONT_FACE) || IS_DIRTY(TS_DOMAIN_ORIGIN) ||
       IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL)) {
      bool front_face_ccw = dyn->rs.front_face != VK_FRONT_FACE_CLOCKWISE;
      if (cmd->state.shaders[MESA_SHADER_TESS_EVAL]) {
         front_face_ccw ^= gfx->tess.info.ccw;
         front_face_ccw ^=
            dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
      }
      mtl_set_front_face_winding(enc, front_face_ccw
                                         ? MTL_WINDING_COUNTER_CLOCKWISE
                                         : MTL_WINDING_CLOCKWISE);
   }

   if (IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(RS_DEPTH_BIAS_ENABLE)) {
      if (dyn->rs.depth_bias.enable)
         mtl_set_depth_bias(enc, dyn->rs.depth_bias.constant_factor,
                            dyn->rs.depth_bias.slope_factor,
                            dyn->rs.depth_bias.clamp);
      else
         mtl_set_depth_bias(enc, 0.0f, 0.0f, 0.0f);
   }

   if (IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE) ||
       IS_DIRTY(VP_VIEWPORT_COUNT) || IS_DIRTY(VP_VIEWPORTS)) {
      /* Mapping of Vulkan clamp/clip combinations to Metal:
       * - Clamp Off, Clip Off:
       *     Use Metal's clamp mode with the viewport depth range set to [0, 1],
       *     effectively disabling both clip and clamp. Emulate the viewport Z
       *     transform in the vertex shader.
       * - Clamp Off, Clip On:
       *     Exact match to Metal's clip mode.
       * - Clamp On, Clip Off:
       *     Exact match to Metal's clamp mode.
       * - Clamp On, Clip On:
       *     Use Metal's clip mode, and emulate clamp in the fragment shader.
       */
      bool clamp = dyn->rs.depth_clamp_enable;
      bool clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);

      enum mtl_depth_clip_mode mode =
         clip ? MTL_DEPTH_CLIP_MODE_CLIP : MTL_DEPTH_CLIP_MODE_CLAMP;
      mtl_set_depth_clip_mode(enc, mode);

      desc->root.draw.emulate_depth_clamp = clamp && clip;
      desc->root.draw.emulate_viewport_z = !clamp && !clip;
      if (desc->root.draw.emulate_depth_clamp ||
          desc->root.draw.emulate_viewport_z) {
         /* Ensure viewport depth ranges are up to date now, since we are
          * dirtying root anyway. */
         for (uint32_t i = 0; i < dyn->vp.viewport_count; i++) {
            const VkViewport *vp = &dyn->vp.viewports[i];

            /* These are mutually exclusive. Clamp expects the actual minimum
             * and maximum values, viewport transform supports inverted depth */
            if (desc->root.draw.emulate_depth_clamp) {
               desc->root.draw.viewport_z_range[i * 2] =
                  MIN2(vp->minDepth, vp->maxDepth);
               desc->root.draw.viewport_z_range[i * 2 + 1] =
                  MAX2(vp->minDepth, vp->maxDepth);
            } else {
               desc->root.draw.viewport_z_range[i * 2] = vp->minDepth;
               desc->root.draw.viewport_z_range[i * 2 + 1] = vp->maxDepth;
            }
         }
      }
      desc->root_dirty = true;
   }

   if ((IS_DIRTY(DS_DEPTH_BOUNDS_TEST_ENABLE) ||
        IS_DIRTY(DS_DEPTH_BOUNDS_TEST_BOUNDS)) &&
       kk_gpu_supports_depth_bounds(&pdev->info)) {
      /* Metal does not expose a separate flag for enabling the depth bounds
       * test. Instead, it treats [0, 1] as disabled. */
      bool bounds_enable = dyn->ds.depth.bounds_test.enable;
      float bounds_min = bounds_enable ? dyn->ds.depth.bounds_test.min : 0.0f;
      float bounds_max = bounds_enable ? dyn->ds.depth.bounds_test.max : 1.0f;
      mtl_set_depth_test_bounds(enc, bounds_min, bounds_max);
   }

   if (IS_DIRTY(DS_STENCIL_REFERENCE))
      mtl_set_stencil_references(
         enc, cmd->vk.dynamic_graphics_state.ds.stencil.front.reference,
         cmd->vk.dynamic_graphics_state.ds.stencil.back.reference);

   if (IS_DIRTY(CB_BLEND_CONSTANTS)) {
      static_assert(sizeof(desc->root.draw.blend_constant) ==
                       sizeof(dyn->cb.blend_constants),
                    "common size");

      memcpy(desc->root.draw.blend_constant, dyn->cb.blend_constants,
             sizeof(dyn->cb.blend_constants));
      desc->root_dirty = true;
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || gfx->dirty & KK_DIRTY_VB) {
      struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
      unsigned slot = 0;
      u_foreach_bit(i, vs->info.vs.attribs_read) {
         if (dyn->vi->attributes_valid & BITFIELD_BIT(i)) {
            struct vk_vertex_attribute_state attr = dyn->vi->attributes[i];
            struct kk_addr_range vb = gfx->vb.addr_range[attr.binding];

            desc->root.draw.attrib_clamps[slot] = kk_calculate_vbo_clamp(
               vb.addr, 0, vk_format_to_pipe_format(attr.format), vb.range,
               dyn->vi_binding_strides[attr.binding], attr.offset,
               &desc->root.draw.attrib_base[slot]);
            desc->root.draw.buffer_strides[attr.binding] =
               dyn->vi_binding_strides[attr.binding];
         }
         slot++;
      }
      desc->root_dirty = true;
   }

   if (desc->root_dirty)
      kk_upload_descriptor_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);

   if (desc->root.addr) {
      kk_cmd_bind_root_to_argument_table(cmd, desc->root.addr);
   }

   if (gfx->dirty & KK_DIRTY_OCCLUSION) {
      mtl_set_visibility_result_mode(enc, gfx->occlusion.mode,
                                     gfx->occlusion.index * sizeof(uint64_t));
   }
}

static void
kk_flush_gfx_state(struct kk_cmd_buffer *cmd)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_descriptor_state *desc = &gfx->descriptors;

   kk_flush_render_pass(cmd);
   kk_flush_pipeline(cmd);

   if (desc->push_dirty)
      kk_cmd_buffer_flush_push_descriptors(cmd, desc);

   kk_flush_dynamic_state(cmd);

   if (debug_get_option_kk_target_submit_log()) {
      struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
      if (vs && vs->msl_data[MESA_SHADER_FRAGMENT].code &&
          kk_is_turning_target_shader(_mesa_hash_string(vs->msl_data[MESA_SHADER_FRAGMENT].code))) {
         cmd->diagnostic_target_draws++;
         if (debug_get_option_kk_target_draw_log())
            mesa_logi("kk-target-record: FS-0e188069 draw=%u", cmd->diagnostic_target_draws);
      }
   }

   struct kk_shader *fs = cmd->state.shaders[MESA_SHADER_FRAGMENT];
   static int watching = -1;
   if (watching < 0)
      watching = getenv("KK_CAPTURE_TRIGGER_FILE") || getenv("XRT_SESSION_RECORD_DIR");
   if (watching) {
      /* The fragment MSL lives with the vertex shader object (as for the target log). */
      struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
      const char *code = vs ? vs->msl_data[MESA_SHADER_FRAGMENT].code : NULL;
      static const char *last_code;
      static uint32_t last_hash;
      if (code && code != last_code) {
         last_code = code;
         last_hash = _mesa_hash_string(code);
      }
      if (code)
         kk_implicit_watch_draw(cmd, last_hash);
   }
   if (debug_get_option_kk_descriptor_diag() && fs &&
       fs->diagnostic_msl_hash == 0x0e188069) {
      static_assert(offsetof(struct kk_root_descriptor_table, sets[0]) == 928, "set 0");
      struct kk_descriptor_set *set = desc->sets[0];
      bool found = false;
      util_dynarray_foreach(&cmd->diagnostic_sets, struct kk_descriptor_set *, prev)
         found |= *prev == set;
      if (!found)
         util_dynarray_append(&cmd->diagnostic_sets, set);
   }

   cmd->state.dirty_shaders = 0u;
   cmd->state.gfx.dirty = 0u;
   vk_dynamic_graphics_state_clear_dirty(&cmd->vk.dynamic_graphics_state);
}
#undef IS_SHADER_DIRTY
#undef IS_DIRTY

/* Returns true if the draw was successfully converted. */
static bool
kk_convert_to_indirect_draw(struct kk_cmd_buffer *cmd,
                            struct kk_draw_command *data)
{
   if (data->indirect)
      return true;

   uint32_t draw_stride = data->indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                        : sizeof(VkDrawIndirectCommand);
   struct kk_ptr indirect_draw =
      kk_pool_upload(cmd, &data->draws[0], data->draw_count * draw_stride, 4u);

   if (unlikely(!indirect_draw.gpu))
      return false;

   data->indirect_command.addr = indirect_draw.gpu;
   data->indirect_command.stride = draw_stride;
   data->indirect = true;

   return true;
}

/* Returns true if the call succeeds. */
static bool
kk_predicate_draws(struct kk_cmd_buffer *cmd, struct kk_draw_command *data)
{
   assert(data->predicate_count);

   if (unlikely(!kk_convert_to_indirect_draw(cmd, data)))
      return false;

   assert((data->indirect_command.stride % sizeof(uint32_t)) == 0 &&
          "stride is not aligned");

   uint32_t out_stride = data->indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                       : sizeof(VkDrawIndirectCommand);
   struct kk_ptr patched =
      kk_pool_alloc(cmd, out_stride * data->draw_count, 4u);
   if (unlikely(!patched.gpu))
      return false;

   uint64_t in_addr = data->indirect_command.addr;
   uint32_t out_stride_el = out_stride / sizeof(uint32_t);
   uint32_t in_stride_el = data->indirect_command.stride / sizeof(uint32_t);

   /* TODO_KOSMICKRISP: This can be accomplished more efficiently using device
    * generated commands, constructing an indirect command buffer on the GPU
    * which only contains the commands to run if the condition is true. For the
    * time being, we apply predicates by zeroing out disabled indirect data */
   struct kk_grid grid = kk_grid_1d(data->draw_count);
   for (uint32_t i = 0; i < data->predicate_count; i++) {
      uint64_t addr = data->predicate_addr[i];
      switch (data->predicate_op[i]) {
      case KK_PREDICATE_GT_DRAW_ID:
         libkk_predicate_indirect_gt_draw_id(cmd, grid, true, patched.gpu,
                                             in_addr, addr, out_stride_el,
                                             in_stride_el);
         break;
      case KK_PREDICATE_EQ_ZERO:
         libkk_predicate_indirect_eq_zero(cmd, grid, true, patched.gpu, in_addr,
                                          addr, out_stride_el, in_stride_el);
         break;
      case KK_PREDICATE_NEQ_ZERO:
         libkk_predicate_indirect_neq_zero(cmd, grid, true, patched.gpu,
                                           in_addr, addr, out_stride_el,
                                           in_stride_el);
         break;
      default:
         UNREACHABLE("Unsupported indirect draw predicate");
      }

      if (i == 0) {
         /* Further predicates will operate on previous patched data */
         in_addr = patched.gpu;
         in_stride_el = out_stride_el;
      }
   }

   data->indirect_command.addr = patched.gpu;
   data->indirect_command.stride = out_stride;
   data->predicate_count = 0;

   return true;
}

/* Points the draw at the 32-bit heap indices an unroll kernel wrote */
static void
kk_set_unrolled_draw(struct kk_cmd_buffer *cmd, struct kk_draw_command *data,
                     enum mesa_prim prim, uint64_t out_draws)
{
   data->prim = prim;
   data->index_buffer = kk_heap_indices(cmd);
   data->restart_index = UINT32_MAX;
   data->index_buffer_el_size_B = 4u;
   data->indirect = true;
   data->indexed = true;
   data->restart = false;
   data->flatshade_first = true;
   data->indirect_command.addr = out_draws;
   data->indirect_command.stride = sizeof(VkDrawIndexedIndirectCommand);
}

/* Unrolling will always be done through indirect rendering, so if this is
 * called from non-indirect calls, we will fake it. */
static bool
kk_unroll_geometry(struct kk_cmd_buffer *cmd, struct kk_draw_command *data)
{
   if (unlikely(!kk_convert_to_indirect_draw(cmd, data)))
      return false;

   assert((data->indirect_command.stride % sizeof(uint32_t)) == 0 &&
          "stride is not aligned");

   struct kk_ptr out_draws = kk_pool_alloc(
      cmd, data->draw_count * sizeof(VkDrawIndexedIndirectCommand), 4u);
   if (unlikely(!out_draws.gpu))
      return false;

   struct libkk_unroll_geometry_args info = {
      .index_buffer = data->index_buffer.addr,
      .heap = kk_heap(cmd),
      .in_draw = data->indirect_command.addr,
      .out_draw = out_draws.gpu,
      .in_draw_stride_el = data->indirect_command.stride / sizeof(uint32_t),
      /* Handle primitive restart disable by forcing index to UINT32_MAX */
      .restart_index = !data->restart ? UINT32_MAX : data->restart_index,
      .index_buffer_size_el =
         data->indexed ? data->index_buffer.range / data->index_buffer_el_size_B
                       : 0u,
      .in_el_size_B = data->index_buffer_el_size_B,
      .out_el_size_B = 4u,
      .flatshade_first = data->flatshade_first,
      .mode = data->prim,
   };

   libkk_unroll_geometry_struct(cmd, kk_grid_1d(1024 * data->draw_count), true,
                                info);

   kk_set_unrolled_draw(cmd, data, u_decomposed_prim(data->prim),
                        out_draws.gpu);

   return true;
}

/* Metal has no adjacency topologies, so draws without a GS draw the lines or
 * triangles they rasterize instead. */
static bool
kk_unroll_adjacency(struct kk_cmd_buffer *cmd, struct kk_draw_command *data)
{
   if (unlikely(!kk_convert_to_indirect_draw(cmd, data)))
      return false;

   struct kk_ptr out_draws = kk_pool_alloc(
      cmd, data->draw_count * sizeof(VkDrawIndexedIndirectCommand), 4u);
   if (unlikely(!out_draws.gpu))
      return false;

   struct libkk_unroll_adjacency_args info = {
      .index_buffer = data->index_buffer.addr,
      .heap = kk_heap(cmd),
      .in_draw = data->indirect_command.addr,
      .out_draw = out_draws.gpu,
      .in_draw_stride_el = data->indirect_command.stride / sizeof(uint32_t),
      .index_buffer_size_el =
         data->indexed ? data->index_buffer.range / data->index_buffer_el_size_B
                       : 0u,
      .in_el_size_B = data->index_buffer_el_size_B,
      .flatshade_first = data->flatshade_first,
      .mode = data->prim,
   };

   libkk_unroll_adjacency_struct(cmd, kk_grid_1d(1024 * data->draw_count), true,
                                 info);

   kk_set_unrolled_draw(cmd, data, u_reduced_prim(data->prim), out_draws.gpu);

   return true;
}

static enum mtl_primitive_type
mesa_prim_to_mtl_primitive_type(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return MTL_PRIMITIVE_TYPE_POINT;
   case MESA_PRIM_LINES:
      return MTL_PRIMITIVE_TYPE_LINE;
   case MESA_PRIM_LINE_STRIP:
      return MTL_PRIMITIVE_TYPE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return MTL_PRIMITIVE_TYPE_TRIANGLE;
   case MESA_PRIM_TRIANGLE_STRIP:
      return MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP;
   default:
      UNREACHABLE("Unsupported primitive type");
   }
}

static uint32_t
build_per_draw_upload_mask(struct kk_cmd_buffer *cmd)
{
   uint32_t mask = 0;

   struct kk_shader *vertex = cmd->state.shaders[MESA_SHADER_VERTEX];
   if (vertex && vertex->info.uses_per_draw_data) {
      mask |= BITFIELD_BIT(MESA_SHADER_VERTEX);
   }

   /* Tessellation will always require per draw data to be submitted. */
   struct kk_shader *tese = cmd->state.shaders[MESA_SHADER_TESS_EVAL];
   if (tese) {
      mask |= BITFIELD_BIT(MESA_SHADER_TESS_EVAL);
   }

   /* So does a GS */
   if (cmd->state.shaders[MESA_SHADER_GEOMETRY]) {
      mask |= BITFIELD_BIT(MESA_SHADER_GEOMETRY);
   }

   struct kk_shader *fragment = cmd->state.shaders[MESA_SHADER_FRAGMENT];
   if (fragment && fragment->info.uses_per_draw_data) {
      mask |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   }

   return mask;
}

static struct kk_addr_range
kk_sanitize_index_range(struct kk_cmd_buffer *cmd, struct kk_addr_range range)
{
   /* Metal does not support an index buffer address or size of 0. Cache a pool
    * allocation of the maximum index size, filled with 0, and use it to handle
    * these cases. Robustness will handle over-reads.
    *
    * Note that we only need to do this at draw dispatch as our other draw logic
    * for unrolling, tessellation, etc. will properly handle these cases. */
   if (range.addr == 0u || range.range == 0u) {
      struct kk_graphics_state *gfx = &cmd->state.gfx;

      if (!gfx->index.null_addr) {
         struct kk_ptr null_alloc =
            kk_pool_alloc(cmd, sizeof(uint32_t), sizeof(uint32_t));
         *(uint32_t *)null_alloc.cpu = 0u;

         gfx->index.null_addr = null_alloc.gpu;
      }

      range.addr = gfx->index.null_addr;
      range.range = sizeof(uint32_t);
   }

   /* KK_WORKAROUND_12: macOS 26 requires addr + range to be 4 byte aligned due
    * to a known bug where Metal drops the last index if addr + range is not 4
    * byte aligned. The extra bytes will not be read. The issue is not present
    * in macOS 27.
    */
   struct kk_physical_device *pdev =
      kk_device_physical(kk_cmd_buffer_device(cmd));
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(12)))
      range.range = align64(range.addr + range.range, 4u) - range.addr;

   return range;
}

static void
kk_dispatch_draw(struct kk_cmd_buffer *cmd, struct kk_draw_data data)
{
   mtl_render_encoder *enc = cs_get_render(cmd);

   if (kk_grid_is_indirect(data.grid)) {
      if (data.index.el_size_B) {
         struct kk_addr_range index_range =
            kk_sanitize_index_range(cmd, data.index.gpu);
         mtl_draw_indexed_primitives_indirect(
            enc, data.primitive_type,
            index_size_in_bytes_to_mtl_index_type(data.index.el_size_B),
            index_range.addr, index_range.range, data.grid.addr);
      } else {
         mtl_draw_primitives_indirect(enc, data.primitive_type, data.grid.addr);
      }
   } else {
      if (data.index.el_size_B) {
         struct kk_addr_range index_range =
            kk_sanitize_index_range(cmd, data.index.gpu);
         mtl_draw_indexed_primitives(
            enc, data.primitive_type, data.grid.size.x,
            index_size_in_bytes_to_mtl_index_type(data.index.el_size_B),
            index_range.addr, index_range.range, data.grid.size.y,
            data.vertex_offset, data.grid.size.z);
      } else {
         /* Avoid Metal validation error. Empty draws from tessellation will
          * have values set to 0. */
         if (data.grid.size.x != 0 && data.grid.size.y != 0)
            mtl_draw_primitives(enc, data.primitive_type, data.vertex_offset,
                                data.grid.size.x, data.grid.size.y,
                                data.grid.size.z);
      }
   }
}

static bool
requires_unroll_index_promotion(const struct kk_draw_command *data)
{
   if (!data->indexed)
      return false;

   /* uint8_t indices must be promoted since they are not natively supported. */
   if (data->index_buffer_el_size_B == sizeof(uint8_t))
      return true;

   /* For primitive types that support primitive restart, if restart is disabled
    * and the index size is less than uint32_t, we need to make sure to perform
    * unrolling as it automatically promotes the type to uint32_t, preventing
    * valid indices from being treated as restarts. For uint32_t indices with
    * restart disabled, we realistically will never have enough vertices for the
    * restart index to be valid anyway. */
   switch (data->prim) {
   case MESA_PRIM_LINE_STRIP:
   case MESA_PRIM_TRIANGLE_STRIP:
   case MESA_PRIM_TRIANGLE_FAN:
      return (!data->restart &&
              data->index_buffer_el_size_B < sizeof(uint32_t));
   default:
      return false;
   }
}

static bool
requires_unroll_robustness(struct kk_cmd_buffer *cmd,
                           const struct kk_draw_command *data)
{
   /* No need for robustness if the draw does not use an index buffer */
   if (!data->indexed)
      return false;

   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   /* KK_WORKAROUND_12, KK_WORKAROUND_13 */
   bool workaround_12 =
      !(pdev->settings.disabled_workarounds & BITFIELD64_BIT(12));
   bool workaround_13 =
      !(pdev->settings.disabled_workarounds & BITFIELD64_BIT(13));

   /* Do nothing if both workarounds are disabled */
   if (!workaround_12 && !workaround_13)
      return false;

   /* KK_WORKAROUND_12: Need to handle null descriptor case here as we can't
    * rely on hardware robustness */
   if (workaround_12 &&
       (data->index_buffer.addr == 0u || data->index_buffer.range == 0u))
      return true;

   /* No need to handle robustness if not enabled
    * TODO_KOSMICKRISP: Narrow pipeline robustness to vertex input only */
   if (!dev->vk.enabled_features.robustBufferAccess2 &&
       !dev->vk.enabled_features.pipelineRobustness)
      return false;

   /* Only need to handle case where index buffer is not aligned to 4 bytes.
    * KK_WORKAROUND_12: Need to handle all alignments. */
   if (!workaround_12 && (data->index_buffer.addr & 3u) == 0u &&
       (data->index_buffer.range & 3u) == 0u)
      return false;

   /* We can't tell if the draw over-reads up-front with indirect draws, so we
    * always have to handle it */
   if (data->indirect)
      return true;

   /* For direct draws, we can check now if any over-read the index buffer */
   for (uint32_t i = 0; i < data->draw_count; i++) {
      const VkDrawIndexedIndirectCommand *draw = &data->indexed_draws[i];
      if ((draw->firstIndex + draw->indexCount) * data->index_buffer_el_size_B >
          data->index_buffer.range)
         return true;
   }

   return false;
}

static bool
requires_unroll_restart(struct kk_cmd_buffer *cmd,
                        const struct kk_draw_command *data)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   if (!data->restart || !data->indexed)
      return false;

   /* The adjacency rewrite reads restart-free input */
   if (mesa_prim_has_adjacency(data->prim))
      return true;

   switch (data->prim) {
   case MESA_PRIM_POINTS:
   case MESA_PRIM_LINES:
   case MESA_PRIM_TRIANGLES:
      /* Unroll list restart only if the user requests it, to avoid associated
       * cost otherwise. Some applications unintentionally leave the primitive
       * restart flag enabled while using list primitives without any restarts,
       * and in these cases we can avoid the cost of unroll, even though they
       * are technically against spec. */
      return dev->vk.enabled_features.primitiveTopologyListRestart;
   default:
      break;
   }

   /* For topologies that natively support restart, unroll if unusual primitive
    * restart index is set by user */
   uint32_t default_idx = BITFIELD_RANGE(0, data->index_buffer_el_size_B * 8);
   return data->restart_index != default_idx;
}

static bool
requires_unroll_flatshade(struct kk_cmd_buffer *cmd,
                          struct kk_draw_command *data)
{
   struct kk_shader *fs = cmd->state.shaders[MESA_SHADER_FRAGMENT];

   /* If the last vertex is provoking and the fragment shader uses flat inputs,
    * we need to unroll to correct the vertex order */
   return !data->flatshade_first && fs && fs->info.fs.uses_flat_varyings;
}

static bool
requires_unroll(struct kk_cmd_buffer *cmd, struct kk_draw_command *data)
{
   /* No need to unroll for tessellation. Robustness is handled by tessellator,
    * restart is not supported for patch lists, and flat-shading does not have a
    * well-defined vertex used when tessellating. A GS reads any index size and
    * topology through poly's bounds-checked input assembly, and
    * kk_draw_without_restart unrolls restart before it runs. */
   bool tess = cmd->state.shaders[MESA_SHADER_TESS_EVAL];
   bool geom = cmd->state.shaders[MESA_SHADER_GEOMETRY];
   if (tess || geom)
      return false;

   /* Metal does not support triangle fans */
   if (data->prim == MESA_PRIM_TRIANGLE_FAN)
      return true;

   /* Check for remaining unroll conditions */
   return requires_unroll_index_promotion(data) ||
          requires_unroll_robustness(cmd, data) ||
          requires_unroll_restart(cmd, data) ||
          requires_unroll_flatshade(cmd, data);
}

static uint64_t
upload_base_vertex(struct kk_cmd_buffer *cmd, const struct kk_draw_data *data)
{
   if (kk_grid_is_indirect(data->grid)) {
      uint64_t first_vertex_offset =
         data->index.el_size_B
            ? offsetof(VkDrawIndexedIndirectCommand, vertexOffset)
            : offsetof(VkDrawIndirectCommand, firstVertex);
      return data->grid.addr + first_vertex_offset;
   } else {
      return kk_pool_upload(cmd, &data->vertex_offset,
                            sizeof(data->vertex_offset), 4u)
         .gpu;
   }
}

static uint64_t
upload_base_instance(struct kk_cmd_buffer *cmd, const struct kk_draw_data *data)
{
   if (kk_grid_is_indirect(data->grid)) {
      uint64_t base_instance_offset =
         data->index.el_size_B
            ? offsetof(VkDrawIndexedIndirectCommand, firstInstance)
            : offsetof(VkDrawIndirectCommand, firstInstance);
      return data->grid.addr + base_instance_offset;
   } else {
      return kk_pool_upload(cmd, &data->grid.size.z, sizeof(data->grid.size.z),
                            4u)
         .gpu;
   }
}

/* Prepares emulation data for stages like tessellation. It also upload system
 * values not present in Metal such as drawID. gs_restart is set when a GS draw
 * unrolls primitive restart. */
static void
kk_upload_per_draw_data(struct kk_cmd_buffer *cmd, uint32_t upload_mask,
                        uint32_t draw_id, const struct kk_draw_data *draw,
                        bool gs_restart)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   gfx->per_draw_data.draw_id = draw_id;

   /* Prepare emulation data for tessellation and GS. */
   bool tess = upload_mask & BITFIELD_BIT(MESA_SHADER_TESS_EVAL);
   bool geom = upload_mask & BITFIELD_BIT(MESA_SHADER_GEOMETRY);
   if (tess || geom) {
      gfx->per_draw_data.index_size = draw->index.el_size_B;
      gfx->per_draw_data.base_vertex_addr = upload_base_vertex(cmd, draw);
      gfx->per_draw_data.base_instance_addr = upload_base_instance(cmd, draw);
      gfx->per_draw_data.vertex_params = kk_upload_vertex_params(cmd, draw);
   }

   if (tess) {
      struct kk_ptr tess_args =
         kk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
      gfx->per_draw_data.tess_params = tess_args.gpu;
      if (tess_args.gpu) {
         kk_upload_tess_params(cmd, tess_args.cpu, draw);
      }
   }

   if (geom) {
      gfx->per_draw_data.geometry_params =
         kk_upload_geometry_params(cmd, draw, gs_restart);

      const struct vk_dynamic_graphics_state *dyn =
         &cmd->vk.dynamic_graphics_state;
      gfx->per_draw_data.provoking_last =
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;
   }

   struct kk_ptr shader_data_gpu =
      kk_pool_upload(cmd, &gfx->per_draw_data, sizeof(gfx->per_draw_data), 8u);
   if (unlikely(!shader_data_gpu.gpu))
      return;

   mtl_set_address(cmd->argument_table, shader_data_gpu.gpu, 2u);
}

static void
kk_dispatch_compute(mtl_compute_encoder *enc, struct kk_grid grid,
                    struct mtl_size local_size)
{
   if (grid.mode == KK_GRID_DIRECT)
      mtl_dispatch_threads(enc, grid.size, local_size);
   else if (grid.mode == KK_GRID_INDIRECT_LOCAL)
      mtl_dispatch_threads_with_indirect_buffer(enc, grid.addr);
   else
      mtl_dispatch_threadgroups_with_indirect_buffer(enc, grid.addr,
                                                     local_size);
}

static struct kk_draw_data
kk_launch_tess(struct kk_cmd_buffer *cmd, struct kk_draw_data draw)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_grid grid_vs, grid_tcs, grid_tess;

   struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
   struct kk_shader *tcs = cmd->state.shaders[MESA_SHADER_TESS_CTRL];

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = gfx->per_draw_data.tess_params;
   struct kk_tess_info info = gfx->tess.info;

   /* Setup grids */
   if (kk_grid_is_indirect(draw.grid)) {
      struct libkk_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.indirect_ptr.gpu,
         .indirect = draw.grid.addr,
         .vp = gfx->per_draw_data.vertex_params,
         .vertex_outputs = vs->info.outputs_written,
         .tcs_statistic = 0,
      };

      if (draw.index.el_size_B) {
         args.in_index_buffer = draw.index.gpu.addr;
         args.in_index_size_B = draw.index.el_size_B;
         args.in_index_buffer_range_el =
            draw.index.gpu.range / args.in_index_size_B;
      }

      libkk_tess_setup_indirect_struct(cmd, kk_grid_1d(1), true, args);

      uint32_t grid_stride = sizeof(uint32_t) * 3;
      grid_vs = kk_grid_indirect(gfx->tess.indirect_ptr.gpu + 0u * grid_stride);
      grid_tcs =
         kk_grid_indirect(gfx->tess.indirect_ptr.gpu + 1u * grid_stride);
      grid_tess =
         kk_grid_indirect(gfx->tess.indirect_ptr.gpu + 2u * grid_stride);
   } else {
      uint32_t patches = draw.grid.size.x / input_patch_size;
      grid_vs = grid_tcs = kk_grid_2d(draw.grid.size.x, draw.grid.size.y);

      grid_tcs.size.x = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = kk_grid_1d(patches * draw.grid.size.y);
   }

   /* First launch the VS and TCS */

   mtl_compute_encoder *enc = cs_get_compute(cmd);
   {
      mtl_compute_pipeline_state *pipeline = vs->pipeline.gfx.pre_render[0];
      struct mtl_size local_size = {KK_POLY_WG_SIZE, 1, 1};
      mtl_barrier_after_encoder_stages(enc, MTL_STAGE_DISPATCH,
                                       MTL_STAGE_DISPATCH);
      mtl_compute_set_pipeline_state(enc, pipeline);
      kk_dispatch_compute(enc, grid_vs, local_size);
   }
   {
      mtl_compute_pipeline_state *pipeline = vs->pipeline.gfx.pre_render[1];
      struct mtl_size local_size = {tcs->info.tess.tcs_output_patch_size, 1, 1};
      /* Avoid Metal validation error by trying to launch empty compute. Return
       * empty data. We set restart to true to avoid unroll. */
      if (grid_tcs.mode == KK_GRID_DIRECT && grid_tcs.size.x == 0u)
         return (struct kk_draw_data){.grid = kk_grid_1d(0u)};
      mtl_barrier_after_encoder_stages(enc, MTL_STAGE_DISPATCH,
                                       MTL_STAGE_DISPATCH);
      mtl_compute_set_pipeline_state(enc, pipeline);
      kk_dispatch_compute(enc, grid_tcs, local_size);
   }

   /* First generate counts, then prefix sum them, and then tessellate. */
   libkk_tessellate(cmd, grid_tess, true, info.mode, POLY_TESS_MODE_COUNT,
                    state);

   libkk_prefix_sum_tess(cmd, kk_grid_1d(1024u), true, state);

   libkk_tessellate(cmd, grid_tess, true, info.mode, POLY_TESS_MODE_WITH_COUNTS,
                    state);

   draw.grid = kk_grid_indirect(gfx->tess.out_draws_addr);

   draw.index.gpu = kk_heap_indices(cmd);
   draw.index.el_size_B = 4u;
   draw.primitive_type = mesa_prim_to_mtl_primitive_type(gfx->tess.prim);
   return draw;
}

/* A GS assembles its input in software, which cannot follow restarts, so the
 * draw is unrolled to lists first. */
static struct kk_draw_data
kk_draw_without_restart(struct kk_cmd_buffer *cmd,
                        const struct kk_draw_command *data,
                        struct kk_draw_data draw)
{
   assert(kk_grid_is_indirect(draw.grid));

   struct kk_ptr out_draw =
      kk_pool_alloc(cmd, sizeof(VkDrawIndexedIndirectCommand), 4u);

   struct libkk_unroll_restart_args args = {
      .index_buffer = draw.index.gpu.addr,
      .heap = kk_heap(cmd),
      .in_draw = draw.grid.addr,
      .out_draw = out_draw.gpu,
      .restart_index = data->restart_index,
      .index_buffer_size_el = draw.index.gpu.range / draw.index.el_size_B,
      .index_size_B = draw.index.el_size_B,
      .flatshade_first = data->flatshade_first,
      .mode = data->prim,
   };
   libkk_unroll_restart_struct(cmd, kk_grid_1d(1024u), true, args);

   draw.grid = kk_grid_indirect(out_draw.gpu);
   draw.index.gpu = kk_heap_indices(cmd);
   return draw;
}

/* Runs the stage feeding the GS and the GS compute pass, returning the draw
 * that rasterizes its output. */
static struct kk_draw_data
kk_launch_gs_prerast(struct kk_cmd_buffer *cmd,
                     const struct kk_draw_command *data,
                     struct kk_draw_data draw, bool restart)
{
   struct kk_physical_device *pdev =
      kk_device_physical(kk_cmd_buffer_device(cmd));
   struct kk_graphics_state *gfx = &cmd->state.gfx;
   struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
   struct kk_shader *tes = cmd->state.shaders[MESA_SHADER_TESS_EVAL];
   struct kk_shader *gs = cmd->state.shaders[MESA_SHADER_GEOMETRY];
   uint64_t vertex_params = gfx->per_draw_data.vertex_params;
   uint64_t geometry_params = gfx->per_draw_data.geometry_params;
   enum poly_gs_shape shape = gs->info.gs.shape;
   struct kk_grid grid_vs, grid_gs;

   enum mesa_prim mode = kk_gs_in_prim(cmd);

   if (restart) {
      draw = kk_draw_without_restart(cmd, data, draw);
      mode = u_decomposed_prim(mode);
   }

   if (kk_grid_is_indirect(draw.grid)) {
      struct libkk_gs_setup_indirect_args args = {
         .index_buffer = draw.index.gpu.addr,
         .draw = draw.grid.addr,
         .vp = vertex_params,
         .p = geometry_params,
         .heap = kk_heap(cmd),
         .vs_outputs = (tes ? tes : vs)->info.outputs_written,
         .prim = mode,
         .max_indices = gs->info.gs.max_indices,
         .shape = shape,
      };

      if (draw.index.el_size_B) {
         args.index_size_B = draw.index.el_size_B;
         args.index_buffer_range_el =
            draw.index.gpu.range / draw.index.el_size_B;
      }

      libkk_gs_setup_indirect_struct(cmd, kk_grid_1d(1u), true, args);

      grid_vs = kk_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));
      grid_gs = kk_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = kk_grid_2d(draw.grid.size.x, draw.grid.size.y);
      grid_gs.size.x = u_decomposed_prims_for_vertices(mode, draw.grid.size.x);

      /* Avoid Metal validation errors from empty dispatches and draws */
      if (grid_gs.size.x == 0u)
         return (struct kk_draw_data){.grid = kk_grid_1d(0u)};
   }

   /* The stage feeding the GS and its compute pass are the last two pre-render
    * pipelines */
   uint32_t pre_render_count = vs->pipeline.gfx.pre_render_count;
   struct mtl_size local_size = {KK_POLY_WG_SIZE, 1, 1};
   mtl_compute_encoder *enc = cs_get_compute(cmd);

   /* KK_WORKAROUND_19 */
   if (!(pdev->settings.disabled_workarounds & BITFIELD64_BIT(19)))
      mtl_set_threadgroup_memory_length(enc, 16u, 0u);

   mtl_barrier_after_encoder_stages(enc, MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);
   mtl_compute_set_pipeline_state(
      enc, vs->pipeline.gfx.pre_render[pre_render_count - 2u]);
   kk_dispatch_compute(enc, grid_vs, local_size);

   mtl_barrier_after_encoder_stages(enc, MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);
   mtl_compute_set_pipeline_state(
      enc, vs->pipeline.gfx.pre_render[pre_render_count - 1u]);
   kk_dispatch_compute(enc, grid_gs, local_size);

   struct kk_draw_data rast = {
      .primitive_type = mesa_prim_to_mtl_primitive_type(gs->info.gs.mode),
   };

   if (poly_gs_indexed(shape)) {
      rast.index.gpu = gfx->gs.index;
      rast.index.el_size_B = shape == POLY_GS_SHAPE_DYNAMIC_INDEXED
                                ? sizeof(uint32_t)
                                : sizeof(uint16_t);
   }

   if (kk_grid_is_indirect(draw.grid)) {
      rast.grid = kk_grid_indirect(geometry_params +
                                   offsetof(struct poly_geometry_params, draw));
   } else {
      rast.grid = kk_grid_3d(gfx->gs.index_count, gfx->gs.instance_count, 0u);
   }

   return rast;
}

/* Get modifiable per draw data. */
static struct kk_draw_data
build_draw_data(struct kk_cmd_buffer *cmd, struct kk_draw_command *data,
                uint32_t draw_id)
{
   /* Emulated stages rasterize a primitive of their own */
   bool emulated = cmd->state.shaders[MESA_SHADER_TESS_EVAL] ||
                   cmd->state.shaders[MESA_SHADER_GEOMETRY];
   struct kk_draw_data draw = {
      .index.gpu = data->index_buffer,
      .index.el_size_B = data->index_buffer_el_size_B,
      .primitive_type =
         emulated ? 0u : mesa_prim_to_mtl_primitive_type(data->prim),
   };

   if (data->indirect) {
      uint64_t addr =
         data->indirect_command.addr + draw_id * data->indirect_command.stride;
      draw.grid = kk_grid_indirect(addr);
   } else if (data->indexed) {
      VkDrawIndexedIndirectCommand draw_cmd = data->indexed_draws[draw_id];
      draw.grid = kk_grid_3d(draw_cmd.indexCount, draw_cmd.instanceCount,
                             draw_cmd.firstInstance);
      draw.vertex_offset = draw_cmd.vertexOffset;

      uint64_t index_offset =
         draw_cmd.firstIndex * data->index_buffer_el_size_B;
      draw.index.gpu.addr += index_offset;
      draw.index.gpu.range = index_offset < draw.index.gpu.range
                                ? draw.index.gpu.range - index_offset
                                : 0u;
   } else {
      VkDrawIndirectCommand draw_cmd = data->draws[draw_id];
      draw.grid = kk_grid_3d(draw_cmd.vertexCount, draw_cmd.instanceCount,
                             draw_cmd.firstInstance);
      draw.vertex_offset = draw_cmd.firstVertex;
   }

   return draw;
}

static void
kk_draw(struct kk_cmd_buffer *cmd, struct kk_draw_command *data)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   kk_flush_gfx_state(cmd);

   data->restart = dyn->ia.primitive_restart_enable;
   data->restart_index = dyn->ia.primitive_restart_index;
   data->flatshade_first =
      dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT;

   /* Convert to indirect and process predicates. Skip draw if we fail. */
   if (data->predicate_count > 0 && !kk_predicate_draws(cmd, data))
      return;

   bool tess = cmd->state.shaders[MESA_SHADER_TESS_EVAL];
   bool geom = cmd->state.shaders[MESA_SHADER_GEOMETRY];

   /* Unroll geometry. Skip draw if we fail. */
   if (requires_unroll(cmd, data) && !kk_unroll_geometry(cmd, data))
      return;

   if (!geom && mesa_prim_has_adjacency(data->prim) &&
       !kk_unroll_adjacency(cmd, data))
      return;

   /* Patch lists do not restart, so only a GS fed by the input assembly needs
    * its restarts unrolled. The unroll kernel reads an indirect draw. */
   bool gs_restart = geom && !tess && data->restart && data->indexed;
   if (gs_restart && !kk_convert_to_indirect_draw(cmd, data))
      return;

   for (uint32_t i = 0; i < data->draw_count; i++) {
      struct kk_draw_data draw_data = build_draw_data(cmd, data, i);

      if (data->upload_mask)
         kk_upload_per_draw_data(cmd, data->upload_mask, i, &draw_data,
                                 gs_restart);

      if (tess)
         draw_data = kk_launch_tess(cmd, draw_data);
      if (geom)
         draw_data = kk_launch_gs_prerast(cmd, data, draw_data, gs_restart);

      /* TODO_KOSMICKRISP Remove this once unroll, tess and any compute does not
       * split render pass */
      if (cmd->state.gfx.need_to_start_render_pass)
         kk_flush_gfx_state(cmd);

      kk_dispatch_draw(cmd, draw_data);
   }
}

static void
kk_seq_draw(struct kk_cmd_buffer *cmd, const char *what, uint32_t count,
            uint32_t instances)
{
   if (!cmd->state.gfx.seq_log)
      return;
   const struct kk_shader *fs = cmd->state.shaders[MESA_SHADER_FRAGMENT];
   const char *msl_dir = debug_get_option("KK_SEQ_MSL_DIR", NULL);
   const struct kk_shader *vs = cmd->state.shaders[MESA_SHADER_VERTEX];
   if (msl_dir && vs && vs->msl_data[MESA_SHADER_FRAGMENT].code) {
      char path[512];
      snprintf(path, sizeof(path), "%s/fs-%p.metal", msl_dir, (const void *)fs);
      FILE *f = fopen(path, "wx");
      if (f) {
         fputs(vs->msl_data[MESA_SHADER_FRAGMENT].code, f);
         fclose(f);
      }
   }
   kk_seq_log(cmd, "%s %u x%u fs %p outputs 0x%llx", what, count, instances,
              (const void *)fs,
              fs ? (unsigned long long)fs->info.fs.outputs_written : 0ull);
   if (count == 4 || count == 3) {
      const struct vk_dynamic_graphics_state *d = &cmd->vk.dynamic_graphics_state;
      kk_seq_log(cmd, "  state: cb mask 0x%x blend %d (%d,%d op %d), depth test %d write %d op %d, "
                      "stencil test %d front fail %d pass %d dfail %d cmp %d cmask 0x%x wmask 0x%x ref %u, "
                      "samples %u sample shading %d",
                 d->cb.attachments[0].write_mask, d->cb.attachments[0].blend_enable,
                 d->cb.attachments[0].src_color_blend_factor,
                 d->cb.attachments[0].dst_color_blend_factor,
                 d->cb.attachments[0].color_blend_op, d->ds.depth.test_enable,
                 d->ds.depth.write_enable, d->ds.depth.compare_op,
                 d->ds.stencil.test_enable, d->ds.stencil.front.op.fail,
                 d->ds.stencil.front.op.pass, d->ds.stencil.front.op.depth_fail,
                 d->ds.stencil.front.op.compare, d->ds.stencil.front.compare_mask,
                 d->ds.stencil.front.write_mask, d->ds.stencil.front.reference,
                 d->ms.rasterization_samples, 0);
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   /* Metal validation dislikes empty calls */
   if (instanceCount == 0 || vertexCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "draw", vertexCount, instanceCount);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   struct kk_draw_command data = {
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .predicate_count = cmd->state.cond_render.enabled ? 1u : 0u,
      .predicate_op[0] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .draw_count = 1,
      .predicate_addr[0] = cmd->state.cond_render.address,
      .draws[0] = {
         .vertexCount = vertexCount,
         .instanceCount = instanceCount,
         .firstVertex = firstVertex,
         .firstInstance = firstInstance,
      }};

   kk_draw(cmd, &data);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   /* Metal validation dislikes empty calls */
   if (drawCount == 0 || instanceCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "DrawMultiEXT", 0, 0);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* Build final draw list from parameters */
   struct kk_draw_command *data =
      rzalloc_size(NULL, sizeof(struct kk_draw_command) +
                            sizeof(VkDrawIndirectCommand) * (drawCount - 1u));
   if (!data) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   *data = (struct kk_draw_command){
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .predicate_count = cmd->state.cond_render.enabled ? 1u : 0u,
      .predicate_op[0] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .predicate_addr[0] = cmd->state.cond_render.address,
   };

   for (uint32_t i = 0; i < drawCount; ++i) {
      /* Metal validation dislikes empty calls */
      if (pVertexInfo->vertexCount > 0) {
         data->draws[data->draw_count] = (VkDrawIndirectCommand){
            .vertexCount = pVertexInfo->vertexCount,
            .instanceCount = instanceCount,
            .firstVertex = pVertexInfo->firstVertex,
            .firstInstance = firstInstance,
         };
         data->draw_count += 1u;
      }

      pVertexInfo = ((void *)pVertexInfo) + stride;
   }

   if (data->draw_count > 0)
      kk_draw(cmd, data);

   ralloc_free(data);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   /* Metal validation dislikes empty calls */
   if (instanceCount == 0 || indexCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "draw indexed", indexCount, instanceCount);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   struct kk_draw_command data = {
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .index_buffer = cmd->state.gfx.index.gpu,
      .index_buffer_el_size_B = cmd->state.gfx.index.bytes_per_index,
      .indexed = true,
      .predicate_count = cmd->state.cond_render.enabled ? 1u : 0u,
      .predicate_op[0] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .draw_count = 1,
      .predicate_addr[0] = cmd->state.cond_render.address,
      .indexed_draws[0] =
         {
            .indexCount = indexCount,
            .instanceCount = instanceCount,
            .firstIndex = firstIndex,
            .vertexOffset = vertexOffset,
            .firstInstance = firstInstance,
         },
   };

   kk_draw(cmd, &data);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   /* Metal validation dislikes empty calls */
   if (drawCount == 0 || instanceCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "DrawMultiIndexedEXT", 0, 0);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* Build final draw list from parameters */
   struct kk_draw_command *data = ralloc_size(
      NULL, sizeof(struct kk_draw_command) +
               sizeof(VkDrawIndexedIndirectCommand) * (drawCount - 1u));
   if (!data) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   *data = (struct kk_draw_command){
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .index_buffer = cmd->state.gfx.index.gpu,
      .index_buffer_el_size_B = cmd->state.gfx.index.bytes_per_index,
      .indexed = true,
      .predicate_count = cmd->state.cond_render.enabled ? 1u : 0u,
      .predicate_op[0] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .predicate_addr[0] = cmd->state.cond_render.address,
   };

   for (uint32_t i = 0; i < drawCount; ++i) {
      /* Metal validation dislikes empty calls */
      if (pIndexInfo->indexCount > 0) {
         data->indexed_draws[data->draw_count] = (VkDrawIndexedIndirectCommand){
            .indexCount = pIndexInfo->indexCount,
            .instanceCount = instanceCount,
            .firstIndex = pIndexInfo->firstIndex,
            .vertexOffset = pVertexOffset != NULL ? *pVertexOffset
                                                  : pIndexInfo->vertexOffset,
            .firstInstance = firstInstance,
         };
         data->draw_count += 1u;
      }

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }

   if (data->draw_count > 0)
      kk_draw(cmd, data);

   ralloc_free(data);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDrawIndirect2KHR(VkCommandBuffer commandBuffer,
                       const VkDrawIndirect2InfoKHR *pInfo)
{
   if (pInfo->drawCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "DrawIndirect2KHR", 0, 0);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   struct kk_draw_command data = {
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .indirect = true,
      .predicate_count = cmd->state.cond_render.enabled ? 1u : 0u,
      .predicate_op[0] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .draw_count = pInfo->drawCount,
      .predicate_addr[0] = cmd->state.cond_render.address,
      .indirect_command.addr = pInfo->addressRange.address,
      .indirect_command.stride = pInfo->addressRange.stride,
   };

   kk_draw(cmd, &data);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDrawIndexedIndirect2KHR(VkCommandBuffer commandBuffer,
                              const VkDrawIndirect2InfoKHR *pInfo)
{
   if (pInfo->drawCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "DrawIndexedIndirect2KHR", 0, 0);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   struct kk_draw_command data = {
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .index_buffer = cmd->state.gfx.index.gpu,
      .index_buffer_el_size_B = cmd->state.gfx.index.bytes_per_index,
      .indirect = true,
      .indexed = true,
      .predicate_count = cmd->state.cond_render.enabled ? 1u : 0u,
      .predicate_op[0] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .draw_count = pInfo->drawCount,
      .predicate_addr[0] = cmd->state.cond_render.address,
      .indirect_command.addr = pInfo->addressRange.address,
      .indirect_command.stride = pInfo->addressRange.stride,
   };

   kk_draw(cmd, &data);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDrawIndirectCount2KHR(VkCommandBuffer commandBuffer,
                            const VkDrawIndirectCount2InfoKHR *pInfo)
{
   if (pInfo->maxDrawCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "DrawIndirectCount2KHR", 0, 0);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   struct kk_draw_command data = {
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .indirect = true,
      .predicate_count = cmd->state.cond_render.enabled ? 2u : 1u,
      .predicate_op[0] = KK_PREDICATE_GT_DRAW_ID,
      .predicate_op[1] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .draw_count = pInfo->maxDrawCount,
      .predicate_addr[0] = pInfo->countAddressRange.address,
      .predicate_addr[1] = cmd->state.cond_render.address,
      .indirect_command.addr = pInfo->addressRange.address,
      .indirect_command.stride = pInfo->addressRange.stride,
   };

   kk_draw(cmd, &data);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdDrawIndexedIndirectCount2KHR(VkCommandBuffer commandBuffer,
                                   const VkDrawIndirectCount2InfoKHR *pInfo)
{
   if (pInfo->maxDrawCount == 0)
      return;

   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   kk_seq_draw(cmd, "DrawIndexedIndirectCount2KHR", 0, 0);
   if (kk_draw_prologue(cmd))
      return;

   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   struct kk_draw_command data = {
      .prim = vk_topology_to_mesa(dyn->ia.primitive_topology),
      .upload_mask = build_per_draw_upload_mask(cmd),
      .index_buffer = cmd->state.gfx.index.gpu,
      .index_buffer_el_size_B = cmd->state.gfx.index.bytes_per_index,
      .indirect = true,
      .indexed = true,
      .predicate_count = cmd->state.cond_render.enabled ? 2u : 1u,
      .predicate_op[0] = KK_PREDICATE_GT_DRAW_ID,
      .predicate_op[1] = cmd->state.cond_render.inverted
                            ? KK_PREDICATE_EQ_ZERO
                            : KK_PREDICATE_NEQ_ZERO,
      .draw_count = pInfo->maxDrawCount,
      .predicate_addr[0] = pInfo->countAddressRange.address,
      .predicate_addr[1] = cmd->state.cond_render.address,
      .indirect_command.addr = pInfo->addressRange.address,
      .indirect_command.stride = pInfo->addressRange.stride,
   };

   kk_draw(cmd, &data);
}
