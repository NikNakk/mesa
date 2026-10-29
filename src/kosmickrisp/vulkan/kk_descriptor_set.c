/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_descriptor_set.h"

#include "kk_bo.h"
#include "kosmickrisp/bridge/mtl_buffer.h"
#include "kk_buffer.h"
#include "kk_buffer_view.h"
#include "kk_descriptor_set_layout.h"
#include "kk_device.h"
#include "kk_image.h"
#include "kk_fdm.h"
#include "kk_entrypoints.h"
#include "kk_image_view.h"
#include "kk_physical_device.h"
#include "kk_sampler.h"

#include "util/format/u_format.h"
#include "util/log.h"
#include "util/u_debug.h"
#include "util/half_float.h"

DEBUG_GET_ONCE_BOOL_OPTION(kk_descriptor_diag, "KK_DESCRIPTOR_DIAG", false);

static inline uint32_t
align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

static inline void *
desc_ubo_data(struct kk_descriptor_set *set, uint32_t binding, uint32_t elem,
              uint32_t *size_out)
{
   const struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   uint32_t offset = binding_layout->offset + elem * binding_layout->stride;
   assert(offset < set->size);

   if (size_out != NULL)
      *size_out = set->size - offset;

   return (char *)set->mapped_ptr + offset;
}

static void
write_desc(struct kk_descriptor_set *set, uint32_t binding, uint32_t elem,
           const void *desc_data, size_t desc_size)
{
   ASSERTED uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   assert(desc_size <= dst_size);
   memcpy(dst, desc_data, desc_size);
}

static void
get_sampled_image_view_desc(VkDescriptorType descriptor_type,
                            const VkDescriptorImageInfo *const info, void *dst,
                            size_t dst_size, bool is_input_attachment)
{
   struct kk_sampled_image_descriptor desc[3] = {};
   uint8_t plane_count = 1;

   if (descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER && info &&
       info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_image_view, view, info->imageView);

      plane_count = view->plane_count;
      for (uint8_t plane = 0; plane < plane_count; plane++) {
         if (is_input_attachment) {
            assert(view->planes[plane].input_gpu_resource_id);
            desc[plane].image_gpu_resource_id =
               view->planes[plane].input_gpu_resource_id;
            if (!strcmp(debug_get_option("KK_CUSTOM_RESOLVE_DEBUG", ""), "log")) {
               static uint32_t logged;
               if (logged++ < 64)
                  mesa_logi("kk-cr-desc: input view=%p samples=%u type=%d "
                            "input=%llx sampled=%llx",
                            (void *)view, view->vk.image->samples,
                            view->vk.view_type,
                            (unsigned long long)view->planes[plane].input_gpu_resource_id,
                            (unsigned long long)view->planes[plane].sampled_gpu_resource_id);
            }
         } else {
            assert(view->planes[plane].sampled_gpu_resource_id);
            desc[plane].image_gpu_resource_id =
               view->planes[plane].sampled_gpu_resource_id;
         }

         const struct kk_image *image =
            container_of(view->vk.image, struct kk_image, vk);
         desc[plane].rate_params =
            image->rate_params && !is_input_attachment ? image->rate_params->gpu
                                                       : 0;
         desc[plane].rate_base_layer = view->vk.base_array_layer;
         desc[plane].rate_image_size[0] =
            u_minify(view->vk.image->extent.width, view->vk.base_mip_level);
         desc[plane].rate_image_size[1] =
            u_minify(view->vk.image->extent.height, view->vk.base_mip_level);
         kk_fdm_note_view(&view->vk, is_input_attachment ? "input" : "sampled");
         if (is_input_attachment)
            kk_fdm_note_use(&image->vk, "input attachment");
         else if (descriptor_type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            kk_fdm_note_use(&image->vk, "sampled image, separate sampler");
         else if (info->sampler == VK_NULL_HANDLE)
            kk_fdm_note_use(&image->vk, "combined, no sampler in write");
         else if (vk_sampler_from_handle(info->sampler)->flags &
                  VK_SAMPLER_CREATE_SUBSAMPLED_BIT_EXT)
            kk_fdm_note_use(&image->vk, "combined, subsampled sampler");
         else
            kk_fdm_note_use(&image->vk, "combined, ordinary sampler");

         float min_lod = MAX2(view->vk.min_lod - view->vk.base_mip_level, 0.0);
         desc[plane].image_min_lod_fp16 = _mesa_float_to_half(min_lod);
         desc[plane].image_min_lod_uint16 = min_lod;
      }
   }

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      VK_FROM_HANDLE(kk_sampler, sampler, info->sampler);

      if (sampler->has_border)
         assert(plane_count == 1);
      else
         plane_count = MAX2(plane_count, sampler->plane_count);

      for (uint8_t plane = 0; plane < plane_count; plane++) {
         /* We need to replicate the last sampler plane out to all image
          * planes due to sampler table entry limitations. See
          * nvk_CreateSampler in nvk_sampler.c for more details.
          */
         uint8_t sampler_plane = MIN2(plane, sampler->plane_count - 1u);
         assert(sampler->planes[sampler_plane].hw->handle);
         desc[plane].sampler_index = sampler->planes[sampler_plane].hw->index;
         desc[plane].sampler_lod_bias_fp16 = sampler->lod_bias_fp16;
         desc[plane].sampler_lod_min_fp16 = sampler->lod_min_fp16;
         desc[plane].sampler_lod_max_fp16 = sampler->lod_max_fp16;
         desc[plane].clamp_0_sampler_index_or_negative = -1;
      }

      if (sampler->has_border) {
         assert(sampler->plane_count == 2);
         desc[0].clamp_0_sampler_index_or_negative =
            sampler->planes[1].hw->index;

         assert(desc[0].clamp_0_sampler_index_or_negative >= 0 &&
                "we have a border colour");

         static_assert(sizeof(desc[0].border) == sizeof(sampler->custom_border),
                       "fixed format");

         memcpy(desc[0].border, sampler->custom_border.uint32,
                sizeof(sampler->custom_border));
      }
   }

   assert(sizeof(desc[0]) * plane_count <= dst_size);
   memcpy(dst, desc, sizeof(desc[0]) * plane_count);
}

static void
write_sampled_image_view_desc(struct kk_descriptor_set *set,
                              const VkDescriptorImageInfo *const _info,
                              uint32_t binding, uint32_t elem,
                              VkDescriptorType descriptor_type)
{
   VkDescriptorImageInfo info = *_info;

   struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];
   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      if (binding_layout->immutable_samplers != NULL) {
         info.sampler =
            kk_sampler_to_handle(binding_layout->immutable_samplers[elem]);
      }
   }

   uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   get_sampled_image_view_desc(
      descriptor_type, &info, dst, dst_size,
      descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
}

static void
get_storage_image_view_desc(
   struct kk_descriptor_set_binding_layout *binding_layout,
   const VkDescriptorImageInfo *const info, void *dst, size_t dst_size)
{
   struct kk_storage_image_descriptor desc = {};

   if (info && info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_image_view, view, info->imageView);

      /* Storage images are always single plane */
      assert(view->plane_count == 1);
      kk_fdm_note_use(view->vk.image, "storage image");
      uint8_t plane = 0;

      assert(view->planes[plane].storage_gpu_resource_id);
      desc.image_gpu_resource_id = view->planes[plane].storage_gpu_resource_id;
   }

   assert(sizeof(desc) <= dst_size);
   memcpy(dst, &desc, sizeof(desc));
}

static void
write_storage_image_view_desc(struct kk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem)
{
   uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];
   get_storage_image_view_desc(binding_layout, info, dst, dst_size);
}

static void
write_buffer_desc(struct kk_descriptor_set *set,
                  const VkDescriptorBufferInfo *const info, uint32_t binding,
                  uint32_t elem)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);

   const struct kk_addr_range addr_range =
      kk_buffer_addr_range(buffer, info->offset, info->range);
   assert(addr_range.range <= UINT32_MAX);

   const struct kk_buffer_address desc = {
      .base_addr = addr_range.addr,
      .size = addr_range.range,
   };
   write_desc(set, binding, elem, &desc, sizeof(desc));
   const struct kk_descriptor_set_binding_layout *bl = &set->layout->binding[binding];
   if (debug_get_option_kk_descriptor_diag() &&
       bl->offset + elem * bl->stride == 64) {
      void *cpu = buffer ? mtl_get_contents(buffer->metal.handle) : NULL;
      set->diagnostic_addr = desc.base_addr;
      set->diagnostic_cpu = cpu && desc.size >= 8
         ? (char *)cpu + kk_buffer_mtl_offset(buffer, info->offset) : NULL;
      static unsigned logged;
      if (p_atomic_inc_return(&logged) <= 256 || !desc.base_addr)
         mesa_logi("kk-desc-write: set=%p layout=%p binding=%u elem=%u type=%u "
                   "buffer=%p offset=%llu range=%llu addr=%llx size=%u cpu=%p",
                   (void *)set, (void *)set->layout, binding, elem, bl->type,
                   (void *)buffer, (unsigned long long)info->offset,
                   (unsigned long long)info->range,
                   (unsigned long long)desc.base_addr, desc.size, set->diagnostic_cpu);
   }
}

void
kk_descriptor_diagnostic_submit(struct kk_descriptor_set *set)
{
   static unsigned logged, anomalies;
   if (!set || set->size < 80)
      return;
   struct kk_buffer_address desc;
   memcpy(&desc, (char *)set->mapped_ptr + 64, sizeof(desc));
   int32_t header[2] = {0};
   bool mapped = set->diagnostic_cpu && set->diagnostic_addr == desc.base_addr;
   if (mapped)
      memcpy(header, set->diagnostic_cpu, sizeof(header));
   bool bad = !desc.base_addr || desc.size < 8 ||
              (mapped && (header[1] < 0 || header[1] > 8 ||
               (uint64_t)352 + (uint64_t)16 * header[1] > desc.size));
   if (p_atomic_inc_return(&logged) <= 128 ||
       (bad && p_atomic_inc_return(&anomalies) <= 128)) {
      mesa_logi("kk-desc-submit: set=%p layout=%p set_addr=%llx addr=%llx size=%u "
                "mapped=%u enabled=%d count=%d suspect=%u root_sets_offset=%zu",
                (void *)set, (void *)set->layout, (unsigned long long)set->addr,
                (unsigned long long)desc.base_addr, desc.size, mapped,
                header[0], header[1], bad, (size_t)928);
      for (unsigned b = 0; b < set->layout->binding_count; b++) {
         const struct kk_descriptor_set_binding_layout *bl = &set->layout->binding[b];
         if (bl->array_size && bl->stride && bl->offset <= 64 &&
             64 - bl->offset < bl->array_size * bl->stride)
            mesa_logi("kk-desc-layout: layout=%p binding=%u type=%u offset=%u stride=%u array=%u",
                      (void *)set->layout, b, bl->type, bl->offset, bl->stride, bl->array_size);
      }
   }
}

static void
write_dynamic_buffer_desc(struct kk_descriptor_set *set,
                          const VkDescriptorBufferInfo *const info,
                          uint32_t binding, uint32_t elem)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);
   const struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   const struct kk_addr_range addr_range =
      kk_buffer_addr_range(buffer, info->offset, info->range);
   assert(addr_range.range <= UINT32_MAX);

   struct kk_buffer_address *desc =
      &set->dynamic_buffers[binding_layout->dynamic_buffer_index + elem];
   *desc = (struct kk_buffer_address){
      .base_addr = addr_range.addr,
      .size = addr_range.range,
   };
}

static void
write_buffer_view_desc(struct kk_descriptor_set *set,
                       const VkBufferView bufferView, uint32_t binding,
                       uint32_t elem)
{
   struct kk_storage_image_descriptor desc = {};
   if (bufferView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_buffer_view, view, bufferView);

      assert(view->mtl_texel_buffer_handle);
      assert(view->texel_buffer_gpu_id);

      desc.image_gpu_resource_id = view->texel_buffer_gpu_id;
   }
   write_desc(set, binding, elem, &desc, sizeof(desc));

}

static void
write_inline_uniform_data(struct kk_descriptor_set *set,
                          const VkWriteDescriptorSetInlineUniformBlock *info,
                          uint32_t binding, uint32_t offset)
{
   assert(set->layout->binding[binding].stride == 1);
   write_desc(set, binding, offset, info->pData, info->dataSize);
}

VKAPI_ATTR void VKAPI_CALL
kk_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                        const VkWriteDescriptorSet *pDescriptorWrites,
                        uint32_t descriptorCopyCount,
                        const VkCopyDescriptorSet *pDescriptorCopies)
{
   for (uint32_t w = 0; w < descriptorWriteCount; w++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[w];
      VK_FROM_HANDLE(kk_descriptor_set, set, write->dstSet);

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(set, write->pBufferInfo + j, write->dstBinding,
                              write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_dynamic_buffer_desc(set, write->pBufferInfo + j,
                                      write->dstBinding,
                                      write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
         const VkWriteDescriptorSetInlineUniformBlock *write_inline =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
         assert(write_inline->dataSize == write->descriptorCount);
         write_inline_uniform_data(set, write_inline, write->dstBinding,
                                   write->dstArrayElement);
         break;
      }

      default:
         break;
      }
   }

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      VK_FROM_HANDLE(kk_descriptor_set, src, copy->srcSet);
      VK_FROM_HANDLE(kk_descriptor_set, dst, copy->dstSet);

      const struct kk_descriptor_set_binding_layout *src_binding_layout =
         &src->layout->binding[copy->srcBinding];
      const struct kk_descriptor_set_binding_layout *dst_binding_layout =
         &dst->layout->binding[copy->dstBinding];

      if (dst_binding_layout->stride > 0 && src_binding_layout->stride > 0) {
         for (uint32_t j = 0; j < copy->descriptorCount; j++) {
            ASSERTED uint32_t dst_max_size, src_max_size;
            void *dst_map = desc_ubo_data(
               dst, copy->dstBinding, copy->dstArrayElement + j, &dst_max_size);
            const void *src_map = desc_ubo_data(
               src, copy->srcBinding, copy->srcArrayElement + j, &src_max_size);
            const uint32_t copy_size =
               MIN2(dst_binding_layout->stride, src_binding_layout->stride);
            assert(copy_size <= dst_max_size && copy_size <= src_max_size);
            memcpy(dst_map, src_map, copy_size);
            if (debug_get_option_kk_descriptor_diag() &&
                dst_binding_layout->offset + (copy->dstArrayElement + j) * dst_binding_layout->stride == 64) {
               bool tracked = src_binding_layout->offset + (copy->srcArrayElement + j) * src_binding_layout->stride == 64;
               dst->diagnostic_addr = tracked ? src->diagnostic_addr : 0;
               dst->diagnostic_cpu = tracked ? src->diagnostic_cpu : NULL;
               mesa_logi("kk-desc-copy: src=%p binding=%u elem=%u dst=%p binding=%u elem=%u tracked=%u",
                         (void *)src, copy->srcBinding, copy->srcArrayElement+j,
                         (void *)dst, copy->dstBinding, copy->dstArrayElement+j, tracked);
            }
         }
      }

      switch (src_binding_layout->type) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
         const uint32_t dst_dyn_start =
            dst_binding_layout->dynamic_buffer_index + copy->dstArrayElement;
         const uint32_t src_dyn_start =
            src_binding_layout->dynamic_buffer_index + copy->srcArrayElement;
         typed_memcpy(&dst->dynamic_buffers[dst_dyn_start],
                      &src->dynamic_buffers[src_dyn_start],
                      copy->descriptorCount);
         break;
      }
      default:
         break;
      }
   }
}

void
kk_push_descriptor_set_update(struct kk_push_descriptor_set *push_set,
                              uint32_t write_count,
                              const VkWriteDescriptorSet *writes)
{
   struct kk_descriptor_set_layout *layout = push_set->layout;
   assert(layout->non_variable_descriptor_buffer_size < sizeof(push_set->data));
   struct kk_descriptor_set set = {
      .layout = push_set->layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };

   for (uint32_t w = 0; w < write_count; w++) {
      const VkWriteDescriptorSet *write = &writes[w];
      assert(write->dstSet == VK_NULL_HANDLE);

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               &set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(&set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(&set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(&set, write->pBufferInfo + j, write->dstBinding,
                              write->dstArrayElement + j);
         }
         break;

      default:
         break;
      }
   }
}

static void kk_descriptor_pool_free(struct kk_descriptor_pool *pool,
                                    uint64_t addr, uint64_t size);

void
kk_descriptor_pool_set_label(struct kk_descriptor_pool *pool, const char *label)
{
   if (pool->bo)
      kk_bo_set_label(pool->bo, label);
}

static void
kk_descriptor_set_destroy(struct kk_device *dev,
                          struct kk_descriptor_pool *pool,
                          struct kk_descriptor_set *set)
{
   list_del(&set->link);
   if (set->size > 0)
      kk_descriptor_pool_free(pool, set->addr, set->size);
   vk_descriptor_set_layout_unref(&dev->vk, &set->layout->vk);

   vk_object_free(&dev->vk, NULL, set);
}

static void
kk_destroy_descriptor_pool(struct kk_device *dev,
                           const VkAllocationCallbacks *pAllocator,
                           struct kk_descriptor_pool *pool)
{
   list_for_each_entry_safe(struct kk_descriptor_set, set, &pool->sets, link)
      kk_descriptor_set_destroy(dev, pool, set);

   util_vma_heap_finish(&pool->heap);

   if (pool->bo != NULL)
      kk_destroy_bo(dev, pool->bo);

   vk_object_free(&dev->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateDescriptorPool(VkDevice _device,
                        const VkDescriptorPoolCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_descriptor_pool *pool;
   VkResult result = VK_SUCCESS;

   pool = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   list_inithead(&pool->sets);

   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   uint32_t max_align = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      kk_descriptor_stride_align_for_type(pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      max_align = MAX2(max_align, alignment);
   }

   uint64_t mem_size = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      kk_descriptor_stride_align_for_type(pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      mem_size +=
         MAX2(stride, max_align) * pCreateInfo->pPoolSizes[i].descriptorCount;
   }

   /* Individual descriptor sets are aligned to the min UBO alignment to
    * ensure that we don't end up with unaligned data access in any shaders.
    * This means that each descriptor buffer allocated may burn up to 16B of
    * extra space to get the right alignment.  (Technically, it's at most 28B
    * because we're always going to start at least 4B aligned but we're being
    * conservative here.)  Allocate enough extra space that we can chop it
    * into maxSets pieces and align each one of them to 32B.
    */
   mem_size += kk_min_cbuf_alignment() * pCreateInfo->maxSets;

   if (mem_size) {
      result = kk_alloc_bo(dev, &dev->vk.base, mem_size, 0u, &pool->bo);
      if (result != VK_SUCCESS) {
         kk_destroy_descriptor_pool(dev, pAllocator, pool);
         return result;
      }

      /* The BO may be larger thanks to GPU page alignment.  We may as well
       * make that extra space available to the client.
       */
      assert(pool->bo->size_B >= mem_size);
      util_vma_heap_init(&pool->heap, pool->bo->gpu, pool->bo->size_B);
   } else {
      util_vma_heap_init(&pool->heap, 0, 0);
   }

   *pDescriptorPool = kk_descriptor_pool_to_handle(pool);
   return result;
}

static VkResult
kk_descriptor_pool_alloc(struct kk_descriptor_pool *pool, uint64_t size,
                         uint64_t alignment, uint64_t *addr_out, void **map_out)
{
   assert(size > 0);
   assert(size % alignment == 0);

   if (size > pool->heap.free_size)
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   uint64_t addr = util_vma_heap_alloc(&pool->heap, size, alignment);
   if (addr == 0)
      return VK_ERROR_FRAGMENTED_POOL;

   assert(addr >= pool->bo->gpu);
   assert(addr + size <= pool->bo->gpu + pool->bo->size_B);
   uint64_t offset = addr - pool->bo->gpu;

   *addr_out = addr;
   *map_out = pool->bo->cpu + offset;

   return VK_SUCCESS;
}

static void
kk_descriptor_pool_free(struct kk_descriptor_pool *pool, uint64_t addr,
                        uint64_t size)
{
   assert(size > 0);
   assert(addr >= pool->bo->gpu);
   assert(addr + size <= pool->bo->gpu + pool->bo->size_B);
   util_vma_heap_free(&pool->heap, addr, size);
}

static VkResult
kk_descriptor_set_create(struct kk_device *dev, struct kk_descriptor_pool *pool,
                         struct kk_descriptor_set_layout *layout,
                         uint32_t variable_count,
                         struct kk_descriptor_set **out_set)
{
   struct kk_descriptor_set *set;
   VkResult result = VK_SUCCESS;

   uint32_t mem_size =
      sizeof(struct kk_descriptor_set) +
      layout->vk.dynamic_descriptor_count * sizeof(struct kk_buffer_address);
   set =
      vk_object_zalloc(&dev->vk, NULL, mem_size, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   set->size = layout->non_variable_descriptor_buffer_size;

   if (layout->binding_count > 0 &&
       (layout->binding[layout->binding_count - 1].flags &
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)) {
      uint32_t stride = layout->binding[layout->binding_count - 1].stride;
      set->size += stride * variable_count;
   }

   uint32_t alignment = kk_min_cbuf_alignment();
   set->size = align64(set->size, alignment);

   if (set->size > 0) {
      result = kk_descriptor_pool_alloc(pool, set->size, alignment, &set->addr,
                                        &set->mapped_ptr);
      if (result != VK_SUCCESS) {
         vk_object_free(&dev->vk, NULL, set);
         return result;
      }
   }

   vk_descriptor_set_layout_ref(&layout->vk);
   set->layout = layout;

   for (uint32_t b = layout->first_immutable_binding; b != UINT32_MAX;
        b = layout->binding[b].next_immutable_binding) {
      uint32_t array_size = layout->binding[b].array_size;
      if (layout->binding[b].flags &
          VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         array_size = variable_count;

      const VkDescriptorImageInfo empty = {};
      for (uint32_t j = 0; j < array_size; j++) {
         write_sampled_image_view_desc(set, &empty, b, j,
                                       layout->binding[b].type);
      }
   }

   list_addtail(&set->link, &pool->sets);
   *out_set = set;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_AllocateDescriptorSets(VkDevice device,
                          const VkDescriptorSetAllocateInfo *pAllocateInfo,
                          VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, pAllocateInfo->descriptorPool);

   VkResult result = VK_SUCCESS;
   uint32_t i;

   struct kk_descriptor_set *set = NULL;

   const VkDescriptorSetVariableDescriptorCountAllocateInfo *var_desc_count =
      vk_find_struct_const(
         pAllocateInfo->pNext,
         DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

   /* allocate a set of buffers for each shader to contain descriptors */
   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(kk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      /* If descriptorSetCount is zero or this structure is not included in
       * the pNext chain, then the variable lengths are considered to be zero.
       */
      const uint32_t variable_count =
         var_desc_count && var_desc_count->descriptorSetCount > 0
            ? var_desc_count->pDescriptorCounts[i]
            : 0;

      result =
         kk_descriptor_set_create(dev, pool, layout, variable_count, &set);
      if (result != VK_SUCCESS)
         break;

      pDescriptorSets[i] = kk_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS) {
      kk_FreeDescriptorSets(device, pAllocateInfo->descriptorPool, i,
                            pDescriptorSets);
      for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
         pDescriptorSets[i] = VK_NULL_HANDLE;
      }
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                      uint32_t descriptorSetCount,
                      const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, descriptorPool);

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      VK_FROM_HANDLE(kk_descriptor_set, set, pDescriptorSets[i]);

      if (set)
         kk_descriptor_set_destroy(dev, pool, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyDescriptorPool(VkDevice device, VkDescriptorPool _pool,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, _pool);

   if (!_pool)
      return;

   kk_destroy_descriptor_pool(dev, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                       VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, descriptorPool);

   list_for_each_entry_safe(struct kk_descriptor_set, set, &pool->sets, link)
      kk_descriptor_set_destroy(dev, pool, set);

   return VK_SUCCESS;
}

static void
kk_descriptor_set_write_template(
   struct kk_descriptor_set *set,
   const struct vk_descriptor_update_template *template, const void *data)
{
   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct vk_descriptor_template_entry *entry = &template->entries[i];

      switch (entry->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_sampled_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j,
                                          entry->type);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_storage_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkBufferView *bview =
               data + entry->offset + j * entry->stride;

            write_buffer_view_desc(set, *bview, entry->binding,
                                   entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_buffer_desc(set, info, entry->binding,
                              entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_dynamic_buffer_desc(set, info, entry->binding,
                                      entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
         write_desc(set, entry->binding, entry->array_element,
                    data + entry->offset, entry->array_count);
         break;

      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_UpdateDescriptorSetWithTemplate(
   VkDevice device, VkDescriptorSet descriptorSet,
   VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void *pData)
{
   VK_FROM_HANDLE(kk_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  descriptorUpdateTemplate);

   kk_descriptor_set_write_template(set, template, pData);
}

void
kk_push_descriptor_set_update_template(
   struct kk_push_descriptor_set *push_set,
   struct kk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data)
{
   struct kk_descriptor_set tmp_set = {
      .layout = layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };
   kk_descriptor_set_write_template(&tmp_set, template, data);
}
