/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_image.h"
#include "kk_image_view.h"
#include "kk_sync.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_metal_objects.h"

#include "util/log.h"
#include "vk_common_entrypoints.h"
#include "vk_semaphore.h"
#include "vk_util.h"

#ifdef VK_USE_PLATFORM_METAL_EXT

static void
kk_residency_add(struct kk_device *dev, mtl_texture *texture)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, texture);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_image_release_metal_import(struct kk_device *dev, struct kk_image *image)
{
   if (!image->metal_imported || !image->planes[0].mtl_handle)
      return;
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_remove_allocation(dev->residency_set.handle,
                                       image->planes[0].mtl_handle);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

static VkResult
kk_image_adopt_texture(struct kk_device *dev, struct kk_image *image,
                       mtl_texture *texture)
{
   struct kk_image_plane *plane = &image->planes[0];
   struct mtl_texture_mismatch mm;
   mtl_texture_check_layout(texture, dev->mtl_handle, &plane->layout, &mm);
   uint32_t missing = mm.missing_usage & ~MTL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;
   if (mm.device || mm.type || mm.format || mm.extent || mm.levels ||
       mm.samples || mm.layers || missing) {
      mesa_loge("kk: VkImportMetalTextureInfoEXT rejected: device=%d type=%d "
                "format=%d extent=%d levels=%d samples=%d layers=%d "
                "missing_usage=0x%x",
                mm.device, mm.type, mm.format, mm.extent, mm.levels,
                mm.samples, mm.layers, mm.missing_usage);
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }
   if (mm.missing_usage)
      mesa_logw("kk: imported texture lacks MTLTextureUsagePixelFormatView");

   plane->mtl_handle = mtl_retain(texture);
   return VK_SUCCESS;
}

VkResult
kk_image_import_metal(struct kk_device *dev, struct kk_image *image,
                      const VkImageCreateInfo *info)
{
   const VkImportMetalTextureInfoEXT *tex_info =
      vk_find_struct_const(info->pNext, IMPORT_METAL_TEXTURE_INFO_EXT);
   const VkImportMetalIOSurfaceInfoEXT *ios_info =
      vk_find_struct_const(info->pNext, IMPORT_METAL_IO_SURFACE_INFO_EXT);

   if (!tex_info && !ios_info)
      return VK_SUCCESS;

   if (tex_info && ios_info)
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   if (image->plane_count != 1 || image->planes[0].layout.linear)
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);

   struct kk_image_plane *plane = &image->planes[0];
   VkResult result;

   if (tex_info) {
      if (!tex_info->mtlTexture ||
          (tex_info->plane != 0 &&
           tex_info->plane != VK_IMAGE_ASPECT_PLANE_0_BIT))
         return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      result = kk_image_adopt_texture(dev, image, tex_info->mtlTexture);
      if (result != VK_SUCCESS)
         return result;
   } else {
      if (!ios_info->ioSurface) {
         mesa_loge("kk: VkImportMetalIOSurfaceInfoEXT with a NULL IOSurface "
                   "is not supported");
         return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      if (info->imageType != VK_IMAGE_TYPE_2D || info->arrayLayers != 1 ||
          info->mipLevels != 1 || info->samples != VK_SAMPLE_COUNT_1_BIT ||
          plane->layout.type != MTL_TEXTURE_TYPE_2D)
         return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
      plane->mtl_handle = mtl_new_texture_with_iosurface(
         dev->mtl_handle, &plane->layout, ios_info->ioSurface);
      if (!plane->mtl_handle) {
         mesa_loge("kk: newTextureWithDescriptor:iosurface:plane: failed");
         return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      }
   }

   plane->mtl_handle_array = NULL;
   plane->addr = mtl_texture_get_gpu_resource_id(plane->mtl_handle);
   if (image->vk.base.object_name)
      mtl_resource_set_label(plane->mtl_handle, image->vk.base.object_name);
   kk_residency_add(dev, plane->mtl_handle);
   image->metal_imported = true;
   return VK_SUCCESS;
}

static uint8_t
kk_metal_plane_index(VkImageAspectFlagBits plane)
{
   switch (plane) {
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
      return 1;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
      return 2;
   default:
      return 0;
   }
}

static mtl_texture *
kk_export_texture(const VkExportMetalTextureInfoEXT *info)
{
   if (info->image != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_image, image, info->image);
      uint8_t p = kk_metal_plane_index(info->plane);
      return p < image->plane_count ? image->planes[p].mtl_handle : NULL;
   }
   if (info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_image_view, view, info->imageView);
      uint8_t p = kk_metal_plane_index(info->plane);
      return p < view->plane_count ? view->planes[p].mtl_handle_sampled : NULL;
   }
   return NULL;
}

static mtl_shared_event *
kk_export_shared_event(const VkExportMetalSharedEventInfoEXT *info)
{
   if (info->semaphore == VK_NULL_HANDLE)
      return NULL;
   VK_FROM_HANDLE(vk_semaphore, sem, info->semaphore);
   struct vk_sync *sync = vk_semaphore_get_active_sync(sem);
   if (sync->type != &kk_sync_type)
      return NULL;
   return container_of(sync, struct kk_sync_timeline, base)->mtl_handle;
}

VKAPI_ATTR void VKAPI_CALL
kk_ExportMetalObjectsEXT(VkDevice _device,
                         VkExportMetalObjectsInfoEXT *pMetalObjectsInfo)
{
   VK_FROM_HANDLE(kk_device, dev, _device);

   vk_foreach_struct(stype, s, pMetalObjectsInfo) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT:
         ((VkExportMetalDeviceInfoEXT *)s)->mtlDevice = dev->mtl_handle;
         break;
      case VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT:
         ((VkExportMetalCommandQueueInfoEXT *)s)->mtlCommandQueue = NULL;
         break;
      case VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT:
         ((VkExportMetalBufferInfoEXT *)s)->mtlBuffer = NULL;
         break;
      case VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT: {
         VkExportMetalTextureInfoEXT *t = (VkExportMetalTextureInfoEXT *)s;
         t->mtlTexture = kk_export_texture(t);
         break;
      }
      case VK_STRUCTURE_TYPE_EXPORT_METAL_IO_SURFACE_INFO_EXT: {
         VkExportMetalIOSurfaceInfoEXT *i = (VkExportMetalIOSurfaceInfoEXT *)s;
         i->ioSurface = NULL;
         if (i->image != VK_NULL_HANDLE) {
            VK_FROM_HANDLE(kk_image, image, i->image);
            if (image->planes[0].mtl_handle)
               i->ioSurface =
                  mtl_texture_get_iosurface(image->planes[0].mtl_handle);
         }
         break;
      }
      case VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT: {
         VkExportMetalSharedEventInfoEXT *e =
            (VkExportMetalSharedEventInfoEXT *)s;
         e->mtlSharedEvent = kk_export_shared_event(e);
         break;
      }
      default:
         break;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkSemaphore *pSemaphore)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   const VkImportMetalSharedEventInfoEXT *import =
      vk_find_struct_const(pCreateInfo->pNext,
                           IMPORT_METAL_SHARED_EVENT_INFO_EXT);

   VkResult result =
      vk_common_CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
   if (result != VK_SUCCESS || !import)
      return result;

   VK_FROM_HANDLE(vk_semaphore, sem, *pSemaphore);
   struct vk_sync *sync = vk_semaphore_get_active_sync(sem);
   if (!import->mtlSharedEvent || sem->type != VK_SEMAPHORE_TYPE_TIMELINE ||
       sync->type != &kk_sync_type) {
      vk_common_DestroySemaphore(device, *pSemaphore, pAllocator);
      *pSemaphore = VK_NULL_HANDLE;
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }

   if (mtl_shared_event_device_registry_id(import->mtlSharedEvent) !=
       mtl_device_get_registry_id(dev->mtl_handle))
      mesa_logw("kk: imported MTLSharedEvent belongs to another MTLDevice");

   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   void *retained = mtl_retain(import->mtlSharedEvent);
   mtl_release(timeline->mtl_handle);
   timeline->mtl_handle = retained;
   return VK_SUCCESS;
}

#endif /* VK_USE_PLATFORM_METAL_EXT */
