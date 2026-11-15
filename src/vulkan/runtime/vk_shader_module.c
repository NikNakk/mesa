/*
 * Copyright © 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "vk_shader_module.h"
#include <stdio.h>
#include <stdlib.h>

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_log.h"
#include "vk_nir.h"
#include "vk_pipeline.h"
#include "vk_util.h"

void vk_shader_module_init(struct vk_device *device,
                           struct vk_shader_module *module,
                           const VkShaderModuleCreateInfo *create_info)
{
   vk_object_base_init(device, &module->base, VK_OBJECT_TYPE_SHADER_MODULE);

   module->nir = NULL;

   module->size = create_info->codeSize;
   memcpy(module->data, create_info->pCode, module->size);

   vk_shader_module_hash(create_info, module->hash);
}

/* KK_COUNT_RELAXED=1 (diagnostic): how much of the application's SPIR-V is decorated
 * RelaxedPrecision (16-bit is enough), per shader stage, logged every 100 modules. */
static void
vk_shader_module_count_relaxed(const uint32_t *code, size_t size)
{
   static int on = -1;
   if (on < 0)
      on = getenv("KK_COUNT_RELAXED") != NULL;
   if (!on || size < 20 || code[0] != 0x07230203)
      return;
   static struct { uint64_t modules, results, float_ops, relaxed; } st[8];
   static uint64_t total;
   static const char *const names[8] = {"vertex", "tess-ctrl", "tess-eval", "geometry",
                                        "fragment", "compute", "kernel", "other"};
   const size_t words = size / 4;
   uint32_t stage = 7;
   uint64_t results = 0, float_ops = 0, relaxed = 0;
   for (size_t i = 5; i < words;) {
      const uint32_t op = code[i] & 0xffff, n = code[i] >> 16;
      if (!n || i + n > words)
         break;
      if (op == 15 /* OpEntryPoint */ && stage == 7)
         stage = MIN2(code[i + 1], 6u);
      else if (op == 71 /* OpDecorate */ && n >= 3 && code[i + 2] == 0 /* RelaxedPrecision */)
         relaxed++;
      else if ((op >= 109 && op <= 127) || (op >= 129 && op <= 151) || op == 12 /* ExtInst */)
         float_ops++;
      if (op >= 59 /* OpVariable and later */ && n >= 3)
         results++;
      i += n;
   }
   st[stage].modules++;
   st[stage].results += results;
   st[stage].float_ops += float_ops;
   st[stage].relaxed += relaxed;
   if (++total % 100 == 0) {
      for (unsigned s = 0; s < 8; s++)
         if (st[s].modules)
            fprintf(stderr, "kk-relaxed: %s: %llu modules, %llu arithmetic/conversion "
                            "instructions, %llu RelaxedPrecision decorations (%.0f%% of "
                            "those instructions)\n",
                    names[s], (unsigned long long)st[s].modules,
                    (unsigned long long)st[s].float_ops, (unsigned long long)st[s].relaxed,
                    st[s].float_ops ? 100.0 * st[s].relaxed / st[s].float_ops : 0.0);
   }
}

void vk_shader_module_hash(const VkShaderModuleCreateInfo *create_info,
                           blake3_hash hash)
{
   vk_shader_module_count_relaxed(create_info->pCode, create_info->codeSize);
   _mesa_blake3_compute(create_info->pCode, create_info->codeSize, hash);
}

VKAPI_ATTR VkResult VKAPI_CALL
vk_common_CreateShaderModule(VkDevice _device,
                             const VkShaderModuleCreateInfo *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator,
                             VkShaderModule *pShaderModule)
{
    VK_FROM_HANDLE(vk_device, device, _device);
    struct vk_shader_module *module;

    assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    assert(pCreateInfo->flags == 0);

    module = vk_alloc2(&device->alloc, pAllocator,
                       sizeof(*module) + pCreateInfo->codeSize, 8,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    if (module == NULL)
       return VK_ERROR_OUT_OF_HOST_MEMORY;

    vk_shader_module_init(device, module, pCreateInfo);

    *pShaderModule = vk_shader_module_to_handle(module);

    return VK_SUCCESS;
}

const uint8_t vk_shaderModuleIdentifierAlgorithmUUID[VK_UUID_SIZE] = "MESA-BLAKE3";

VKAPI_ATTR void VKAPI_CALL
vk_common_GetShaderModuleIdentifierEXT(VkDevice _device,
                                       VkShaderModule _module,
                                       VkShaderModuleIdentifierEXT *pIdentifier)
{
   VK_FROM_HANDLE(vk_shader_module, module, _module);
   memcpy(pIdentifier->identifier, module->hash, sizeof(module->hash));
   pIdentifier->identifierSize = sizeof(module->hash);
}

VKAPI_ATTR void VKAPI_CALL
vk_common_GetShaderModuleCreateInfoIdentifierEXT(VkDevice _device,
                                                 const VkShaderModuleCreateInfo *pCreateInfo,
                                                 VkShaderModuleIdentifierEXT *pIdentifier)
{
   _mesa_blake3_compute(pCreateInfo->pCode, pCreateInfo->codeSize,
                        pIdentifier->identifier);
   pIdentifier->identifierSize = sizeof(blake3_hash);
}

VKAPI_ATTR void VKAPI_CALL
vk_common_DestroyShaderModule(VkDevice _device,
                              VkShaderModule _module,
                              const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vk_device, device, _device);
   VK_FROM_HANDLE(vk_shader_module, module, _module);

   if (!module)
      return;

   /* NIR modules (which are only created internally by the driver) are not
    * dynamically allocated so we should never call this for them.
    * Instead the driver is responsible for freeing the NIR code when it is
    * no longer needed.
    */
   assert(module->nir == NULL);

   vk_object_free(device, pAllocator, module);
}

#define SPIR_V_MAGIC_NUMBER 0x07230203

uint32_t
vk_shader_module_spirv_version(const struct vk_shader_module *mod)
{
   if (mod->nir != NULL)
      return 0;

   return vk_spirv_version((uint32_t *)mod->data, mod->size);
}
