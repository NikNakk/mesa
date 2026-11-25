/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_device.h"

/* TODO_KOSMICKRISP Remove */
#include "kk_image_layout.h"
#include "kk_private.h"

#include <Metal/MTL4Counters.h>
#include <Metal/MTLCaptureManager.h>
#include <Metal/MTLDevice.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Device creation */
mtl_device *
mtl_device_create()
{
   mtl_device *device = nil;

   @autoreleasepool {
      NSArray<id<MTLDevice>> *devs = [MTLCopyAllDevices() autorelease];
      uint32_t device_count = [devs count];
      
      for (uint32_t i = 0u; i < device_count; ++i) {
         if ([devs[i] supportsFamily:MTLGPUFamilyMetal4]) {
            device = (mtl_device *)[devs[i] retain];
            break;
         }
      }
   }

   return device;
}

/* Device operations */
static bool capture_window_attempted;
static bool capture_window_active;
static double capture_window_end;

static double
capture_time(void)
{
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   return now.tv_sec + now.tv_nsec / 1e9;
}

static bool
start_gpu_capture(mtl_device *mtl_dev_handle, const char *directory)
{
   @autoreleasepool {
      id<MTLDevice> mtl_dev = (id<MTLDevice>)mtl_dev_handle;
      MTLCaptureManager *captureMgr = [MTLCaptureManager sharedCaptureManager];

      MTLCaptureDescriptor *captureDesc = [[MTLCaptureDescriptor new] autorelease];
      captureDesc.captureObject = mtl_dev;
      captureDesc.destination = MTLCaptureDestinationDeveloperTools;

      if (directory && [captureMgr supportsDestination: MTLCaptureDestinationGPUTraceDocument]) {
         NSString *dir = [NSString stringWithUTF8String:directory];
         NSString *pname = [[NSProcessInfo processInfo] processName];
         NSString *capture_path = [NSString stringWithFormat:@"%@/%@.gputrace", dir, pname];
         captureDesc.destination = MTLCaptureDestinationGPUTraceDocument;
         captureDesc.outputURL = [NSURL fileURLWithPath: capture_path];
      }

      NSError *err = nil;
      if (![captureMgr startCaptureWithDescriptor:captureDesc error:&err]) {
         fprintf(stderr, "Failed to automatically start GPU capture session (Error code %li) using startCaptureWithDescriptor: %s\n",
                 (long)err.code, err.localizedDescription.UTF8String);
         return false;
      }
      fprintf(stderr, "kk-capture: started directory=%s\n", directory ? directory : "Xcode");
      return true;
   }
}

void
mtl_start_gpu_capture(mtl_device *device, const char *directory)
{
   @synchronized ([MTLCaptureManager sharedCaptureManager]) {
      if (getenv("KK_CAPTURE_TRIGGER_FILE")) {
         fprintf(stderr, "kk-capture: armed trigger=%s\n", getenv("KK_CAPTURE_TRIGGER_FILE"));
      } else {
         start_gpu_capture(device, directory);
      }
   }
}

bool
mtl_trigger_gpu_capture(mtl_device *device)
{
   bool started = false;
   @synchronized ([MTLCaptureManager sharedCaptureManager]) {
      if (!capture_window_attempted && !capture_window_active) {
         capture_window_attempted = true;
         capture_window_active = start_gpu_capture(
            device, getenv("MESA_KK_GPU_CAPTURE_DIRECTORY"));
         capture_window_end = 0.0;
         started = capture_window_active;
      }
   }
   return started;
}

bool
mtl_gpu_capture_active(void)
{
   bool active;
   @synchronized ([MTLCaptureManager sharedCaptureManager]) {
      active = capture_window_active;
   }
   return active;
}

/* `foveated`: the caller is submitting work that used implicit foveation. With
 * KK_CAPTURE_FOVEATED_ONLY=1 only such a submission starts the triggered capture, so
 * it records the device that renders the eyes rather than whichever of the process's
 * devices (an OpenXR layer's, say) begins a command buffer first. */
void
mtl_poll_gpu_capture_foveated(mtl_device *device, bool foveated)
{
   const char *trigger = getenv("KK_CAPTURE_TRIGGER_FILE");
   if (!trigger)
      return;
   static int foveated_only = -1;
   if (foveated_only < 0) {
      const char *v = getenv("KK_CAPTURE_FOVEATED_ONLY");
      foveated_only = v && !strcmp(v, "1");
   }
   @synchronized ([MTLCaptureManager sharedCaptureManager]) {
      if (foveated_only && !foveated && !capture_window_active)
         return;
      if (capture_window_active && capture_window_end > 0.0 &&
          capture_time() >= capture_window_end) {
         [[MTLCaptureManager sharedCaptureManager] stopCapture];
         capture_window_active = false;
         fprintf(stderr, "kk-capture: stopped\n");
      }
      if (!capture_window_attempted && access(trigger, F_OK) == 0) {
         /* The trigger file may carry settings, so they can change without relaunching:
          * "foveated" (as KK_CAPTURE_FOVEATED_ONLY=1) and "seconds=N". */
         char text[128] = {0};
         FILE *file = fopen(trigger, "r");
         if (file) {
            fread(text, 1, sizeof(text) - 1, file);
            fclose(file);
         }
         if (strstr(text, "foveated") && !foveated)
            return;
         capture_window_attempted = true;
         fprintf(stderr, "kk-capture: trigger observed%s\n", foveated ? " (foveated work)" : "");
         capture_window_active = start_gpu_capture(device,
                            getenv("MESA_KK_GPU_CAPTURE_DIRECTORY"));
         const char *seconds = strstr(text, "seconds=") ? strstr(text, "seconds=") + 8
                                                        : getenv("KK_CAPTURE_SECONDS");
         double duration = seconds ? atof(seconds) : 0.25;
         capture_window_end = capture_time() + (duration > 0 && duration <= 5 ? duration : 0.25);
      }
   }
}

void
mtl_poll_gpu_capture(mtl_device *device)
{
   mtl_poll_gpu_capture_foveated(device, false);
}

void
mtl_stop_gpu_capture()
{
   @autoreleasepool {
      @synchronized ([MTLCaptureManager sharedCaptureManager]) {
         [[MTLCaptureManager sharedCaptureManager] stopCapture];
         capture_window_active = false;
      }
   }
}

/* Device feature query */
void
mtl_device_get_name(mtl_device *dev, char buffer[256])
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      [device.name getCString:buffer maxLength:(sizeof(char) * 256) encoding:NSUTF8StringEncoding];
   }
}

void
mtl_device_get_architecture_name(mtl_device *dev, char buffer[256])
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      [device.architecture.name getCString:buffer maxLength:(sizeof(char) * 256) encoding:NSUTF8StringEncoding];
   }
}

uint32_t
mtl_device_get_gpu_apple_family(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      uint32_t gpu_family = 0u;
      MTLGPUFamily family = MTLGPUFamilyApple1;
      while([device supportsFamily:family]) {
         family += 1u;
         gpu_family += 1u;
      }
      return gpu_family;
   }
}

uint64_t
mtl_device_get_registry_id(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.registryID;
   }
}

bool
mtl_device_supports_sample_count(mtl_device *dev, uint32_t sample_count)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return [device supportsTextureSampleCount:sample_count];
   }
}

struct mtl_size
mtl_device_max_threads_per_threadgroup(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return (struct mtl_size){.x = device.maxThreadsPerThreadgroup.width,
                               .y = device.maxThreadsPerThreadgroup.height,
                               .z = device.maxThreadsPerThreadgroup.depth};
   }
}

uint32_t
mtl_device_max_threadgroup_memory_length(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.maxThreadgroupMemoryLength;
   }
}

uint64_t
mtl_device_max_buffer_length(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.maxBufferLength;
   }
}

uint64_t
mtl_device_recommended_max_working_set_size(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.recommendedMaxWorkingSetSize;
   }
}

uint64_t
mtl_device_current_allocated_size(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.currentAllocatedSize;
   }
}

uint32_t
mtl_device_max_argument_buffer_sampler_count(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.maxArgumentBufferSamplerCount;
   }
}

bool
mtl_device_supports_bc_texture_compression(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.supportsBCTextureCompression;
   }
}

bool
mtl_device_supports_32_bit_float_filtering(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.supports32BitFloatFiltering;
   }
}

/* Timestamp query */
uint64_t
mtl_device_get_gpu_timestamp(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      MTLTimestamp cpu_ts, gpu_ts;

      [device sampleTimestamps:&cpu_ts gpuTimestamp:&gpu_ts];

      return (uint64_t)gpu_ts;
   }
}

uint64_t
mtl_device_timestamp_frequency(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return [device queryTimestampFrequency];
   }
}

mtl_counter_heap *
mtl_new_timestamp_counter_heap(mtl_device *dev, uint32_t count)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      MTL4CounterHeapDescriptor *desc =
         [[MTL4CounterHeapDescriptor alloc] init];
      desc.type = MTL4CounterHeapTypeTimestamp;
      desc.count = count;

      NSError *error = nil;
      id<MTL4CounterHeap> heap = [device newCounterHeapWithDescriptor:desc
                                                                error:&error];
      [desc release];

      if (heap == nil) {
         fprintf(stderr, "Failed to create timestamp counter heap: %s\n",
                 error ? error.localizedDescription.UTF8String : "unknown");
         return NULL;
      }

      return (mtl_counter_heap *)heap;
   }
}

/* Resource queries */
/* TODO_KOSMICKRISP Return a struct */
void
mtl_heap_buffer_size_and_align_with_length(mtl_device *device, uint64_t *size_B,
                                           uint64_t *align_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLSizeAndAlign size_align = [dev heapBufferSizeAndAlignWithLength:*size_B options:KK_MTL_RESOURCE_OPTIONS];
      *size_B = size_align.size;
      *align_B = size_align.align;
   }
}

/* TODO_KOSMICKRISP Remove */
static MTLTextureDescriptor *
mtl_new_texture_descriptor(const struct kk_image_layout *layout)
{
   @autoreleasepool {
      MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
      descriptor.textureType = (MTLTextureType)layout->type;
      descriptor.pixelFormat = layout->format.mtl;
      descriptor.width = layout->width_px;
      descriptor.height = layout->height_px;
      descriptor.depth = layout->depth_px;
      descriptor.mipmapLevelCount = layout->levels;
      descriptor.sampleCount = layout->sample_count_sa;
      descriptor.arrayLength = layout->layers;
      descriptor.allowGPUOptimizedContents = layout->optimized_layout;
      descriptor.usage = (MTLTextureUsage)layout->usage;
      /* We don't set the swizzle because Metal complains when the usage has store or render target with swizzle... */
      
      return descriptor;
   }
}

uint64_t
mtl_minimum_linear_texture_alignment_for_pixel_format(
   mtl_device *device, enum mtl_pixel_format format)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format];
   }
}

void
mtl_heap_texture_size_and_align_with_descriptor(mtl_device *device,
                                                struct kk_image_layout *layout,
                                                uint64_t *size_B,
                                                uint64_t *align_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLTextureDescriptor *descriptor = [mtl_new_texture_descriptor(layout) autorelease];
      descriptor.resourceOptions = KK_MTL_RESOURCE_OPTIONS;

      MTLSizeAndAlign size_align = [dev heapTextureSizeAndAlignWithDescriptor:descriptor];
      if (size_B)
         *size_B = size_align.size;
      if (align_B)
         *align_B = size_align.align;
   }
}

uint32_t
mtl_sparse_tile_size_in_bytes(mtl_device *device) {
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev sparseTileSizeInBytes];
   }
}

struct mtl_size
mtl_sparse_tile_size(mtl_device *device, struct kk_image_layout *layout) {
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLSize tile_size = [dev sparseTileSizeWithTextureType:(MTLTextureType)layout->type
                                                 pixelFormat:layout->format.mtl
                                                 sampleCount:layout->sample_count_sa];
      return (struct mtl_size){tile_size.width, tile_size.height, tile_size.depth};
   }
}

struct mtl_size
mtl_sparse_tile_count(mtl_device *device, struct kk_image_layout *layout,
                      struct mtl_size tile_size) {
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLRegion pixel_region = MTLRegionMake3D(
          0, 0, 0, layout->width_px, layout->height_px, layout->depth_px);
      MTLRegion tile_region;
      [dev convertSparsePixelRegions:&pixel_region
                       toTileRegions:&tile_region
                        withTileSize:MTLSizeMake(tile_size.x, tile_size.y, tile_size.z)
                       alignmentMode:MTLSparseTextureRegionAlignmentModeOutward
                          numRegions:1];

      return (struct mtl_size){tile_region.size.width, tile_region.size.height,
                               tile_region.size.depth};
   }
}

/* Resource creation */
mtl_buffer *
mtl_new_buffer_with_bytes_no_copy(mtl_device *device, void* ptr,
                                  uint64_t size_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newBufferWithBytesNoCopy:ptr length:size_B options:KK_MTL_RESOURCE_OPTIONS deallocator:nil];
   }
}

mtl_command_allocator *
mtl_new_command_allocator(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newCommandAllocator];
   }
}

mtl_command_buffer *
mtl_new_command_buffer(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev newCommandBuffer];
   }
}
