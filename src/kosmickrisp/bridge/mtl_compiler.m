/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_compiler.h"
#include "mtl_format.h"

#include <Metal/MTL4ComputePipeline.h>
#include <Metal/MTL4LibraryDescriptor.h>
#include <Metal/MTL4LibraryFunctionDescriptor.h>
#include <Metal/MTL4RenderPipeline.h>
#include <Metal/MTLDevice.h>
#include <CommonCrypto/CommonDigest.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
   KK_LIBRARY_CACHE_ENTRIES = 128,
   KK_LIBRARY_CACHE_SOURCE_BYTES = 8 * 1024 * 1024,
   KK_LIBRARY_CACHE_PENDING = 16,
};

static _Atomic uint64_t kk_library_scope;

typedef struct {
   id<MTLLibrary> library;
   bool complete;
   size_t source_bytes;
   uint64_t serial;
} KKLibraryEntry;

typedef struct {
   NSCondition *condition;
   NSMutableDictionary *entries;
   size_t source_bytes;
   unsigned pending;
   uint64_t scope, serial, hits, waits, misses, bypasses;
} KKLibraryCache;

static char kk_library_cache_key;

static bool
kk_library_option(const char *name)
{
   const char *value = getenv(name);
   return value && strcmp(value, "1") == 0;
}

static KKLibraryCache *
kk_library_cache(id<MTL4Compiler> compiler)
{
   @synchronized (compiler) {
      NSDictionary *storage = objc_getAssociatedObject(compiler, &kk_library_cache_key);
      if (!storage) {
         NSMutableData *state = [NSMutableData dataWithLength:sizeof(KKLibraryCache)];
         KKLibraryCache *cache = state.mutableBytes;
         cache->scope = atomic_fetch_add(&kk_library_scope, 1) + 1;
         cache->condition = [[NSCondition new] autorelease];
         cache->entries = [NSMutableDictionary dictionary];
         storage = @{ @"state": state, @"condition": cache->condition,
                       @"entries": cache->entries };
         objc_setAssociatedObject(compiler, &kk_library_cache_key, storage,
                                  OBJC_ASSOCIATION_RETAIN_NONATOMIC);
      }
      return (KKLibraryCache *)[[storage objectForKey:@"state"] mutableBytes];
   }
}

static id<MTLLibrary>
kk_compile_library(id<MTL4Compiler> compiler, NSString *source,
                   enum mtl_language_version msl_version,
                   enum mtl_math_mode math_mode,
                   enum mtl_math_floating_point_functions math_fp_fns)
{
   MTLCompileOptions *opts = [[MTLCompileOptions new] autorelease];
   opts.languageVersion = (MTLLanguageVersion)msl_version;
   opts.mathMode = (MTLMathMode)math_mode;
   opts.mathFloatingPointFunctions = (MTLMathFloatingPointFunctions)math_fp_fns;

   MTL4LibraryDescriptor *desc = [[MTL4LibraryDescriptor new] autorelease];
   desc.source = source;
   desc.options = opts;

   NSError *error = nil;
   id<MTLLibrary> lib = [compiler newLibraryWithDescriptor:desc error:&error];
   if (lib == nil)
      fprintf(stderr, "Failed to create MTLLibrary: %s\n", [error.localizedDescription UTF8String]);
   return lib;
}

static id<MTLLibrary>
kk_cached_library(KKLibraryCache *cache, id<MTL4Compiler> compiler,
                  NSString *source, size_t bytes,
                  enum mtl_language_version msl_version,
                  enum mtl_math_mode math_mode,
                  enum mtl_math_floating_point_functions math_fp_fns,
                  const char **outcome)
{
   NSArray *key = @[source, @(msl_version), @(math_mode), @(math_fp_fns)];
   [cache->condition lock];
   NSMutableDictionary *entry_storage = [[cache->entries objectForKey:key] retain];
   KKLibraryEntry *entry = (KKLibraryEntry *)[[entry_storage objectForKey:@"state"] mutableBytes];
   if (entry) {
      if (!entry->complete) {
         cache->waits++;
         *outcome = "wait";
         while (!entry->complete)
            [cache->condition wait];
      } else {
         cache->hits++;
         *outcome = "hit";
      }
      entry->serial = ++cache->serial;
      id<MTLLibrary> lib = [entry->library retain];
      [cache->condition unlock];
      [entry_storage release];
      return lib;
   }

   if (bytes <= KK_LIBRARY_CACHE_SOURCE_BYTES &&
       cache->pending < KK_LIBRARY_CACHE_PENDING) {
      while (cache->entries.count >= KK_LIBRARY_CACHE_ENTRIES ||
             cache->source_bytes + bytes > KK_LIBRARY_CACHE_SOURCE_BYTES) {
         NSArray *old_key = nil;
         KKLibraryEntry *oldest = nil;
         for (NSArray *candidate in cache->entries) {
            NSDictionary *storage = [cache->entries objectForKey:candidate];
            KKLibraryEntry *value = (KKLibraryEntry *)[[storage objectForKey:@"state"] mutableBytes];
            if (value->complete && (!oldest || value->serial < oldest->serial)) {
               old_key = candidate;
               oldest = value;
            }
         }
         if (!oldest)
            break;
         cache->source_bytes -= oldest->source_bytes;
         [cache->entries removeObjectForKey:old_key];
      }
   }

   if (bytes > KK_LIBRARY_CACHE_SOURCE_BYTES ||
       cache->pending >= KK_LIBRARY_CACHE_PENDING ||
       cache->entries.count >= KK_LIBRARY_CACHE_ENTRIES ||
       cache->source_bytes + bytes > KK_LIBRARY_CACHE_SOURCE_BYTES) {
      cache->bypasses++;
      [cache->condition unlock];
      *outcome = "bypass";
      return kk_compile_library(compiler, source, msl_version, math_mode, math_fp_fns);
   }

   NSMutableData *state = [NSMutableData dataWithLength:sizeof(KKLibraryEntry)];
   entry_storage = [[NSMutableDictionary alloc] initWithObjectsAndKeys:state, @"state", nil];
   entry = state.mutableBytes;
   entry->source_bytes = bytes;
   [cache->entries setObject:entry_storage forKey:key];
   cache->source_bytes += bytes;
   cache->pending++;
   cache->misses++;
   [cache->condition unlock];

   *outcome = "compile";
   id<MTLLibrary> lib = kk_compile_library(compiler, source, msl_version,
                                          math_mode, math_fp_fns);
   [cache->condition lock];
   if (lib)
      [entry_storage setObject:lib forKey:@"library"];
   entry->library = lib;
   entry->serial = ++cache->serial;
   entry->complete = true;
   cache->pending--;
   if (!lib) {
      cache->source_bytes -= entry->source_bytes;
      [cache->entries removeObjectForKey:key];
   }
   [cache->condition broadcast];
   [cache->condition unlock];
   [entry_storage release];
   return lib;
}

static void
kk_library_trace(mtl_compiler *compiler, uint64_t scope, const char *source, uint64_t start_ns,
                  enum mtl_language_version msl_version,
                  enum mtl_math_mode math_mode,
                  enum mtl_math_floating_point_functions math_fp_fns,
                  const char *outcome, bool success)
{
   uint64_t end_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
   uint64_t tid = 0;
   pthread_threadid_np(NULL, &tid);
   unsigned char digest[CC_SHA256_DIGEST_LENGTH];
   CC_SHA256(source, (CC_LONG)strlen(source), digest);
   char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
   for (unsigned i = 0; i < sizeof(digest); i++)
      snprintf(hex + 2 * i, 3, "%02x", digest[i]);
   fprintf(stderr, "kk-library: pid=%d compiler=%p scope=%llu start_ns=%llu end_ns=%llu thread=%llu source=%s "
                   "msl=%u math=%u fp=%u outcome=%s success=%u\n",
           getpid(), compiler, (unsigned long long)scope,
           (unsigned long long)start_ns, (unsigned long long)end_ns,
           (unsigned long long)tid, hex, (unsigned)msl_version,
           (unsigned)math_mode, (unsigned)math_fp_fns, outcome, success);
}

/* Compiler */
mtl_compiler *
mtl_new_compiler(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;

      MTL4CompilerDescriptor *desc = [[MTL4CompilerDescriptor new] autorelease];

      NSError *error;
      id<MTL4Compiler> compiler = [dev newCompilerWithDescriptor:desc error:&error];

      if (compiler == nil) {
         fprintf(stderr, "Failed to create MTL4Compiler: %s\n", [error.localizedDescription UTF8String]);
      }

      return compiler;
   }
}

/* Library */
mtl_library *
mtl_new_library(mtl_compiler *compiler, const char *src,
                enum mtl_language_version msl_version,
                enum mtl_math_mode math_mode,
                enum mtl_math_floating_point_functions math_fp_fns)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      NSString *ns_src = [NSString stringWithCString:src encoding:NSASCIIStringEncoding];
      bool trace = kk_library_option("KK_MTL_LIBRARY_TRACE");
      bool enabled = kk_library_option("KK_MTL_LIBRARY_CACHE");
      uint64_t start_ns = trace ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
      const char *outcome = "uncached";
      KKLibraryCache *cache = comp && (trace || enabled) ? kk_library_cache(comp) : NULL;
      id<MTLLibrary> lib;
      if (ns_src && cache && enabled)
         lib = kk_cached_library(cache, comp, ns_src, strlen(src) + 1,
                                  msl_version, math_mode, math_fp_fns, &outcome);
      else
         lib = kk_compile_library(comp, ns_src, msl_version, math_mode, math_fp_fns);
      if (trace)
         kk_library_trace(compiler, cache ? cache->scope : 0, src, start_ns,
                           msl_version, math_mode, math_fp_fns,
                           outcome, lib != nil);
      return lib;
   }
}

mtl_function_descriptor *
mtl_new_library_function_descriptor(mtl_library *library, const char *entry_point)
{
   @autoreleasepool {
      id<MTLLibrary> lib = (id<MTLLibrary>)library;
      NSString *ns_entry_point = [NSString stringWithCString:entry_point encoding:NSASCIIStringEncoding];

      MTL4LibraryFunctionDescriptor *desc = [MTL4LibraryFunctionDescriptor new];
      desc.name = ns_entry_point;
      desc.library = lib;
      return desc;
   }
}

/* Compute pipeline */
mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state(mtl_compiler *compiler,
                               mtl_function_descriptor *function,
                               uint64_t max_total_threads_per_threadgroup)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;

      MTL4ComputePipelineDescriptor *desc = [[MTL4ComputePipelineDescriptor new] autorelease];
      desc.computeFunctionDescriptor = (MTL4FunctionDescriptor *)function;
      desc.maxTotalThreadsPerThreadgroup = max_total_threads_per_threadgroup;

      NSError *error;
      id<MTLComputePipelineState> pipeline = [comp newComputePipelineStateWithDescriptor:desc compilerTaskOptions:nil
                                                                                   error:&error];

      if (pipeline == nil) {
         fprintf(stderr, "Failed to create MTLComputePipelineState: %s\n", [error.localizedDescription UTF8String]);
      }

      return pipeline;
   }
}

/* Render pipeline descriptor */
mtl_render_pipeline_descriptor *
mtl_new_render_pipeline_descriptor()
{
   @autoreleasepool {
      return [MTL4RenderPipelineDescriptor new];
   }
}

void
mtl_render_pipeline_descriptor_set_vertex_shader(mtl_render_pipeline_descriptor *descriptor,
                                                 mtl_function_descriptor *shader)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.vertexFunctionDescriptor = (MTL4FunctionDescriptor *)shader;
   }
}

void
mtl_render_pipeline_descriptor_set_fragment_shader(mtl_render_pipeline_descriptor *descriptor,
                                                   mtl_function_descriptor *shader)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.fragmentFunctionDescriptor = (MTL4FunctionDescriptor *)shader;
   }
}

void
mtl_render_pipeline_descriptor_set_input_primitive_topology(mtl_render_pipeline_descriptor *descriptor,
                                                            enum mtl_primitive_topology_class class)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.inputPrimitiveTopology = (MTLPrimitiveTopologyClass)class;
   }
}

void
mtl_render_pipeline_descriptor_set_color_attachment_format(mtl_render_pipeline_descriptor *descriptor,
                                                           uint8_t index,
                                                           enum mtl_pixel_format format)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.colorAttachments[index].pixelFormat = (MTLPixelFormat)format;
   }
}

void
mtl_render_pipeline_descriptor_set_color_attachment_write_mask(mtl_render_pipeline_descriptor *descriptor,
                                                               uint8_t index, uint32_t mask)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      MTLColorWriteMask m = MTLColorWriteMaskNone;
      if (mask & 1)
         m |= MTLColorWriteMaskRed;
      if (mask & 2)
         m |= MTLColorWriteMaskGreen;
      if (mask & 4)
         m |= MTLColorWriteMaskBlue;
      if (mask & 8)
         m |= MTLColorWriteMaskAlpha;
      desc.colorAttachments[index].writeMask = m;
   }
}

void
mtl_render_pipeline_descriptor_set_color_attachment_blend(
   mtl_render_pipeline_descriptor *descriptor, uint8_t index,
   const struct mtl_color_blend_state *blend)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      MTL4RenderPipelineColorAttachmentDescriptor *att = desc.colorAttachments[index];
      att.blendingState = blend->enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
      att.rgbBlendOperation = (MTLBlendOperation)blend->rgb_operation;
      att.alphaBlendOperation = (MTLBlendOperation)blend->alpha_operation;
      att.sourceRGBBlendFactor = (MTLBlendFactor)blend->source_rgb;
      att.destinationRGBBlendFactor = (MTLBlendFactor)blend->destination_rgb;
      att.sourceAlphaBlendFactor = (MTLBlendFactor)blend->source_alpha;
      att.destinationAlphaBlendFactor = (MTLBlendFactor)blend->destination_alpha;
      mtl_render_pipeline_descriptor_set_color_attachment_write_mask(
         descriptor, index, blend->write_mask);
   }
}

void
mtl_render_pipeline_descriptor_set_multisample_alpha(
   mtl_render_pipeline_descriptor *descriptor, bool coverage, bool one)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.alphaToCoverageState = coverage ? MTL4AlphaToCoverageStateEnabled
                                          : MTL4AlphaToCoverageStateDisabled;
      desc.alphaToOneState = one ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
   }
}

void
mtl_render_pipeline_descriptor_set_raster_sample_count(mtl_render_pipeline_descriptor *descriptor,
                                                       uint32_t sample_count)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.rasterSampleCount = sample_count;
   }
}

void
mtl_render_pipeline_descriptor_set_rasterization_enabled(mtl_render_pipeline_descriptor *descriptor,
                                                         bool enabled)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.rasterizationEnabled = enabled;
   }
}

void
mtl_render_pipeline_descriptor_set_max_vertex_amplification_count(mtl_render_pipeline_descriptor *descriptor,
                                                                  uint32_t count)
{
   @autoreleasepool {
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;
      desc.maxVertexAmplificationCount = count;
   }
}

/* Render pipeline */
mtl_render_pipeline_state *
mtl_new_render_pipeline(mtl_compiler *compiler, mtl_render_pass_descriptor *descriptor)
{
   @autoreleasepool {
      id<MTL4Compiler> comp = (id<MTL4Compiler>)compiler;
      MTL4RenderPipelineDescriptor *desc = (MTL4RenderPipelineDescriptor *)descriptor;

      NSError *error = nil;
      id<MTLRenderPipelineState> pipeline = [comp newRenderPipelineStateWithDescriptor:desc compilerTaskOptions:nil
                                                                                 error:&error];

      if (pipeline == nil) {
         fprintf(stderr, "Failed to create MTLRenderPipelineState: %s\n", [error.localizedDescription UTF8String]);
      }

      return pipeline;
   }
}
