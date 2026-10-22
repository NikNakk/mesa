/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "mtl_tile.h"

#include <Metal/MTL4CommandBuffer.h>
#include <Metal/MTL4Compiler.h>
#include <Metal/MTL4RenderCommandEncoder.h>
#include <Metal/MTL4RenderPass.h>
#include <Metal/MTL4TileRenderPipeline.h>

#include <stdio.h>

mtl_render_pipeline_state *
mtl_new_tile_render_pipeline(mtl_compiler *compiler,
                             mtl_function_descriptor *function,
                             uint32_t sample_count,
                             const enum mtl_pixel_format *formats,
                             uint32_t format_count)
{
   @autoreleasepool {
      MTL4TileRenderPipelineDescriptor *desc = [MTL4TileRenderPipelineDescriptor new];
      desc.tileFunctionDescriptor = (MTL4FunctionDescriptor *)function;
      desc.rasterSampleCount = sample_count;
      desc.threadgroupSizeMatchesTileSize = YES;
      for (uint32_t i = 0; i < format_count; i++)
         desc.colorAttachments[i].pixelFormat = (MTLPixelFormat)formats[i];
      NSError *error = nil;
      id<MTLRenderPipelineState> pipeline =
         [(id<MTL4Compiler>)compiler newRenderPipelineStateWithDescriptor:desc
                                                      compilerTaskOptions:nil
                                                                    error:&error];
      if (pipeline == nil)
         fprintf(stderr, "Failed to create tile MTLRenderPipelineState: %s\n",
                 [error.localizedDescription UTF8String]);
      [desc release];
      return pipeline;
   }
}

void
mtl_render_pass_descriptor_set_tile_size(mtl_render_pass_descriptor *descriptor,
                                         uint32_t width, uint32_t height)
{
   @autoreleasepool {
      MTL4RenderPassDescriptor *desc = (MTL4RenderPassDescriptor *)descriptor;
      desc.tileWidth = width;
      desc.tileHeight = height;
   }
}

void
mtl_render_pass_descriptor_set_visibility_accumulate(
   mtl_render_pass_descriptor *descriptor, bool accumulate)
{
   @autoreleasepool {
      MTL4RenderPassDescriptor *desc = (MTL4RenderPassDescriptor *)descriptor;
      desc.visibilityResultType =
         accumulate ? MTLVisibilityResultTypeAccumulate : MTLVisibilityResultTypeReset;
   }
}

mtl_render_encoder *
mtl_new_render_command_encoder_with_options(mtl_command_buffer *command_buffer,
                                            mtl_render_pass_descriptor *descriptor,
                                            uint32_t options)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd = (id<MTL4CommandBuffer>)command_buffer;
      MTL4RenderPassDescriptor *desc = (MTL4RenderPassDescriptor *)descriptor;
      MTL4RenderEncoderOptions o = MTL4RenderEncoderOptionNone;
      if (options & MTL_RENDER_ENCODER_SUSPENDING)
         o |= MTL4RenderEncoderOptionSuspending;
      if (options & MTL_RENDER_ENCODER_RESUMING)
         o |= MTL4RenderEncoderOptionResuming;
      return [[cmd renderCommandEncoderWithDescriptor:desc options:o] retain];
   }
}

void
mtl_dispatch_threads_per_tile(mtl_render_encoder *encoder, uint32_t width,
                              uint32_t height)
{
   @autoreleasepool {
      [(id<MTL4RenderCommandEncoder>)encoder
         dispatchThreadsPerTile:MTLSizeMake(width, height, 1)];
   }
}
