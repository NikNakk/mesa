/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_queue.h"
#include "mtl_alloc_check.h"

#include <Metal/MTLDevice.h>
#include <Metal/MTLCommandQueue.h>
#include <objc/message.h>
#include "util/u_debug.h"
#include "util/log.h"

/* MTL4CommitOptions */
mtl_commit_options *
mtl_new_commit_options(void)
{
   @autoreleasepool {
      return [MTL4CommitOptions new];
   }
}

void
mtl_commit_options_add_feedback_handler(mtl_commit_options *options,
                                        mtl_feedback_handler_callback callback,
                                        void *data)
{
   @autoreleasepool {
      MTL4CommitOptions* opt = (MTL4CommitOptions *)options;
      [opt addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
         @autoreleasepool {
            NSError *error = feedback.error;
            struct mtl_feedback_data feedback_data = (struct mtl_feedback_data){
               .user_data = data,
               .error_message =
                  error ? error.localizedDescription.UTF8String : NULL,
               .gpu_start = feedback.GPUStartTime,
               .gpu_end = feedback.GPUEndTime,
               .error = error
                           ? (enum mtl_command_queue_error)error.code
                           : MTL_COMMAND_QUEUE_ERROR_NONE,
            };
            callback(&feedback_data);
         }
      }];
   }
}

/* MTL4CommandQueue */
mtl_command_queue *
mtl_new_command_queue(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      id<MTL4CommandQueue> queue = [dev newMTL4CommandQueue];
      long priority = debug_get_num_option("KK_GPU_QUEUE_PRIORITY", -1);
      if (priority >= 0 && priority <= 2) {
         SEL set_fg = sel_registerName("setGPUPriority:");
         SEL set_bg = sel_registerName("setBackgroundGPUPriority:");
         SEL get_fg = sel_registerName("getGPUPriority");
         SEL get_bg = sel_registerName("getBackgroundGPUPriority");
         if ([queue respondsToSelector:set_fg] &&
             [queue respondsToSelector:set_bg] &&
             [queue respondsToSelector:get_fg] &&
             [queue respondsToSelector:get_bg]) {
            ((void (*)(id, SEL, long))objc_msgSend)(queue, set_fg, priority);
            ((void (*)(id, SEL, long))objc_msgSend)(queue, set_bg, priority);
            mesa_logi("kk: GPU queue priority requested=%ld foreground=%ld "
                      "background=%ld", priority,
                      ((long (*)(id, SEL))objc_msgSend)(queue, get_fg),
                      ((long (*)(id, SEL))objc_msgSend)(queue, get_bg));
         } else {
            mesa_logw("kk: GPU queue priority selectors unavailable");
         }
      } else if (priority != -1) {
         mesa_logw("kk: invalid KK_GPU_QUEUE_PRIORITY=%ld; expected -1..2",
                   priority);
      }
      return queue;
   }
}

void
mtl_command_queue_add_residency_set(mtl_command_queue *cmd_queue,
                                    mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue addResidencySet:s];
   }
}

void
mtl_command_queue_remove_residency_set(mtl_command_queue *cmd_queue,
                                       mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue removeResidencySet:s];
   }
}

void
mtl_signal_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q signalEvent:e value:value];
   }
}

void
mtl_wait_for_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q waitForEvent:e value:value];
   }
}

void
mtl_command_queue_commit(mtl_command_queue *queue,
                         mtl_command_buffer **command_buffers, uint32_t count,
                         mtl_commit_options *options)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTL4CommandBuffer> *cmds = (id<MTL4CommandBuffer> *)command_buffers;
      MTL4CommitOptions *opt = (MTL4CommitOptions *)options;
      mtl_alloc_check_commit(options, (void *const *)command_buffers, count);
      [q commit:cmds count:count options:opt];
   }
}

void
mtl_command_queue_wait_for_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q waitForDrawable:d];
   }
}

void
mtl_command_queue_signal_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q signalDrawable:d];
   }
}
