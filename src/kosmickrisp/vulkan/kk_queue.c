/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_queue.h"
#include "kk_debug.h"
#include "util/log.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"
#include "kk_sync.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_cmd_queue.h"

struct kk_commit_data {
   struct kk_queue *queue;
   struct kk_cmd_buffer *cmd;
   uint32_t refs;
};

static void
kk_queue_wait_for_commits(struct kk_queue *queue)
{
   mtx_lock(&queue->mutex);
   while (queue->commits_in_flight > 0u)
      cnd_wait(&queue->cond, &queue->mutex);
   mtx_unlock(&queue->mutex);
}

static void
kk_queue_release_cmd_buffer_locked(struct kk_queue *queue,
                                   struct kk_cmd_buffer *cmd)
{
   kk_cmd_buffer_ops.reset(&cmd->vk, 0u);
   util_dynarray_append(&queue->free_cmd_buffers, &cmd->vk);
}

static void
check_device_lost(struct kk_device *dev, struct mtl_feedback_data *data)
{
   if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE) {
      mesa_loge("kk: command queue error %s: %s",
                mtl_command_queue_error_to_string(data->error),
                data->error_message ? data->error_message : "(no message)");
      vk_device_set_lost(
         &dev->vk, "Command queue error: %s, with message \"%s\"",
         mtl_command_queue_error_to_string(data->error), data->error_message);
   }
}

static void
commit_callback(struct mtl_feedback_data *data)
{
   check_device_lost((struct kk_device *)data->user_data, data);
}

static void
rerecord_commit_callback(struct mtl_feedback_data *data)
{
   struct kk_commit_data *commit = (struct kk_commit_data *)data->user_data;
   struct kk_queue *queue = commit->queue;
   struct kk_device *dev = kk_queue_device(queue);

   check_device_lost(dev, data);

   if (p_atomic_dec_return(&commit->refs) > 0)
      return;

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   kk_queue_release_cmd_buffer_locked(queue, commit->cmd);

   assert(queue->commits_in_flight > 0u);
   if (--queue->commits_in_flight == 0u) {
      kk_ResetCommandPool(kk_device_to_handle(dev),
                          kk_cmd_pool_to_handle(queue->cmd_pool), 0u);
      cnd_broadcast(&queue->cond);
   }

   mtx_unlock(&queue->mutex);

   vk_free(&dev->vk.alloc, commit);
}

static bool
kk_cmd_buffer_has_work(struct kk_cmd_buffer *cmd)
{
   return cmd->metal.cmd_buf;
}

static void
kk_queue_commit(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                mtl_feedback_handler_callback callback, void *user_data)
{
   assert(kk_cmd_buffer_has_work(cmd));

   mtl_commit_options *options = mtl_new_commit_options();
   mtl_commit_options_add_feedback_handler(options, callback, user_data);

   mtl_command_queue_commit(queue->mtl_handle, &cmd->metal.cmd_buf, 1u,
                            options);
   mtl_release(options);
}

static VkResult
rerecord_cmd_buffer(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                    bool no_stitch, struct kk_commit_data **out);

static VkResult
rerecord_and_commit_cmd_buffer(struct kk_queue *queue,
                               struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct vk_command_buffer *vk_cmd = NULL;
   struct kk_cmd_buffer *rerecord;
   VkResult result = VK_SUCCESS;

   /* Completion callbacks are called from multiple threads, so we need to
    * ensure the access to queue resources is safe. */
   mtx_lock(&queue->mutex);

   if (util_dynarray_num_elements(&queue->free_cmd_buffers,
                                  struct vk_command_buffer *) > 0u) {
      vk_cmd = util_dynarray_pop(&queue->free_cmd_buffers,
                                 struct vk_command_buffer *);
   } else {
      result = kk_cmd_buffer_ops.create(
         &queue->cmd_pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &vk_cmd);
      if (result != VK_SUCCESS)
         goto unlock;
   }

   rerecord = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   VkCommandBuffer rerecord_handle = kk_cmd_buffer_to_handle(rerecord);
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   result = kk_BeginCommandBuffer(rerecord_handle, &begin_info);
   if (result != VK_SUCCESS)
      goto release;

   vk_cmd_queue_execute(&cmd->vk.cmd_queue, rerecord_handle,
                        &dev->vk.dispatch_table);

   result = kk_EndCommandBuffer(rerecord_handle);
   if (result != VK_SUCCESS)
      goto release;

   if (!kk_cmd_buffer_has_work(rerecord))
      goto release;

   struct kk_commit_data *commit = vk_alloc(&dev->vk.alloc, sizeof(*commit), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (commit == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto release;
   }
   commit->queue = queue;
   commit->cmd = rerecord;
   commit->refs = 1;

   /* Need to ensure the new buffers allocated at record are resident. */
   kk_device_make_resources_resident(dev);

   queue->commits_in_flight++;
   kk_queue_commit(queue, rerecord, rerecord_commit_callback, commit);

   mtx_unlock(&queue->mutex);
   return VK_SUCCESS;

release:
   kk_queue_release_cmd_buffer_locked(queue, rerecord);
unlock:
   mtx_unlock(&queue->mutex);
   return result;
}

static VkResult
rerecord_cmd_buffer(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                    bool no_stitch, struct kk_commit_data **out)
{
   struct kk_device *dev = kk_queue_device(queue);
   struct vk_command_buffer *vk_cmd = NULL;
   struct kk_cmd_buffer *rerecord;
   VkResult result = VK_SUCCESS;

   *out = NULL;
   mtx_lock(&queue->mutex);

   if (util_dynarray_num_elements(&queue->free_cmd_buffers,
                                  struct vk_command_buffer *) > 0u) {
      vk_cmd = util_dynarray_pop(&queue->free_cmd_buffers,
                                 struct vk_command_buffer *);
   } else {
      result = kk_cmd_buffer_ops.create(
         &queue->cmd_pool->vk, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &vk_cmd);
      if (result != VK_SUCCESS)
         goto unlock;
   }

   rerecord = container_of(vk_cmd, struct kk_cmd_buffer, vk);
   VkCommandBuffer rerecord_handle = kk_cmd_buffer_to_handle(rerecord);
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   rerecord->no_stitch = no_stitch;
   result = kk_BeginCommandBuffer(rerecord_handle, &begin_info);
   if (result != VK_SUCCESS)
      goto release;

   vk_cmd_queue_execute(&cmd->vk.cmd_queue, rerecord_handle,
                        &dev->vk.dispatch_table);

   result = kk_EndCommandBuffer(rerecord_handle);
   if (result != VK_SUCCESS)
      goto release;

   if (!kk_cmd_buffer_has_work(rerecord))
      goto release;

   struct kk_commit_data *commit = vk_alloc(&dev->vk.alloc, sizeof(*commit), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (commit == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto release;
   }
   commit->queue = queue;
   commit->cmd = rerecord;
   commit->refs = 1;
   kk_device_make_resources_resident(dev);
   queue->commits_in_flight++;
   *out = commit;
   mtx_unlock(&queue->mutex);
   return VK_SUCCESS;

release:
   kk_queue_release_cmd_buffer_locked(queue, rerecord);
unlock:
   mtx_unlock(&queue->mutex);
   return result;
}

struct kk_commit_group {
   mtl_command_buffer *bufs[32];
   struct kk_commit_data *commits[32];
   mtl_command_allocator *held[32];
   uint32_t count;
   uint32_t held_count;
};

struct kk_held_allocators {
   struct kk_device *dev;
   uint32_t count;
   mtl_command_allocator *allocators[32];
};

static void
kk_held_allocators_callback(struct mtl_feedback_data *data)
{
   struct kk_held_allocators *held = data->user_data;
   simple_mtx_lock(&held->dev->stitch_allocators.mutex);
   for (uint32_t i = 0; i < held->count; i++)
      util_dynarray_append(&held->dev->stitch_allocators.free,
                           held->allocators[i]);
   simple_mtx_unlock(&held->dev->stitch_allocators.mutex);
   free(held);
}

static void
kk_commit_group_flush(struct kk_queue *queue, struct kk_commit_group *group)
{
   if (group->count == 0)
      return;
   struct kk_device *dev = kk_queue_device(queue);
   mtl_commit_options *options = mtl_new_commit_options();
   mtl_commit_options_add_feedback_handler(options, commit_callback, dev);
   for (uint32_t i = 0; i < group->count; i++) {
      if (group->commits[i])
         mtl_commit_options_add_feedback_handler(options, rerecord_commit_callback,
                                                 group->commits[i]);
   }
   if (group->held_count) {
      struct kk_held_allocators *held = calloc(1, sizeof(*held));
      if (held) {
         held->dev = dev;
         held->count = group->held_count;
         memcpy(held->allocators, group->held,
                group->held_count * sizeof(group->held[0]));
         mtl_commit_options_add_feedback_handler(
            options, kk_held_allocators_callback, held);
      }
   }
   mtl_command_queue_commit(queue->mtl_handle, group->bufs, group->count,
                            options);
   mtl_release(options);
   group->count = 0;
   group->held_count = 0;
}

static void
kk_commit_group_add(struct kk_queue *queue, struct kk_commit_group *group,
                    mtl_command_buffer *buf, struct kk_commit_data *commit,
                    struct kk_cmd_buffer *owner)
{
   if (group->count == ARRAY_SIZE(group->bufs) ||
       (owner && owner->metal.held_allocator &&
        group->held_count == ARRAY_SIZE(group->held)))
      kk_commit_group_flush(queue, group);
   group->bufs[group->count] = buf;
   group->commits[group->count] = commit;
   group->count++;
   if (owner && owner->metal.held_allocator) {
      group->held[group->held_count++] = owner->metal.held_allocator;
      owner->metal.held_allocator = NULL;
   }
}

static bool
kk_cmd_buffer_is_stitched(const struct kk_cmd_buffer *cmd)
{
   return cmd->state.gfx.stitch.starts_resumed ||
          cmd->state.gfx.stitch.ends_suspended;
}

static VkResult
kk_submit_unstitched(struct kk_queue *queue, struct kk_cmd_buffer *cmd,
                     bool rerecord)
{
   if (cmd->drawable)
      mtl_command_queue_wait_for_drawable(queue->mtl_handle, cmd->drawable);

   struct kk_cmd_buffer *committed = cmd;
   struct kk_commit_data *commit = NULL;
   if (cmd->submitted || rerecord) {
      VkResult result = rerecord_cmd_buffer(queue, cmd, rerecord, &commit);
      if (result != VK_SUCCESS)
         return result;
      committed = commit ? commit->cmd : NULL;
   } else if (!kk_cmd_buffer_has_work(cmd)) {
      committed = NULL;
   }
   cmd->submitted = true;

   if (committed) {
      struct kk_commit_group group = {0};
      kk_commit_group_add(queue, &group, committed->metal.cmd_buf, commit,
                          committed);
      kk_commit_group_flush(queue, &group);
   }

   if (cmd->drawable) {
      mtl_command_queue_signal_drawable(queue->mtl_handle, cmd->drawable);
      mtl_release(cmd->drawable);
      cmd->drawable = NULL;
   }
   return VK_SUCCESS;
}

static bool
kk_chain_can_stitch(struct kk_cmd_buffer **chain, uint32_t count)
{
   if (count < 2 || count > 32)
      return false;
   for (uint32_t i = 0; i < count; i++) {
      const struct kk_cmd_buffer *cmd = chain[i];
      bool first = i == 0, last = i == count - 1;
      if (cmd->state.gfx.stitch.violation || cmd->drawable)
         return false;
      if (cmd->state.gfx.stitch.starts_resumed == first)
         return false;
      if (cmd->state.gfx.stitch.ends_suspended == last)
         return false;
      if (cmd->metal.head && !last)
         return false;
   }
   return true;
}

struct kk_pending {
   struct kk_cmd_buffer *cmds[48];
   bool in_chain[48];
   uint32_t count;
   uint32_t chain_count;
   bool broken;
};

static VkResult
kk_submit_pending(struct kk_queue *queue, struct kk_pending *pending)
{
   struct kk_cmd_buffer *chain[48];
   uint32_t n = 0;
   for (uint32_t i = 0; i < pending->count; i++) {
      if (pending->in_chain[i])
         chain[n++] = pending->cmds[i];
   }

   VkResult result = VK_SUCCESS;
   if (pending->broken || !kk_chain_can_stitch(chain, n)) {
      static uint32_t logged;
      if (logged++ < 4)
         mesa_logw("kk: re-recording %u command buffers without render pass "
                   "stitching",
                   n);
      kk_encoder_stat("stitch fallback", false);
      for (uint32_t i = 0; i < pending->count && result == VK_SUCCESS; i++)
         result = kk_submit_unstitched(queue, pending->cmds[i],
                                       pending->in_chain[i]);
      goto done;
   }

   kk_encoder_stat("stitched chain", false);
   struct kk_commit_group group = {0};
   for (uint32_t i = 0; i < n; i++) {
      struct kk_cmd_buffer *cmd = chain[i];
      struct kk_cmd_buffer *committed = cmd;
      struct kk_commit_data *commit = NULL;
      if (cmd->submitted) {
         result = rerecord_cmd_buffer(queue, cmd, false, &commit);
         if (result != VK_SUCCESS)
            goto done;
         committed = commit ? commit->cmd : NULL;
      }
      cmd->submitted = true;
      if (!committed)
         continue;

      if (!committed->metal.head) {
         kk_commit_group_add(queue, &group, committed->metal.cmd_buf, commit,
                             committed);
         continue;
      }

      bool tail = committed->metal.tail_used || commit;
      if (commit)
         commit->refs++;
      kk_commit_group_add(queue, &group, committed->metal.head, commit,
                          tail ? NULL : committed);
      kk_commit_group_flush(queue, &group);
      if (tail)
         kk_commit_group_add(queue, &group, committed->metal.cmd_buf, commit,
                             committed);
   }
   kk_commit_group_flush(queue, &group);

   for (uint32_t i = 0; i < pending->count && result == VK_SUCCESS; i++) {
      if (!pending->in_chain[i])
         result = kk_submit_unstitched(queue, pending->cmds[i], false);
   }

done:
   pending->count = 0;
   pending->chain_count = 0;
   pending->broken = false;
   return result;
}

static VkResult
kk_queue_submit_stitched(struct kk_queue *queue, struct vk_queue_submit *submit)
{
   struct kk_pending pending = {0};
   VkResult result;

   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      struct kk_cmd_buffer *cmd =
         container_of(submit->command_buffers[i], struct kk_cmd_buffer, vk);
      bool stitched = kk_cmd_buffer_is_stitched(cmd);
      bool resumes = cmd->state.gfx.stitch.starts_resumed;

      if (pending.chain_count && !resumes && !stitched && !cmd->drawable &&
          pending.count < ARRAY_SIZE(pending.cmds)) {
         pending.cmds[pending.count] = cmd;
         pending.in_chain[pending.count++] = false;
         continue;
      }

      if (pending.chain_count && !resumes) {
         pending.broken = true;
         result = kk_submit_pending(queue, &pending);
         if (result != VK_SUCCESS)
            return result;
      }

      if (!pending.chain_count && !stitched) {
         result = kk_submit_unstitched(queue, cmd, false);
         if (result != VK_SUCCESS)
            return result;
         continue;
      }

      if (pending.count == ARRAY_SIZE(pending.cmds)) {
         pending.broken = true;
         result = kk_submit_pending(queue, &pending);
         if (result != VK_SUCCESS)
            return result;
         result = kk_submit_unstitched(queue, cmd, true);
         if (result != VK_SUCCESS)
            return result;
         continue;
      }

      pending.cmds[pending.count] = cmd;
      pending.in_chain[pending.count++] = true;
      pending.chain_count++;

      if (!cmd->state.gfx.stitch.ends_suspended) {
         result = kk_submit_pending(queue, &pending);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   if (pending.count) {
      pending.broken = true;
      return kk_submit_pending(queue, &pending);
   }
   return VK_SUCCESS;
}

static VkResult
kk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct kk_queue *queue = container_of(vk_queue, struct kk_queue, vk);
   struct kk_device *dev = kk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   for (struct vk_sync_wait *wait = submit->waits,
                            *end = submit->waits + submit->wait_count;
        wait != end; ++wait) {
      struct kk_sync_timeline *sync =
         container_of(wait->sync, struct kk_sync_timeline, base);
      mtl_wait_for_event(queue->mtl_handle, sync->mtl_handle, wait->wait_value);
   }

   /* Ensure any changes to residency are propagated before we submit any
    * work. All resources should have been allocated before submission.
    * Otherwise, users are playing with fire. */
   kk_device_make_resources_resident(dev);

   if (KK_EXPERIMENTAL(STITCH_PASSES)) {
      VkResult result = kk_queue_submit_stitched(queue, submit);
      if (result != VK_SUCCESS)
         return result;
      goto signal;
   }

   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      struct kk_cmd_buffer *cmd_buffer =
         container_of(submit->command_buffers[i], struct kk_cmd_buffer, vk);

      if (cmd_buffer->drawable) {
         mtl_command_queue_wait_for_drawable(queue->mtl_handle,
                                             cmd_buffer->drawable);
      }

      /* Metal's command buffers are one time use, re-record multiple
       * submissions. */
      if (cmd_buffer->submitted) {
         VkResult result = rerecord_and_commit_cmd_buffer(queue, cmd_buffer);
         if (result != VK_SUCCESS)
            return result;
      } else if (kk_cmd_buffer_has_work(cmd_buffer)) {
         kk_queue_commit(queue, cmd_buffer, commit_callback, dev);
      }

      cmd_buffer->submitted = true;

      if (cmd_buffer->drawable) {
         mtl_command_queue_signal_drawable(queue->mtl_handle,
                                           cmd_buffer->drawable);
         mtl_release(cmd_buffer->drawable);
         cmd_buffer->drawable = NULL;
      }
   }

signal:
   for (uint32_t i = 0u; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];
      struct kk_sync_timeline *sync =
         container_of(signal->sync, struct kk_sync_timeline, base);
      mtl_signal_event(queue->mtl_handle, sync->mtl_handle,
                       signal->signal_value);
   }

   return VK_SUCCESS;
}

VkResult
kk_queue_init(struct kk_device *dev, struct kk_queue *queue,
              const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   const VkDeviceQueueGlobalPriorityCreateInfo *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);
   const VkQueueGlobalPriority global_priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;

   /* From the Vulkan 1.3.295 spec:
    *
    *    "If the globalPriorityQuery feature is enabled and the requested
    *    global priority is not reported via
    *    VkQueueFamilyGlobalPriorityPropertiesKHR, the driver implementation
    *    must fail the queue creation. In this scenario,
    *    VK_ERROR_INITIALIZATION_FAILED is returned."
    */
   if (dev->vk.enabled_features.globalPriorityQuery &&
       global_priority != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_INITIALIZATION_FAILED;

   if (global_priority > VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_NOT_PERMITTED;

   VkResult result;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = pCreateInfo->queueFamilyIndex,
   };
   VkCommandPool pool_handle;
   result = kk_CreateCommandPool(kk_device_to_handle(dev), &pool_info, NULL,
                                 &pool_handle);
   if (result != VK_SUCCESS) {
      vk_queue_finish(&queue->vk);
      return result;
   }
   queue->cmd_pool = kk_cmd_pool_from_handle(pool_handle);

   queue->free_cmd_buffers = UTIL_DYNARRAY_INIT;
   queue->commits_in_flight = 0u;
   mtx_init(&queue->mutex, mtx_plain);
   cnd_init(&queue->cond);

   queue->mtl_handle = mtl_new_command_queue(dev->mtl_handle);
   mtl_command_queue_add_residency_set(queue->mtl_handle,
                                       dev->residency_set.handle);

   queue->vk.driver_submit = kk_queue_submit;

   return VK_SUCCESS;
}

void
kk_queue_finish(struct kk_device *dev, struct kk_queue *queue)
{
   /* There may be in-flight command buffers, wait for them to finish. */
   kk_queue_wait_for_commits(queue);

   mtl_command_queue_remove_residency_set(queue->mtl_handle,
                                          dev->residency_set.handle);
   mtl_release(queue->mtl_handle);

   kk_DestroyCommandPool(kk_device_to_handle(dev),
                         kk_cmd_pool_to_handle(queue->cmd_pool), NULL);
   util_dynarray_fini(&queue->free_cmd_buffers);
   cnd_destroy(&queue->cond);
   mtx_destroy(&queue->mutex);

   vk_queue_finish(&queue->vk);
}
