/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_FDM_H
#define KK_FDM_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

struct kk_cmd_buffer;
struct vk_cmd_queue;
void kk_fdm_execute_commands(struct kk_cmd_buffer *cmd, struct vk_cmd_queue *queue);

void kk_fdm_setup_rendering(struct kk_cmd_buffer *cmd,
                            const VkRenderingInfo *pRenderingInfo,
                            mtl_render_pass_descriptor *descriptor,
                            VkExtent2D screen);

void kk_fdm_carry_to_next_rendering(struct kk_cmd_buffer *cmd);
struct kk_image;
void kk_fdm_clear_image_mapping(struct kk_cmd_buffer *cmd, struct kk_image *image);

/* Gaze point per view, in [0, 1] across the render area, for gaze-centred
 * rate maps. Exported for the OpenXR layer in the same process. */
PUBLIC void kk_fdm_set_gaze(uint32_t view, float x, float y, bool valid);

/* Takes the gaze used by rate maps recorded until the next latch. Called at
 * queue submission, at most once every 10 ms, so that the passes of a frame
 * share one map. */
void kk_fdm_latch_gaze(void);

void kk_seq_log(struct kk_cmd_buffer *cmd, const char *fmt, ...);

struct vk_image;
void kk_fdm_note_use(const struct vk_image *image, const char *what);
struct vk_image_view;
void kk_fdm_note_view(const struct vk_image_view *view, const char *what);

#endif
