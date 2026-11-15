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

struct kk_fdm_params {
   struct kk_device *dev;
   struct kk_bo *bo;
   uint32_t refs;
   /* The rate map these parameters describe (retained), so a pass can still render an
    * image in the layout it was drawn in after the density map has moved on. */
   void *map;
   uint64_t serial;
};

void kk_fdm_cmd_release(struct kk_cmd_buffer *cmd);
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

/* Implicit foveation (KK_IMPLICIT_FDM=WIDTHxHEIGHT[,OUTER[,CORE,EDGE]]): multisampled
 * attachments of exactly that size are treated as subsampled images and their passes
 * get a driver-built rate map, for applications that do not use fragment density
 * maps. The target holds both eyes side by side. OUTER is the peripheral rate per
 * axis (default 0.3); CORE and EDGE are the full-rate and outer radii as fractions
 * of half an eye's width (defaults 0.35, 0.75). Needs MESA_KK_EXPERIMENTAL=fdm_rate. */
bool kk_implicit_fdm_enabled(void);
bool kk_implicit_fdm_image(const VkImageCreateInfo *info);
struct kk_image;
/* Whether a candidate is an eye target, once the view size is known. */
bool kk_implicit_fdm_match(struct kk_image *image);
struct kk_rendering_state;
/* KK_IMPLICIT_SINGLE: foveate a pass's attachments together or not at all. */
void kk_implicit_match_pass(struct kk_cmd_buffer *cmd, struct kk_rendering_state *render);
/* Set while the driver blits a foveated image into itself (mip generation): the copy
 * keeps the warped layout, so it is neither translated nor rate mapped. */
bool kk_implicit_raw_active(void);
bool kk_implicit_storage_enabled(void);
/* KK_IMPLICIT_COMPACT=1 (with KK_IMPLICIT_STORAGE): a compute dispatch that covers a
 * foveated image and only writes foveated storage images runs one invocation per
 * physical texel; each finds the screen texel it stands for (kk_shader.c). */
bool kk_implicit_compact_enabled(void);
/* KK_IMPLICIT_COMPACT_FILE: re-read the run-time switches (called at presents). */
void kk_fdm_runtime_poll(void);
/* A present happened: implicit foveation may switch to a new profile at the next pass. */
void kk_implicit_fdm_present(void);
/* The OpenXR layer's frame boundary (the eye image released), for applications that do
 * not present. */
PUBLIC void kk_implicit_fdm_frame_boundary(void);
struct kk_shader;
/* Rate parameters to remap the dispatch with, shrinking the group counts to the
 * physical size, or 0 to run it as recorded. */
uint64_t kk_implicit_compact_dispatch(struct kk_cmd_buffer *cmd, const struct kk_shader *cs,
                                      uint32_t *group_x, uint32_t *group_y, uint32_t group_z);
void kk_implicit_set_raw(bool raw);
/* KK_IMPLICIT_FDM=auto: the view size and profile come from the OpenXR layer. */
PUBLIC void kk_implicit_fdm_set_profile(uint32_t view_width, uint32_t view_height, uint32_t count,
                                        const float *horizontal0, const float *horizontal1,
                                        const float *vertical);
struct kk_image_view;
/* The hidden resolve view for a foveated multisampled image (created on first use),
 * or NULL when `dst` can take the resolve directly. */
struct kk_image_view *kk_implicit_resolve_view(struct kk_device *dev, struct kk_image *src,
                                               const struct kk_image_view *dst);
void kk_implicit_image_finish(struct kk_device *dev, struct kk_image *image);
struct kk_cmd_buffer;
/* A raw copy (transfer, not a shader) is about to read `src`. If it is foveated, record
 * a shader blit of it into an ordinary image of the same format and return that image
 * for the copy to read instead; otherwise return VK_NULL_HANDLE. */
VkImage kk_implicit_unwarp_for_copy(struct kk_cmd_buffer *cmd, struct kk_image *src,
                                    VkImageLayout src_layout, const struct kk_image *dst);

/* A raw copy from one foveated image into another (Talos keeps a copy of an eye): the
 * destination takes the source's map with its texels. */
void kk_implicit_copy_map(struct kk_cmd_buffer *cmd, struct kk_image *src, struct kk_image *dst);

struct vk_image;
void kk_implicit_trace(const char *what, const struct vk_image *a, const struct vk_image *b);

struct vk_image;
void kk_fdm_note_use(const struct vk_image *image, const char *what);
struct vk_image_view;
void kk_fdm_note_view(const struct vk_image_view *view, const char *what);

struct kk_rendering_state;
/* KK_FAKE_VRS: note a pass's shading-rate requests (attachment at begin, pipeline or
 * dynamic rates at draws) and log every pass shape with them; KK_VRS_CAPTURE_DIR saves
 * the rate images. */
void kk_vrs_note_draw(struct kk_cmd_buffer *cmd);
void kk_vrs_end(struct kk_cmd_buffer *cmd);
void kk_vrs_frame(void);

#endif
