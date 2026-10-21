/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_FDM_H
#define KK_FDM_H 1

#include "kk_private.h"

#include "kosmickrisp/bridge/mtl_types.h"

struct kk_cmd_buffer;

void kk_fdm_setup_rendering(struct kk_cmd_buffer *cmd,
                            const VkRenderingInfo *pRenderingInfo,
                            mtl_render_pass_descriptor *descriptor,
                            VkExtent2D screen);

void kk_fdm_carry_to_next_rendering(struct kk_cmd_buffer *cmd);

#endif
