/*
 * Copyright 2026 Nick Kennedy
 * SPDX-License-Identifier: MIT
 */

#include "mtl_alloc_check.h"

#include <Metal/MTL4CommandQueue.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util/log.h"
#include "util/u_debug.h"

#define ALLOC_CHECK_SLOTS 8192
#define ALLOC_CHECK_LOG_LIMIT 200

struct alloc_entry {
   const void *allocator;
   uint64_t encoding_thread;
   const void *encoding_cmd;
   int64_t in_flight;
   uint64_t begins;
};

struct cmd_entry {
   const void *cmd;
   const void *allocator;
};

static pthread_mutex_t check_lock = PTHREAD_MUTEX_INITIALIZER;
static struct alloc_entry allocs[ALLOC_CHECK_SLOTS];
static struct cmd_entry cmds[ALLOC_CHECK_SLOTS];
static uint64_t n_begins, n_commits, n_completions;
static uint64_t n_concurrent, n_nested, n_reuse_in_flight, n_unmatched_end;
static uint64_t n_commit_while_encoding, n_underflow, n_reset_in_flight;
static uint64_t n_untracked;
static unsigned n_logged;
static double last_summary;

static void exit_summary(void);

static int check_enabled;
static pthread_once_t check_once = PTHREAD_ONCE_INIT;

static void
check_init(void)
{
   check_enabled = debug_get_bool_option("KK_ALLOCATOR_CHECK", false);
   if (check_enabled)
      atexit(exit_summary);
}

bool
mtl_alloc_check_enabled(void)
{
   pthread_once(&check_once, check_init);
   return check_enabled;
}

static uint64_t
current_thread(void)
{
   uint64_t tid = 0;
   pthread_threadid_np(NULL, &tid);
   return tid;
}

static double
now_seconds(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec / 1e9;
}

static size_t
slot_of(const void *p)
{
   return (size_t)(((uintptr_t)p >> 4) * 2654435761u) % ALLOC_CHECK_SLOTS;
}

static struct alloc_entry *
alloc_entry(const void *allocator)
{
   size_t s = slot_of(allocator);
   for (size_t i = 0; i < ALLOC_CHECK_SLOTS; i++) {
      struct alloc_entry *e = &allocs[(s + i) % ALLOC_CHECK_SLOTS];
      if (e->allocator == allocator)
         return e;
      if (!e->allocator) {
         e->allocator = allocator;
         return e;
      }
   }
   n_untracked++;
   return NULL;
}

static struct cmd_entry *
cmd_entry(const void *cmd)
{
   size_t s = slot_of(cmd);
   for (size_t i = 0; i < ALLOC_CHECK_SLOTS; i++) {
      struct cmd_entry *e = &cmds[(s + i) % ALLOC_CHECK_SLOTS];
      if (e->cmd == cmd)
         return e;
      if (!e->cmd) {
         e->cmd = cmd;
         return e;
      }
   }
   n_untracked++;
   return NULL;
}

static void
report(const char *format, ...)
{
   if (n_logged >= ALLOC_CHECK_LOG_LIMIT)
      return;
   n_logged++;
   char line[512];
   va_list args;
   va_start(args, format);
   vsnprintf(line, sizeof(line), format, args);
   va_end(args);
   mesa_logw("kk-alloc-check: %s", line);
}

static void
print_summary(void)
{
   mesa_logi("kk-alloc-check: summary begins=%llu commits=%llu completions=%llu "
             "concurrent=%llu nested=%llu reuse_in_flight=%llu "
             "unmatched_end=%llu commit_while_encoding=%llu underflow=%llu "
             "reset_in_flight=%llu untracked=%llu",
             (unsigned long long)n_begins, (unsigned long long)n_commits,
             (unsigned long long)n_completions, (unsigned long long)n_concurrent,
             (unsigned long long)n_nested,
             (unsigned long long)n_reuse_in_flight,
             (unsigned long long)n_unmatched_end,
             (unsigned long long)n_commit_while_encoding,
             (unsigned long long)n_underflow,
             (unsigned long long)n_reset_in_flight,
             (unsigned long long)n_untracked);
}

static void
maybe_summary(void)
{
   double now = now_seconds();
   if (now - last_summary < 10.0)
      return;
   last_summary = now;
   print_summary();
}

static void
exit_summary(void)
{
   pthread_mutex_lock(&check_lock);
   print_summary();
   pthread_mutex_unlock(&check_lock);
}

void
mtl_alloc_check_begin(const void *command_buffer, const void *allocator)
{
   if (!mtl_alloc_check_enabled() || !allocator || !command_buffer)
      return;
   uint64_t tid = current_thread();
   pthread_mutex_lock(&check_lock);
   n_begins++;
   struct alloc_entry *a = alloc_entry(allocator);
   struct cmd_entry *c = cmd_entry(command_buffer);
   if (a) {
      if (a->encoding_cmd && a->encoding_thread != tid) {
         n_concurrent++;
         report("CONCURRENT allocator %p: cmd %p begins on thread %llu while "
                "cmd %p is still encoding on thread %llu",
                allocator, command_buffer, (unsigned long long)tid,
                a->encoding_cmd, (unsigned long long)a->encoding_thread);
      } else if (a->encoding_cmd && a->encoding_cmd != command_buffer) {
         n_nested++;
         report("NESTED allocator %p: cmd %p begins on thread %llu while cmd "
                "%p from the same allocator has not ended",
                allocator, command_buffer, (unsigned long long)tid,
                a->encoding_cmd);
      }
      if (a->in_flight > 0) {
         n_reuse_in_flight++;
         if (n_reuse_in_flight <= 5)
            report("reuse allocator %p for cmd %p on thread %llu with %lld "
                   "earlier command buffers still in flight",
                   allocator, command_buffer, (unsigned long long)tid,
                   (long long)a->in_flight);
      }
      a->encoding_cmd = command_buffer;
      a->encoding_thread = tid;
      a->begins++;
   }
   if (c)
      c->allocator = allocator;
   maybe_summary();
   pthread_mutex_unlock(&check_lock);
}

void
mtl_alloc_check_end(const void *command_buffer)
{
   if (!mtl_alloc_check_enabled() || !command_buffer)
      return;
   uint64_t tid = current_thread();
   pthread_mutex_lock(&check_lock);
   struct cmd_entry *c = cmd_entry(command_buffer);
   struct alloc_entry *a = c && c->allocator ? alloc_entry(c->allocator) : NULL;
   if (a && a->encoding_cmd == command_buffer) {
      if (a->encoding_thread != tid)
         report("cmd %p ends on thread %llu but began on thread %llu "
                "(allocator %p)",
                command_buffer, (unsigned long long)tid,
                (unsigned long long)a->encoding_thread, a->allocator);
      a->encoding_cmd = NULL;
      a->encoding_thread = 0;
   } else {
      n_unmatched_end++;
      report("END without matching begin: cmd %p on thread %llu (allocator %p "
             "now encoding %p)",
             command_buffer, (unsigned long long)tid,
             c ? c->allocator : NULL, a ? a->encoding_cmd : NULL);
   }
   pthread_mutex_unlock(&check_lock);
}

void
mtl_alloc_check_reset(const void *allocator)
{
   if (!mtl_alloc_check_enabled() || !allocator)
      return;
   pthread_mutex_lock(&check_lock);
   struct alloc_entry *a = alloc_entry(allocator);
   if (a && (a->in_flight > 0 || a->encoding_cmd)) {
      n_reset_in_flight++;
      report("RESET allocator %p with %lld command buffers in flight, "
             "encoding cmd %p",
             allocator, (long long)a->in_flight, a->encoding_cmd);
   }
   pthread_mutex_unlock(&check_lock);
}

static void
complete(const void **allocators, uint32_t count)
{
   pthread_mutex_lock(&check_lock);
   n_completions++;
   for (uint32_t i = 0; i < count; i++) {
      struct alloc_entry *a = allocators[i] ? alloc_entry(allocators[i]) : NULL;
      if (!a)
         continue;
      if (--a->in_flight < 0) {
         n_underflow++;
         report("UNDERFLOW allocator %p completed more command buffers than "
                "were committed",
                a->allocator);
         a->in_flight = 0;
      }
   }
   pthread_mutex_unlock(&check_lock);
}

void
mtl_alloc_check_commit(void *options, void *const *command_buffers,
                       uint32_t count)
{
   if (!mtl_alloc_check_enabled() || !options || !count)
      return;
   const void **allocators = calloc(count, sizeof(*allocators));
   if (!allocators)
      return;
   uint64_t tid = current_thread();
   pthread_mutex_lock(&check_lock);
   n_commits++;
   for (uint32_t i = 0; i < count; i++) {
      struct cmd_entry *c = cmd_entry(command_buffers[i]);
      struct alloc_entry *a = c && c->allocator ? alloc_entry(c->allocator) : NULL;
      if (!a)
         continue;
      if (a->encoding_cmd == command_buffers[i]) {
         n_commit_while_encoding++;
         report("COMMIT of cmd %p on thread %llu before it ended (allocator "
                "%p)",
                command_buffers[i], (unsigned long long)tid, a->allocator);
      }
      a->in_flight++;
      allocators[i] = a->allocator;
   }
   pthread_mutex_unlock(&check_lock);
   MTL4CommitOptions *opt = (MTL4CommitOptions *)options;
   [opt addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
      (void)feedback;
      complete(allocators, count);
      free(allocators);
   }];
}
