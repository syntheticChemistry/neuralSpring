// SPDX-License-Identifier: AGPL-3.0-or-later

//! ToadStool-style dispatch patterns — local evolution for absorption.
//!
//! Three patterns evolved locally, mirroring upstream `barracuda::session`:
//!
//! - [`ShaderCache`]: Compiles WGSL sources once, returns cached modules.
//! - [`StreamingDispatch`]: compile → bind → dispatch → readback in one call.
//! - [`WorkloadQueue`]: Priority-sorted batch of heterogeneous GPU workloads
//!   submitted in a single `queue.submit()`.
//!
//! ## Absorption target
//!
//! `ToadStool` absorbs these into `barracuda::session` / `barracuda::pipeline`.
//! The local copies validate the patterns on real hardware before absorption.
//!
//! ## Design rationale
//!
//! GPU dispatch overhead on Vulkan is ~1.5 ms fixed cost per `queue.submit()`.
//! Batching N dispatches into one encoder eliminates (N-1)×1.5 ms overhead.
//! `WorkloadQueue` sorts by priority so latency-sensitive work runs first
//! within the same submission.

use barracuda::device::WgpuDevice;
use std::collections::HashMap;
use std::sync::Arc;

// ─── ShaderCache ─────────────────────────────────────────────────────────

/// Session-scoped shader compilation cache.
///
/// Avoids recompiling the same WGSL source on every dispatch. Keyed by
/// label string — callers must use consistent labels for the same source.
///
/// Mirrors `barracuda::session::pipelines::SessionPipelines` which caches
/// compiled pipelines at the session level.
pub struct ShaderCache {
    device: Arc<WgpuDevice>,
    modules: HashMap<String, wgpu::ShaderModule>,
}

impl ShaderCache {
    /// Create an empty cache bound to a device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            device,
            modules: HashMap::new(),
        }
    }

    /// Get or compile a shader module from WGSL source.
    ///
    /// On first call for a given label, compiles via `WgpuDevice::compile_shader`.
    /// Subsequent calls return the cached module.
    pub fn get_or_compile(&mut self, label: &str, source: &str) -> &wgpu::ShaderModule {
        if !self.modules.contains_key(label) {
            let module = self.device.compile_shader(source, Some(label));
            self.modules.insert(label.to_owned(), module);
        }
        &self.modules[label]
    }

    /// Get or compile a shader through the df64 hybrid path.
    pub fn get_or_compile_df64(&mut self, label: &str, source: &str) -> &wgpu::ShaderModule {
        if !self.modules.contains_key(label) {
            let module = self.device.compile_shader_df64(source, Some(label));
            self.modules.insert(label.to_owned(), module);
        }
        &self.modules[label]
    }

    /// Number of cached shader modules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Check if a shader is already cached.
    #[must_use]
    pub fn contains(&self, label: &str) -> bool {
        self.modules.contains_key(label)
    }
}

// ─── StreamingDispatch ───────────────────────────────────────────────────

/// Arguments for a streaming f64 compute dispatch.
pub struct DispatchArgs<'a, P: bytemuck::Pod> {
    /// Shader label for caching.
    pub label: &'a str,
    /// WGSL source (compiled once, cached thereafter).
    pub source: &'a str,
    /// Compute entry point name.
    pub entry: &'a str,
    /// f64 input data (uploaded to storage buffer).
    pub input: &'a [f64],
    /// Number of f64 values in the output.
    pub output_count: usize,
    /// Uniform parameters.
    pub params: &'a P,
    /// (x, y, z) workgroup counts.
    pub workgroups: (u32, u32, u32),
}

/// Single-shot streaming dispatch: compile → bind → dispatch → readback.
///
/// Wraps the common pattern of uploading data, running a compute shader,
/// and reading back results. Eliminates boilerplate around buffer creation,
/// bind group layout, pipeline creation, and readback.
///
/// Mirrors `barracuda::staging::UnidirectionalPipeline` which provides
/// a similar upload → compute → readback flow.
pub struct StreamingDispatch {
    device: Arc<WgpuDevice>,
    cache: ShaderCache,
}

impl StreamingDispatch {
    /// Create a new dispatch context with an empty shader cache.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            cache: ShaderCache::new(Arc::clone(&device)),
            device,
        }
    }

    /// Execute a compute shader on f64 data and return the results.
    ///
    /// # Errors
    ///
    /// Returns `Err` if buffer creation or GPU readback fails.
    pub fn dispatch_f64<P: bytemuck::Pod>(
        &mut self,
        args: &DispatchArgs<'_, P>,
    ) -> Result<Vec<f64>, String> {
        self.dispatch_f64_inner(
            args.label,
            args.source,
            args.entry,
            args.input,
            args.output_count,
            args.params,
            args.workgroups,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "private helper, args from DispatchArgs"
    )]
    fn dispatch_f64_inner<P: bytemuck::Pod>(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        input: &[f64],
        output_count: usize,
        params: &P,
        workgroups: (u32, u32, u32),
    ) -> Result<Vec<f64>, String> {
        let module = self.cache.get_or_compile(label, source).clone();

        let in_buf = self.device.create_buffer_f64_init("stream_in", input);
        let out_buf = self
            .device
            .create_buffer_f64(output_count)
            .map_err(|e| format!("output buffer: {e}"))?;
        let param_buf = self.device.create_uniform_buffer("stream_params", params);

        let raw = self.device.device();

        let bgl = raw.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
        });

        let bg = raw.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let pl = raw.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let mut encoder =
            raw.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.device
            .queue()
            .submit(std::iter::once(encoder.finish()));

        self.device
            .read_buffer_f64(&out_buf, output_count)
            .map_err(|e| format!("readback: {e}"))
    }

    /// Access the underlying shader cache.
    #[must_use]
    pub const fn cache(&self) -> &ShaderCache {
        &self.cache
    }
}

// ─── WorkloadQueue ───────────────────────────────────────────────────────

/// Priority level for queued workloads.
///
/// Higher-priority work is encoded first within the same command encoder,
/// ensuring it begins execution earlier on the GPU command processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Background computation (large batches, offline analysis).
    Low = 0,
    /// Default priority (most science workloads).
    Normal = 1,
    /// Latency-sensitive (real-time inference, interactive viz).
    High = 2,
    /// Critical path (convergence checks, gating decisions).
    Critical = 3,
}

/// A queued GPU workload — shader + bindings + dispatch dimensions.
///
/// Workloads are accumulated via [`WorkloadQueue::enqueue`] and executed
/// together in a single `queue.submit()` via [`WorkloadQueue::flush`].
pub struct QueuedWorkload {
    /// Human-readable label for debugging.
    pub label: String,
    /// Execution priority (higher runs first within the batch).
    pub priority: Priority,
    /// Pre-compiled pipeline for this workload.
    pipeline: wgpu::ComputePipeline,
    /// Bind group with all buffer bindings.
    bind_group: wgpu::BindGroup,
    /// Workgroup dimensions (x, y, z).
    workgroups: (u32, u32, u32),
}

/// Batched workload queue with priority-sorted single-submission dispatch.
///
/// Accumulates heterogeneous GPU workloads, then flushes them all in one
/// `queue.submit()` call. Within the submission, higher-priority workloads
/// are encoded first.
///
/// Mirrors `barracuda::session::TensorSession` which batches tensor ops
/// into a single command encoder.
pub struct WorkloadQueue {
    device: Arc<WgpuDevice>,
    workloads: Vec<QueuedWorkload>,
}

impl WorkloadQueue {
    /// Create an empty workload queue.
    #[must_use]
    pub const fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            device,
            workloads: Vec::new(),
        }
    }

    /// Add a workload to the queue.
    pub fn enqueue(&mut self, workload: QueuedWorkload) {
        self.workloads.push(workload);
    }

    /// Number of pending workloads.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.workloads.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.workloads.is_empty()
    }

    /// Execute all queued workloads in a single GPU submission.
    ///
    /// Workloads are sorted by priority (highest first) and encoded into
    /// one command encoder. The queue is drained after flush.
    pub fn flush(&mut self) {
        if self.workloads.is_empty() {
            return;
        }

        self.workloads.sort_by(|a, b| b.priority.cmp(&a.priority));

        let raw = self.device.device();
        let mut encoder = raw.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("WorkloadQueue::flush"),
        });

        for wl in &self.workloads {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&wl.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&wl.pipeline);
            pass.set_bind_group(0, Some(&wl.bind_group), &[]);
            pass.dispatch_workgroups(wl.workgroups.0, wl.workgroups.1, wl.workgroups.2);
        }

        self.device
            .queue()
            .submit(std::iter::once(encoder.finish()));
        self.workloads.clear();
    }

    /// Build a [`QueuedWorkload`] from components.
    ///
    /// Helper for constructing workloads without exposing pipeline creation
    /// boilerplate to callers.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "builder pattern — callers assemble from GPU primitives"
    )]
    pub fn build_workload(
        &self,
        label: &str,
        priority: Priority,
        module: &wgpu::ShaderModule,
        entry: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
        bind_group: wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) -> QueuedWorkload {
        let raw = self.device.device();
        let pl = raw.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        QueuedWorkload {
            label: label.to_owned(),
            priority,
            pipeline,
            bind_group,
            workgroups,
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────

const fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test assertions")]
mod tests {
    use super::*;

    fn test_device() -> Arc<WgpuDevice> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        Arc::new(rt.block_on(async { WgpuDevice::new().await.unwrap() }))
    }

    #[test]
    fn shader_cache_caches() {
        let device = test_device();

        let mut cache = ShaderCache::new(device);
        assert!(cache.is_empty());

        let src = "@compute @workgroup_size(1) fn main() {}";
        let _ = cache.get_or_compile("test_trivial", src);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains("test_trivial"));

        let _ = cache.get_or_compile("test_trivial", src);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn workload_queue_starts_empty() {
        let queue = WorkloadQueue::new(test_device());
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn empty_flush_is_noop() {
        let mut queue = WorkloadQueue::new(test_device());
        queue.flush();
        assert!(queue.is_empty());
    }
}
