use std::{collections::HashSet, ffi::CStr, mem::ManuallyDrop, os::raw::c_char, sync::Arc};

use super::physical_device::{PhysicalDevice, QueueFamily};
use crate::{
    CommandBuffer, Fence, Semaphore,
    error::{Error, Result},
};
use ash::{
    ext::shader_object,
    khr,
    vk::{self},
};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use parking_lot::Mutex;

const DEVICE_EXTENSIONS: [&CStr; 5] = [
    khr::swapchain::NAME,
    khr::maintenance5::NAME,
    khr::maintenance6::NAME,
    khr::push_descriptor::NAME,
    shader_object::NAME,
];

pub struct Queue {
    pub raw: vk::Queue,
    pub family: QueueFamily,
}

pub struct Immediate {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

#[derive(Clone)]
pub struct Device(pub(crate) Arc<DeviceInner>);

impl std::ops::Deref for Device {
    type Target = DeviceInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Device {
    pub fn new(physical_device: PhysicalDevice) -> Result<Self> {
        Ok(Self(Arc::new(DeviceInner::new(physical_device)?)))
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.raw.device_wait_idle()? };
        Ok(())
    }

    pub fn submit(
        &self,
        cmd: &CommandBuffer,
        wait: &Semaphore,
        signal: &Semaphore,
        fence: &Fence,
    ) -> Result<()> {
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(wait.raw)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(signal.raw)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw)];

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&cmds)
            .signal_semaphore_infos(&signals);

        unsafe {
            self.raw
                .queue_submit2(self.main_queue.raw, &[submit], fence.raw)?;
        }
        Ok(())
    }

    pub fn immediate_submit(
        &self,
        record: impl FnOnce(&CommandBuffer) -> Result<()>,
    ) -> Result<()> {
        unsafe {
            self.raw
                .wait_for_fences(&[self.immediate.fence], true, u64::MAX)?;

            self.raw.reset_command_pool(
                self.immediate.command_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
        }

        let cmd = CommandBuffer {
            raw: self.immediate.command_buffer,
            device: self.clone(),
        };

        cmd.begin()?;

        record(&cmd)?;

        cmd.end()?;

        let command_buffer_infos = [
            vk::CommandBufferSubmitInfo::default().command_buffer(self.immediate.command_buffer)
        ];

        let submit_info = vk::SubmitInfo2::default().command_buffer_infos(&command_buffer_infos);

        unsafe {
            self.raw.reset_fences(&[self.immediate.fence])?;
            self.raw
                .queue_submit2(self.main_queue.raw, &[submit_info], self.immediate.fence)?;
            self.raw
                .wait_for_fences(&[self.immediate.fence], true, u64::MAX)?;
        }

        Ok(())
    }
}

pub struct DeviceInner {
    pub(crate) raw: ash::Device,
    pub main_queue: Queue,
    pub(crate) allocator: ManuallyDrop<Mutex<Allocator>>,

    pub(crate) shader_object: shader_object::Device,

    immediate: Immediate,

    // keep alive
    pub(crate) physical_device: PhysicalDevice,
}

impl DeviceInner {
    fn new(physical_device: PhysicalDevice) -> Result<Self> {
        let instance = &physical_device.instance;

        let extension_properties = unsafe {
            instance
                .raw
                .enumerate_device_extension_properties(physical_device.raw)?
        };
        let supported_extensions: HashSet<&CStr> = extension_properties
            .iter()
            .filter_map(|p| p.extension_name_as_c_str().ok())
            .collect();
        let unsupported_extensions: Vec<String> = DEVICE_EXTENSIONS
            .iter()
            .filter(|ext| !supported_extensions.contains(*ext))
            .map(|ext| ext.to_string_lossy().into_owned())
            .collect();

        if !unsupported_extensions.is_empty() {
            return Err(Error::UnsupportedExtensions(unsupported_extensions));
        }

        let main_queue_family = physical_device
            .queue_families
            .iter()
            .filter(|queue_family| {
                queue_family
                    .properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            })
            .copied()
            .next();
        let Some(main_queue_family) = main_queue_family else {
            return Err(Error::NoSuitableQueueFamily("graphics + compute"));
        };

        let priorities = [1.0];
        let main_queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(main_queue_family.index)
            .queue_priorities(&priorities)];

        let extension_ptrs: Vec<*const c_char> =
            DEVICE_EXTENSIONS.iter().map(|err| err.as_ptr()).collect();

        let mut enabled_features = EnabledFeatures::new();
        let mut features2 = enabled_features.features2();

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&main_queue_info)
            .enabled_extension_names(&extension_ptrs)
            .push_next(&mut features2);

        let device = unsafe {
            instance
                .raw
                .create_device(physical_device.raw, &device_info, None)
                .unwrap()
        };
        tracing::info!("Vulkan device created");

        let main_queue = Queue {
            raw: unsafe { device.get_device_queue(main_queue_family.index, 0) },
            family: main_queue_family,
        };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.raw.clone(),
            device: device.clone(),
            physical_device: physical_device.raw,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })?;

        let shader_object = shader_object::Device::new(&instance.raw, &device);

        let immediate_command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(main_queue.family.index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?
        };

        let immediate_command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(immediate_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };

        let immediate_fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        Ok(Self {
            raw: device,
            main_queue,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            shader_object,
            physical_device,
            immediate: Immediate {
                command_pool: immediate_command_pool,
                command_buffer: immediate_command_buffer,
                fence: immediate_fence,
            },
        })
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.raw.destroy_fence(self.immediate.fence, None);
            self.raw
                .destroy_command_pool(self.immediate.command_pool, None); // frees its buffers too
            ManuallyDrop::drop(&mut self.allocator);
            self.raw.destroy_device(None);
        }
        tracing::info!("Vulkan device destroyed");
    }
}

struct EnabledFeatures<'a> {
    v11: vk::PhysicalDeviceVulkan11Features<'a>,
    v12: vk::PhysicalDeviceVulkan12Features<'a>,
    v13: vk::PhysicalDeviceVulkan13Features<'a>,
    maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR<'a>,
    maintenance6: vk::PhysicalDeviceMaintenance6FeaturesKHR<'a>,
    shader_object: vk::PhysicalDeviceShaderObjectFeaturesEXT<'a>,
}

impl<'a> EnabledFeatures<'a> {
    fn new() -> Self {
        Self {
            v11: vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true),
            v12: vk::PhysicalDeviceVulkan12Features::default()
                .descriptor_indexing(true)
                .runtime_descriptor_array(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_variable_descriptor_count(true)
                .descriptor_binding_update_unused_while_pending(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_storage_image_update_after_bind(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .shader_storage_image_array_non_uniform_indexing(true)
                .shader_storage_buffer_array_non_uniform_indexing(true)
                .timeline_semaphore(true)
                .buffer_device_address(true)
                .scalar_block_layout(true),
            v13: vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true)
                .maintenance4(true),
            maintenance5: vk::PhysicalDeviceMaintenance5FeaturesKHR::default().maintenance5(true),
            maintenance6: vk::PhysicalDeviceMaintenance6FeaturesKHR::default().maintenance6(true),
            shader_object: vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true),
        }
    }

    fn features2(&'a mut self) -> vk::PhysicalDeviceFeatures2<'a> {
        vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().shader_int64(true))
            .push_next(&mut self.v11)
            .push_next(&mut self.v12)
            .push_next(&mut self.v13)
            .push_next(&mut self.maintenance5)
            .push_next(&mut self.maintenance6)
            .push_next(&mut self.shader_object)
    }
}
