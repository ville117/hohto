use super::command_buffer::CommandBuffer;
use crate::{error::Result, vulkan::device::Device};
use ash::vk;

pub struct CommandPool {
    pub(crate) raw: vk::CommandPool,
    pub(crate) device: Device,
}

impl CommandPool {
    pub fn new(device: Device) -> Result<Self> {
        let raw = unsafe {
            device.raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.main_queue.family.index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?
        };
        Ok(Self { raw, device })
    }

    pub fn allocate_command_buffers(&self, count: u32) -> Result<Vec<CommandBuffer>> {
        let raws = unsafe {
            self.device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.raw)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(count),
            )?
        };
        Ok(raws
            .into_iter()
            .map(|raw| CommandBuffer {
                raw,
                device: self.device.clone(),
            })
            .collect())
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.device
                .raw
                .reset_command_pool(self.raw, vk::CommandPoolResetFlags::empty())?;
        }
        Ok(())
    }
}
