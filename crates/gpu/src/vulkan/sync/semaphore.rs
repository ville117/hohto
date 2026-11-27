use ash::vk;

use crate::{error::Result, vulkan::device::Device};
pub struct Semaphore {
    pub(crate) raw: vk::Semaphore,
    _device: Device,
}

impl Semaphore {
    pub fn new(device: Device) -> Result<Self> {
        let raw = unsafe {
            device
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };
        Ok(Self {
            raw,
            _device: device,
        })
    }
}
