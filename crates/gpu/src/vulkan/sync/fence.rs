use ash::vk;

use crate::{error::Result, vulkan::device::Device};

pub struct Fence {
    pub(crate) raw: vk::Fence,
    device: Device,
}

impl Fence {
    pub fn new(device: Device, signaled: bool) -> Result<Self> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let raw = unsafe {
            device
                .raw
                .create_fence(&vk::FenceCreateInfo::default().flags(flags), None)?
        };
        Ok(Self { raw, device })
    }

    pub fn wait(&self) -> Result<()> {
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.raw], true, u64::MAX)?;
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        unsafe { self.device.raw.reset_fences(&[self.raw])? };
        Ok(())
    }
}
