use std::sync::Arc;

use crate::error::Result;

use super::instance::Instance;
use ash::vk;

#[derive(Clone)]
pub struct PhysicalDevice(pub(crate) Arc<PhysicalDeviceInner>);

impl std::ops::Deref for PhysicalDevice {
    type Target = PhysicalDeviceInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Copy)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

#[derive(Clone)]
pub struct PhysicalDeviceInner {
    pub(crate) raw: vk::PhysicalDevice,
    pub queue_families: Vec<QueueFamily>,

    pub properties: vk::PhysicalDeviceProperties,

    // keep alive
    pub(crate) instance: Instance,
}

impl Instance {
    pub fn enumerate_physical_devices(&self) -> Result<Vec<PhysicalDevice>> {
        unsafe {
            let physical_devices = self.0.raw.enumerate_physical_devices()?;

            Ok(physical_devices
                .into_iter()
                .map(|physical_device| {
                    let properties = self.0.raw.get_physical_device_properties(physical_device);

                    let queue_families = self
                        .0
                        .raw
                        .get_physical_device_queue_family_properties(physical_device)
                        .into_iter()
                        .enumerate()
                        .map(|(index, queue_properties)| QueueFamily {
                            #[allow(clippy::cast_possible_truncation)]
                            index: index as u32,
                            properties: queue_properties,
                        })
                        .collect();

                    PhysicalDevice(Arc::new(PhysicalDeviceInner {
                        raw: physical_device,
                        queue_families,
                        properties,
                        instance: self.clone(),
                    }))
                })
                .collect())
        }
    }
}
