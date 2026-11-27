use crate::vulkan::{instance::Instance, physical_device::PhysicalDevice};
use std::{ffi::CStr, sync::Arc};

use crate::error::Result;
use ash::{khr, vk};

#[derive(Clone)]
pub struct Surface(pub(crate) Arc<Inner>);

impl std::ops::Deref for Surface {
    type Target = Inner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Surface {
    pub fn new(
        instance: Instance,
        window: &(impl raw_window_handle::HasDisplayHandle + raw_window_handle::HasWindowHandle),
    ) -> Result<Self> {
        Ok(Self(Arc::new(Inner::new(instance, window)?)))
    }

    pub fn window_required_extensions(
        window: &impl raw_window_handle::HasDisplayHandle,
    ) -> Result<Vec<&'static CStr>> {
        Ok(
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?
                .iter()
                .copied()
                .map(|ext| unsafe { CStr::from_ptr(ext) })
                .collect(),
        )
    }
}

pub struct Inner {
    pub(crate) raw: vk::SurfaceKHR,
    pub(crate) loader: khr::surface::Instance,

    _instance: Instance,
}

impl Inner {
    pub fn new(
        instance: Instance,
        window: &(impl raw_window_handle::HasDisplayHandle + raw_window_handle::HasWindowHandle),
    ) -> Result<Self> {
        let window_handle = window.window_handle()?;
        let display_handle = window.display_handle()?;

        let surface = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.raw,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )?
        };

        let loader = khr::surface::Instance::new(&instance.entry, &instance.raw);

        Ok(Self {
            raw: surface,
            loader,
            _instance: instance,
        })
    }

    pub fn capabilities(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR> {
        Ok(unsafe {
            self.loader
                .get_physical_device_surface_capabilities(physical_device.raw, self.raw)?
        })
    }

    pub fn formats(&self, physical_device: &PhysicalDevice) -> Result<Vec<vk::SurfaceFormatKHR>> {
        Ok(unsafe {
            self.loader
                .get_physical_device_surface_formats(physical_device.raw, self.raw)?
        })
    }

    pub fn present_modes(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>> {
        Ok(unsafe {
            self.loader
                .get_physical_device_surface_present_modes(physical_device.raw, self.raw)?
        })
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.raw, None);
        }
    }
}
