use crate::vulkan::{
    device::{Device, Queue},
    presentation::surface::Surface,
    resources::{Image, ImageDesc, ImageKind, ImageView, ImageViewDesc},
    sync::Semaphore,
};
use std::sync::Arc;

use crate::error::{Error, Result};
use ash::{khr, vk};

pub struct SwapchainDesc {
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub vsync: bool,
}

pub struct AcquiredImage {
    pub index: u32,
    pub suboptimal: bool,
}

pub enum AcquireResult {
    Ok(AcquiredImage),
    OutOfDate,
}

pub struct Swapchain {
    pub(crate) raw: vk::SwapchainKHR,
    pub(crate) loader: khr::swapchain::Device,
    pub desc: SwapchainDesc,

    pub images: Vec<Arc<Image>>,
    pub views: Vec<ImageView>,

    pub extent: vk::Extent2D,

    // keep alive
    device: Device,
    surface: Surface,
}

impl Swapchain {
    pub fn new(device: Device, surface: Surface, desc: SwapchainDesc) -> Result<Self> {
        Self::create(device, surface, desc, vk::SwapchainKHR::null())
    }

    fn create(
        device: Device,
        surface: Surface,
        desc: SwapchainDesc,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Self> {
        let capabilities = surface.capabilities(&device.physical_device)?;

        let max_images = if capabilities.max_image_count == 0 {
            u32::MAX
        } else {
            capabilities.max_image_count
        };
        let desired_image_count = 3.clamp(capabilities.min_image_count, max_images);

        let surface_extent = if capabilities.current_extent.width == u32::MAX {
            desc.extent
        } else {
            capabilities.current_extent
        };

        if surface_extent.width == 0 || surface_extent.height == 0 {
            return Err(Error::InvalidExtent {
                width: surface_extent.width,
                height: surface_extent.height,
            });
        }

        let preferred_present_modes = if desc.vsync {
            [vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO]
        } else {
            [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
        };

        let supported_present_modes = surface.present_modes(&device.physical_device)?;
        let present_mode = preferred_present_modes
            .into_iter()
            .find(|mode| supported_present_modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let pre_transform = if capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            capabilities.current_transform
        };

        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .old_swapchain(old_swapchain)
            .surface(surface.raw)
            .min_image_count(desired_image_count)
            .image_color_space(desc.format.color_space)
            .image_format(desc.format.format)
            .image_extent(surface_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        let loader = khr::swapchain::Device::new(&device.physical_device.instance.raw, &device.raw);
        let swapchain = unsafe { loader.create_swapchain(&swapchain_info, None) }?;

        let raw_images = unsafe { loader.get_swapchain_images(swapchain)? };
        let images = raw_images
            .into_iter()
            .map(|raw_image| {
                Arc::new(Image {
                    raw: raw_image,
                    desc: ImageDesc {
                        kind: ImageKind::D2,
                        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        flags: vk::ImageCreateFlags::empty(),
                        format: desc.format.format,
                        extent: desc.extent.into(),
                        mip_levels: 1,
                        array_elements: 1,
                        samples: vk::SampleCountFlags::TYPE_1,
                    },
                    allocation: None,
                })
            })
            .collect::<Vec<_>>();

        let views = images
            .clone()
            .into_iter()
            .map(|image| {
                device
                    .create_image_view(
                        &image,
                        &ImageViewDesc {
                            ..Default::default()
                        },
                    )
                    .unwrap()
            })
            .collect();

        Ok(Self {
            raw: swapchain,
            loader,
            desc,
            images,
            views,

            extent: surface_extent,

            device,
            surface,
        })
    }

    pub fn recreate(&mut self, extent: vk::Extent2D) -> Result<bool> {
        if extent.width == 0 || extent.height == 0 {
            return Ok(false);
        }
        self.device.wait_idle()?;

        let desc = SwapchainDesc {
            extent,
            ..self.desc
        };
        let new = Self::create(self.device.clone(), self.surface.clone(), desc, self.raw)?;
        *self = new;
        Ok(true)
    }

    pub fn acquire_next_image(&self, signal: &Semaphore) -> Result<AcquireResult> {
        let result = unsafe {
            self.loader
                .acquire_next_image(self.raw, u64::MAX, signal.raw, vk::Fence::null())
        };
        match result {
            Ok((index, suboptimal)) => Ok(AcquireResult::Ok(AcquiredImage { index, suboptimal })),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(AcquireResult::OutOfDate),
            Err(err) => Err(err.into()),
        }
    }

    pub fn present(&self, queue: &Queue, index: u32, wait: &Semaphore) -> Result<bool> {
        let swapchains = [self.raw];
        let indices = [index];
        let waits = [wait.raw];
        let info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(&waits);

        match unsafe { self.loader.queue_present(queue.raw, &info) } {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
            Err(err) => Err(err.into()),
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for &view in &self.views {
                self.device.raw.destroy_image_view(view.raw, None);
            }
            self.loader.destroy_swapchain(self.raw, None);
        }
    }
}
