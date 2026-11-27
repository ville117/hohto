use crate::vulkan::device::Device;
use crate::{ImageState, error::Result};
use ash::vk::{self, Extent3D};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageKind {
    D1,
    D1Array,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

impl ImageKind {
    #[allow(clippy::must_use_candidate)]
    pub fn view_type(self) -> vk::ImageViewType {
        match self {
            Self::D1 => vk::ImageViewType::TYPE_1D,
            Self::D1Array => vk::ImageViewType::TYPE_1D_ARRAY,
            Self::D2 => vk::ImageViewType::TYPE_2D,
            Self::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::D3 => vk::ImageViewType::TYPE_3D,
            Self::Cube => vk::ImageViewType::CUBE,
            Self::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}

pub(crate) fn aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub kind: ImageKind,
    pub format: vk::Format,
    pub extent: Extent3D,
    pub mip_levels: u32,
    pub array_elements: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}

impl ImageDesc {
    pub(crate) fn create_info(&self) -> vk::ImageCreateInfo<'static> {
        let [width, height, depth] = [self.extent.width, self.extent.height, self.extent.depth];
        let count = self.array_elements;

        let (image_type, extent, array_layers) = match self.kind {
            ImageKind::D1 => (vk::ImageType::TYPE_1D, [width, 1, 1], 1),
            ImageKind::D1Array => (vk::ImageType::TYPE_1D, [width, 1, 1], count),
            ImageKind::D2 => (vk::ImageType::TYPE_2D, [width, height, 1], 1),
            ImageKind::D2Array => (vk::ImageType::TYPE_2D, [width, height, 1], count),
            ImageKind::D3 => (vk::ImageType::TYPE_3D, [width, height, depth], 1),
            ImageKind::Cube => (vk::ImageType::TYPE_2D, [width, height, 1], 6),
            ImageKind::CubeArray => (vk::ImageType::TYPE_2D, [width, height, 1], 6 * count),
        };

        let mut flags = self.flags;
        if matches!(self.kind, ImageKind::Cube | ImageKind::CubeArray) {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }

        vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(self.format)
            .extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            })
            .mip_levels(self.mip_levels)
            .array_layers(array_layers)
            .samples(self.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(self.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct ImageViewDesc {
    pub kind: Option<ImageKind>,
    pub format: Option<vk::Format>,
    pub aspect: Option<vk::ImageAspectFlags>,
    pub base_mip: u32,
    pub mip_count: Option<u32>,
    pub base_layer: u32,
    pub layer_count: Option<u32>,
}

pub struct Image {
    pub(crate) raw: vk::Image,
    pub desc: ImageDesc,
    pub allocation: Option<Allocation>,
}

impl Device {
    pub fn create_image(&self, desc: ImageDesc, name: &str) -> Result<Image> {
        tracing::debug!("Creating image `{name}`: {desc:?}");

        let mut create_info = desc.create_info();
        create_info.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;

        let image = unsafe { self.raw.create_image(&create_info, None)? };
        let requirements = unsafe { self.raw.get_image_memory_requirements(image) };

        let allocation = match self.allocator.lock().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(alloc) => alloc,
            Err(err) => {
                unsafe { self.raw.destroy_image(image, None) };
                return Err(err.into());
            }
        };

        unsafe {
            self.raw
                .bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        Ok(Image {
            raw: image,
            desc,
            allocation: Some(allocation),
        })
    }

    pub fn create_image_view(&self, image: &Image, desc: &ImageViewDesc) -> Result<ImageView> {
        let kind = desc.kind.unwrap_or(image.desc.kind);
        let format = desc.format.unwrap_or(image.desc.format);
        let aspect = desc.aspect.unwrap_or_else(|| aspect_mask(format));

        let info = vk::ImageViewCreateInfo::default()
            .image(image.raw)
            .view_type(kind.view_type())
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: desc.base_mip,
                level_count: desc.mip_count.unwrap_or(vk::REMAINING_MIP_LEVELS),
                base_array_layer: desc.base_layer,
                layer_count: desc.layer_count.unwrap_or(vk::REMAINING_ARRAY_LAYERS),
            });

        Ok(unsafe {
            ImageView {
                raw: self.raw.create_image_view(&info, None)?,
            }
        })
    }

    pub fn create_image_with(&self, mut desc: ImageDesc, data: &[u8], name: &str) -> Result<Image> {
        desc.usage |= vk::ImageUsageFlags::TRANSFER_DST;

        let image = self.create_image(desc, name)?;

        let staging =
            self.create_buffer_with(&format!("{name}_staging"), data, MemoryLocation::CpuToGpu)?;

        self.immediate_submit(|cmd| {
            cmd.transition(&image, ImageState::Undefined, ImageState::TransferDst);

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(desc.extent);

            unsafe {
                self.raw.cmd_copy_buffer_to_image(
                    cmd.raw,
                    staging.raw,
                    image.raw,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }

            cmd.transition(&image, ImageState::TransferDst, ImageState::ShaderRead);
            Ok(())
        })?;

        Ok(image)
    }
}

#[derive(Clone, Copy)]
pub struct ImageView {
    pub(crate) raw: vk::ImageView,
}
