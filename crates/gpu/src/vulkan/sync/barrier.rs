use ash::vk;

use crate::{Buffer, CommandBuffer, Image, aspect_mask};

pub type StageFlags = vk::PipelineStageFlags2;
pub type AccessFlags = vk::AccessFlags2;

#[derive(Clone, Copy)]
pub enum ImageState {
    Undefined,
    General,
    ColorAttachment,
    ShaderRead,
    TransferSrc,
    TransferDst,
    Present,
}

impl ImageState {
    fn layout(self) -> vk::ImageLayout {
        match self {
            Self::Undefined => vk::ImageLayout::UNDEFINED,
            Self::General => vk::ImageLayout::GENERAL,
            Self::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Self::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            Self::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Self::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Self::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    fn sync(self) -> (StageFlags, AccessFlags) {
        match self {
            Self::Undefined | Self::Present => (StageFlags::empty(), AccessFlags::empty()),
            Self::General => (
                StageFlags::ALL_COMMANDS,
                AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
            ),
            Self::ColorAttachment => (
                StageFlags::COLOR_ATTACHMENT_OUTPUT,
                AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
            ),
            Self::ShaderRead => (
                StageFlags::FRAGMENT_SHADER | StageFlags::COMPUTE_SHADER,
                AccessFlags::SHADER_SAMPLED_READ,
            ),
            Self::TransferSrc => (StageFlags::ALL_TRANSFER, AccessFlags::TRANSFER_READ),
            Self::TransferDst => (StageFlags::ALL_TRANSFER, AccessFlags::TRANSFER_WRITE),
        }
    }
}

pub struct MemoryBarrier {
    pub src_stage: StageFlags,
    pub src_access: AccessFlags,

    pub dst_stage: StageFlags,
    pub dst_access: AccessFlags,
}

pub struct BufferBarrier<'a> {
    pub buffer: &'a Buffer,

    pub src_stage: StageFlags,
    pub src_access: AccessFlags,

    pub dst_stage: StageFlags,
    pub dst_access: AccessFlags,

    pub offset: u64,
    pub size: Option<u64>,
}

pub struct ImageBarrier<'a> {
    pub image: &'a Image,
    pub from: ImageState,
    pub to: ImageState,
}

impl CommandBuffer {
    pub fn barriers(
        &self,
        memory: &[MemoryBarrier],
        buffer: &[BufferBarrier],
        images: &[ImageBarrier],
    ) {
        let memory_barriers: Vec<_> = memory
            .iter()
            .map(|barrier| {
                vk::MemoryBarrier2::default()
                    .src_stage_mask(barrier.src_stage)
                    .src_access_mask(barrier.src_access)
                    .dst_stage_mask(barrier.dst_stage)
                    .dst_access_mask(barrier.dst_access)
            })
            .collect();

        let buffer_barriers: Vec<_> = buffer
            .iter()
            .map(|barrier| {
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(barrier.src_stage)
                    .src_access_mask(barrier.src_access)
                    .dst_stage_mask(barrier.dst_stage)
                    .dst_access_mask(barrier.dst_access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(barrier.buffer.raw)
                    .offset(barrier.offset)
                    .size(barrier.size.unwrap_or(vk::WHOLE_SIZE))
            })
            .collect();

        let image_barriers: Vec<_> = images
            .iter()
            .map(|barrier| {
                let (src_stage, src_access) = barrier.from.sync();
                let (dst_stage, dst_access) = barrier.to.sync();

                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
                    .old_layout(barrier.from.layout())
                    .new_layout(barrier.to.layout())
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(barrier.image.raw)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect_mask(barrier.image.desc.format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
            })
            .collect();

        let dependency = vk::DependencyInfo::default()
            .memory_barriers(&memory_barriers)
            .buffer_memory_barriers(&buffer_barriers)
            .image_memory_barriers(&image_barriers);

        unsafe {
            self.device.raw.cmd_pipeline_barrier2(self.raw, &dependency);
        }
    }

    pub fn transition(&self, image: &Image, from: ImageState, to: ImageState) {
        self.barriers(&[], &[], &[ImageBarrier { image, from, to }]);
    }
}
