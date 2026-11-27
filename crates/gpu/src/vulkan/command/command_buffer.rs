use ash::vk;

use crate::{
    Buffer,
    error::Result,
    vulkan::{
        device::Device,
        resources::{ImageView, Shader},
    },
};

#[derive(Clone)]
pub struct CommandBuffer {
    pub(crate) raw: vk::CommandBuffer,
    pub(crate) device: Device,
}

impl CommandBuffer {
    pub fn begin(&self) -> Result<()> {
        unsafe {
            self.device.raw.begin_command_buffer(
                self.raw,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(())
    }
    pub fn end(&self) -> Result<()> {
        unsafe { self.device.raw.end_command_buffer(self.raw)? };
        Ok(())
    }

    pub fn begin_rendering(&self, view: ImageView, extent: vk::Extent2D, clear: [f32; 4]) {
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view.raw)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: clear },
            });
        let attachments = [attachment];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);

        unsafe { self.device.raw.cmd_begin_rendering(self.raw, &info) };
        self.set_viewport_scissor(extent);
    }
    pub fn end_rendering(&self) {
        unsafe { self.device.raw.cmd_end_rendering(self.raw) };
    }

    pub fn bind_shaders(&self, shaders: &[&Shader]) {
        let stages: Vec<vk::ShaderStageFlags> =
            shaders.iter().map(|shader| shader.desc.stage).collect();
        let raws: Vec<vk::ShaderEXT> = shaders.iter().map(|shader| shader.raw).collect();
        unsafe {
            self.device
                .shader_object
                .cmd_bind_shaders(self.raw, &stages, &raws);
        }
    }

    pub fn draw(&self, vertices: u32, instances: u32) {
        unsafe {
            self.device
                .raw
                .cmd_draw(self.raw, vertices, instances, 0, 0);
        }
    }

    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device
                .raw
                .cmd_dispatch(self.raw, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn dispatch_indirect(&self, buffer: &Buffer, offset: u64) {
        unsafe {
            self.device
                .raw
                .cmd_dispatch_indirect(self.raw, buffer.raw, offset);
        }
    }

    pub fn fill_buffer(&self, buffer: &Buffer, offset: u64, size: u64, data: u32) {
        unsafe {
            self.device
                .raw
                .cmd_fill_buffer(self.raw, buffer.raw, offset, size, data);
        }
    }
}
