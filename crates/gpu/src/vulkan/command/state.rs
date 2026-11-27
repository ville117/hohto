use ash::vk;

use crate::CommandBuffer;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RasterState {
    pub topology: vk::PrimitiveTopology,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
}

impl Default for RasterState {
    fn default() -> Self {
        Self {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::CLOCKWISE,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DepthState {
    pub test: bool,
    pub write: bool,
    pub compare_op: vk::CompareOp,
}

impl DepthState {
    pub const DISABLED: Self = Self {
        test: false,
        write: false,
        compare_op: vk::CompareOp::LESS,
    };

    pub const READ_WRITE: Self = Self {
        test: true,
        write: true,
        compare_op: vk::CompareOp::LESS,
    };

    pub const READ_ONLY: Self = Self {
        test: true,
        write: false,
        compare_op: vk::CompareOp::LESS,
    };
}

impl Default for DepthState {
    fn default() -> Self {
        Self::DISABLED
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlendState {
    pub enable: bool,
    pub src_color: vk::BlendFactor,
    pub dst_color: vk::BlendFactor,
    pub color_op: vk::BlendOp,
    pub src_alpha: vk::BlendFactor,
    pub dst_alpha: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
    pub write_mask: vk::ColorComponentFlags,
}

impl BlendState {
    pub const OPAQUE: Self = Self {
        enable: false,
        src_color: vk::BlendFactor::ONE,
        dst_color: vk::BlendFactor::ZERO,
        color_op: vk::BlendOp::ADD,
        src_alpha: vk::BlendFactor::ONE,
        dst_alpha: vk::BlendFactor::ZERO,
        alpha_op: vk::BlendOp::ADD,
        write_mask: vk::ColorComponentFlags::RGBA,
    };

    pub const ALPHA: Self = Self {
        enable: true,
        src_color: vk::BlendFactor::SRC_ALPHA,
        dst_color: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_op: vk::BlendOp::ADD,
        src_alpha: vk::BlendFactor::ONE,
        dst_alpha: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_op: vk::BlendOp::ADD,
        write_mask: vk::ColorComponentFlags::RGBA,
    };
}

impl Default for BlendState {
    fn default() -> Self {
        Self::OPAQUE
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct DrawState {
    pub raster: RasterState,
    pub depth: DepthState,
    pub blend: BlendState,
}

impl CommandBuffer {
    pub fn set_draw_state(&self, state: &DrawState, color_attachments: u32) {
        self.set_base_state();
        self.set_raster_state(&state.raster);
        self.set_depth_state(state.depth);
        for attachment in 0..color_attachments {
            self.set_blend_state(attachment, &state.blend);
        }
    }

    pub fn set_viewport_scissor(&self, extent: vk::Extent2D) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            #[allow(clippy::cast_precision_loss)]
            width: extent.width as f32,
            #[allow(clippy::cast_precision_loss)]
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        unsafe {
            self.device
                .raw
                .cmd_set_viewport_with_count(self.raw, &[viewport]);
            self.device
                .raw
                .cmd_set_scissor_with_count(self.raw, &[scissor]);
        }
    }

    fn set_base_state(&self) {
        let device = &self.device.raw;
        let shader_object = &self.device.shader_object;
        let command_buffer = self.raw;

        unsafe {
            shader_object.cmd_set_vertex_input(command_buffer, &[], &[]);

            device.cmd_set_primitive_restart_enable(command_buffer, false);
            device.cmd_set_rasterizer_discard_enable(command_buffer, false);
            device.cmd_set_depth_bias_enable(command_buffer, false);
            device.cmd_set_depth_bounds_test_enable(command_buffer, false);
            device.cmd_set_stencil_test_enable(command_buffer, false);
            shader_object.cmd_set_depth_clamp_enable(command_buffer, false);

            device.cmd_set_line_width(command_buffer, 1.0);

            shader_object
                .cmd_set_rasterization_samples(command_buffer, vk::SampleCountFlags::TYPE_1);
            shader_object.cmd_set_sample_mask(
                command_buffer,
                vk::SampleCountFlags::TYPE_1,
                &[u32::MAX],
            );
            shader_object.cmd_set_alpha_to_coverage_enable(command_buffer, false);

            shader_object.cmd_set_logic_op_enable(command_buffer, false);
        }
    }

    pub(super) fn set_raster_state(&self, state: &RasterState) {
        let device = &self.device.raw;
        let shader_object = &self.device.shader_object;

        unsafe {
            device.cmd_set_primitive_topology(self.raw, state.topology);
            shader_object.cmd_set_polygon_mode(self.raw, state.polygon_mode);
            device.cmd_set_cull_mode(self.raw, state.cull_mode);
            device.cmd_set_front_face(self.raw, state.front_face);
        }
    }

    pub(super) fn set_depth_state(&self, state: DepthState) {
        let device = &self.device.raw;

        unsafe {
            device.cmd_set_depth_test_enable(self.raw, state.test);
            device.cmd_set_depth_write_enable(self.raw, state.write);
            device.cmd_set_depth_compare_op(self.raw, state.compare_op);
        }
    }

    pub(super) fn set_blend_state(&self, attachment: u32, state: &BlendState) {
        let shader_object = &self.device.shader_object;
        let equation = vk::ColorBlendEquationEXT::default()
            .src_color_blend_factor(state.src_color)
            .dst_color_blend_factor(state.dst_color)
            .color_blend_op(state.color_op)
            .src_alpha_blend_factor(state.src_alpha)
            .dst_alpha_blend_factor(state.dst_alpha)
            .alpha_blend_op(state.alpha_op);

        unsafe {
            shader_object.cmd_set_color_blend_enable(
                self.raw,
                attachment,
                &[vk::Bool32::from(state.enable)],
            );
            shader_object.cmd_set_color_blend_equation(self.raw, attachment, &[equation]);
            shader_object.cmd_set_color_write_mask(self.raw, attachment, &[state.write_mask]);
        }
    }
}
