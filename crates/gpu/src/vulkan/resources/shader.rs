use std::{ffi::CString, io::Cursor};

use crate::vulkan::device::Device;
use crate::{
    BindlessLayout,
    error::{Error, Result},
};
use ash::vk;

pub type ShaderStages = vk::ShaderStageFlags;

pub struct ShaderDesc {
    pub stage: ShaderStages,
    pub next_stage: ShaderStages,
    pub entry_point: String,
}

pub struct Shader {
    pub(crate) raw: vk::ShaderEXT,
    pub desc: ShaderDesc,

    // keep alive
    device: Device,
}

impl Shader {
    pub fn load_spirv(bytes: &[u8]) -> Result<Vec<u32>> {
        ash::util::read_spv(&mut Cursor::new(bytes))
            .map_err(|err| Error::InvalidSpirv(err.to_string()))
    }

    pub fn new(
        device: Device,
        desc: ShaderDesc,
        spirv: &[u32],
        layout: &BindlessLayout,
    ) -> Result<Self> {
        let entry_point = CString::new(desc.entry_point.as_str())
            .map_err(|_| Error::InvalidEntryPoint(desc.entry_point.clone()))?;
        let code = bytemuck::cast_slice(spirv);

        let push_constant_ranges = [layout.push_constant_range()];
        let set_layouts = [layout.set_layout()];

        let create_info = vk::ShaderCreateInfoEXT::default()
            .stage(desc.stage)
            .next_stage(desc.next_stage)
            .code_type(vk::ShaderCodeTypeEXT::SPIRV)
            .code(code)
            .name(&entry_point)
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&set_layouts);

        let raw = match unsafe { device.shader_object.create_shaders(&[create_info], None) } {
            Ok(mut shaders) => shaders
                .pop()
                .ok_or(Error::Internal("create_shaders returned no handles"))?,

            Err((shaders, error)) => {
                for shader in shaders {
                    if shader != vk::ShaderEXT::null() {
                        unsafe {
                            device.shader_object.destroy_shader(shader, None);
                        }
                    }
                }
                return Err(error.into());
            }
        };

        Ok(Self { raw, desc, device })
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.shader_object.destroy_shader(self.raw, None);
        }
    }
}

#[derive(Clone)]
pub struct PipelineLayout {
    pub(crate) raw: vk::PipelineLayout,
    device: Device,
}

impl PipelineLayout {
    pub fn new(device: Device, push_constant_ranges: &[vk::PushConstantRange]) -> Result<Self> {
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(push_constant_ranges);

        let raw = unsafe { device.raw.create_pipeline_layout(&layout_info, None)? };

        Ok(Self { raw, device })
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_pipeline_layout(self.raw, None);
        }
    }
}
