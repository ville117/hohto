use std::{borrow::Cow, collections::HashSet, ffi::CStr, os::raw::c_char, sync::Arc};

use crate::error::{Error, Result};
use ash::{ext::debug_utils, vk};

#[derive(Default)]
pub struct InstanceDesc {
    pub required_extensions: Vec<&'static CStr>,
    pub debugging: bool,
}

#[derive(Clone)]
pub struct Instance(pub(crate) Arc<Inner>);

impl Instance {
    pub fn new(desc: InstanceDesc) -> Result<Self> {
        Ok(Self(Inner::new(desc)?))
    }
}

impl std::ops::Deref for Instance {
    type Target = Inner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Inner {
    pub(crate) raw: ash::Instance,
    pub(crate) entry: ash::Entry,
    debug: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl Inner {
    fn new(mut desc: InstanceDesc) -> Result<Arc<Self>> {
        let entry = unsafe { ash::Entry::load()? };

        desc.required_extensions
            .extend(Self::additional_extensions(&desc));

        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let supported_extensions: HashSet<&CStr> = available_extensions
            .iter()
            .filter_map(|props| props.extension_name_as_c_str().ok())
            .collect();

        let unsupported_extensions: Vec<String> = desc
            .required_extensions
            .iter()
            .filter(|ext| !supported_extensions.contains(*ext))
            .map(|ext| ext.to_string_lossy().into_owned())
            .collect();

        if !unsupported_extensions.is_empty() {
            return Err(Error::UnsupportedExtensions(unsupported_extensions));
        }

        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };

        let layers = Self::additional_layers(&desc, &available_layers);
        let layers: Vec<*const c_char> = layers
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));

        let extension_ptrs: Vec<*const c_char> = desc
            .required_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect();

        let inst_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_ptrs)
            .enabled_layer_names(&layers);

        let instance = unsafe { entry.create_instance(&inst_info, None)? };
        tracing::info!("Vulkan instance created");

        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug = if desc.debugging {
            let loader = debug_utils::Instance::new(&entry, &instance);
            match unsafe { loader.create_debug_utils_messenger(&debug_info, None) } {
                Ok(messenger) => Some((loader, messenger)),
                Err(err) => {
                    tracing::warn!("Failed to create debug messenger: {err}");
                    None
                }
            }
        } else {
            None
        };

        Ok(Arc::new(Self {
            entry,
            raw: instance,
            debug,
        }))
    }

    fn additional_extensions(desc: &InstanceDesc) -> Vec<&'static CStr> {
        let mut names = vec![];
        if desc.debugging {
            names.push(debug_utils::NAME);
        }
        names
    }

    fn additional_layers(
        desc: &InstanceDesc,
        available_layers: &[vk::LayerProperties],
    ) -> Vec<&'static CStr> {
        const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

        let mut names = vec![];
        if desc.debugging {
            let found = available_layers.iter().any(|layer| {
                layer
                    .layer_name_as_c_str()
                    .is_ok_and(|name| name == VALIDATION_LAYER_NAME)
            });
            if found {
                names.push(VALIDATION_LAYER_NAME);
            } else {
                tracing::warn!(
                    "{VALIDATION_LAYER_NAME:?} layer not available; continuing without validation"
                );
            }
        }
        names
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.raw.destroy_instance(None);
        }
        tracing::info!("Vulkan instance destroyed");
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { &*p_callback_data };
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name) }.to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    let extended_message =
        format!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}");

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => tracing::error!("{extended_message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => tracing::warn!("{extended_message}"),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => tracing::info!("{extended_message}"),
        _ => {}
    }

    vk::FALSE
}
