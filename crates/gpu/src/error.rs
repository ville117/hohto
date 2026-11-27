use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load vulkan library: {0}")]
    LoadingError(#[from] ash::LoadingError),

    #[error("out of device memory")]
    OutOfDeviceMemory,

    #[error("out of host memory")]
    OutOfHostMemory,

    #[error("unsupported extensions: {}", .0.join(", "))]
    UnsupportedExtensions(Vec<String>),

    #[error("unsupported device feature: {0}")]
    UnsupportedFeature(String),

    #[error("no suitable queue family found with support for {0}")]
    NoSuitableQueueFamily(&'static str),

    #[error("device lost")]
    DeviceLost,

    #[error("surface lost")]
    SurfaceLost,

    #[error("no suitable physical device found")]
    NoSuitableDevice,

    #[error("unsupported layer: {0}")]
    UnsupportedLayer(String),

    #[error("invalid spir-v: {0}")]
    InvalidSpirv(String),

    #[error("invalid shader entry point {0:?}: contains a nul byte")]
    InvalidEntryPoint(String),

    #[error("invalid surface extent {width}x{height}: dimensions must be non-zero")]
    InvalidExtent { width: u32, height: u32 },

    #[error("internal error: {0}")]
    Internal(&'static str),

    #[error("initial data ({data} bytes) exceeds buffer size ({size} bytes)")]
    InitialDataTooLarge { data: usize, size: usize },

    #[error("failed to get window handle: {0}")]
    WindowHandle(#[from] raw_window_handle::HandleError),

    #[error("allocation failed: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),

    #[error("vulkan error: {0}")]
    Vulkan(vk::Result),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<vk::Result> for Error {
    fn from(result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Error::OutOfDeviceMemory,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => Error::OutOfHostMemory,
            vk::Result::ERROR_FEATURE_NOT_PRESENT => Error::UnsupportedFeature(result.to_string()),
            vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
            vk::Result::ERROR_SURFACE_LOST_KHR => Error::SurfaceLost,
            vk::Result::ERROR_LAYER_NOT_PRESENT => Error::UnsupportedLayer(result.to_string()),
            _ => Error::Vulkan(result),
        }
    }
}
