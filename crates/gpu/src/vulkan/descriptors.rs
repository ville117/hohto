use std::{collections::HashMap, sync::Arc};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use parking_lot::{Mutex, MutexGuard};

use crate::{CommandBuffer, Device, Error, Image, Result, aspect_mask};

const BINDING_TEXTURES: u32 = 0;
const BINDING_SAMPLERS: u32 = 1;
const BINDING_STORAGE: u32 = 2;
const BINDING_COUNT: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct ViewDesc {
    pub view_type: vk::ImageViewType,
    pub format: Option<vk::Format>,
    pub aspect: Option<vk::ImageAspectFlags>,
    pub base_mip: u32,
    pub mip_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
    pub sampled_layout: vk::ImageLayout,
}

impl Default for ViewDesc {
    fn default() -> Self {
        Self {
            view_type: vk::ImageViewType::TYPE_2D,
            format: None,
            aspect: None,
            base_mip: 0,
            mip_count: vk::REMAINING_MIP_LEVELS,
            base_layer: 0,
            layer_count: 1,
            sampled_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}

impl ViewDesc {
    pub fn depth() -> Self {
        Self {
            aspect: Some(vk::ImageAspectFlags::DEPTH),
            sampled_layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerDesc {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode: vk::SamplerAddressMode,
    pub anisotropy: u8,
    pub border_color: vk::BorderColor,
    pub compare_op: Option<vk::CompareOp>,
}

impl SamplerDesc {
    pub const LINEAR_REPEAT: Self = Self {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode: vk::SamplerAddressMode::REPEAT,
        anisotropy: 0,
        border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        compare_op: None,
    };

    pub const LINEAR_CLAMP: Self = Self {
        address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ..Self::LINEAR_REPEAT
    };

    pub const NEAREST_CLAMP: Self = Self {
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ..Self::LINEAR_REPEAT
    };
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Pod, Zeroable)]
pub struct TextureId(u32);

impl TextureId {
    pub const NONE: Self = Self(0);

    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Pod, Zeroable)]
pub struct SamplerId(u32);

impl SamplerId {
    pub const DEFAULT: Self = Self(0);

    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Pod, Zeroable)]
pub struct StorageId(u32);

impl StorageId {
    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    pub textures: u32,
    pub storage: u32,
    pub samplers: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct BindlessLayoutDesc {
    pub max_textures: u32,
    pub max_storage: u32,
    pub max_samplers: u32,
    pub push_constant_size: u32,
}

impl Default for BindlessLayoutDesc {
    fn default() -> Self {
        Self {
            max_textures: 65_536,
            max_storage: 8_192,
            max_samplers: 1_024,
            push_constant_size: 8,
        }
    }
}

struct LayoutInner {
    device: Device,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    push_constant_size: u32,
    capacity: Capacity,
}

impl Drop for LayoutInner {
    fn drop(&mut self) {
        unsafe {
            let raw_device = &self.device.raw;
            raw_device.destroy_pipeline_layout(self.pipeline_layout, None);
            raw_device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[derive(Clone)]
pub struct BindlessLayout(Arc<LayoutInner>);

impl BindlessLayout {
    pub fn new(device: &Device, desc: &BindlessLayoutDesc) -> Result<Self> {
        debug_assert!(
            desc.push_constant_size.is_multiple_of(4),
            "push constant size must be a multiple of 4"
        );

        require_device_support(device)?;
        let limits = MaxDeviceLimits::query(device);

        if desc.push_constant_size > limits.push_constants_size {
            return Err(Error::Internal(
                "push constant block exceeds maxPushConstantsSize",
            ));
        }
        let capacity = limits.clamp(desc);
        if capacity.textures == 0 || capacity.samplers == 0 {
            return Err(Error::Internal("device cannot back a bindless table"));
        }

        let mut inner = LayoutInner {
            device: device.clone(),
            set_layout: vk::DescriptorSetLayout::default(),
            pipeline_layout: vk::PipelineLayout::default(),
            push_constant_size: desc.push_constant_size,
            capacity,
        };
        let raw_device = &device.raw;

        let bindings = [
            (
                BINDING_TEXTURES,
                vk::DescriptorType::SAMPLED_IMAGE,
                capacity.textures,
            ),
            (
                BINDING_SAMPLERS,
                vk::DescriptorType::SAMPLER,
                capacity.samplers,
            ),
            (
                BINDING_STORAGE,
                vk::DescriptorType::STORAGE_IMAGE,
                capacity.storage,
            ),
        ]
        .map(|(binding, descriptor_type, count)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(count)
                .stage_flags(vk::ShaderStageFlags::ALL)
        });

        let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
            BINDING_COUNT];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        inner.set_layout = unsafe {
            raw_device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .bindings(&bindings)
                    .push_next(&mut flags_info),
                None,
            )?
        };

        let set_layouts = [inner.set_layout];
        let push_range_array = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(desc.push_constant_size)];
        let push_ranges: &[_] = if desc.push_constant_size > 0 {
            &push_range_array
        } else {
            &[]
        };
        inner.pipeline_layout = unsafe {
            raw_device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(push_ranges),
                None,
            )?
        };

        Ok(Self(Arc::new(inner)))
    }

    #[allow(clippy::must_use_candidate)]
    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.0.set_layout
    }

    #[allow(clippy::must_use_candidate)]
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.0.pipeline_layout
    }

    pub fn push_constant_range(&self) -> vk::PushConstantRange {
        vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(self.0.push_constant_size)
    }

    #[must_use]
    pub fn push_constant_size(&self) -> u32 {
        self.0.push_constant_size
    }

    #[must_use]
    pub fn capacity(&self) -> Capacity {
        self.0.capacity
    }
}
struct MaxDeviceLimits {
    push_constants_size: u32,
    per_stage_resources: u32,
    set_sampled_images: u32,
    set_storage_images: u32,
    set_samplers: u32,
    stage_sampled_images: u32,
    stage_storage_images: u32,
    stage_samplers: u32,
}

impl MaxDeviceLimits {
    fn query(device: &Device) -> Self {
        let instance = &device.physical_device.instance.raw;
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut vulkan12);
        unsafe {
            instance.get_physical_device_properties2(device.physical_device.raw, &mut properties2);
        }
        let push_constants_size = properties2.properties.limits.max_push_constants_size;

        Self {
            push_constants_size,
            per_stage_resources: vulkan12.max_per_stage_update_after_bind_resources,
            set_sampled_images: vulkan12.max_descriptor_set_update_after_bind_sampled_images,
            set_storage_images: vulkan12.max_descriptor_set_update_after_bind_storage_images,
            set_samplers: vulkan12.max_descriptor_set_update_after_bind_samplers,
            stage_sampled_images: vulkan12
                .max_per_stage_descriptor_update_after_bind_sampled_images,
            stage_storage_images: vulkan12
                .max_per_stage_descriptor_update_after_bind_storage_images,
            stage_samplers: vulkan12.max_per_stage_descriptor_update_after_bind_samplers,
        }
    }

    fn clamp(&self, desc: &BindlessLayoutDesc) -> Capacity {
        let image_budget = self.per_stage_resources / 2;
        Capacity {
            textures: desc
                .max_textures
                .min(self.set_sampled_images)
                .min(self.stage_sampled_images)
                .min(image_budget),
            storage: desc
                .max_storage
                .min(self.set_storage_images)
                .min(self.stage_storage_images)
                .min(image_budget),
            samplers: desc
                .max_samplers
                .min(self.set_samplers)
                .min(self.stage_samplers),
        }
    }
}

fn require_device_support(device: &Device) -> Result<()> {
    let instance = &device.physical_device.instance.raw;
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    {
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features12);
        unsafe {
            instance.get_physical_device_features2(device.physical_device.raw, &mut features2);
        }
    }
    let required = [
        (
            features12.runtime_descriptor_array,
            "missing device feature: runtimeDescriptorArray",
        ),
        (
            features12.descriptor_binding_partially_bound,
            "missing device feature: descriptorBindingPartiallyBound",
        ),
        (
            features12.descriptor_binding_update_unused_while_pending,
            "missing device feature: descriptorBindingUpdateUnusedWhilePending",
        ),
        (
            features12.descriptor_binding_sampled_image_update_after_bind,
            "missing device feature: descriptorBindingSampledImageUpdateAfterBind",
        ),
        (
            features12.descriptor_binding_storage_image_update_after_bind,
            "missing device feature: descriptorBindingStorageImageUpdateAfterBind",
        ),
        (
            features12.shader_sampled_image_array_non_uniform_indexing,
            "missing device feature: shaderSampledImageArrayNonUniformIndexing",
        ),
        (
            features12.shader_storage_image_array_non_uniform_indexing,
            "missing device feature: shaderStorageImageArrayNonUniformIndexing",
        ),
    ];
    match required
        .iter()
        .find(|(supported, _)| *supported == vk::FALSE)
    {
        None => Ok(()),
        Some(&(_, message)) => Err(Error::Internal(message)),
    }
}

#[derive(Clone, Copy, Debug)]
enum SlotKind {
    Texture,
    Storage,
}

impl SlotKind {
    fn table_full_error(self) -> Error {
        Error::Internal(match self {
            Self::Texture => "texture table is full",
            Self::Storage => "storage image table is full",
        })
    }
}

struct Slot {
    table: Arc<TableInner>,
    kind: SlotKind,
    index: u32,
    view: vk::ImageView,
}

impl Drop for Slot {
    fn drop(&mut self) {
        let mut state = self.table.state.lock();
        let retired_in_frame = state.current_frame;
        state.pending.push(RetiredSlot {
            retired_in_frame,
            kind: self.kind,
            index: self.index,
            view: self.view,
        });
    }
}

#[must_use = "dropping the handle releases the texture slot"]
pub struct TextureHandle(Slot);

impl TextureHandle {
    #[must_use]
    pub fn id(&self) -> TextureId {
        TextureId(self.0.index)
    }
}

#[must_use = "dropping the handle releases the storage slot"]
pub struct StorageHandle(Slot);

impl StorageHandle {
    #[must_use]
    pub fn id(&self) -> StorageId {
        StorageId(self.0.index)
    }
}

struct SlotAllocator {
    next_unused: u32,
    capacity: u32,
    free_list: Vec<u32>,
}

impl SlotAllocator {
    fn new(first_slot: u32, capacity: u32) -> Self {
        Self {
            next_unused: first_slot,
            capacity,
            free_list: Vec::new(),
        }
    }

    fn allocate(&mut self) -> Option<u32> {
        if let Some(index) = self.free_list.pop() {
            return Some(index);
        }
        if self.next_unused < self.capacity {
            let index = self.next_unused;
            self.next_unused += 1;
            return Some(index);
        }
        None
    }
}

struct RetiredSlot {
    retired_in_frame: u64,
    kind: SlotKind,
    index: u32,
    view: vk::ImageView,
}

struct TableState {
    textures: SlotAllocator,
    storage: SlotAllocator,

    sampler_cache: HashMap<SamplerDesc, u32>,
    sampler_objects: Vec<vk::Sampler>,

    fallback_view: vk::ImageView,
    pending: Vec<RetiredSlot>,
    current_frame: u64,
}

struct TableInner {
    layout: BindlessLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    state: Mutex<TableState>,
}

impl TableInner {
    fn raw_device(&self) -> &ash::Device {
        &self.layout.0.device.raw
    }

    fn write_image_descriptor(
        &self,
        _guard: &MutexGuard<'_, TableState>,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        index: u32,
        image_info: vk::DescriptorImageInfo,
    ) {
        let image_infos = [image_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .dst_array_element(index)
            .descriptor_type(descriptor_type)
            .image_info(&image_infos);
        unsafe { self.raw_device().update_descriptor_sets(&[write], &[]) };
    }

    fn write_texture(
        &self,
        guard: &MutexGuard<'_, TableState>,
        index: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        self.write_image_descriptor(
            guard,
            BINDING_TEXTURES,
            vk::DescriptorType::SAMPLED_IMAGE,
            index,
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(layout),
        );
    }

    fn write_storage(&self, guard: &MutexGuard<'_, TableState>, index: u32, view: vk::ImageView) {
        self.write_image_descriptor(
            guard,
            BINDING_STORAGE,
            vk::DescriptorType::STORAGE_IMAGE,
            index,
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL),
        );
    }

    fn write_sampler(&self, guard: &MutexGuard<'_, TableState>, index: u32, sampler: vk::Sampler) {
        self.write_image_descriptor(
            guard,
            BINDING_SAMPLERS,
            vk::DescriptorType::SAMPLER,
            index,
            vk::DescriptorImageInfo::default().sampler(sampler),
        );
    }

    fn create_view(&self, image: &Image, desc: &ViewDesc) -> Result<vk::ImageView> {
        let format = desc.format.unwrap_or(image.desc.format);
        let aspect = desc.aspect.unwrap_or_else(|| {
            let full_aspect = aspect_mask(format);
            if full_aspect.contains(vk::ImageAspectFlags::DEPTH) {
                vk::ImageAspectFlags::DEPTH
            } else {
                full_aspect
            }
        });
        let info = vk::ImageViewCreateInfo::default()
            .image(image.raw)
            .view_type(desc.view_type)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: desc.base_mip,
                level_count: desc.mip_count,
                base_array_layer: desc.base_layer,
                layer_count: desc.layer_count,
            });
        Ok(unsafe { self.raw_device().create_image_view(&info, None)? })
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> Result<vk::Sampler> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(desc.mag_filter)
            .min_filter(desc.min_filter)
            .mipmap_mode(desc.mipmap_mode)
            .address_mode_u(desc.address_mode)
            .address_mode_v(desc.address_mode)
            .address_mode_w(desc.address_mode)
            .anisotropy_enable(desc.anisotropy > 0)
            .max_anisotropy(f32::from(desc.anisotropy.max(1)))
            .compare_enable(desc.compare_op.is_some())
            .compare_op(desc.compare_op.unwrap_or(vk::CompareOp::ALWAYS))
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .border_color(desc.border_color);
        Ok(unsafe { self.raw_device().create_sampler(&info, None)? })
    }
}

impl Drop for TableInner {
    fn drop(&mut self) {
        unsafe {
            let raw_device = self.raw_device().clone();
            let _ = raw_device.device_wait_idle();

            let state = self.state.get_mut();
            for retired in state.pending.drain(..) {
                raw_device.destroy_image_view(retired.view, None);
            }
            for &sampler in &state.sampler_objects {
                raw_device.destroy_sampler(sampler, None);
            }
            raw_device.destroy_image_view(state.fallback_view, None);

            raw_device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

#[derive(Clone)]
pub struct DescriptorTable(Arc<TableInner>);

impl DescriptorTable {
    pub fn new(layout: &BindlessLayout, fallback: &Image) -> Result<Self> {
        let capacity = layout.capacity();

        let mut inner = TableInner {
            layout: layout.clone(),
            pool: vk::DescriptorPool::default(),
            set: vk::DescriptorSet::default(),
            state: Mutex::new(TableState {
                textures: SlotAllocator::new(1, capacity.textures), // slot 0 = fallback
                storage: SlotAllocator::new(0, capacity.storage),
                sampler_cache: HashMap::new(),
                sampler_objects: Vec::new(),
                fallback_view: vk::ImageView::default(),
                pending: Vec::new(),
                current_frame: 1,
            }),
        };

        let pool_sizes = [
            (vk::DescriptorType::SAMPLED_IMAGE, capacity.textures),
            (vk::DescriptorType::SAMPLER, capacity.samplers),
            (vk::DescriptorType::STORAGE_IMAGE, capacity.storage),
        ]
        .map(
            |(descriptor_type, descriptor_count)| vk::DescriptorPoolSize {
                ty: descriptor_type,
                descriptor_count,
            },
        );

        let raw_device = &layout.0.device.raw;
        inner.pool = unsafe {
            raw_device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        let set_layouts = [layout.set_layout()];
        inner.set = unsafe {
            raw_device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(inner.pool)
                    .set_layouts(&set_layouts),
            )?[0]
        };

        let table = Self(Arc::new(inner));

        let default_sampler = table.sampler(&SamplerDesc::LINEAR_REPEAT)?;
        debug_assert_eq!(default_sampler, SamplerId::DEFAULT);

        let fallback_view = table.0.create_view(fallback, &ViewDesc::default())?;
        {
            let mut state = table.0.state.lock();
            state.fallback_view = fallback_view;
            table.0.write_texture(
                &state,
                TextureId::NONE.index(),
                fallback_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }

        Ok(table)
    }

    #[must_use]
    pub fn layout(&self) -> &BindlessLayout {
        &self.0.layout
    }

    pub fn create_texture(&self, image: &Image, view_desc: &ViewDesc) -> Result<TextureHandle> {
        self.alloc_slot(SlotKind::Texture, image, view_desc)
            .map(TextureHandle)
    }

    pub fn create_storage(&self, image: &Image, view_desc: &ViewDesc) -> Result<StorageHandle> {
        self.alloc_slot(SlotKind::Storage, image, view_desc)
            .map(StorageHandle)
    }

    fn alloc_slot(&self, kind: SlotKind, image: &Image, view_desc: &ViewDesc) -> Result<Slot> {
        let view = self.0.create_view(image, view_desc)?;
        let mut state = self.0.state.lock();

        let allocator = match kind {
            SlotKind::Texture => &mut state.textures,
            SlotKind::Storage => &mut state.storage,
        };
        let Some(index) = allocator.allocate() else {
            drop(state);
            unsafe { self.0.raw_device().destroy_image_view(view, None) };
            return Err(kind.table_full_error());
        };

        match kind {
            SlotKind::Texture => {
                self.0
                    .write_texture(&state, index, view, view_desc.sampled_layout);
            }
            SlotKind::Storage => self.0.write_storage(&state, index, view),
        }
        Ok(Slot {
            table: self.0.clone(),
            kind,
            index,
            view,
        })
    }

    pub fn sampler(&self, desc: &SamplerDesc) -> Result<SamplerId> {
        const TABLE_FULL: Error = Error::Internal("sampler table is full");

        let mut state = self.0.state.lock();
        if let Some(&index) = state.sampler_cache.get(desc) {
            return Ok(SamplerId(index));
        }
        let index = u32::try_from(state.sampler_objects.len()).map_err(|_| TABLE_FULL)?;
        if index >= self.0.layout.capacity().samplers {
            return Err(TABLE_FULL);
        }
        let sampler = self.0.create_sampler(desc)?;
        state.sampler_objects.push(sampler);
        state.sampler_cache.insert(*desc, index);
        self.0.write_sampler(&state, index, sampler);
        Ok(SamplerId(index))
    }

    pub fn begin_frame(&self, current: u64, completed: u64) {
        debug_assert!(current >= 1, "frame numbering starts at 1");
        debug_assert!(completed < current, "GPU cannot be ahead of the CPU");

        let mut state = self.0.state.lock();
        state.current_frame = current;

        let (released, still_pending): (Vec<_>, Vec<_>) = std::mem::take(&mut state.pending)
            .into_iter()
            .partition(|retired| retired.retired_in_frame <= completed);
        state.pending = still_pending;

        for retired in released {
            match retired.kind {
                SlotKind::Texture => {
                    self.0.write_texture(
                        &state,
                        retired.index,
                        state.fallback_view,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    );
                    state.textures.free_list.push(retired.index);
                }
                SlotKind::Storage => state.storage.free_list.push(retired.index),
            }
            unsafe { self.0.raw_device().destroy_image_view(retired.view, None) };
        }
    }
}

impl CommandBuffer {
    pub fn bind_descriptors(&self, table: &DescriptorTable, bind_point: vk::PipelineBindPoint) {
        unsafe {
            self.device.raw.cmd_bind_descriptor_sets(
                self.raw,
                bind_point,
                table.layout().pipeline_layout(),
                0,
                &[table.0.set],
                &[],
            );
        }
    }

    pub fn push_constants<Constants: Pod>(&self, layout: &BindlessLayout, value: &Constants) {
        debug_assert!(
            u32::try_from(size_of::<Constants>())
                .is_ok_and(|size| size <= layout.push_constant_size()),
            "push constants exceed BindlessLayoutDesc::push_constant_size"
        );
        unsafe {
            self.device.raw.cmd_push_constants(
                self.raw,
                layout.pipeline_layout(),
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(value),
            );
        }
    }
}
