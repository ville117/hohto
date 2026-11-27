use crate::error::{Error, Result};
use crate::vulkan::device::Device;
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

const DEFAULT_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
);

pub struct Buffer {
    pub raw: vk::Buffer,
    pub size: u64,
    pub address: u64,
    allocation: Option<Allocation>,
    device: Device,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { self.device.raw.destroy_buffer(self.raw, None) };
        if let Some(allocation) = self.allocation.take() {
            let _ = self.device.allocator.lock().free(allocation);
        }
    }
}

impl Buffer {
    pub fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        self.allocation
            .as_mut()
            .and_then(|alloc| alloc.mapped_slice_mut())
    }
}

impl Device {
    fn create_buffer_raw(
        &self,
        name: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Buffer> {
        if size == 0 {
            return Err(Error::Internal("cannot create an empty buffer"));
        }

        let buffer_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { self.raw.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { self.raw.get_buffer_memory_requirements(buffer) };

        let allocation = match self.allocator.lock().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
        }) {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { self.raw.destroy_buffer(buffer, None) };
                return Err(err.into());
            }
        };

        unsafe {
            self.raw
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        let address = unsafe {
            self.raw
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        Ok(Buffer {
            raw: buffer,
            size,
            address,
            allocation: Some(allocation),
            device: self.clone(),
        })
    }

    fn create_readback_buffer(&self, name: &str, size: u64) -> Result<Buffer> {
        self.create_buffer_raw(
            name,
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )
    }

    pub fn create_buffer(&self, name: &str, size: u64, location: MemoryLocation) -> Result<Buffer> {
        let buffer = self.create_buffer_raw(name, size, DEFAULT_USAGE, location)?;
        Ok(buffer)
    }

    pub fn create_buffer_with<T: bytemuck::Pod>(
        &self,
        name: &str,
        data: &[T],
        location: MemoryLocation,
    ) -> Result<Buffer> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let mut buffer = self.create_buffer(name, bytes.len() as u64, location)?;

        if let Some(mapped) = buffer.mapped_slice_mut() {
            mapped[..bytes.len()].copy_from_slice(bytes);
            return Ok(buffer);
        }
        self.upload_buffer(&buffer, 0, data)?;

        Ok(buffer)
    }

    pub fn readback_buffer<T: bytemuck::Pod>(
        &self,
        src: &Buffer,
        offset: u64,
        output: &mut [T],
    ) -> Result<()> {
        let bytes = bytemuck::cast_slice_mut(output);
        let size = bytes.len() as u64;

        if offset.checked_add(size).is_none_or(|end| end > src.size) {
            return Err(Error::Internal("buffer readback out of bounds"));
        }

        let mut readback = self.create_readback_buffer("readback", size)?;

        self.immediate_submit(|cmd| {
            let regions = [vk::BufferCopy2::default()
                .src_offset(offset)
                .dst_offset(0)
                .size(size)];

            let copy_info = vk::CopyBufferInfo2::default()
                .src_buffer(src.raw)
                .dst_buffer(readback.raw)
                .regions(&regions);

            unsafe {
                self.raw.cmd_copy_buffer2(cmd.raw, &copy_info);
            }

            Ok(())
        })?;

        let mapped = readback
            .mapped_slice_mut()
            .ok_or(Error::Internal("readback buffer is not mapped"))?;

        bytes.copy_from_slice(&mapped[..bytes.len()]);

        Ok(())
    }

    pub fn upload_buffer<T: bytemuck::Pod>(
        &self,
        dst: &Buffer,
        offset: u64,
        data: &[T],
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let bytes: &[u8] = bytemuck::cast_slice(data);

        if offset
            .checked_add(bytes.len() as u64)
            .is_none_or(|end| end > dst.size)
        {
            return Err(Error::Internal("buffer upload out of bounds"));
        }

        let mut staging_buffer = self.create_buffer_raw(
            "initial data",
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;

        let mapped = staging_buffer
            .mapped_slice_mut()
            .ok_or(Error::Internal("staging buffer is not mapped"))?;
        mapped[..bytes.len()].copy_from_slice(bytes);

        tracing::info!("uploaded {} bytes into staging buffer", bytes.len());

        self.immediate_submit(|cmd| {
            let regions = [vk::BufferCopy2::default()
                .src_offset(0)
                .dst_offset(offset)
                .size(bytes.len() as u64)];

            let copy_info = vk::CopyBufferInfo2::default()
                .src_buffer(staging_buffer.raw)
                .dst_buffer(dst.raw)
                .regions(&regions);

            unsafe { self.raw.cmd_copy_buffer2(cmd.raw, &copy_info) };
            Ok(())
        })?;

        Ok(())
    }
}
