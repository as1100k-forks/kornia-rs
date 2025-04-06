use crate::ParentDeallocator;
use auto_impl::auto_impl;
use std::alloc;
use std::alloc::Layout;
use thiserror::Error;

/// An error type for tensor allocator operations.
#[derive(Debug, Error, PartialEq)]
pub enum TensorAllocatorError {
    /// An error occurred during memory allocation.
    #[error("Invalid tensor layout {0}")]
    LayoutError(core::alloc::LayoutError),

    /// An error occurred during memory allocation.
    #[error("Null pointer")]
    NullPointer,
}

/// A trait for allocating and deallocating memory for tensors.
///
/// # Safety
///
/// The tensor allocator must be thread-safe.
///
/// # Methods
///
/// * `alloc` - Allocates memory for a tensor with the given layout.
/// * `dealloc` - Deallocates memory for a tensor with the given layout.
#[auto_impl(&mut, Box)]
pub trait TensorAllocator: Clone {
    /// Allocates memory for a tensor with the given layout.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError>;

    /// Deallocates memory for a tensor with the given layout.
    fn dealloc<PD: ParentDeallocator>(
        &self,
        ptr: *mut u8,
        parent_ptr: Option<(Option<*mut ()>, PD)>,
        layout: Layout,
    );
}

#[derive(Clone)]
/// A tensor allocator that uses the system allocator.
pub struct CpuAllocator;

/// Implement the `Default` trait for the `CpuAllocator` struct.
impl Default for CpuAllocator {
    fn default() -> Self {
        Self
    }
}

/// Implement the `TensorAllocator` trait for the `CpuAllocator` struct.
impl TensorAllocator for CpuAllocator {
    /// Allocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `layout` - The layout of the tensor.
    ///
    /// # Returns
    ///
    /// A non-null pointer to the allocated memory if successful, otherwise an error.
    fn alloc(&self, layout: Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)?
        }
        Ok(ptr)
    }

    /// Deallocates memory for a tensor with the given layout.
    ///
    /// # Arguments
    ///
    /// * `ptr` - A non-null pointer to the allocated memory.
    /// * `layout` - The layout of the tensor.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the layout must be correct.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn dealloc<PD: ParentDeallocator>(
        &self,
        ptr: *mut u8,
        parent_ptr: Option<(Option<*mut ()>, PD)>,
        layout: Layout,
    ) {
        if let Some((parent_ptr, mut parent_deallocator)) = parent_ptr {
            if !ptr.is_null() {
                parent_deallocator.dealloc(parent_ptr);
            }
        } else {
            if !ptr.is_null() {
                unsafe {
                    alloc::dealloc(ptr, layout);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct DefaultParentDeallocator;

    impl ParentDeallocator for DefaultParentDeallocator {
        fn dealloc(&mut self, _parent_ptr: Option<*mut ()>) {
            // Do nothing, just a placeholder type
        }
    }

    #[test]
    fn test_cpu_allocator() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let layout = Layout::from_size_align(1024, 64).unwrap();
        let ptr = allocator.alloc(layout)?;
        allocator.dealloc::<DefaultParentDeallocator>(ptr, None, layout);
        Ok(())
    }
}
