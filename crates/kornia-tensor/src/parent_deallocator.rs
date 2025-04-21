/// A trait for deallocating memory of parent for tensor.
///
/// # Safety
///
/// The parent deallocator must be thread-safe.
pub trait ParentDeallocator {
    /// Deallocates the parent.
    fn dealloc(&self);
}
