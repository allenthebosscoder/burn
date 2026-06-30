use burn_tensor::Device;
use typing_rules::*; // import filament ifc

/// A trait for batching items of type `I` into items of type `O`.
pub trait Batcher<I, O, L: Label>: Send + Sync {
    /// Batches the given items on the specified device.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to batch.
    /// * `device` - The backend device to use.
    ///
    /// # Returns
    ///
    /// The batched items.
    fn batch(&self, items: Vec<Labeled<I, L>>, device: &Device) -> Labeled<O, L>;
}

/// Test batcher
#[cfg(test)]
#[derive(new, Clone)]
pub struct TestBatcher;

#[cfg(test)]
impl<I, L: Label> Batcher<I, Vec<I>, L> for TestBatcher {
    fn batch(&self, items: Vec<Labeled<I, L>>, _device: &Device) -> Labeled<Vec<I>, L> {
        let items = items
            .into_iter()
            .map(declassify)
            .collect::<Vec<I>>();

        Labeled::<Vec<I>, L>::new(items)
    }
}
