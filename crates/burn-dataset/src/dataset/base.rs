use std::sync::Arc;

use crate::DatasetIterator;

use typing_rules::*; // import filament ifc

/// The dataset trait defines a basic collection of items with a predefined size.
pub trait Dataset<I, L: Label>: Send + Sync {
    /// Gets the item at the given index.
    /// Data extracted should inherit label.
    fn get(&self, index: usize) -> Option<Labeled<I, L>>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    /// Iterator should be labeled
    fn iter(&self) -> DatasetIterator<'_, I, L>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}


/// Arc and Box wrappers of dataset. Label carries from dataset.
impl<D, I, L: Label> Dataset<I, L> for Arc<D>
where
    D: Dataset<I, L>,
{
    fn get(&self, index: usize) -> Option<Labeled<I, L>> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I, L: Label> Dataset<I, L> for Arc<dyn Dataset<I, L>> {
    fn get(&self, index: usize) -> Option<Labeled<I, L>> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<D, I, L: Label> Dataset<I, L> for Box<D>
where
    D: Dataset<I, L>,
{
    fn get(&self, index: usize) -> Option<Labeled<I, L>> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I, L: Label> Dataset<I, L> for Box<dyn Dataset<I, L>> {
    fn get(&self, index: usize) -> Option<Labeled<I, L>> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}
