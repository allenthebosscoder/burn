use crate::dataset::Dataset;
use std::iter::Iterator;

use typing_rules::*; // import filament ifc

/// Dataset iterator.
pub struct DatasetIterator<'a, I, L: Label> {
    current: usize,
    dataset: &'a dyn Dataset<I, L>,
}

impl<'a, I, L: Label> DatasetIterator<'a, I, L> {
    /// Creates a new dataset iterator.
    pub fn new<D>(dataset: &'a D) -> Self
    where
        D: Dataset<I, L>,
    {
        DatasetIterator {
            current: 0,
            dataset,
        }
    }
}

impl<I, L: Label> Iterator for DatasetIterator<'_, I, L> {
    type Item = Labeled<I, L>;

    fn next(&mut self) -> Option<Labeled<I, L>> {
        let item = self.dataset.get(self.current);
        self.current += 1;
        item
    }
}
