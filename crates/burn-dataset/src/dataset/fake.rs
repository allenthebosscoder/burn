use crate::{Dataset, DatasetIterator, InMemDataset};
use fake::{Dummy, Fake, Faker};
use typing_rules::*; // import filament ifc

/// Dataset filled with fake items generated from the [fake](fake) crate.
pub struct FakeDataset<I, L: Label> {
    dataset: InMemDataset<I, L>,
}

impl<I: Dummy<Faker>, L: Label> FakeDataset<I, L> {
    /// Create a new fake dataset with the given size.
    pub fn new(size: usize) -> Self {
        let mut items = Vec::with_capacity(size);
        for _ in 0..size {
            items.push(Labeled::<_, L>::new(Faker.fake()));
        }
        let dataset = InMemDataset::new(items);

        Self { dataset }
    }
}

impl<I: Send + Sync + Clone, L:Label> Dataset<I, L> for FakeDataset<I, L> {
    fn iter(&self) -> DatasetIterator<'_, I, L> {
        DatasetIterator::new(self)
    }

    fn get(&self, index: usize) -> Option<Labeled<I, L>> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}
