use crate::Dataset;
use std::marker::PhantomData;
use typing_rules::*; // import filament ifc

/// Basic mapper trait to be used with the [mapper dataset](MapperDataset).
pub trait Mapper<I, O>: Send + Sync {
    /// Maps an item of type I to an item of type O.
    fn map(&self, item: &I) -> O;
}

/// Dataset mapping each element in an inner dataset to another element type lazily.
#[derive(new)]
pub struct MapperDataset<D, M, I, L: Label> {
    dataset: D,
    mapper: M,
    input: PhantomData<Labeled<I, L>>,
}

impl<D, M, I, O, L: Label> Dataset<O, L> for MapperDataset<D, M, I, L>
where
    D: Dataset<I, L>,
    M: Mapper<I, O> + Send + Sync,
    I: Send + Sync,
    O: Send + Sync,
{
    fn get(&self, index: usize) -> Option<Labeled<O, L>> {
        let item = self.dataset.get(index);

        item.map(|item| {
            item.__map_ref(|inner| {
                self.mapper.map(inner)
            })
        })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InMemDataset, test_data};

    #[test]
    pub fn given_mapper_dataset_when_iterate_should_iterate_though_all_map_items() {
        struct StringToFirstChar;

        impl Mapper<String, String> for StringToFirstChar {
            fn map(&self, item: &String) -> String {
                // let mut item = item.clone();
                // item.truncate(1);
                // item
                item.__map_ref(|s| {
                    let mut out = s.clone();
                    out.truncate(1);
                    out
                })
            }
        }

        let items_original = test_data::string_items();
        let dataset = InMemDataset::new(items_original);
        let dataset = MapperDataset::new(dataset, StringToFirstChar);

        let items: Vec<String> = dataset.iter().collect();

        assert_eq!(vec!["1", "2", "3", "4"], items);
    }
}
