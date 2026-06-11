use std::{cmp::max, marker::PhantomData, num::NonZeroUsize};

use crate::Dataset;

use typing_rules::*; // import filament ifc

/// Functionality to create a window.
pub trait Window<I, L: Label> {
    /// Creates a window of a collection.
    ///
    /// # Returns
    ///
    /// A `Vec<I>` representing the window.
    fn window(&self, current: usize, size: NonZeroUsize) -> Option<Labeled<Vec<I>, L>>;
}

impl<I, L: Label, T: Dataset<I, L> + ?Sized> Window<I, L> for T {
    fn window(&self, current: usize, size: NonZeroUsize) -> Option<Labeled<Vec<I>, L>> {
        let items: Option<Vec<Labeled<I, L>>> = 
            (current..current + size.get())
                .map(|x| self.get(x))
                .collect();
        items.map(|items| {
            Labeled::new(
                items
                    .into_iter()
                    .map(|item| item.__private_into_value())
                    .collect()
            )
        })
    }
}

/// Functionality to create a `WindowsIterator`.
pub trait Windows<I, L: Label> {
    /// Creates and returns an iterator over all the windows of length `size`.
    fn windows(&self, size: usize) -> WindowsIterator<'_, I, L>;
}

impl<I, L: Label, T: Dataset<I, L>> Windows<I, L> for T {
    /// Is empty if the `Dataset` is shorter than `size`.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.    
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::burn_dataset::{
    ///    transform::{Windows, WindowsDataset},
    ///    Dataset, InMemDataset,
    /// };
    ///
    /// let items = [1, 2, 3, 4].to_vec();
    /// let dataset = InMemDataset::new(items.clone());
    ///
    /// for window in dataset.windows(2) {
    ///  // do sth with window
    /// }
    /// ```
    fn windows(&self, size: usize) -> WindowsIterator<'_, I, L> {
        let size = NonZeroUsize::new(size).expect("window size must be non-zero");
        WindowsIterator::new(self, size)
    }
}

/// Overlapping windows iterator.
pub struct WindowsIterator<'a, I, L: Label> {
    /// The size of the windows.
    pub size: NonZeroUsize,
    current: usize,
    dataset: &'a dyn Dataset<I, L>,
}

impl<'a, I, L: Label> WindowsIterator<'a, I, L> {
    /// Creates a new `WindowsIterator` instance. The windows overlap.
    /// Is empty if the input `Dataset` is shorter than `size`.
    ///
    /// # Parameters
    ///
    /// - `dataset`: The dataset over which windows will be created.
    /// - `size`: The size of the windows.
    pub fn new(dataset: &'a dyn Dataset<I, L>, size: NonZeroUsize) -> Self {
        WindowsIterator {
            current: 0,
            dataset,
            size,
        }
    }
}

impl<I, L: Label> Iterator for WindowsIterator<'_, I, L> {
    type Item = Labeled<Vec<I>, L>;

    fn next(&mut self) -> Option<Labeled<Vec<I>, L>> {
        self.current += 1;
        self.dataset.window(self.current - 1, self.size)
    }
}

impl<I, L: Label> Clone for WindowsIterator<'_, I, L> {
    fn clone(&self) -> Self {
        WindowsIterator {
            size: self.size,
            dataset: self.dataset,
            current: self.current,
        }
    }
}

/// Dataset designed to work with overlapping windows of data.
pub struct WindowsDataset<D, I, L: Label> {
    /// The size of the windows.
    pub size: NonZeroUsize,
    dataset: D,
    input: PhantomData<Labeled<I, L>>,
}

impl<D, I, L: Label> WindowsDataset<D, I, L>
where
    D: Dataset<I, L>,
{
    /// Creates a new `WindowsDataset` instance. The windows overlap.
    /// Is empty if the input `Dataset` is shorter than `size`.
    ///
    /// # Parameters
    ///
    /// - `dataset`: The dataset over which windows will be created.
    /// - `size`: The size of the windows.
    pub fn new(dataset: D, size: usize) -> Self
    {
        let size = NonZeroUsize::new(size).expect("window size must be non-zero");
        WindowsDataset::<D, I, L> {
            size,
            dataset,
            input: PhantomData,
        }
    }
}

impl<D, I, L: Label> Dataset<Vec<I>, L> for WindowsDataset<D, I, L>
where
    D: Dataset<I, L>,
    I: Send + Sync,
{
    /// Retrieves a window of items from the dataset.
    ///
    /// # Parameters
    ///
    /// - `index`: The index of the window.
    ///
    /// # Returns
    ///
    /// A vector representing the window.
    fn get(&self, index: usize) -> Option<Labeled<Vec<I>, L>> {
        self.dataset.window(index, self.size)
    }

    /// Retrieves the number of windows in the dataset.
    ///
    /// # Returns
    ///
    /// A size representing the number of windows.
    fn len(&self) -> usize {
        let len = self.dataset.len() as isize - self.size.get() as isize + 1;
        max(len, 0) as usize
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{
        Dataset, InMemDataset,
        transform::{Windows, WindowsDataset},
    };

    #[rstest]
    pub fn windows_should_be_equal_to_vec_windows() {
        let items = [1, 2, 3, 4, 5].to_vec();
        let dataset = InMemDataset::new(items.clone());
        let expected = items
            .windows(3)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<i32>>>();

        let result = dataset.windows(3).collect::<Vec<Vec<i32>>>();

        assert_eq!(result, expected);
    }

    #[rstest]
    pub fn windows_dataset_should_be_equal_to_vec_windows() {
        let items = [1, 2, 3, 4, 5].to_vec();
        let dataset = InMemDataset::new(items.clone());
        let expected = items
            .windows(3)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<i32>>>();

        let result = WindowsDataset::new(dataset, 3)
            .iter()
            .collect::<Vec<Vec<i32>>>();

        assert_eq!(result, expected);
    }

    #[rstest]
    pub fn cloned_iterator_should_be_equal() {
        let items = [1, 2, 3, 4, 5].to_vec();
        let dataset = InMemDataset::new(items.clone());
        let original = dataset.windows(4);

        let cloned = original.clone();

        assert!(std::ptr::eq(cloned.dataset, original.dataset));
        assert_eq!(cloned.size, original.size);
        assert_eq!(cloned.current, original.current);
    }

    #[rstest]
    pub fn cloned_iterator_should_be_unaffected() {
        let items = [1, 2, 3, 4, 5].to_vec();
        let dataset = InMemDataset::new(items.clone());
        let mut original = dataset.windows(4);

        let cloned = original.clone();
        original.current = 2;

        assert_ne!(cloned.current, original.current);
    }

    #[rstest]
    #[should_panic(expected = "window size must be non-zero")]
    pub fn windows_should_panic() {
        let items = [1, 2].to_vec();
        let dataset = InMemDataset::new(items.clone());

        dataset.windows(0);
    }

    #[rstest]
    #[should_panic(expected = "window size must be non-zero")]
    pub fn new_window_dataset_should_panic() {
        let items = [1, 2].to_vec();
        let dataset = InMemDataset::new(items.clone());

        WindowsDataset::new(dataset, 0);
    }

    #[rstest]
    pub fn window_dataset_len_should_be_equal() {
        let dataset = InMemDataset::new([1, 2, 3, 4].to_vec());

        let result = WindowsDataset::new(dataset, 2).len();

        assert_eq!(result, 3);
    }

    #[rstest]
    pub fn window_iterator_should_be_empty() {
        let dataset = InMemDataset::new([1, 2].to_vec());
        let mut peekable = dataset.windows(4).peekable();

        let result = peekable.peek();

        assert_eq!(result, None);
    }

    #[rstest]
    pub fn window_dataset_len_should_be_zero() {
        let dataset = InMemDataset::new([1, 2].to_vec());

        let result = WindowsDataset::new(dataset, 4).len();

        assert_eq!(result, 0);
    }

    #[rstest]
    pub fn window_dataset_get_should_be_equal() {
        let dataset = InMemDataset::new([1, 2, 3, 4].to_vec());
        let expected = Some([1, 2, 3].to_vec());

        let result = WindowsDataset::new(dataset, 3).get(0);

        assert_eq!(result, expected);
    }

    #[rstest]
    pub fn window_dataset_get_should_be_none() {
        let dataset = InMemDataset::new([1, 2].to_vec());

        let result = WindowsDataset::new(dataset, 4).get(0);

        assert_eq!(result, None);
    }
}
