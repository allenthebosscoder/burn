use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use serde::de::DeserializeOwned;

use crate::Dataset;

use typing_rules::*; // import filament ifc

/// Dataset where all items are stored in ram.
pub struct InMemDataset<I, L: Label> {
    items: Vec<Labeled<I, L>>,
}

impl<I, L: Label> InMemDataset<I, L> {
    /// Creates a new in memory dataset from the given labeled items.
    pub fn new(items: Vec<Labeled<I, L>>) -> Self {
        InMemDataset { items }
    }
}

impl<I, L> Dataset<I, L> for InMemDataset<I, L>
where
    I: Clone + Send + Sync,
    L: Label,
{
    fn get(&self, index: usize) -> Option<Labeled<I, L>> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }

}

impl<I, L> InMemDataset<I, L>
where
    I: Clone + Send + Sync + DeserializeOwned,
    L: Label,
{
    /// Create from a dataset. All items are loaded in memory.
    pub fn from_dataset(dataset: &impl Dataset<I, L>) -> Self {
        let len = dataset.len();
        let mut items = Vec::with_capacity(len);

        for index in 0..len {
            if let Some(item) = dataset.get(index) {
                items.push(item);
            }
        }

        Self::new(items)
    }

    /// Create from a json rows file (one json per line).
    ///
    /// [Supported field types](https://docs.rs/serde_json/latest/serde_json/value/enum.Value.html)
    pub fn from_json_rows<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut items = Vec::new();

        for line in reader.lines() {
            let item = serde_json::from_str(line?.as_str()).unwrap();
            items.push(Labeled::<I, L>::new(item));
        }

        Ok(Self::new(items))
    }

    /// Create from a csv file.
    ///
    /// The provided `csv::ReaderBuilder` can be configured to fit your csv format.
    ///
    /// The supported field types are: String, integer, float, and bool.
    ///
    /// See:
    /// - [Reading with Serde](https://docs.rs/csv/latest/csv/tutorial/index.html#reading-with-serde)
    /// - [Delimiters, quotes and variable length records](https://docs.rs/csv/latest/csv/tutorial/index.html#delimiters-quotes-and-variable-length-records)
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        builder: &csv::ReaderBuilder,
    ) -> Result<Self, std::io::Error> {
        let mut rdr = builder.from_path(path)?;
        let mut items = Vec::new();

        for result in rdr.deserialize() {
            let item: I = result?;
            items.push(Labeled::<I, L>::new(item));
        }

        Ok(Self::new(items))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_data, SqliteDataset};

    use rstest::{fixture, rstest};
    use serde::{Deserialize, Serialize};

    const DB_FILE: &str = "tests/data/sqlite-dataset.db";
    const JSON_FILE: &str = "tests/data/dataset.json";
    const CSV_FILE: &str = "tests/data/dataset.csv";
    const CSV_FMT_FILE: &str = "tests/data/dataset-fmt.csv";

    type SqlDs = SqliteDataset<Sample, A>;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Sample {
        column_str: String,
        column_bytes: Vec<u8>,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct SampleCsv {
        column_str: String,
        column_int: i64,
        column_bool: bool,
        column_float: f64,
    }

    #[fixture]
    fn train_dataset() -> SqlDs {
        SqliteDataset::from_db_file(DB_FILE, "train").unwrap()
    }

    #[rstest]
    pub fn from_dataset(train_dataset: SqlDs) {
        let dataset = InMemDataset::from_dataset(&train_dataset);

        let non_existing_record_index: usize = 10;
        let record_index: usize = 0;

        assert!(train_dataset.get(non_existing_record_index).is_none());
        assert_eq!(declassify(dataset.get(record_index).unwrap()).column_str, "HI1");
    }

    #[test]
    pub fn from_json_rows() {
        let dataset = InMemDataset::<Sample, A>::from_json_rows(JSON_FILE).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        assert!(dataset.get(non_existing_record_index).is_none());

        let item = declassify(dataset.get(record_index).unwrap());
        assert_eq!(item.column_str, "HI2");
        assert!(!item.column_bool);
    }

    #[test]
    pub fn from_csv_rows() {
        let rdr = csv::ReaderBuilder::new();
        let dataset = InMemDataset::<SampleCsv, A>::from_csv(CSV_FILE, &rdr).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        let item = dataset.get(non_existing_record_index);

        assert!(item.is_none());

        let item = declassify(dataset.get(record_index).unwrap());
        assert_eq!(item.column_str, "HI2");
        assert_eq!(item.column_int, 1);
        assert!(!item.column_bool);
        assert_eq!(item.column_float, 1.0);
    }

    #[test]
    pub fn from_csv_rows_fmt() {
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b' ').has_headers(false);
        let dataset = InMemDataset::<SampleCsv, A>::from_csv(CSV_FMT_FILE, rdr).unwrap();

        let non_existing_record_index: usize = 10;
        let record_index: usize = 1;

        assert!(dataset.get(non_existing_record_index).is_none());
        let item = declassify(dataset.get(record_index).unwrap());
        assert_eq!(item.column_str, "HI2");
        assert_eq!(item.column_int, 1);
        assert!(!item.column_bool);
        assert_eq!(item.column_float, 1.0);
    }

    #[test]
    pub fn given_in_memory_dataset_when_iterate_should_iterate_though_all_items() {
        let items_original = test_data::string_items();
        let labeled_items = items_original
            .iter()
            .cloned()
            .map(Labeled::<String, A>::new)
            .collect::<Vec<_>>();

        let dataset = InMemDataset::<String, A>::new(labeled_items);

        let items: Vec<String> = dataset
            .iter()
            .map(declassify)
            .collect();

        assert_eq!(items_original, items);
    }
}