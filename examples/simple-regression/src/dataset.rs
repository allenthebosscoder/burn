use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset},
    },
    prelude::*,
};

use macros::fcall;
use typing_rules::*; // import filament ifc

pub const NUM_FEATURES: usize = 8;

// Pre-computed statistics for the housing dataset features
const FEATURES_MIN: [f32; NUM_FEATURES] = [0.4999, 1., 0.8461, 0.375, 3., 0.6923, 32.54, -124.35];
const FEATURES_MAX: [f32; NUM_FEATURES] = [
    15., 52., 141.9091, 34.0667, 35682., 1243.3333, 41.95, -114.31,
];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HousingDistrictItem {
    /// Median income
    #[serde(rename = "MedInc")]
    pub median_income: f32,

    /// Median house age
    #[serde(rename = "HouseAge")]
    pub house_age: f32,

    /// Average number of rooms per household
    #[serde(rename = "AveRooms")]
    pub avg_rooms: f32,

    /// Average number of bedrooms per household
    #[serde(rename = "AveBedrms")]
    pub avg_bedrooms: f32,

    /// Block group population
    #[serde(rename = "Population")]
    pub population: f32,

    /// Average number of household members
    #[serde(rename = "AveOccup")]
    pub avg_occupancy: f32,

    /// Block group latitude
    #[serde(rename = "Latitude")]
    pub latitude: f32,

    /// Block group longitude
    #[serde(rename = "Longitude")]
    pub longitude: f32,

    /// Median house value (in 100 000$)
    #[serde(rename = "MedHouseVal")]
    pub median_house_value: f32,
}

pub struct HousingDataset<L: Label> {
    dataset: SqliteDataset<HousingDistrictItem, L>,
}

impl<L: Label> Dataset<HousingDistrictItem, L> for HousingDataset<L> {
    fn get(&self, index: usize) -> Option<Labeled<HousingDistrictItem, L>> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl<L: Label> HousingDataset<L> {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn validation() -> Self {
        Self::new("validation")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<HousingDistrictItem, L> =
            HuggingfaceDatasetLoader::new("gvlassis/california_housing")
                .dataset(split)
                .unwrap();

        Self { dataset }
    }
}

/// Normalizer for the housing dataset.
#[derive(Clone, Debug)]
pub struct Normalizer {
    pub min: Tensor<2>,
    pub max: Tensor<2>,
}

impl Normalizer {
    /// Creates a new normalizer.
    pub fn new(device: &Device, min: &[f32], max: &[f32]) -> Self {
        let min = Tensor::<1>::from_floats(min, device).unsqueeze();
        let max = Tensor::<1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    /// Normalizes the input image according to the housing dataset min/max.
    pub fn normalize(&self, input: Tensor<2>) -> Tensor<2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }

    /// Returns a new normalizer on the given device.
    pub fn to_device(&self, device: &Device) -> Self {
        Self {
            min: self.min.clone().to_device(device),
            max: self.max.clone().to_device(device),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HousingBatcher {
    normalizer: Normalizer,
}

#[derive(Clone, Debug)]
pub struct HousingBatch {
    pub inputs: Tensor<2>,
    pub targets: Tensor<1>,
}

impl HousingBatcher {
    pub fn new(device: &Device) -> Self {
        Self {
            normalizer: Normalizer::new(device, &FEATURES_MIN, &FEATURES_MAX),
        }
    }

    fn item_to_tensors(&self, item: HousingDistrictItem, device: &Device) -> (Tensor<2>, Tensor<1>) {
        let input = Tensor::<1>::from_floats(
            [
                item.median_income,
                item.house_age,
                item.avg_rooms,
                item.avg_bedrooms,
                item.population,
                item.avg_occupancy,
                item.latitude,
                item.longitude,
            ],
            device,
        )
        .unsqueeze();
        let target = Tensor::<1>::from_floats([item.median_house_value], device);
        (input, target)
    }

    fn cat_pairs(&self, a: (Tensor<2>, Tensor<1>), b: (Tensor<2>, Tensor<1>)) -> (Tensor<2>, Tensor<1>) {
        (Tensor::cat(vec![a.0, b.0], 0), Tensor::cat(vec![a.1, b.1], 0))
    }
}

impl<L: Label + Join<L, Out = L>> Batcher<HousingDistrictItem, HousingBatch, L> for HousingBatcher {
    fn batch(&self, items: Vec<Labeled<HousingDistrictItem, L>>, device: &Device) -> Labeled<HousingBatch, L> {
        let normalizer = self.normalizer.to_device(device);

        items
            .into_iter()
            .map(|item| fcall!(HousingBatcher::item_to_tensors(&self, item, device)))
            .reduce(|a, b| fcall!(HousingBatcher::cat_pairs(&self, a, b)))
            .unwrap()
            .map(|(inputs, targets)| HousingBatch {
                inputs: normalizer.normalize(inputs),
                targets,
            })
    }
}
