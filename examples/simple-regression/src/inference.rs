use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    prelude::*,
    store::ModuleRecord,
    tensor::Device,
};
use macros::fcall;
use rgb::RGB8;
use textplots::{Chart, ColorPlot, Shape};
use typing_rules::*; // import filament ifc

use crate::{
    dataset::{HousingBatch, HousingBatcher, HousingDataset, HousingDistrictItem},
    model::{RegressionModel, RegressionModelConfig},
};

fn run_forward(batch: HousingBatch, model: &RegressionModel) -> (Tensor<1>, Tensor<1>) {
    (model.forward(batch.inputs).squeeze_dim::<1>(1), batch.targets)
}

pub fn infer(artifact_dir: &str, device: impl Into<Device>) {
    let device = device.into();
    let record = ModuleRecord::load(format!("{artifact_dir}/model"))
        .expect("Trained model should exist; run train first");

    let model = RegressionModelConfig::new()
        .init(&device)
        .load_record(record);

    // Use a sample of 1000 items from the test split
    let dataset = HousingDataset::<Secret>::test();
    let items: Vec<Labeled<HousingDistrictItem, Secret>> = dataset.iter().take(1000).collect();

    let batcher = HousingBatcher::new(&device);
    let batch = batcher.batch(items, &device);

    let (labeled_predicted, labeled_targets) = fcall!(run_forward(batch, &model)).split();

    // Declassify at the public output boundary — predictions are being displayed
    let predicted = declassify(labeled_predicted).into_data();
    let expected = declassify(labeled_targets).into_data();

    let points = predicted
        .iter::<f32>()
        .zip(expected.iter::<f32>())
        .collect::<Vec<_>>();

    println!("Predicted vs. Expected Median House Value (in 100,000$)");
    Chart::new_with_y_range(120, 60, 0., 5., 0., 5.)
        .linecolorplot(
            &Shape::Points(&points),
            RGB8 {
                r: 255,
                g: 85,
                b: 85,
            },
        )
        .display();

    // Print a single numeric value as an example
    println!("Predicted {} Expected {}", points[0].0, points[0].1);
}
