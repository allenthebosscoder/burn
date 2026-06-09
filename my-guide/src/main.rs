#![recursion_limit = "131"]
use burn::{data::dataset::Dataset, optim::AdamConfig, prelude::*};
use my_guide::{
    inference,
    model::ModelConfig,
    training::{self, TrainingConfig},
};

fn main() {
    // Create a default Wgpu-backed device.
    let device = Device::wgpu(DeviceKind::DefaultDevice);

    // Create model and print it
    // let model = ModelConfig::new(10, 512).init(&device);
    // println!("{model}");

    // All the training artifacts will be saved in this directory
    let artifact_dir = "artifacts";

    // Train the model
    // training::train(
    //     artifact_dir,
    //     TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    //     device.clone(),
    // );

    // Infer the model
    inference::infer(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(4365)
            .unwrap(),
    );
}