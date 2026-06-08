use crate::{data::MnistBatcher, training::TrainingConfig};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer(artifact_dir: &str, device: impl Into<Device>, item: MnistItem) {
    let device = device.into();                                                 // convert device into real Burn device
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))    // load config
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()                                         // load trained weights
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained nodel should exist; run train first");

    let model = config.model.init(&device).load_record(record);                 // recreate model

    let label = item.label;                                                     // correct number
    let batcher = MnistBatcher::default();
    let batch = batcher.batch(vec![item], &device);                             // make batch with just 1 item
    let output = model.forward(batch.images);                                   // run model
    let predicted: u8 = output.argmax(1).flatten::<1>(0, 1).into_scalar();      // output guess

    println!("Predicted {predicted} Expected {label}");
}