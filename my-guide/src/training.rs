use crate::{
    data::{MnistBatch, MnistBatcher},
    model::{Model, ModelConfig},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

impl Model {
    pub fn forward_classification(
        &self,
        images: Tensor<3>,
        targets: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let output = self.forward(images);              // outputs logits
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());  // Clone is NOT copying. Just sharing to not lose ownership

        ClassificationOutput::new(loss, output, targets)
    }
}

impl TrainStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_classification(batch.images, batch.targets);

        // return model, gradients, (loss, output, targets)
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> ClassificationOutput {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,          // Epoch: one complete pass through dataset
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train(artifact_dir: &str, config: TrainingConfig, device: impl Into<Device>) {
    
    // Create artifact directory and save training config
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // Converts the device passed into function to a real Device
    let device = device.into();
    device.seed(config.seed);                                       // sets random seed
    let autodiff_device = device.clone().autodiff();                // creates version of device that supports gradients

    let batcher = MnistBatcher::default();                          // creates batcher that can convert MnistItem to MnistBatch

    // Create iterator over TRAINING batches. Output: MnistBatch
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())  // use this batcher to convert items to batches
        .batch_size(config.batch_size)                              // batch size from config
        .shuffle(config.seed)                                       // shuffle based on seeds
        .num_workers(config.num_workers)                            // use N worker threads
        .build(MnistDataset::train());                              // read from MNIST training set

    // Create iterator over TEST batches. Output: MnistBatch
    let dataloader_test = DataLoaderBuilder::new(batcher)           // use this batcher to convert items to batches
        .batch_size(config.batch_size)                              // batch size from config
        .shuffle(config.seed)                                       // shuffle based on seeds
        .num_workers(config.num_workers)                            // use N worker threads
        .build(MnistDataset::test());                               // read from MNIST test set

    // Configure training loop: data, metrics, checkpointing, epochs, summary
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test) // create training runner that knows where to store files and what batches to train and test on
        .metrics((AccuracyMetric::new(), LossMetric::new()))        // track accuracy and loss during training/evaluation
        .with_file_checkpointer(CompactRecorder::new())             // save checkpoints to disk
        .num_epochs(config.num_epochs)                              // run through full dataset multiple times
        .summary();                                                 // prep summary of training setup/results

    let model = config.model.init(&autodiff_device);                // create model with weights on a device that supports gradients
    
    // Create learner containg model, Adam optimizer, and learning rate
    // This is where actual training loop runs
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    // save the final trained model weights to disk
    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}