use crate::LearnerModel;
use crate::{TrainOutput, TrainStep, TrainingModelInput, TrainingModelOutput};
use burn_core::data::dataloader::DataLoaderIterator;
use burn_core::module::Module;
use burn_core::data::dataloader::Progress;
use burn_core::tensor::Device;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;
use typing_rules::*; // import filament ifc
use macros::fcall; // import ifc macros

/// Multi devices train step.
pub struct MultiDevicesTrainStep<M: LearnerModel, L: Label> {
    workers: Vec<Worker<M, L>>,
    receiver: Receiver<MultiTrainOutput<TrainingModelOutput<M>, L>>,
}

struct Message<M, TI, L: Label> {
    item: TI,
    model: Labeled<M, L>,
}

struct Worker<M: LearnerModel, L: Label> {
    // Not that complex. Extracting into another type would only make it more confusing.
    // #[allow(clippy::type_complexity)]
    sender_input: Sender<Message<M, Labeled<TrainingModelInput<M>, L>, L>>,
    device: Device,
    device_id: usize,
}

impl<M: LearnerModel, L: Label> Worker<M, L> {
    fn register(&self, item: Labeled<TrainingModelInput<M>, L>, model: Labeled<M, L>) {
        let message = Message { item, model };
        self.sender_input.send(message).unwrap();
    }

    // Not that complex. Extracting into another type would only make it more confusing.
    // #[allow(clippy::type_complexity)]
    fn start(
        &self,
        sender_output: Sender<MultiTrainOutput<TrainingModelOutput<M>, L>>,
        receiver_input: Receiver<Message<M, Labeled<TrainingModelInput<M>, L>, L>>,
    ) {
        let device = self.device.clone();
        let device_id = self.device_id;

        spawn(move || {
            loop {
                match receiver_input.recv() {
                    Ok(item) => {
                        // fork() consumes self, so fcall! (owned chain) is used instead of mcall!.
                        let model = fcall!(Module::fork(item.model, (&device)));
                        let input = item.item;
                        // TrainStep::step is named the same as InferenceStep::step, so it must be
                        // called UFCS-qualified (same pattern as Learner::train_step in base.rs).
                        let output: Labeled<_, _> = fcall!(TrainStep::step(&model, input));
                        let item = MultiTrainOutput { output, device_id };

                        sender_output.send(item).unwrap();
                    }
                    Err(_err) => {
                        log::info!("Closing thread on device {device:?}");
                        break;
                    }
                }
            }
        });
    }
}

/// Multiple output items.
pub struct MultiTrainOutput<TO, L: Label> {
    /// The training output (labeled with the input's security label).
    pub output: Labeled<TrainOutput<TO>, L>,
    /// The worker/device on which the computing happened.
    pub(crate) device_id: usize,
}

impl<M: LearnerModel, L: Label> MultiDevicesTrainStep<M, L> {
    /// Create a new multi devices train step.
    ///
    /// # Arguments
    ///
    /// * `devices` - Devices.
    ///
    /// # Returns
    ///
    /// MultiDevicesTrainStep instance.
    pub fn new(devices: &[Device]) -> Self {
        let (sender_output, receiver_output) = std::sync::mpsc::channel();
        let workers = devices
            .iter()
            .enumerate()
            .map(|(device_id, device)| {
                let (sender_input, receiver_input) = std::sync::mpsc::channel();
                let worker = Worker {
                    sender_input,
                    device: device.clone(),
                    device_id,
                };

                worker.start(sender_output.clone(), receiver_input);
                worker
            })
            .collect();

        Self {
            workers,
            receiver: receiver_output,
        }
    }

    /// Collect outputs from workers for one step.
    ///
    /// # Arguments
    ///
    /// * `model` - Model.
    /// * `dataloaders` - The data loader for each worker.
    ///
    /// # Returns
    ///
    /// Outputs.
    pub fn step<'a>(
        &self,
        dataloaders: &mut [Box<dyn DataLoaderIterator<TrainingModelInput<M>, L> + 'a>],
        model: &Labeled<M, L>,
    ) -> (Vec<MultiTrainOutput<TrainingModelOutput<M>, L>>, Progress) {
        let mut num_send = 0;

        let mut items_total = 0;
        let mut items_processed = 0;
        let unit: Option<String> = Some("items".to_string());

        for (i, worker) in self.workers.iter().enumerate() {
            let dataloader = &mut dataloaders[i];
            if let Some(item) = dataloader.next() {
                worker.register(item, Clone::clone(model));
                num_send += 1;
                let progress = dataloader.progress();
                items_total += progress.items_total;
                items_processed += progress.items_processed;
            }
        }

        let mut outputs = Vec::with_capacity(num_send);

        for _ in 0..num_send {
            let output = self.receiver.recv().unwrap();
            outputs.push(output);
        }

        (outputs, Progress::new(items_processed, items_total, unit))
    }
}
