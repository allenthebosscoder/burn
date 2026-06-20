use crate::{LearningComponentsTypes, TrainingModel};
use crate::{TrainOutput, TrainStep, TrainingModelInput, TrainingModelOutput};
use burn_core::data::dataloader::DataLoaderIterator;
use burn_core::data::dataloader::Progress;
use burn_core::module::Module;
use burn_core::tensor::Device;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::spawn;
use typing_rules::*; // import filament ifc
use macros::mcall; // import ifc macros

/// Multi devices train step.
pub struct MultiDevicesTrainStep<LC: LearningComponentsTypes, L: Label> {
    workers: Vec<Worker<LC, L>>,
    receiver: Receiver<MultiTrainOutput<TrainingModelOutput<LC>, L>>,
}

struct Message<M, TI> {
    item: TI,
    model: M,
}

struct Worker<LC: LearningComponentsTypes, L: Label> {
    // Not that complex. Extracting into another type would only make it more confusing.
    // #[allow(clippy::type_complexity)]
    sender_input: Sender<Message<TrainingModel<LC>, Labeled<TrainingModelInput<LC>, L>>>,
    device: Device,
    device_id: usize,
}

impl<LC: LearningComponentsTypes, L: Label> Worker<LC, L> {
    fn register(&self, item: Labeled<TrainingModelInput<LC>, L>, model: &TrainingModel<LC>) {
        let message = Message {
            item,
            model: model.clone(),
        };
        self.sender_input.send(message).unwrap();
    }

    // Not that complex. Extracting into another type would only make it more confusing.
    // #[allow(clippy::type_complexity)]
    fn start(
        &self,
        sender_output: Sender<MultiTrainOutput<TrainingModelOutput<LC>, L>>,
        receiver_input: Receiver<Message<TrainingModel<LC>, Labeled<TrainingModelInput<LC>, L>>>,
    ) {
        let device = self.device.clone();
        let device_id = self.device_id;

        spawn(move || {
            loop {
                match receiver_input.recv() {
                    Ok(item) => {
                        let model = item.model.fork(&device);
                        let input = item.item;
                        // let output = fcall!(TrainStep::step(&model, input));
                        let output: Labeled<_, _> = mcall!(model.step(input));
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

impl<LC: LearningComponentsTypes, L: Label> MultiDevicesTrainStep<LC, L> {
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
        dataloaders: &mut [Box<dyn DataLoaderIterator<TrainingModelInput<LC>, L> + 'a>],
        model: &TrainingModel<LC>,
    ) -> (Vec<MultiTrainOutput<TrainingModelOutput<LC>, L>>, Progress) {
        let mut num_send = 0;

        let mut items_total = 0;
        let mut items_processed = 0;
        let unit: Option<String> = Some("items".to_string());

        for (i, worker) in self.workers.iter().enumerate() {
            let dataloader = &mut dataloaders[i];
            if let Some(item) = dataloader.next() {
                worker.register(item, model);
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
