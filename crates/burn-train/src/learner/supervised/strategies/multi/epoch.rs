use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::train::MultiDevicesTrainStep;
use crate::{
    Learner, LearningComponentsTypes, MultiDeviceOptim, SupervisedTrainingEventProcessor,
    TrainLoader,
};
use burn_core::data::dataloader::Progress;
use burn_core::tensor::Device;
use burn_optim::{GradientsAccumulator, GradientsParams, MultiGradientsParams};
use typing_rules::*; // import filament ifc
use macros::fcall; // import ifc macros

/// A training epoch.
#[derive(new)]
pub struct MultiDeviceTrainEpoch<LC: LearningComponentsTypes, L: Label> {
    dataloaders: Vec<TrainLoader<LC, L>>,
    grad_accumulation: Option<usize>,
}

impl<LC: LearningComponentsTypes, L: Label> MultiDeviceTrainEpoch<LC, L> {
    /// Runs the training epoch on multiple devices.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `lr_scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    /// * `devices` - The devices to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        learner: &mut Learner<LC, L>,
        global_progress: &Progress,
        event_processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
        devices: Vec<Device>,
        strategy: MultiDeviceOptim,
    ) {
        match strategy {
            MultiDeviceOptim::OptimMainDevice => self.run_optim_main(
                learner,
                global_progress,
                event_processor,
                interrupter,
                devices,
            ),
            MultiDeviceOptim::OptimSharded => self.run_optim_distr(
                learner,
                global_progress,
                event_processor,
                interrupter,
                devices,
            ),
        }
    }

    fn run_optim_main(
        &self,
        learner: &mut Learner<LC, L>,
        global_progress: &Progress,
        event_processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
        devices: Vec<Device>,
    ) {
        let epoch = global_progress.items_processed;
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            epoch,
            devices
        );

        let mut iterators = self
            .dataloaders
            .iter()
            .map(|d| d.iter())
            .collect::<Vec<_>>();
        let mut iteration = 0;
        // Labeled from the start so __chain_mut can track the label across accumulate() calls.
        let mut accumulator: Labeled<GradientsAccumulator<LC::Model>, L> = Labeled::new(GradientsAccumulator::new());
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1);
        let step = MultiDevicesTrainStep::<LC, L>::new(&devices);

        // The main device is always the first in the list.
        let device_main = devices.first().expect("A minimum of one device.").clone();

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &learner.model);
            if items.is_empty() {
                break;
            }

            learner.lr_step();

            let mut progress_items = Vec::with_capacity(items.len());
            for item in items.into_iter() {
                // Decompose labeled output into grads and item (pure structural split).
                let (labeled_raw_grads, labeled_item) = item.output.map(|o| (o.grads, o.item)).split();
                // Move grads to main device; fcall! passes &LC::Model via __chain_ref on learner.model.
                let labeled_grads = fcall!(GradientsParams::to_device(labeled_raw_grads, (&device_main), &learner.model));
                // __chain_mut unwraps labeled_grads, gives &mut inner to accumulate(),
                // then reassigns accumulator with label joined with labeled_grads' label.
                fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model, labeled_grads));
                progress_items.push(labeled_item);
            }

            accumulation_current += 1;

            if accumulation <= accumulation_current {
                // grads() resets the accumulator internally; __chain_mut reassigns
                // accumulator (now empty) and returns Labeled<GradientsParams, L>.
                let labeled_grads_combined = fcall!(GradientsAccumulator::grads(&mut accumulator));
                fcall!(Learner::optimizer_step(&mut *learner, labeled_grads_combined));
                accumulation_current = 0;
            }

            for item in progress_items {
                iteration += 1;
                let labeled_training_item = fcall!(TrainingItem::new(item, progress.clone(), Some(iteration), Some(learner.lr_current())));
                let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
                fcall!(EventProcessorTraining::process_train(event_processor, labeled_event));
            }

            if interrupter.should_stop() {
                break;
            }
        }
    }

    fn run_optim_distr(
        &self,
        learner: &mut Learner<LC, L>,
        global_progress: &Progress,
        event_processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
        devices: Vec<Device>,
    ) {
        let epoch = global_progress.items_processed;
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            epoch,
            devices
        );

        let mut iterators = self
            .dataloaders
            .iter()
            .map(|d| d.iter())
            .collect::<Vec<_>>();
        let mut iteration = 0;
        let mut accumulators: Vec<GradientsAccumulator<_>> = (0..devices.len())
            .map(|_| GradientsAccumulator::new())
            .collect();
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1);
        let step = MultiDevicesTrainStep::<LC, L>::new(&devices);

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &learner.model);
            if items.is_empty() {
                break;
            }

            learner.lr_step();

            let mut progress_items = Vec::with_capacity(items.len());
            for item in items.into_iter() {
                let accumulator = &mut accumulators[item.device_id];
                let (labeled_grads, labeled_item) = item.output.map(|o| (o.grads, o.item)).split();
                fcall!(GradientsAccumulator::accumulate(accumulator, &learner.model, labeled_grads));
                progress_items.push(labeled_item);
            }

            accumulation_current += 1;

            if accumulation <= accumulation_current {
                let mut grads = MultiGradientsParams::default();
                for (device_id, accumulator) in accumulators.iter_mut().enumerate() {
                    let grad = accumulator.grads();
                    grads.grads.push((grad, devices[device_id].clone()));
                }
                learner.optimizer_step_multi(grads);
                accumulation_current = 0;
            }

            for item in progress_items {
                iteration += 1;
                let labeled_training_item = fcall!(TrainingItem::new(item, progress.clone(), Some(iteration), Some(learner.lr_current())));
                let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
                fcall!(EventProcessorTraining::process_train(event_processor, labeled_event));
            }

            if interrupter.should_stop() {
                break;
            }
        }
    }
}
