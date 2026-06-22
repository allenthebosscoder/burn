use burn_core::data::dataloader::Progress;
use burn_core::module::AutodiffModule;
use burn_optim::GradientsAccumulator;
use std::sync::{Arc, Mutex};
use typing_rules::*; // import filament ifc
use macros::fcall; // import ifc macros

use crate::SupervisedTrainingEventProcessor;
use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::{InferenceStep, Learner, LearningComponentsTypes, TrainLoader, ValidLoader};

/// A validation epoch.
#[derive(new)]
pub struct DdpValidEpoch<LC: LearningComponentsTypes, L: Label> {
    dataloader: ValidLoader<LC, L>,
}

/// A training epoch.
#[derive(new)]
pub struct DdpTrainEpoch<LC: LearningComponentsTypes, L: Label> {
    dataloader: TrainLoader<LC, L>,
    grad_accumulation: Option<usize>,
}

impl<LC: LearningComponentsTypes, L: Label> DdpValidEpoch<LC, L> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        model: &<LC as LearningComponentsTypes>::Model,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing validation step for epoch {}", epoch);
        let model = model.valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = fcall!(InferenceStep::step(&model, item));
            let labeled_event = item.map(|o| LearnerEvent::ProcessedItem(TrainingItem::new(o, progress, Some(iteration), None)));
            fcall!(EventProcessorTraining::process_valid(processor, labeled_event));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
    }
}

impl<LC: LearningComponentsTypes, L: Label> DdpTrainEpoch<LC, L> {
    /// Runs the training epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    pub fn run(
        &self,
        learner: &mut Learner<LC>,
        global_progress: &Progress,
        processor: Arc<Mutex<SupervisedTrainingEventProcessor<LC>>>,
        interrupter: &Interrupter,
        peer_count: usize,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing training step for epoch {}", epoch,);

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            for _ in 0..peer_count {
                iteration += 1;
                learner.lr_step();
            }
            log::info!("Iteration {iteration}");

            let mut progress = iterator.progress();
            progress.items_processed *= peer_count;
            progress.items_total *= peer_count;

            //let item = learner.train_step(item);
            let item = fcall!(Learner::train_step(&learner, item));
            let (labeled_grads, labeled_item) = item.map(|o| (o.grads, o.item)).split();

            match self.grad_accumulation {
                Some(accumulation) => {
                    fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();

                        learner.optimizer_step(grads);
                        accumulation_current = 0;
                    }
                }
                None => {
                    fcall!(Learner::optimizer_step(&mut *learner, labeled_grads));
                }
            }

            let labeled_event = labeled_item
                .map(|o| TrainingItem::new(o, progress, Some(iteration), Some(learner.lr_current())))
                .map(LearnerEvent::ProcessedItem);

            {
                let mut processor = processor.lock().unwrap();
                fcall!(EventProcessorTraining::process_train(&mut *processor, labeled_event));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
    }
}
