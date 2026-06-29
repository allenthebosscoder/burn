use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::{
    InferenceStep, Learner, LearningComponentsTypes, SupervisedTrainingEventProcessor, TrainLoader,
    ValidLoader,
};
use burn_core::data::dataloader::Progress;
use burn_core::module::AutodiffModule;
use burn_optim::GradientsAccumulator;
use macros::fcall; // import ifc macros
use typing_rules::*; // import filament ifc

/// A validation epoch.
#[derive(new)]
pub struct SingleDeviceValidEpoch<LC: LearningComponentsTypes, L: Label> {
    dataloader: ValidLoader<LC, L>,
}

/// A training epoch.
#[derive(new)]
pub struct SingleDeviceTrainEpoch<LC: LearningComponentsTypes, L: Label> {
    dataloader: TrainLoader<LC, L>,
    grad_accumulation: Option<usize>,
}

impl<LC: LearningComponentsTypes, L: Label> SingleDeviceValidEpoch<LC, L> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        learner: &Learner<LC>,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing validation step for epoch {}", epoch);
        let model = learner.model().valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = fcall!(InferenceStep::step(&model, item));
            let labeled_training_item = fcall!(TrainingItem::new(item, progress, Some(iteration), None));
            let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
            fcall!(EventProcessorTraining::process_valid(processor, labeled_event));

            if interrupter.should_stop() {
                break;
            }
        }
    }
}

impl<LC: LearningComponentsTypes, L: Label> SingleDeviceTrainEpoch<LC, L> {
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
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing training step for epoch {}", epoch,);

        // Single device / dataloader
        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            learner.lr_step();
            log::info!("Iteration {iteration}");

            let progress = iterator.progress();
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
                None => { fcall!(Learner::optimizer_step(&mut *learner, labeled_grads)); }
            }

            let labeled_training_item = fcall!(TrainingItem::new(labeled_item, progress, Some(iteration), Some(learner.lr_current())));
            let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
            fcall!(EventProcessorTraining::process_train(processor, labeled_event));

            if interrupter.should_stop() {
                break;
            }
        }
    }
}
