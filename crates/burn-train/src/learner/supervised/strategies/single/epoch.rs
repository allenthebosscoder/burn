use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::{
    InferenceStep, Learner, LearningComponentsTypes, SupervisedTrainingEventProcessor, TrainLoader,
    ValidLoader,
};
use burn_core::data::dataloader::Progress;
use burn_core::module::AutodiffModule;
use burn_optim::GradientsAccumulator;
use macros::{fcall, mcall}; // import ifc macros
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
        learner: &Learner<LC, L>,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing validation step for epoch {}", epoch);
        let labeled_model = learner.model();
        let model = mcall!(labeled_model.valid());

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
        learner: &mut Learner<LC, L>,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing training step for epoch {}", epoch,);


        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        // Labeled from the start so __chain_mut can track the label across accumulate() calls.
        let mut accumulator: Labeled<GradientsAccumulator<LC::Model>, L> = Labeled::new(GradientsAccumulator::new());
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            learner.lr_step();
            log::info!("Iteration {iteration}");

            let progress = iterator.progress();
            let item = learner.train_step(item);
            let (labeled_grads, labeled_item) = item.map(|o| (o.grads, o.item)).split();

            match self.grad_accumulation {
                Some(accumulation) => {
                    // __chain_mut unwraps labeled_grads, gives &mut inner to accumulate(),
                    // then reassigns accumulator with label joined with labeled_grads' label.
                    fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        // grads() resets the accumulator internally; __chain_mut reassigns
                        // accumulator (now empty) and returns Labeled<GradientsParams, L>.
                        let labeled_grads_combined = fcall!(GradientsAccumulator::grads(&mut accumulator));
                        fcall!(Learner::optimizer_step(&mut *learner, labeled_grads_combined));
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
