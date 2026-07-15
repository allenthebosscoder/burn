use crate::checkpoint::{
    AsyncCheckpointer, Checkpointer, CheckpointingAction, CheckpointingStrategy,
};
use crate::metric::store::EventStoreClient;
use crate::{
    CloneEarlyStoppingStrategy, LearnerModel, TrainOutput, TrainStep, TrainingModelInput,
    TrainingModelOutput,
};
use burn_core::module::Module;
use burn_core::store::ModuleRecord;
use burn_core::tensor::Device;
use burn_optim::lr_scheduler::LrSchedulerRecord;
use burn_optim::lr_scheduler::module_lr_scheduler::{ModuleLearningRate, ModuleLrScheduler};
use burn_optim::{GradientsParams, ModuleOptimizer, MultiGradientsParams, OptimizerRecord};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use typing_rules::*; // import filament ifc
use macros::fcall;

/// Learner struct encapsulating all components necessary to train a Neural Network model.
pub struct Learner<M: LearnerModel, L: Label> {
    pub(crate) model: Labeled<M, L>,
    optim: ModuleOptimizer,
    lr_scheduler: ModuleLrScheduler,
    lr_module: ModuleLearningRate,
}

impl<M: LearnerModel, L: Label> Clone for Learner<M, L> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            optim: self.optim.clone(),
            lr_scheduler: self.lr_scheduler.clone(),
            lr_module: self.lr_module.clone(),
        }
    }
}

impl<M: LearnerModel, L: Label> Learner<M, L> {
    /// Create a learner. The model must be pre-labeled to establish the IFC security level.
    pub fn new(
        model: Labeled<M, L>,
        optim: ModuleOptimizer,
        lr_scheduler: impl Into<ModuleLrScheduler>,
    ) -> Self {
        Self {
            model,
            optim,
            lr_scheduler: lr_scheduler.into(),
            lr_module: 0.0.into(),
        }
    }
}

impl<M: LearnerModel, L: Label> Learner<M, L> {
    /// Fork the learner's model to the given device, preserving the IFC label.
    pub fn fork(&mut self, device: &Device) {
        self.model = fcall!(Module::fork(Clone::clone(&self.model), device));
    }

    /// Returns the labeled model.
    pub fn model(&self) -> Labeled<M, L> {
        self.model.clone()
    }

    /// Returns the current learning rate.
    pub fn lr_current(&self) -> ModuleLearningRate {
        self.lr_module.clone()
    }

    /// Executes a step of the learning rate scheduler.
    pub fn lr_step(&mut self) {
        self.lr_module = self.lr_scheduler.step();
    }

    /// Runs a training step. Takes a labeled item and returns a labeled output so
    /// epoch.rs calls this directly without an outer fcall! wrapper.
    pub fn train_step(&self, item: Labeled<TrainingModelInput<M>, L>) -> Labeled<TrainOutput<TrainingModelOutput<M>>, L> {
        fcall!(TrainStep::step(&self.model, item))
    }

    /// Optimize the current module with the provided gradients and learning rate,
    /// preserving the IFC label on the model.
    pub fn optimizer_step(&mut self, grads: GradientsParams) {
        let model = Clone::clone(&self.model);
        let lr = self.lr_module.clone(); // pre-extract: can't borrow self.lr_module and &mut self.optim simultaneously
        self.model = fcall!(TrainStep::optimize(model, &mut self.optim, lr, grads));
    }

    /// Optimize with multiple gradient sets, preserving the IFC label on the model.
    pub fn optimizer_step_multi(&mut self, grads: MultiGradientsParams) {
        let model = Clone::clone(&self.model);
        let lr = self.lr_module.clone(); // pre-extract: can't borrow self.lr_module and &mut self.optim simultaneously
        self.model = fcall!(TrainStep::optimize_multi(model, &mut self.optim, lr, grads));
    }

    /// Load the module state from a record, preserving the IFC label on the model.
    pub fn load_model(&mut self, record: ModuleRecord) {
        self.model = fcall!(Module::load_record(Clone::clone(&self.model), record));
    }

    /// Load the state of the learner's optimizer from a [record](OptimizerRecord).
    ///
    /// No device is needed: the optimizer state is migrated to each parameter's device on the next
    /// step (see [`ModuleOptimizer::load_record`](burn_optim::ModuleOptimizer::load_record)).
    pub fn load_optim(&mut self, record: OptimizerRecord) {
        self.optim = self.optim.clone().load_record(record);
    }

    /// Load the state of the learner's scheduler from a [record](LrSchedulerRecord).
    pub fn load_scheduler(&mut self, record: LrSchedulerRecord) {
        self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
    }
}

/// Used to create, delete, or load checkpoints of the training process.
pub struct LearningCheckpointer<M: LearnerModel> {
    model: AsyncCheckpointer<ModuleRecord>,
    optim: AsyncCheckpointer<OptimizerRecord>,
    lr_scheduler: AsyncCheckpointer<LrSchedulerRecord>,
    strategy: Box<dyn CheckpointingStrategy>,
    _phantom: PhantomData<M>,
}

impl<M: LearnerModel> LearningCheckpointer<M> {
    /// Create a new learning checkpointer.
    pub fn new(
        model: AsyncCheckpointer<ModuleRecord>,
        optim: AsyncCheckpointer<OptimizerRecord>,
        lr_scheduler: AsyncCheckpointer<LrSchedulerRecord>,
        strategy: Box<dyn CheckpointingStrategy>,
    ) -> Self {
        Self {
            model,
            optim,
            lr_scheduler,
            strategy,
            _phantom: PhantomData,
        }
    }

    /// Create checkpoint for the training process.
    pub fn checkpoint<L: Label>(&mut self, learner: &Learner<M, L>, epoch: usize, store: &EventStoreClient) {
        let actions = self.strategy.checkpointing(epoch, store);

        for action in actions {
            match action {
                CheckpointingAction::Delete(epoch) => {
                    self.model
                        .delete(epoch)
                        .expect("Can delete model checkpoint.");
                    self.optim
                        .delete(epoch)
                        .expect("Can delete optimizer checkpoint.");
                    self.lr_scheduler
                        .delete(epoch)
                        .expect("Can delete learning rate scheduler checkpoint.");
                }
                CheckpointingAction::Save => {
                    self.model
                        .save(epoch, declassify(Clone::clone(&learner.model)).into_record())
                        .expect("Can save model checkpoint.");
                    self.optim
                        .save(epoch, learner.optim.to_record())
                        .expect("Can save optimizer checkpoint.");
                    self.lr_scheduler
                        .save(epoch, learner.lr_scheduler.to_record())
                        .expect("Can save learning rate scheduler checkpoint.");
                }
            }
        }
    }

    /// Load a training checkpoint.
    ///
    /// No device is taken: checkpoints are device-free burnpack records (file-backed bytes). On
    /// load, the model keeps the device of the learner's existing parameters, and the optimizer
    /// state is migrated to each parameter's device on the next step. The training device is fixed
    /// earlier, when the learner's model is created/forked.
    pub fn load_checkpoint<L: Label>(&self, mut learner: Learner<M, L>, epoch: usize) -> Learner<M, L> {
        let record = self
            .model
            .restore(epoch)
            .expect("Can load model checkpoint.");
        learner.load_model(record);

        let record = self
            .optim
            .restore(epoch)
            .expect("Can load optimizer checkpoint.");
        learner.load_optim(record);

        let record = self
            .lr_scheduler
            .restore(epoch)
            .expect("Can load learning rate scheduler checkpoint.");
        learner.load_scheduler(record);

        learner
    }
}

/// Cloneable reference to an early stopping strategy
pub(crate) type EarlyStoppingStrategyRef = Box<dyn CloneEarlyStoppingStrategy>;

#[derive(Clone, Default)]
/// A handle that allows aborting the training/evaluation process early.
pub struct Interrupter {
    state: Arc<AtomicBool>,
    message: Arc<Mutex<Option<String>>>,
}

impl Interrupter {
    /// Create a new instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Notify the learner that it should stop.
    /// # Arguments
    /// * `reason` - A string describing the reason the training was stopped.
    pub fn stop(&self, reason: Option<&str>) {
        self.state.store(true, Ordering::Relaxed);
        reason.inspect(|r| {
            let mut message = self.message.lock().unwrap();
            *message = Some(String::from(*r));
        });
    }

    /// Reset the interrupter.
    pub fn reset(&self) {
        self.state.store(false, Ordering::Relaxed);
    }

    /// True if .stop() has been called.
    pub fn should_stop(&self) -> bool {
        self.state.load(Ordering::Relaxed)
    }

    /// Get the message associated with the interrupt.
    pub fn get_message(&self) -> Option<String> {
        let message = self.message.lock().unwrap();
        message.clone()
    }
}
