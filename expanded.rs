#![feature(prelude_import)]
#![warn(missing_docs)]
//! A library for training neural networks using the burn crate.
extern crate std;
#[prelude_import]
use std::prelude::rust_2024::*;
#[macro_use]
extern crate derive_new;
/// The checkpoint module.
pub mod checkpoint {
    mod async_checkpoint {
        use super::{Checkpointer, CheckpointerError};
        use crate::Interrupter;
        use burn_core::{record::Record, tensor::Device};
        use std::sync::mpsc;
        enum Message<R> {
            Restore(
                usize,
                Device,
                mpsc::SyncSender<Result<R, CheckpointerError>>,
                Option<Interrupter>,
            ),
            Save(usize, R, Option<Interrupter>),
            Delete(usize, Option<Interrupter>),
            End,
        }
        struct CheckpointerThread<C, R> {
            checkpointer: C,
            receiver: mpsc::Receiver<Message<R>>,
        }
        impl<C, R> CheckpointerThread<C, R> {
            ///Constructs a new `CheckpointerThread`.
            pub fn new(checkpointer: C, receiver: mpsc::Receiver<Message<R>>) -> Self {
                CheckpointerThread {
                    checkpointer: checkpointer,
                    receiver: receiver,
                }
            }
        }
        impl<C, R> CheckpointerThread<C, R>
        where
            C: Checkpointer<R>,
            R: Record,
        {
            fn run(self) {
                for item in self.receiver.iter() {
                    match item {
                        Message::Restore(epoch, device, callback, interrupter) => {
                            let record = self.checkpointer.restore(epoch, &device);
                            callback
                                .send(record)
                                .unwrap_or_else(|err| {
                                    interrupter
                                        .map_or_else(
                                            || {
                                                {
                                                    ::core::panicking::panic_fmt(
                                                        format_args!(
                                                            "Error when sending response through callback channel: {0}",
                                                            err,
                                                        ),
                                                    );
                                                }
                                            },
                                            |int| int.stop(Some(&err.to_string())),
                                        )
                                });
                        }
                        Message::Save(epoch, state, interrupter) => {
                            self.checkpointer
                                .save(epoch, state)
                                .unwrap_or_else(|err| {
                                    interrupter
                                        .map_or_else(
                                            || {
                                                ::core::panicking::panic_fmt(
                                                    format_args!("Error when saving the state: {0}", err),
                                                );
                                            },
                                            |int| int.stop(Some(&err.to_string())),
                                        )
                                });
                        }
                        Message::Delete(epoch, interrupter) => {
                            self.checkpointer
                                .delete(epoch)
                                .unwrap_or_else(|err| {
                                    interrupter
                                        .map_or_else(
                                            || {
                                                ::core::panicking::panic_fmt(
                                                    format_args!("Error when deleting the state: {0}", err),
                                                );
                                            },
                                            |int| int.stop(Some(&err.to_string())),
                                        )
                                });
                        }
                        Message::End => {
                            return;
                        }
                    };
                }
            }
        }
        /// Async checkpointer.
        pub struct AsyncCheckpointer<Record> {
            sender: mpsc::SyncSender<Message<Record>>,
            handler: Option<std::thread::JoinHandle<()>>,
            interrupter: Option<Interrupter>,
        }
        impl<R> AsyncCheckpointer<R>
        where
            R: Record + 'static,
        {
            /// Create a new async checkpointer.
            ///
            /// # Arguments
            ///
            /// * `checkpointer` - The checkpointer.
            ///
            /// # Returns
            ///
            /// The async checkpointer.
            pub fn new<C>(checkpointer: C) -> Self
            where
                C: Checkpointer<R> + Send + 'static,
            {
                let (sender, receiver) = mpsc::sync_channel(0);
                let thread = CheckpointerThread::new(checkpointer, receiver);
                let handler = Some(std::thread::spawn(move || thread.run()));
                Self {
                    sender,
                    handler,
                    interrupter: None,
                }
            }
            /// Assign a handle used to interrupt training in case of checkpointing error.
            pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
                self.interrupter = Some(interrupter);
                self
            }
        }
        impl<R> Checkpointer<R> for AsyncCheckpointer<R>
        where
            R: Record + 'static,
        {
            fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
                self.sender
                    .send(Message::Save(epoch, record, self.interrupter.clone()))
                    .expect("Can send message to checkpointer thread.");
                Ok(())
            }
            fn restore(
                &self,
                epoch: usize,
                device: &Device,
            ) -> Result<R, CheckpointerError> {
                let (sender, receiver) = mpsc::sync_channel(1);
                self.sender
                    .send(
                        Message::Restore(
                            epoch,
                            device.clone(),
                            sender,
                            self.interrupter.clone(),
                        ),
                    )
                    .map_err(|e| CheckpointerError::Unknown(e.to_string()))?;
                if let Ok(record) = receiver.recv() {
                    return record;
                }
                Err(CheckpointerError::Unknown("Channel error.".to_string()))
            }
            fn delete(&self, epoch: usize) -> Result<(), CheckpointerError> {
                self.sender
                    .send(Message::Delete(epoch, self.interrupter.clone()))
                    .map_err(|e| CheckpointerError::Unknown(e.to_string()))?;
                Ok(())
            }
        }
        impl<E> Drop for AsyncCheckpointer<E> {
            fn drop(&mut self) {
                self.sender
                    .send(Message::End)
                    .expect("Can send the end message to the checkpointer thread.");
                let handler = self.handler.take();
                if let Some(handler) = handler {
                    handler.join().expect("The checkpointer thread should stop.");
                }
            }
        }
    }
    mod base {
        use burn_core::{
            record::{Record, RecorderError},
            tensor::Device,
        };
        use thiserror::Error;
        /// The error type for checkpointer.
        pub enum CheckpointerError {
            /// IO error.
            #[error("I/O Error: `{0}`")]
            IOError(std::io::Error),
            /// Recorder error.
            #[error("Recorder error: `{0}`")]
            RecorderError(RecorderError),
            /// Other errors.
            #[error("Unknown error: `{0}`")]
            Unknown(String),
        }
        #[allow(unused_qualifications)]
        #[automatically_derived]
        impl ::thiserror::__private18::Error for CheckpointerError {}
        #[allow(unused_qualifications)]
        #[automatically_derived]
        impl ::core::fmt::Display for CheckpointerError {
            fn fmt(
                &self,
                __formatter: &mut ::core::fmt::Formatter,
            ) -> ::core::fmt::Result {
                use ::thiserror::__private18::AsDisplay as _;
                #[allow(unused_variables, deprecated, clippy::used_underscore_binding)]
                match self {
                    CheckpointerError::IOError(_0) => {
                        match (_0.as_display(),) {
                            (__display0,) => {
                                __formatter
                                    .write_fmt(format_args!("I/O Error: `{0}`", __display0))
                            }
                        }
                    }
                    CheckpointerError::RecorderError(_0) => {
                        match (_0.as_display(),) {
                            (__display0,) => {
                                __formatter
                                    .write_fmt(
                                        format_args!("Recorder error: `{0}`", __display0),
                                    )
                            }
                        }
                    }
                    CheckpointerError::Unknown(_0) => {
                        match (_0.as_display(),) {
                            (__display0,) => {
                                __formatter
                                    .write_fmt(format_args!("Unknown error: `{0}`", __display0))
                            }
                        }
                    }
                }
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for CheckpointerError {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    CheckpointerError::IOError(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "IOError",
                            &__self_0,
                        )
                    }
                    CheckpointerError::RecorderError(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "RecorderError",
                            &__self_0,
                        )
                    }
                    CheckpointerError::Unknown(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Unknown",
                            &__self_0,
                        )
                    }
                }
            }
        }
        /// The trait for checkpointer.
        pub trait Checkpointer<R>: Send + Sync
        where
            R: Record,
        {
            /// Save the record.
            ///
            /// # Arguments
            ///
            /// * `epoch` - The epoch.
            /// * `record` - The record.
            fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError>;
            /// Delete the record at the given epoch if present.
            fn delete(&self, epoch: usize) -> Result<(), CheckpointerError>;
            /// Restore the record.
            ///
            /// # Arguments
            ///
            /// * `epoch` - The epoch.
            /// * `device` - The device used to restore the record.
            ///
            /// # Returns
            ///
            /// The record.
            fn restore(
                &self,
                epoch: usize,
                device: &Device,
            ) -> Result<R, CheckpointerError>;
        }
    }
    mod file {
        use std::path::{Path, PathBuf};
        use super::{Checkpointer, CheckpointerError};
        use burn_core::{
            record::{FileRecorder, Record},
            tensor::Device,
        };
        /// The file checkpointer.
        pub struct FileCheckpointer<FR> {
            directory: PathBuf,
            name: String,
            recorder: FR,
        }
        impl<FR> FileCheckpointer<FR> {
            /// Creates a new file checkpointer.
            ///
            /// # Arguments
            ///
            /// * `recorder` - The file recorder.
            /// * `directory` - The directory to save the checkpoints.
            /// * `name` - The name of the checkpoint.
            pub fn new(recorder: FR, directory: impl AsRef<Path>, name: &str) -> Self {
                let directory = directory.as_ref();
                std::fs::create_dir_all(directory).ok();
                Self {
                    directory: directory.to_path_buf(),
                    name: name.to_string(),
                    recorder,
                }
            }
            fn path_for_epoch(&self, epoch: usize) -> PathBuf {
                self.directory
                    .join(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0}-{1}", self.name, epoch),
                            )
                        }),
                    )
            }
        }
        impl<FR, R> Checkpointer<R> for FileCheckpointer<FR>
        where
            R: Record,
            FR: FileRecorder,
        {
            fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
                let file_path = self.path_for_epoch(epoch);
                {
                    {
                        let lvl = ::log::Level::Trace;
                        if lvl <= ::log::STATIC_MAX_LEVEL && lvl <= ::log::max_level() {
                            ::log::__private_api::log(
                                { ::log::__private_api::GlobalLogger },
                                format_args!(
                                    "Saving checkpoint {0} to {1}",
                                    epoch,
                                    file_path.display(),
                                ),
                                lvl,
                                &(
                                    "burn_train::checkpoint::file",
                                    "burn_train::checkpoint::file",
                                    ::log::__private_api::loc(),
                                ),
                                (),
                            );
                        }
                    }
                };
                self.recorder
                    .record(record, file_path)
                    .map_err(CheckpointerError::RecorderError)?;
                Ok(())
            }
            fn restore(
                &self,
                epoch: usize,
                device: &Device,
            ) -> Result<R, CheckpointerError> {
                let file_path = self.path_for_epoch(epoch);
                {
                    {
                        let lvl = ::log::Level::Info;
                        if lvl <= ::log::STATIC_MAX_LEVEL && lvl <= ::log::max_level() {
                            ::log::__private_api::log(
                                { ::log::__private_api::GlobalLogger },
                                format_args!(
                                    "Restoring checkpoint {0} from {1}",
                                    epoch,
                                    file_path.display(),
                                ),
                                lvl,
                                &(
                                    "burn_train::checkpoint::file",
                                    "burn_train::checkpoint::file",
                                    ::log::__private_api::loc(),
                                ),
                                (),
                            );
                        }
                    }
                };
                let record = self
                    .recorder
                    .load(file_path, device)
                    .map_err(CheckpointerError::RecorderError)?;
                Ok(record)
            }
            fn delete(&self, epoch: usize) -> Result<(), CheckpointerError> {
                let file_to_remove = ::alloc::__export::must_use({
                    ::alloc::fmt::format(
                        format_args!(
                            "{0}.{1}",
                            self.path_for_epoch(epoch).display(),
                            FR::file_extension(),
                        ),
                    )
                });
                if std::path::Path::new(&file_to_remove).exists() {
                    {
                        {
                            let lvl = ::log::Level::Trace;
                            if lvl <= ::log::STATIC_MAX_LEVEL
                                && lvl <= ::log::max_level()
                            {
                                ::log::__private_api::log(
                                    { ::log::__private_api::GlobalLogger },
                                    format_args!("Removing checkpoint {0}", file_to_remove),
                                    lvl,
                                    &(
                                        "burn_train::checkpoint::file",
                                        "burn_train::checkpoint::file",
                                        ::log::__private_api::loc(),
                                    ),
                                    (),
                                );
                            }
                        }
                    };
                    std::fs::remove_file(file_to_remove)
                        .map_err(CheckpointerError::IOError)?;
                }
                Ok(())
            }
        }
    }
    mod strategy {
        mod base {
            use std::ops::DerefMut;
            use crate::metric::store::EventStoreClient;
            /// Action to be taken by a [checkpointer](crate::checkpoint::Checkpointer).
            pub enum CheckpointingAction {
                /// Delete the given epoch.
                Delete(usize),
                /// Save the current record.
                Save,
            }
            #[automatically_derived]
            impl ::core::clone::Clone for CheckpointingAction {
                #[inline]
                fn clone(&self) -> CheckpointingAction {
                    match self {
                        CheckpointingAction::Delete(__self_0) => {
                            CheckpointingAction::Delete(
                                ::core::clone::Clone::clone(__self_0),
                            )
                        }
                        CheckpointingAction::Save => CheckpointingAction::Save,
                    }
                }
            }
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for CheckpointingAction {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for CheckpointingAction {
                #[inline]
                fn eq(&self, other: &CheckpointingAction) -> bool {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    __self_discr == __arg1_discr
                        && match (self, other) {
                            (
                                CheckpointingAction::Delete(__self_0),
                                CheckpointingAction::Delete(__arg1_0),
                            ) => __self_0 == __arg1_0,
                            _ => true,
                        }
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for CheckpointingAction {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    match self {
                        CheckpointingAction::Delete(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "Delete",
                                &__self_0,
                            )
                        }
                        CheckpointingAction::Save => {
                            ::core::fmt::Formatter::write_str(f, "Save")
                        }
                    }
                }
            }
            /// Define when checkpoint should be saved and deleted.
            pub trait CheckpointingStrategy: Send {
                /// Based on the epoch, determine if the checkpoint should be saved.
                fn checkpointing(
                    &mut self,
                    epoch: usize,
                    collector: &EventStoreClient,
                ) -> Vec<CheckpointingAction>;
            }
            impl CheckpointingStrategy for Box<dyn CheckpointingStrategy> {
                fn checkpointing(
                    &mut self,
                    epoch: usize,
                    collector: &EventStoreClient,
                ) -> Vec<CheckpointingAction> {
                    self.deref_mut().checkpointing(epoch, collector)
                }
            }
        }
        mod composed {
            use crate::metric::store::EventStoreClient;
            use super::{CheckpointingAction, CheckpointingStrategy};
            use std::collections::HashSet;
            /// Compose multiple checkpointing strategy and only delete checkpoints when both strategy flag an
            /// epoch to be deleted.
            pub struct ComposedCheckpointingStrategy {
                strategies: Vec<Box<dyn CheckpointingStrategy>>,
                deleted: Vec<HashSet<usize>>,
            }
            /// Help building a [checkpointing strategy](CheckpointingStrategy) by combining multiple ones.
            pub struct ComposedCheckpointingStrategyBuilder {
                strategies: Vec<Box<dyn CheckpointingStrategy>>,
            }
            #[automatically_derived]
            impl ::core::default::Default for ComposedCheckpointingStrategyBuilder {
                #[inline]
                fn default() -> ComposedCheckpointingStrategyBuilder {
                    ComposedCheckpointingStrategyBuilder {
                        strategies: ::core::default::Default::default(),
                    }
                }
            }
            impl ComposedCheckpointingStrategyBuilder {
                /// Add a new [checkpointing strategy](CheckpointingStrategy).
                #[allow(clippy::should_implement_trait)]
                pub fn add<S>(mut self, strategy: S) -> Self
                where
                    S: CheckpointingStrategy + 'static,
                {
                    self.strategies.push(Box::new(strategy));
                    self
                }
                /// Create a new [composed checkpointing strategy](ComposedCheckpointingStrategy).
                pub fn build(self) -> ComposedCheckpointingStrategy {
                    ComposedCheckpointingStrategy::new(self.strategies)
                }
            }
            impl ComposedCheckpointingStrategy {
                fn new(strategies: Vec<Box<dyn CheckpointingStrategy>>) -> Self {
                    Self {
                        deleted: strategies.iter().map(|_| HashSet::new()).collect(),
                        strategies,
                    }
                }
                /// Create a new builder which help compose multiple
                /// [checkpointing strategies](CheckpointingStrategy).
                pub fn builder() -> ComposedCheckpointingStrategyBuilder {
                    ComposedCheckpointingStrategyBuilder::default()
                }
            }
            impl CheckpointingStrategy for ComposedCheckpointingStrategy {
                fn checkpointing(
                    &mut self,
                    epoch: usize,
                    collector: &EventStoreClient,
                ) -> Vec<CheckpointingAction> {
                    let mut saved = false;
                    let mut actions = Vec::new();
                    let mut epochs_to_check = Vec::new();
                    for (i, strategy) in self.strategies.iter_mut().enumerate() {
                        let actions = strategy.checkpointing(epoch, collector);
                        if actions.is_empty() {
                            self.deleted
                                .get_mut(i)
                                .expect("As many 'deleted' as 'strategies'.")
                                .insert(epoch);
                        }
                        for action in actions {
                            match action {
                                CheckpointingAction::Delete(epoch) => {
                                    self.deleted
                                        .get_mut(i)
                                        .expect("As many 'deleted' as 'strategies'.")
                                        .insert(epoch);
                                    epochs_to_check.push(epoch);
                                }
                                CheckpointingAction::Save => saved = true,
                            }
                        }
                    }
                    if saved {
                        actions.push(CheckpointingAction::Save);
                    }
                    for epoch in epochs_to_check.into_iter() {
                        let mut num_true = 0;
                        for i in 0..self.strategies.len() {
                            if self
                                .deleted
                                .get(i)
                                .expect("Ad many 'deleted' as 'strategies'.")
                                .contains(&epoch)
                            {
                                num_true += 1;
                            }
                        }
                        if num_true == self.strategies.len() {
                            actions.push(CheckpointingAction::Delete(epoch));
                            for i in 0..self.strategies.len() {
                                self.deleted
                                    .get_mut(i)
                                    .expect("As many 'deleted' as 'strategies'.")
                                    .remove(&epoch);
                            }
                        }
                    }
                    actions
                }
            }
        }
        mod lastn {
            use super::CheckpointingStrategy;
            use crate::{
                checkpoint::CheckpointingAction, metric::store::EventStoreClient,
            };
            /// Keep the last N checkpoints.
            ///
            /// Very useful when training, minimizing disk space while ensuring that the training can be
            /// resumed even if something goes wrong.
            pub struct KeepLastNCheckpoints {
                num_keep: usize,
            }
            impl KeepLastNCheckpoints {
                ///Constructs a new `KeepLastNCheckpoints`.
                pub fn new(num_keep: usize) -> Self {
                    KeepLastNCheckpoints {
                        num_keep: num_keep,
                    }
                }
            }
            impl CheckpointingStrategy for KeepLastNCheckpoints {
                fn checkpointing(
                    &mut self,
                    epoch: usize,
                    _store: &EventStoreClient,
                ) -> Vec<CheckpointingAction> {
                    let mut actions = ::alloc::boxed::box_assume_init_into_vec_unsafe(
                        ::alloc::intrinsics::write_box_via_move(
                            ::alloc::boxed::Box::new_uninit(),
                            [CheckpointingAction::Save],
                        ),
                    );
                    if let Some(epoch) = usize::checked_sub(epoch, self.num_keep)
                        && epoch > 0
                    {
                        actions.push(CheckpointingAction::Delete(epoch));
                    }
                    actions
                }
            }
        }
        mod metric {
            use super::CheckpointingStrategy;
            use crate::{
                checkpoint::CheckpointingAction,
                metric::{
                    Metric, MetricName,
                    store::{Aggregate, Direction, EventStoreClient, Split},
                },
            };
            /// Keep the best checkpoint based on a metric.
            pub struct MetricCheckpointingStrategy {
                current: Option<usize>,
                aggregate: Aggregate,
                direction: Direction,
                split: Split,
                name: MetricName,
            }
            impl MetricCheckpointingStrategy {
                /// Create a new metric checkpointing strategy.
                pub fn new<M>(
                    metric: &M,
                    aggregate: Aggregate,
                    direction: Direction,
                    split: Split,
                ) -> Self
                where
                    M: Metric,
                {
                    Self {
                        current: None,
                        name: metric.name(),
                        aggregate,
                        direction,
                        split,
                    }
                }
            }
            impl CheckpointingStrategy for MetricCheckpointingStrategy {
                fn checkpointing(
                    &mut self,
                    epoch: usize,
                    store: &EventStoreClient,
                ) -> Vec<CheckpointingAction> {
                    let best_epoch = match store
                        .find_epoch(
                            &self.name,
                            self.aggregate,
                            self.direction,
                            &self.split,
                        )
                    {
                        Some(epoch_best) => epoch_best,
                        None => epoch,
                    };
                    let mut actions = Vec::new();
                    if let Some(current) = self.current && current != best_epoch {
                        actions.push(CheckpointingAction::Delete(current));
                    }
                    if best_epoch == epoch {
                        actions.push(CheckpointingAction::Save);
                    }
                    self.current = Some(best_epoch);
                    actions
                }
            }
        }
        pub use base::*;
        pub use composed::*;
        pub use lastn::*;
        pub use metric::*;
    }
    pub use async_checkpoint::*;
    pub use base::*;
    pub use file::*;
    pub use strategy::*;
}
pub(crate) mod components {
    use crate::{InferenceStep, TrainStep};
    use burn_core::module::AutodiffModule;
    use burn_optim::{Optimizer, lr_scheduler::LrScheduler};
    use std::marker::PhantomData;
    /// Components used for a model to learn, grouped in one trait.
    pub trait LearningComponentsTypes {
        /// The learning rate scheduler used for training.
        type LrScheduler: LrScheduler + 'static;
        /// The model to train.
        type Model: TrainStep
            + InferenceStep
            + AutodiffModule
            + core::fmt::Display
            + 'static;
        /// The optimizer used for training.
        type Optimizer: Optimizer<Self::Model> + 'static;
    }
    /// Concrete type that implements the [LearningComponentsTypes](LearningComponentsTypes) trait.
    pub struct LearningComponentsMarker<LR, M, O> {
        _lr_scheduler: PhantomData<LR>,
        _model: PhantomData<M>,
        _optimizer: PhantomData<O>,
    }
    impl<LR, M, O> LearningComponentsTypes for LearningComponentsMarker<LR, M, O>
    where
        LR: LrScheduler + 'static,
        M: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static,
        O: Optimizer<M> + 'static,
    {
        type LrScheduler = LR;
        type Model = M;
        type Optimizer = O;
    }
    /// The model used for training.
    pub type TrainingModel<LC> = <LC as LearningComponentsTypes>::Model;
    /// The non-autodiff model.
    pub(crate) type InferenceModel<LC> = <LC as LearningComponentsTypes>::Model;
    /// Type for training input.
    pub(crate) type TrainingModelInput<LC> = <<LC as LearningComponentsTypes>::Model as TrainStep>::Input;
    /// Type for inference input.
    pub(crate) type InferenceModelInput<LC> = <<LC as LearningComponentsTypes>::Model as InferenceStep>::Input;
    /// Type for training output.
    pub(crate) type TrainingModelOutput<LC> = <<LC as LearningComponentsTypes>::Model as TrainStep>::Output;
    /// Type for inference output.
    pub(crate) type InferenceModelOutput<LC> = <<LC as LearningComponentsTypes>::Model as InferenceStep>::Output;
}
/// Renderer modules to display metrics and training information.
pub mod renderer {
    use std::io::IsTerminal;
    mod base {
        use std::sync::Arc;
        use crate::{
            LearnerSummary, logger::{EvaluationProgressLogger, TrainingProgressLogger},
            metric::{MetricDefinition, MetricEntry, NumericEntry},
        };
        /// Trait for rendering metrics.
        pub trait MetricsRendererTraining: Send + Sync {
            /// Updates the training metric state.
            ///
            /// # Arguments
            ///
            /// * `state` - The metric state.
            fn update_train(&mut self, state: MetricState);
            /// Updates the validation metric state.
            ///
            /// # Arguments
            ///
            /// * `state` - The metric state.
            fn update_valid(&mut self, state: MetricState);
            /// Callback method invoked when training ends, whether it
            /// completed successfully or was interrupted.
            ///
            /// # Returns
            ///
            /// A result indicating whether the end-of-training actions were successful.
            fn on_train_end(
                &mut self,
                summary: Option<LearnerSummary>,
            ) -> Result<(), Box<dyn core::error::Error>> {
                default_summary_action(summary);
                Ok(())
            }
        }
        /// A renderer that can be used for both training and evaluation.
        pub trait MetricsRenderer: MetricsRendererEvaluation + MetricsRendererTraining + TrainingProgressLogger + EvaluationProgressLogger {
            /// Keep the renderer from automatically closing, requiring manual action to close it.
            fn manual_close(&mut self);
            /// Register a new metric.
            fn register_metric(&mut self, definition: MetricDefinition);
        }
        /// The name of an evaluation.
        ///
        /// This is going to group metrics together for easier analysis.
        pub struct EvaluationName {
            pub(crate) name: Arc<String>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for EvaluationName {
            #[inline]
            fn clone(&self) -> EvaluationName {
                EvaluationName {
                    name: ::core::clone::Clone::clone(&self.name),
                }
            }
        }
        impl EvaluationName {
            /// Creates a new evaluation name.
            pub fn new<S: core::fmt::Display>(s: S) -> Self {
                Self {
                    name: Arc::new(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0}", s))
                        }),
                    ),
                }
            }
            /// Returns the evaluation name.
            pub fn as_str(&self) -> &str {
                &self.name
            }
        }
        impl core::fmt::Display for EvaluationName {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&self.name)
            }
        }
        /// Trait for rendering metrics.
        pub trait MetricsRendererEvaluation: Send + Sync {
            /// Updates the testing metric state.
            ///
            /// # Arguments
            ///
            /// * `state` - The metric state.
            fn update_test(&mut self, name: EvaluationName, state: MetricState);
            /// Callback method invoked when testing ends, whether it
            /// completed successfully or was interrupted.
            ///
            /// # Returns
            ///
            /// A result indicating whether the end-of-testing actions were successful.
            fn on_test_end(
                &mut self,
                summary: Option<LearnerSummary>,
            ) -> Result<(), Box<dyn core::error::Error>> {
                default_summary_action(summary);
                Ok(())
            }
        }
        /// The state of a metric.
        pub enum MetricState {
            /// A generic metric.
            Generic(MetricEntry),
            /// A numeric metric.
            Numeric(MetricEntry, NumericEntry),
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for MetricState {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    MetricState::Generic(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Generic",
                            &__self_0,
                        )
                    }
                    MetricState::Numeric(__self_0, __self_1) => {
                        ::core::fmt::Formatter::debug_tuple_field2_finish(
                            f,
                            "Numeric",
                            __self_0,
                            &__self_1,
                        )
                    }
                }
            }
        }
        fn default_summary_action(summary: Option<LearnerSummary>) {
            if let Some(summary) = summary {
                {
                    ::std::io::_print(format_args!("{0}\n", summary));
                };
            }
        }
    }
    pub use base::*;
    pub(crate) mod cli {
        use burn_core::data::dataloader::Progress;
        use crate::{
            logger::{EvaluationProgressLogger, ProgressSnapshot, TrainingProgressLogger},
            renderer::{
                MetricState, MetricsRenderer, MetricsRendererEvaluation,
                MetricsRendererTraining,
            },
        };
        /// A simple renderer for when the cli feature is not enabled.
        pub struct CliMetricsRenderer {
            training_progress: ProgressSnapshot,
            eval_progress: ProgressSnapshot,
        }
        #[allow(clippy::new_without_default)]
        impl CliMetricsRenderer {
            /// Create a new instance.
            pub fn new() -> Self {
                let init = Progress::new(0, 0, Some(String::new()));
                Self {
                    training_progress: ProgressSnapshot::new(init.clone(), init.clone()),
                    eval_progress: ProgressSnapshot::new(init.clone(), init),
                }
            }
        }
        impl MetricsRendererTraining for CliMetricsRenderer {
            fn update_train(&mut self, _state: MetricState) {}
            fn update_valid(&mut self, _state: MetricState) {}
        }
        impl TrainingProgressLogger for CliMetricsRenderer {
            fn start(&mut self, total_epochs: usize, total_items: Option<usize>) {
                self.training_progress.global = Progress::new(
                    1,
                    total_epochs,
                    Some("epochs".to_string()),
                );
                if let Some(items) = total_items {
                    self.training_progress.split = Progress::new(
                        0,
                        items,
                        Some("items".to_string()),
                    );
                }
                {
                    ::std::io::_print(
                        format_args!("Starting training for {0} epochs.\n", total_epochs),
                    );
                };
            }
            fn start_split(&mut self, split_name: &str, total_items: usize) {
                self.training_progress.split = Progress::new(
                    0,
                    total_items,
                    Some("items".to_string()),
                );
                {
                    ::std::io::_print(
                        format_args!(
                            "Starting split \'{0}\' with {1} items.\n",
                            split_name,
                            total_items,
                        ),
                    );
                };
            }
            fn update_split(&mut self, items_processed: usize) {
                let total = self.training_progress.split.items_total;
                let unit = self.training_progress.split.unit.clone();
                self.training_progress.split = Progress::new(
                    items_processed,
                    total,
                    unit,
                );
                if self.training_progress.global.items_total == 0 {
                    self.training_progress.global = self.training_progress.split.clone();
                }
                {
                    ::std::io::_print(format_args!("{0:?}\n", self.training_progress));
                };
            }
            fn update_epoch(&mut self, epoch: usize) {
                let total = self.training_progress.global.items_total;
                let unit = self.training_progress.global.unit.clone();
                self.training_progress.global = Progress::new(epoch + 1, total, unit);
            }
            fn end_split(&mut self) {
                {
                    ::std::io::_print(format_args!("Split ended.\n"));
                };
            }
            fn end(&mut self) {
                {
                    ::std::io::_print(format_args!("Training ended.\n"));
                };
            }
            fn log_event_training(&mut self, _event: String) {}
        }
        impl EvaluationProgressLogger for CliMetricsRenderer {
            fn start_global_progress(&mut self, total_tests: usize) {
                self.eval_progress.global = Progress::new(
                    0,
                    total_tests,
                    Some("tests".to_string()),
                );
                {
                    ::std::io::_print(
                        format_args!(
                            "Starting evaluation with {0} test(s).\n",
                            total_tests,
                        ),
                    );
                };
            }
            fn start_test(&mut self, name: &str, total_items: usize) {
                let current = self.eval_progress.global.items_processed + 1;
                let total = self.eval_progress.global.items_total;
                self.eval_progress.global = Progress::new(
                    current,
                    total,
                    Some("tests".to_string()),
                );
                self.eval_progress.split = Progress::new(
                    0,
                    total_items,
                    Some("items".to_string()),
                );
                {
                    ::std::io::_print(
                        format_args!(
                            "Starting test \'{0}\' with {1} items.\n",
                            name,
                            total_items,
                        ),
                    );
                };
            }
            fn update_test_progress(&mut self, items_processed: usize) {
                let total = self.eval_progress.split.items_total;
                let unit = self.eval_progress.split.unit.clone();
                self.eval_progress.split = Progress::new(items_processed, total, unit);
                {
                    ::std::io::_print(format_args!("{0:?}\n", self.eval_progress));
                };
            }
            fn end_test(&mut self) {}
            fn end_global_progress(&mut self) {}
            fn log_event_evaluation(&mut self, _event: String) {}
        }
        impl MetricsRendererEvaluation for CliMetricsRenderer {
            fn update_test(
                &mut self,
                _name: super::EvaluationName,
                _state: MetricState,
            ) {}
        }
        impl MetricsRenderer for CliMetricsRenderer {
            fn manual_close(&mut self) {}
            fn register_metric(&mut self, _definition: crate::metric::MetricDefinition) {}
        }
    }
    pub use cli::*;
    /// The tui renderer
    pub mod tui {
        mod base {
            use std::sync::Arc;
            use super::{
                ControlsView, NumericMetricView, ProgressBarView, StatusView,
                TerminalFrame, TextMetricView,
            };
            use ratatui::{
                prelude::{Constraint, Direction, Layout, Rect},
                style::Color,
            };
            pub(crate) struct MetricsView<'a> {
                metric_numeric: NumericMetricView<'a>,
                metric_text: TextMetricView<'a>,
                progress: ProgressBarView,
                controls: ControlsView,
                status: StatusView,
            }
            impl<'a> MetricsView<'a> {
                ///Constructs a new `MetricsView`.
                pub fn new(
                    metric_numeric: NumericMetricView<'a>,
                    metric_text: TextMetricView<'a>,
                    progress: ProgressBarView,
                    controls: ControlsView,
                    status: StatusView,
                ) -> Self {
                    MetricsView {
                        metric_numeric: metric_numeric,
                        metric_text: metric_text,
                        progress: progress,
                        controls: controls,
                        status: status,
                    }
                }
            }
            impl MetricsView<'_> {
                pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Min(16), Constraint::Max(4)].as_ref())
                        .split(size);
                    let size_other = chunks[0];
                    let size_progress = chunks[1];
                    let chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints(
                            [Constraint::Percentage(38), Constraint::Percentage(62)]
                                .as_ref(),
                        )
                        .split(size_other);
                    let size_other = chunks[0];
                    let size_metric_numeric = chunks[1];
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints(
                            [Constraint::Max(5), Constraint::Min(6), Constraint::Max(6)]
                                .as_ref(),
                        )
                        .split(size_other);
                    let size_controls = chunks[0];
                    let size_metric_text = chunks[1];
                    let size_status = chunks[2];
                    self.metric_numeric.render(frame, size_metric_numeric);
                    self.metric_text.render(frame, size_metric_text);
                    self.controls.render(frame, size_controls);
                    self.progress.render(frame, size_progress);
                    self.status.render(frame, size_status);
                }
            }
            pub(crate) enum TuiSplit {
                Train,
                Valid,
                Test,
            }
            #[automatically_derived]
            impl ::core::hash::Hash for TuiSplit {
                #[inline]
                fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    ::core::hash::Hash::hash(&__self_discr, state)
                }
            }
            #[automatically_derived]
            #[doc(hidden)]
            unsafe impl ::core::clone::TrivialClone for TuiSplit {}
            #[automatically_derived]
            impl ::core::clone::Clone for TuiSplit {
                #[inline]
                fn clone(&self) -> TuiSplit {
                    *self
                }
            }
            #[automatically_derived]
            impl ::core::marker::Copy for TuiSplit {}
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for TuiSplit {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for TuiSplit {
                #[inline]
                fn eq(&self, other: &TuiSplit) -> bool {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    __self_discr == __arg1_discr
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for TuiSplit {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {}
            }
            #[automatically_derived]
            impl ::core::cmp::PartialOrd for TuiSplit {
                #[inline]
                fn partial_cmp(
                    &self,
                    other: &TuiSplit,
                ) -> ::core::option::Option<::core::cmp::Ordering> {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    ::core::cmp::PartialOrd::partial_cmp(&__self_discr, &__arg1_discr)
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Ord for TuiSplit {
                #[inline]
                fn cmp(&self, other: &TuiSplit) -> ::core::cmp::Ordering {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    ::core::cmp::Ord::cmp(&__self_discr, &__arg1_discr)
                }
            }
            pub(crate) enum TuiGroup {
                Default,
                Named(Arc<String>),
            }
            #[automatically_derived]
            impl ::core::hash::Hash for TuiGroup {
                #[inline]
                fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    ::core::hash::Hash::hash(&__self_discr, state);
                    match self {
                        TuiGroup::Named(__self_0) => {
                            ::core::hash::Hash::hash(__self_0, state)
                        }
                        _ => {}
                    }
                }
            }
            #[automatically_derived]
            impl ::core::clone::Clone for TuiGroup {
                #[inline]
                fn clone(&self) -> TuiGroup {
                    match self {
                        TuiGroup::Default => TuiGroup::Default,
                        TuiGroup::Named(__self_0) => {
                            TuiGroup::Named(::core::clone::Clone::clone(__self_0))
                        }
                    }
                }
            }
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for TuiGroup {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for TuiGroup {
                #[inline]
                fn eq(&self, other: &TuiGroup) -> bool {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    __self_discr == __arg1_discr
                        && match (self, other) {
                            (TuiGroup::Named(__self_0), TuiGroup::Named(__arg1_0)) => {
                                __self_0 == __arg1_0
                            }
                            _ => true,
                        }
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for TuiGroup {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {
                    let _: ::core::cmp::AssertParamIsEq<Arc<String>>;
                }
            }
            #[automatically_derived]
            impl ::core::cmp::PartialOrd for TuiGroup {
                #[inline]
                fn partial_cmp(
                    &self,
                    other: &TuiGroup,
                ) -> ::core::option::Option<::core::cmp::Ordering> {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    match (self, other) {
                        (TuiGroup::Named(__self_0), TuiGroup::Named(__arg1_0)) => {
                            ::core::cmp::PartialOrd::partial_cmp(__self_0, __arg1_0)
                        }
                        _ => {
                            ::core::cmp::PartialOrd::partial_cmp(
                                &__self_discr,
                                &__arg1_discr,
                            )
                        }
                    }
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Ord for TuiGroup {
                #[inline]
                fn cmp(&self, other: &TuiGroup) -> ::core::cmp::Ordering {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    match ::core::cmp::Ord::cmp(&__self_discr, &__arg1_discr) {
                        ::core::cmp::Ordering::Equal => {
                            match (self, other) {
                                (TuiGroup::Named(__self_0), TuiGroup::Named(__arg1_0)) => {
                                    ::core::cmp::Ord::cmp(__self_0, __arg1_0)
                                }
                                _ => ::core::cmp::Ordering::Equal,
                            }
                        }
                        cmp => cmp,
                    }
                }
            }
            pub(crate) struct TuiTag {
                pub(crate) split: TuiSplit,
                pub(crate) group: TuiGroup,
            }
            impl TuiTag {
                ///Constructs a new `TuiTag`.
                pub fn new(split: TuiSplit, group: TuiGroup) -> Self {
                    TuiTag {
                        split: split,
                        group: group,
                    }
                }
            }
            #[automatically_derived]
            impl ::core::hash::Hash for TuiTag {
                #[inline]
                fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) {
                    ::core::hash::Hash::hash(&self.split, state);
                    ::core::hash::Hash::hash(&self.group, state)
                }
            }
            #[automatically_derived]
            impl ::core::clone::Clone for TuiTag {
                #[inline]
                fn clone(&self) -> TuiTag {
                    TuiTag {
                        split: ::core::clone::Clone::clone(&self.split),
                        group: ::core::clone::Clone::clone(&self.group),
                    }
                }
            }
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for TuiTag {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for TuiTag {
                #[inline]
                fn eq(&self, other: &TuiTag) -> bool {
                    self.split == other.split && self.group == other.group
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for TuiTag {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {
                    let _: ::core::cmp::AssertParamIsEq<TuiSplit>;
                    let _: ::core::cmp::AssertParamIsEq<TuiGroup>;
                }
            }
            #[automatically_derived]
            impl ::core::cmp::PartialOrd for TuiTag {
                #[inline]
                fn partial_cmp(
                    &self,
                    other: &TuiTag,
                ) -> ::core::option::Option<::core::cmp::Ordering> {
                    match ::core::cmp::PartialOrd::partial_cmp(
                        &self.split,
                        &other.split,
                    ) {
                        ::core::option::Option::Some(::core::cmp::Ordering::Equal) => {
                            ::core::cmp::PartialOrd::partial_cmp(
                                &self.group,
                                &other.group,
                            )
                        }
                        cmp => cmp,
                    }
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Ord for TuiTag {
                #[inline]
                fn cmp(&self, other: &TuiTag) -> ::core::cmp::Ordering {
                    match ::core::cmp::Ord::cmp(&self.split, &other.split) {
                        ::core::cmp::Ordering::Equal => {
                            ::core::cmp::Ord::cmp(&self.group, &other.group)
                        }
                        cmp => cmp,
                    }
                }
            }
            impl core::fmt::Display for TuiTag {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match &self.group {
                        TuiGroup::Default => f.write_fmt(format_args!("{0}", self.split)),
                        TuiGroup::Named(group) => {
                            f.write_fmt(format_args!("{0} - {1}", self.split, group))
                        }
                    }
                }
            }
            impl core::fmt::Display for TuiGroup {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        TuiGroup::Default => f.write_str(""),
                        TuiGroup::Named(group) => {
                            f.write_fmt(format_args!("{0} ", group))
                        }
                    }
                }
            }
            impl core::fmt::Display for TuiSplit {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        TuiSplit::Train => f.write_str("Train"),
                        TuiSplit::Valid => f.write_str("Valid"),
                        TuiSplit::Test => f.write_str("Test"),
                    }
                }
            }
            impl TuiSplit {
                pub(crate) fn color(&self) -> Color {
                    match self {
                        TuiSplit::Train => Color::LightRed,
                        TuiSplit::Valid => Color::LightBlue,
                        TuiSplit::Test => Color::LightGreen,
                    }
                }
            }
        }
        mod controls {
            use super::TerminalFrame;
            use ratatui::{
                prelude::{Alignment, Rect},
                style::{Color, Style, Stylize},
                text::{Line, Span},
                widgets::{Block, Borders, Paragraph, Wrap},
            };
            /// Controls view.
            pub(crate) struct ControlsView;
            impl ControlsView {
                /// Render the view.
                pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
                    let lines = ::alloc::boxed::box_assume_init_into_vec_unsafe(
                        ::alloc::intrinsics::write_box_via_move(
                            ::alloc::boxed::Box::new_uninit(),
                            [
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            Span::from(" Quit          : ").yellow().bold(),
                                            Span::from("q  ").bold(),
                                            Span::from("  Stop the training.").italic(),
                                        ],
                                    ),
                                ),
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            Span::from(" Plots Metrics : ").yellow().bold(),
                                            Span::from("⬅ ➡").bold(),
                                            Span::from("  Switch between metrics.").italic(),
                                        ],
                                    ),
                                ),
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            Span::from(" Plots Type    : ").yellow().bold(),
                                            Span::from("⬆ ⬇").bold(),
                                            Span::from("  Switch between types.").italic(),
                                        ],
                                    ),
                                ),
                            ],
                        ),
                    );
                    let paragraph = Paragraph::new(
                            lines.into_iter().map(Line::from).collect::<Vec<_>>(),
                        )
                        .alignment(Alignment::Left)
                        .wrap(Wrap { trim: false })
                        .style(Style::default().fg(Color::Gray))
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .style(Style::default().fg(Color::Gray))
                                .title_alignment(Alignment::Left)
                                .title("Controls"),
                        );
                    frame.render_widget(paragraph, size);
                }
            }
        }
        mod full_history {
            use super::PlotAxes;
            use crate::{metric::NumericEntry, renderer::tui::{TuiSplit, TuiTag}};
            use ratatui::{
                style::{Color, Style},
                symbols, widgets::{Bar, Dataset, GraphType},
            };
            use std::collections::BTreeMap;
            /// A plot that shows the full history at a reduced resolution.
            pub(crate) struct FullHistoryPlot {
                pub(crate) axes: PlotAxes,
                points: BTreeMap<TuiTag, FullHistoryPoints>,
                max_samples: usize,
                max_samples_ratio: BTreeMap<TuiSplit, f64>,
                next_x_state: usize,
            }
            struct FullHistoryPoints {
                min_x: f64,
                max_x: f64,
                min_y: f64,
                max_y: f64,
                avg_sum: f64,
                avg_counter: f64,
                points: Vec<(f64, f64)>,
                max_samples: usize,
                step_size: usize,
            }
            impl FullHistoryPlot {
                /// Create a new history plot.
                pub(crate) fn new(max_samples: usize) -> Self {
                    Self {
                        points: BTreeMap::default(),
                        axes: PlotAxes::default(),
                        max_samples,
                        max_samples_ratio: BTreeMap::default(),
                        next_x_state: 0,
                    }
                }
                /// Update the maximum amount of sample to display for the validation points.
                ///
                /// This is necessary if we want the validation line to have the same point density as the
                /// training line.
                pub(crate) fn update_max_sample(&mut self, split: TuiSplit, ratio: f64) {
                    self.max_samples_ratio.insert(split, ratio);
                    self.points
                        .iter_mut()
                        .filter(|(tag, _)| tag.split == split)
                        .for_each(|(_, points)| {
                            points.max_samples = (self.max_samples as f64 * ratio)
                                as usize;
                        });
                }
                /// Register a training data point.
                pub(crate) fn push(&mut self, tag: TuiTag, data: NumericEntry) {
                    let x_current = self.next_x();
                    let points = match self.points.get_mut(&tag) {
                        Some(val) => val,
                        None => {
                            let max_samples = self
                                .max_samples_ratio
                                .get(&tag.split)
                                .map(|ratio| (*ratio * self.max_samples as f64) as usize)
                                .unwrap_or(self.max_samples);
                            self.points
                                .insert(tag.clone(), FullHistoryPoints::new(max_samples));
                            self.points.get_mut(&tag).unwrap()
                        }
                    };
                    points.push((x_current, data));
                    self.update_bounds();
                }
                pub(crate) fn datasets(&self) -> Vec<Dataset<'_>> {
                    let mut datasets = Vec::with_capacity(2);
                    for (tag, points) in self.points.iter() {
                        datasets
                            .push(
                                points
                                    .dataset(
                                        ::alloc::__export::must_use({
                                            ::alloc::fmt::format(format_args!("{0}", tag))
                                        }),
                                        tag.split.color(),
                                    ),
                            );
                    }
                    datasets
                }
                pub(crate) fn bars(
                    &self,
                    max: u64,
                    bar_width: &mut usize,
                ) -> Vec<Bar<'_>> {
                    let mut bars = Vec::new();
                    for (tag, points) in self.points.iter() {
                        if let Some((bar, width)) = points.bar(tag, max) {
                            *bar_width = usize::max(*bar_width, width);
                            bars.push(bar);
                        }
                    }
                    bars
                }
                fn next_x(&mut self) -> f64 {
                    let value = self.next_x_state;
                    self.next_x_state += 1;
                    value as f64
                }
                fn update_bounds(&mut self) {
                    let (mut x_min, mut x_max) = (f64::MAX, f64::MIN);
                    let (mut y_min, mut y_max) = (f64::MAX, f64::MIN);
                    for points in self.points.values() {
                        x_min = f64::min(x_min, points.min_x);
                        x_max = f64::max(x_max, points.max_x);
                        y_min = f64::min(y_min, points.min_y);
                        y_max = f64::max(y_max, points.max_y);
                    }
                    self.axes.update_bounds((x_min, x_max), (y_min, y_max));
                }
            }
            impl FullHistoryPoints {
                fn new(max_samples: usize) -> Self {
                    Self {
                        min_x: 0.,
                        max_x: 0.,
                        min_y: f64::MAX,
                        max_y: f64::MIN,
                        avg_sum: 0.0,
                        avg_counter: 0.0,
                        points: Vec::with_capacity(max_samples),
                        max_samples,
                        step_size: 1,
                    }
                }
                fn push(&mut self, (x, y): (f64, NumericEntry)) {
                    if !(x as usize).is_multiple_of(self.step_size) {
                        return;
                    }
                    let y = match y {
                        NumericEntry::Value(val) => {
                            self.avg_sum += val;
                            self.avg_counter += 1.0;
                            val
                        }
                        NumericEntry::Aggregated { aggregated_value, count } => {
                            self.avg_sum += aggregated_value * count as f64;
                            self.avg_counter += count as f64;
                            aggregated_value
                        }
                    };
                    if x > self.max_x {
                        self.max_x = x;
                    }
                    if x < self.min_x {
                        self.min_x = x;
                    }
                    if y > self.max_y {
                        self.max_y = y;
                    }
                    if y < self.min_y {
                        self.min_y = y;
                    }
                    self.points.push((x, y));
                    if self.points.len() > self.max_samples {
                        self.resize();
                    }
                }
                /// We keep only half the points and we double the step size.
                ///
                /// This ensure that we have the same amount of points across the X axis.
                fn resize(&mut self) {
                    let mut points = Vec::with_capacity(self.max_samples / 2);
                    let mut max_x = f64::MIN;
                    let mut max_y = f64::MIN;
                    let mut min_x = f64::MAX;
                    let mut min_y = f64::MAX;
                    for (i, (x, y)) in self
                        .points
                        .drain(0..self.points.len())
                        .enumerate()
                    {
                        if i % 2 == 0 {
                            if x > max_x {
                                max_x = x;
                            }
                            if x < min_x {
                                min_x = x;
                            }
                            if y > max_y {
                                max_y = y;
                            }
                            if y < min_y {
                                min_y = y;
                            }
                            points.push((x, y));
                        }
                    }
                    self.points = points;
                    self.step_size *= 2;
                    self.min_x = min_x;
                    self.max_x = max_x;
                    self.min_y = min_y;
                    self.max_y = max_y;
                }
                fn dataset<'a>(&'a self, name: String, color: Color) -> Dataset<'a> {
                    Dataset::default()
                        .name(name)
                        .marker(symbols::Marker::Braille)
                        .style(Style::default().fg(color).bold())
                        .graph_type(GraphType::Line)
                        .data(&self.points)
                }
                fn bar<'a>(
                    &'a self,
                    tag: &TuiTag,
                    max: u64,
                ) -> Option<(Bar<'a>, usize)> {
                    if self.avg_sum == 0.0 {
                        return None;
                    }
                    let label = ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0}", tag))
                    });
                    let width = usize::max(label.len(), 7);
                    let factor = max as f64;
                    let avg = self.avg_sum / self.avg_counter;
                    Some((
                        Bar::default()
                            .value((avg * factor) as u64)
                            .style(tag.split.color())
                            .text_value(
                                ::alloc::__export::must_use({
                                    ::alloc::fmt::format(format_args!("{0:.2}", avg))
                                }),
                            )
                            .label(label),
                        width,
                    ))
                }
            }
        }
        mod metric_numeric {
            use crate::{
                logger::ProgressSnapshot, metric::{MetricName, NumericEntry},
                renderer::tui::TuiTag,
            };
            use super::{FullHistoryPlot, RecentHistoryPlot, TerminalFrame, TuiSplit};
            use ratatui::{
                crossterm::event::{
                    Event, KeyCode, KeyEvent, KeyEventKind, MouseButton, MouseEvent,
                    MouseEventKind,
                },
                prelude::{Alignment, Constraint, Direction, Layout, Position, Rect},
                style::{Color, Modifier, Style, Stylize},
                text::Line,
                widgets::{
                    Axis, BarChart, BarGroup, Block, Borders, Chart, LegendPosition,
                    Padding, Paragraph, Tabs, Widget,
                },
            };
            use std::collections::BTreeMap;
            use unicode_width::UnicodeWidthStr;
            /// 1 cell of padding on each side of a tab title, matching ratatui's default `Tabs` widget.
            const TAB_PADDING: u16 = 2;
            /// 1-cell `│` divider between adjacent tabs in ratatui's default `Tabs` widget.
            const TAB_DIVIDER: u16 = 1;
            /// 1000 seems to be required to see some improvement.
            const MAX_NUM_SAMPLES_RECENT: usize = 1000;
            /// 250 seems to be the right resolution when plotting all history.
            /// Otherwise, there is too much points and the lines arent't smooth enough.
            const MAX_NUM_SAMPLES_FULL: usize = 250;
            enum ChevronSide {
                Left,
                Right,
            }
            #[automatically_derived]
            #[doc(hidden)]
            unsafe impl ::core::clone::TrivialClone for ChevronSide {}
            #[automatically_derived]
            impl ::core::clone::Clone for ChevronSide {
                #[inline]
                fn clone(&self) -> ChevronSide {
                    *self
                }
            }
            #[automatically_derived]
            impl ::core::marker::Copy for ChevronSide {}
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for ChevronSide {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for ChevronSide {
                #[inline]
                fn eq(&self, other: &ChevronSide) -> bool {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    __self_discr == __arg1_discr
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for ChevronSide {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {}
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for ChevronSide {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    ::core::fmt::Formatter::write_str(
                        f,
                        match self {
                            ChevronSide::Left => "Left",
                            ChevronSide::Right => "Right",
                        },
                    )
                }
            }
            enum HoverTarget {
                Tab(usize),
                Chevron(ChevronSide),
            }
            #[automatically_derived]
            #[doc(hidden)]
            unsafe impl ::core::clone::TrivialClone for HoverTarget {}
            #[automatically_derived]
            impl ::core::clone::Clone for HoverTarget {
                #[inline]
                fn clone(&self) -> HoverTarget {
                    let _: ::core::clone::AssertParamIsClone<usize>;
                    let _: ::core::clone::AssertParamIsClone<ChevronSide>;
                    *self
                }
            }
            #[automatically_derived]
            impl ::core::marker::Copy for HoverTarget {}
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for HoverTarget {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for HoverTarget {
                #[inline]
                fn eq(&self, other: &HoverTarget) -> bool {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    __self_discr == __arg1_discr
                        && match (self, other) {
                            (HoverTarget::Tab(__self_0), HoverTarget::Tab(__arg1_0)) => {
                                __self_0 == __arg1_0
                            }
                            (
                                HoverTarget::Chevron(__self_0),
                                HoverTarget::Chevron(__arg1_0),
                            ) => __self_0 == __arg1_0,
                            _ => unsafe { ::core::intrinsics::unreachable() }
                        }
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for HoverTarget {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {
                    let _: ::core::cmp::AssertParamIsEq<usize>;
                    let _: ::core::cmp::AssertParamIsEq<ChevronSide>;
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for HoverTarget {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    match self {
                        HoverTarget::Tab(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "Tab",
                                &__self_0,
                            )
                        }
                        HoverTarget::Chevron(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "Chevron",
                                &__self_0,
                            )
                        }
                    }
                }
            }
            /// Hit-test geometry and hover state for the tab strip, populated by `render_tab_strip`
            /// on every frame and consumed by `on_mouse_event`.
            pub(crate) struct TabStripState {
                hovered: Option<HoverTarget>,
                tab_rects: Vec<Rect>,
                chevron_left: Option<Rect>,
                chevron_right: Option<Rect>,
            }
            #[automatically_derived]
            impl ::core::default::Default for TabStripState {
                #[inline]
                fn default() -> TabStripState {
                    TabStripState {
                        hovered: ::core::default::Default::default(),
                        tab_rects: ::core::default::Default::default(),
                        chevron_left: ::core::default::Default::default(),
                        chevron_right: ::core::default::Default::default(),
                    }
                }
            }
            /// Numeric metrics state that handles creating plots.
            pub(crate) struct NumericMetricsState {
                data: BTreeMap<MetricName, (RecentHistoryPlot, FullHistoryPlot)>,
                names: Vec<MetricName>,
                selected: usize,
                kind: PlotKind,
                num_samples_train: Option<usize>,
                num_samples_valid: Option<usize>,
                num_samples_test: Option<usize>,
                epoch: usize,
                strip: TabStripState,
            }
            #[automatically_derived]
            impl ::core::default::Default for NumericMetricsState {
                #[inline]
                fn default() -> NumericMetricsState {
                    NumericMetricsState {
                        data: ::core::default::Default::default(),
                        names: ::core::default::Default::default(),
                        selected: ::core::default::Default::default(),
                        kind: ::core::default::Default::default(),
                        num_samples_train: ::core::default::Default::default(),
                        num_samples_valid: ::core::default::Default::default(),
                        num_samples_test: ::core::default::Default::default(),
                        epoch: ::core::default::Default::default(),
                        strip: ::core::default::Default::default(),
                    }
                }
            }
            /// The kind of plot to display.
            pub(crate) enum PlotKind {
                /// Display the full history of the metric with reduced resolution.
                #[default]
                Full,
                /// Display only the recent history of the metric, but with more resolution.
                Recent,
                Summary,
            }
            #[automatically_derived]
            impl ::core::default::Default for PlotKind {
                #[inline]
                fn default() -> PlotKind {
                    Self::Full
                }
            }
            #[automatically_derived]
            #[doc(hidden)]
            unsafe impl ::core::clone::TrivialClone for PlotKind {}
            #[automatically_derived]
            impl ::core::clone::Clone for PlotKind {
                #[inline]
                fn clone(&self) -> PlotKind {
                    *self
                }
            }
            #[automatically_derived]
            impl ::core::marker::Copy for PlotKind {}
            impl NumericMetricsState {
                /// Register a new training value for the metric with the given name.
                pub(crate) fn push(
                    &mut self,
                    tag: TuiTag,
                    name: MetricName,
                    data: NumericEntry,
                ) {
                    if let Some((recent, full)) = self.data.get_mut(name.as_ref()) {
                        recent.push(tag.clone(), data.current());
                        full.push(tag, data);
                    } else {
                        let mut recent = RecentHistoryPlot::new(MAX_NUM_SAMPLES_RECENT);
                        let mut full = FullHistoryPlot::new(MAX_NUM_SAMPLES_FULL);
                        recent.push(tag.clone(), data.current());
                        full.push(tag, data);
                        self.names.push(name.clone());
                        self.data.insert(name, (recent, full));
                    }
                }
                /// Update the state with the training progress.
                pub(crate) fn update_progress_train(
                    &mut self,
                    progress: &ProgressSnapshot,
                ) {
                    self.epoch = progress.global.items_processed;
                    if self.num_samples_train.is_some() {
                        return;
                    }
                    self.num_samples_train = Some(progress.split.items_total);
                }
                /// Update the state with the validation progress.
                pub(crate) fn update_progress_valid(
                    &mut self,
                    progress: &ProgressSnapshot,
                ) {
                    if self.num_samples_valid.is_some() {
                        return;
                    }
                    if let Some(num_sample_train) = self.num_samples_train {
                        for (_, (_recent, full)) in self.data.iter_mut() {
                            let ratio = progress.split.items_total as f64
                                / num_sample_train as f64;
                            full.update_max_sample(TuiSplit::Valid, ratio);
                        }
                    }
                    self.epoch = progress.global.items_processed;
                    self.num_samples_valid = Some(progress.split.items_total);
                }
                /// Update the state with the testing progress.
                pub(crate) fn update_progress_test(
                    &mut self,
                    progress: &ProgressSnapshot,
                ) {
                    if self.num_samples_test.is_some() {
                        return;
                    }
                    if let Some(num_sample_train) = self.num_samples_train {
                        for (_, (_recent, full)) in self.data.iter_mut() {
                            let ratio = progress.split.items_total as f64
                                / num_sample_train as f64;
                            full.update_max_sample(TuiSplit::Test, ratio);
                        }
                    }
                    self.num_samples_test = Some(progress.split.items_total);
                }
                /// Create a view to display the numeric metrics.
                pub(crate) fn view(&mut self) -> NumericMetricView<'_> {
                    if self.names.is_empty() {
                        return NumericMetricView::None;
                    }
                    match self.kind {
                        PlotKind::Summary => {
                            let chart = Self::bar_chart(
                                &self.names,
                                &self.data,
                                self.selected,
                            );
                            NumericMetricView::BarPlots {
                                titles: &self.names,
                                selected: self.selected,
                                chart,
                                strip: &mut self.strip,
                            }
                        }
                        kind => {
                            let chart = Self::line_chart(
                                &self.names,
                                &self.data,
                                self.selected,
                                kind,
                            );
                            NumericMetricView::LinePlots {
                                titles: &self.names,
                                selected: self.selected,
                                chart,
                                kind,
                                strip: &mut self.strip,
                            }
                        }
                    }
                }
                /// Handle the current event. Returns `true` when visible state changed and a
                /// redraw is warranted, `false` for events that produced no observable change
                /// (so the caller can skip an unnecessary redraw).
                pub(crate) fn on_event(&mut self, event: &Event) -> bool {
                    match event {
                        Event::Key(key) => self.on_key_event(key),
                        Event::Mouse(mouse) => self.on_mouse_event(mouse),
                        _ => false,
                    }
                }
                fn on_key_event(&mut self, key: &KeyEvent) -> bool {
                    match key.kind {
                        KeyEventKind::Release | KeyEventKind::Repeat => {}
                        KeyEventKind::Press => {}
                    }
                    match key.code {
                        KeyCode::Right => {
                            self.next_metric();
                            true
                        }
                        KeyCode::Left => {
                            self.previous_metric();
                            true
                        }
                        KeyCode::Up | KeyCode::Down => {
                            self.switch_kind();
                            true
                        }
                        _ => false,
                    }
                }
                fn on_mouse_event(&mut self, mouse: &MouseEvent) -> bool {
                    let pos = Position::new(mouse.column, mouse.row);
                    let target = hover_target_at(&self.strip, pos);
                    match mouse.kind {
                        MouseEventKind::Moved => {
                            if self.strip.hovered == target {
                                false
                            } else {
                                self.strip.hovered = target;
                                true
                            }
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            match target {
                                Some(HoverTarget::Tab(idx)) => {
                                    self.selected = idx;
                                    true
                                }
                                Some(HoverTarget::Chevron(ChevronSide::Left)) => {
                                    self.previous_metric();
                                    true
                                }
                                Some(HoverTarget::Chevron(ChevronSide::Right)) => {
                                    self.next_metric();
                                    true
                                }
                                None => false,
                            }
                        }
                        _ => false,
                    }
                }
                pub(crate) fn select_by_name(&mut self, name: &MetricName) {
                    if let Some(idx) = self.names.iter().position(|n| n == name) {
                        self.selected = idx;
                    }
                }
                fn switch_kind(&mut self) {
                    self.kind = match self.kind {
                        PlotKind::Full => PlotKind::Recent,
                        PlotKind::Recent => PlotKind::Summary,
                        PlotKind::Summary => PlotKind::Full,
                    };
                }
                fn next_metric(&mut self) {
                    let len = self.data.len();
                    if len == 0 {
                        return;
                    }
                    self.selected = (self.selected + 1) % len;
                }
                fn previous_metric(&mut self) {
                    let len = self.data.len();
                    if len == 0 {
                        return;
                    }
                    if self.selected > 0 {
                        self.selected -= 1;
                    } else {
                        self.selected = len - 1;
                    }
                }
                fn line_chart<'a>(
                    names: &'a [MetricName],
                    data: &'a BTreeMap<MetricName, (RecentHistoryPlot, FullHistoryPlot)>,
                    selected: usize,
                    kind: PlotKind,
                ) -> Chart<'a> {
                    let name = names.get(selected).unwrap();
                    let (recent, full) = data.get(name).unwrap();
                    let (datasets, axes) = match kind {
                        PlotKind::Full => (full.datasets(), &full.axes),
                        PlotKind::Recent => (recent.datasets(), &recent.axes),
                        _ => {
                            ::core::panicking::panic(
                                "internal error: entered unreachable code",
                            )
                        }
                    };
                    Chart::<'a>::new(datasets)
                        .block(Block::default())
                        .x_axis(
                            Axis::default()
                                .style(Style::default().fg(Color::DarkGray))
                                .labels(axes.labels_x.clone().into_iter().map(|s| s.bold()))
                                .bounds(axes.bounds_x),
                        )
                        .y_axis(
                            Axis::default()
                                .style(Style::default().fg(Color::DarkGray))
                                .labels(axes.labels_y.clone().into_iter().map(|s| s.bold()))
                                .bounds(axes.bounds_y),
                        )
                        .legend_position(Some(LegendPosition::Right))
                }
                fn bar_chart<'a>(
                    names: &'a [MetricName],
                    data: &'a BTreeMap<MetricName, (RecentHistoryPlot, FullHistoryPlot)>,
                    selected: usize,
                ) -> BarChart<'a> {
                    let name = names.get(selected).unwrap();
                    let (_recent, full) = data.get(name).unwrap();
                    let mut bar_width = 0;
                    let bars = full.bars(100, &mut bar_width);
                    let data = BarGroup::default().bars(&bars);
                    BarChart::default()
                        .block(Block::default().padding(Padding::new(2, 2, 2, 0)))
                        .bar_width(bar_width as u16)
                        .bar_gap(2)
                        .data(data)
                }
            }
            #[allow(clippy::large_enum_variant)]
            pub(crate) enum NumericMetricView<'a> {
                LinePlots {
                    titles: &'a [MetricName],
                    selected: usize,
                    chart: Chart<'a>,
                    kind: PlotKind,
                    strip: &'a mut TabStripState,
                },
                BarPlots {
                    titles: &'a [MetricName],
                    selected: usize,
                    chart: BarChart<'a>,
                    strip: &'a mut TabStripState,
                },
                None,
            }
            impl NumericMetricView<'_> {
                pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
                    match self {
                        Self::LinePlots { titles, selected, chart, kind, strip } => {
                            let plot_title = match kind {
                                PlotKind::Full => "Full History",
                                PlotKind::Recent => "Recent History",
                                _ => {
                                    ::core::panicking::panic(
                                        "internal error: entered unreachable code",
                                    )
                                }
                            };
                            render_plot_panel(
                                frame,
                                size,
                                "Plots",
                                plot_title,
                                titles,
                                selected,
                                strip,
                                chart,
                            );
                        }
                        Self::BarPlots { titles, selected, chart, strip } => {
                            render_plot_panel(
                                frame,
                                size,
                                "Summary",
                                "Summary",
                                titles,
                                selected,
                                strip,
                                chart,
                            );
                        }
                        Self::None => {}
                    }
                }
            }
            /// Draw the bordered plot panel: tab strip on top, centered plot title, then the chart.
            #[allow(clippy::too_many_arguments)]
            fn render_plot_panel<W: Widget>(
                frame: &mut TerminalFrame<'_>,
                size: Rect,
                block_title: &str,
                plot_title: &str,
                titles: &[MetricName],
                selected: usize,
                strip: &mut TabStripState,
                chart: W,
            ) {
                let block = Block::default()
                    .borders(Borders::ALL)
                    .title(block_title)
                    .title_alignment(Alignment::Left);
                let inner = block.inner(size);
                frame.render_widget(block, size);
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(2),
                        Constraint::Length(1),
                        Constraint::Min(0),
                    ])
                    .split(inner);
                render_tab_strip(frame, chunks[0], titles, selected, strip);
                let title = Paragraph::new(Line::from(plot_title.bold()))
                    .alignment(Alignment::Center);
                frame.render_widget(title, chunks[1]);
                frame.render_widget(chart, chunks[2]);
            }
            /// Render the metric tabs in `area`, scrolling horizontally so the `selected` tab is always
            /// visible. A `‹` / `›` indicator is drawn in a reserved cell on each side when tabs are
            /// hidden off that edge. The hovered tab gets an extra underline, a hovered chevron gets a
            /// brighter foreground. Hit-test rects for the visible tabs and the two chevrons are written
            /// back into `strip` so `on_mouse_event` can route clicks.
            fn render_tab_strip(
                frame: &mut TerminalFrame<'_>,
                area: Rect,
                titles: &[MetricName],
                selected: usize,
                strip: &mut TabStripState,
            ) {
                strip.tab_rects.clear();
                strip.tab_rects.resize(titles.len(), Rect::default());
                strip.chevron_left = None;
                strip.chevron_right = None;
                if titles.is_empty() || area.width == 0 {
                    return;
                }
                let titles_str: Vec<String> = titles
                    .iter()
                    .map(|t| t.to_string())
                    .collect();
                let widths: Vec<u16> = titles_str
                    .iter()
                    .map(|s| tab_cell_width(s))
                    .collect();
                let inner_width = area.width.saturating_sub(2);
                let (start, end) = visible_tab_window(&widths, selected, inner_width);
                if start > 0 {
                    let left = Rect { width: 1, ..area };
                    let color = chevron_color(strip.hovered, ChevronSide::Left);
                    frame
                        .render_widget(
                            Paragraph::new("‹").style(Style::default().fg(color)),
                            left,
                        );
                    strip.chevron_left = Some(left);
                }
                if end < titles.len() {
                    let right = Rect {
                        x: area.x + area.width - 1,
                        width: 1,
                        ..area
                    };
                    let color = chevron_color(strip.hovered, ChevronSide::Right);
                    frame
                        .render_widget(
                            Paragraph::new("›").style(Style::default().fg(color)),
                            right,
                        );
                    strip.chevron_right = Some(right);
                }
                let tabs_area = Rect {
                    x: area.x + 1,
                    width: inner_width,
                    ..area
                };
                let mut x = tabs_area.x;
                let tabs_end = tabs_area.x.saturating_add(tabs_area.width);
                for (i, &w) in (start..end).zip(&widths[start..end]) {
                    let remaining = tabs_end.saturating_sub(x);
                    strip.tab_rects[i] = Rect {
                        x: x.min(tabs_end),
                        y: tabs_area.y,
                        width: w.min(remaining),
                        height: tabs_area.height,
                    };
                    x = x.saturating_add(w).saturating_add(TAB_DIVIDER);
                }
                let tabs = Tabs::new(
                        titles_str[start..end]
                            .iter()
                            .enumerate()
                            .map(|(local, s)| {
                                let span = s.clone().yellow();
                                let span = if strip.hovered
                                    == Some(HoverTarget::Tab(start + local))
                                {
                                    span.underlined()
                                } else {
                                    span
                                };
                                Line::from(
                                    ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                        ::alloc::intrinsics::write_box_via_move(
                                            ::alloc::boxed::Box::new_uninit(),
                                            [span],
                                        ),
                                    ),
                                )
                            }),
                    )
                    .select(selected - start)
                    .highlight_style(
                        Style::default()
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                            .fg(Color::LightYellow),
                    );
                frame.render_widget(tabs, tabs_area);
            }
            fn chevron_color(hovered: Option<HoverTarget>, side: ChevronSide) -> Color {
                if hovered == Some(HoverTarget::Chevron(side)) {
                    Color::Gray
                } else {
                    Color::DarkGray
                }
            }
            fn hover_target_at(
                strip: &TabStripState,
                pos: Position,
            ) -> Option<HoverTarget> {
                if strip.chevron_left.is_some_and(|r| r.contains(pos)) {
                    return Some(HoverTarget::Chevron(ChevronSide::Left));
                }
                if strip.chevron_right.is_some_and(|r| r.contains(pos)) {
                    return Some(HoverTarget::Chevron(ChevronSide::Right));
                }
                strip
                    .tab_rects
                    .iter()
                    .position(|r| r.contains(pos))
                    .map(HoverTarget::Tab)
            }
            /// Cells consumed by one tab. Title display width plus ratatui's default padding.
            fn tab_cell_width(title: &str) -> u16 {
                u16::try_from(UnicodeWidthStr::width(title) + TAB_PADDING as usize)
                    .unwrap_or(u16::MAX)
            }
            /// Pick the `[start, end)` slice of `widths` to render so the tab at `selected` is visible
            /// inside `available` cells. The selected tab is pinned as far right as fits. `end` is then
            /// grown rightward as far as the remaining space allows. Always returns
            /// `start <= selected < end` when `widths` is non-empty. If a single tab exceeds
            /// `available`, clipping is delegated to ratatui's `Tabs`.
            fn visible_tab_window(
                widths: &[u16],
                selected: usize,
                available: u16,
            ) -> (usize, usize) {
                if widths.is_empty() {
                    return (0, 0);
                }
                let selected = selected.min(widths.len() - 1);
                let available = available as u32;
                let divider = TAB_DIVIDER as u32;
                let mut width: u32 = widths[..=selected]
                    .iter()
                    .map(|&w| w as u32)
                    .sum::<u32>() + selected as u32 * divider;
                let mut start = 0;
                while width > available && start < selected {
                    width -= widths[start] as u32 + divider;
                    start += 1;
                }
                let mut end = selected + 1;
                while end < widths.len() {
                    let next = width + widths[end] as u32 + divider;
                    if next > available {
                        break;
                    }
                    width = next;
                    end += 1;
                }
                (start, end)
            }
        }
        mod metric_text {
            use super::TerminalFrame;
            use crate::{
                metric::{MetricEntry, MetricName},
                renderer::tui::{TuiGroup, TuiSplit},
            };
            use ratatui::{
                crossterm::event::{Event, MouseButton, MouseEventKind},
                prelude::{Alignment, Position, Rect},
                style::{Color, Style, Stylize},
                text::{Line, Span},
                widgets::{Block, Borders, Paragraph, Wrap},
            };
            use std::{collections::BTreeMap, ops::Range, sync::Arc};
            /// Hit-test geometry and hover state for the metrics pane, populated by
            /// `TextMetricView::render` on every frame and consumed by `on_event`.
            pub(crate) struct TextHitState {
                hovered: Option<MetricName>,
                rect: Option<Rect>,
                header_rows: Vec<(MetricName, Range<u16>)>,
            }
            #[automatically_derived]
            impl ::core::default::Default for TextHitState {
                #[inline]
                fn default() -> TextHitState {
                    TextHitState {
                        hovered: ::core::default::Default::default(),
                        rect: ::core::default::Default::default(),
                        header_rows: ::core::default::Default::default(),
                    }
                }
            }
            pub(crate) struct TextMetricsState {
                data: BTreeMap<String, MetricGroup>,
                names: Vec<MetricName>,
                pane: TextHitState,
            }
            #[automatically_derived]
            impl ::core::default::Default for TextMetricsState {
                #[inline]
                fn default() -> TextMetricsState {
                    TextMetricsState {
                        data: ::core::default::Default::default(),
                        names: ::core::default::Default::default(),
                        pane: ::core::default::Default::default(),
                    }
                }
            }
            /// What a mouse event meant for the left pane. Drives both selection routing
            /// (the `Clicked` arm carries the metric name to switch to) and redraw gating
            /// (anything other than `Ignored` should cause a redraw).
            pub(crate) enum TextEventOutcome {
                Clicked(MetricName),
                HoverChanged,
                Ignored,
            }
            struct MetricGroup {
                groups: BTreeMap<TuiGroup, MetricSplits>,
            }
            impl MetricGroup {
                fn new(group: TuiGroup, metric: MetricSplits) -> Self {
                    Self {
                        groups: BTreeMap::from_iter(Some((group, metric))),
                    }
                }
                fn update(
                    &mut self,
                    split: TuiSplit,
                    group: TuiGroup,
                    metric: MetricEntry,
                ) {
                    match self.groups.get_mut(&group) {
                        Some(value) => value.update(split, metric),
                        None => {
                            let value = MetricSplits::new(split, metric);
                            self.groups.insert(group, value);
                        }
                    }
                }
            }
            struct MetricSplits {
                splits: BTreeMap<TuiSplit, MetricEntry>,
            }
            impl MetricSplits {
                fn new(split: TuiSplit, metric: MetricEntry) -> Self {
                    Self {
                        splits: BTreeMap::from_iter(Some((split, metric))),
                    }
                }
                fn update(&mut self, split: TuiSplit, metric: MetricEntry) {
                    self.splits.insert(split, metric);
                }
            }
            impl TextMetricsState {
                pub(crate) fn update(
                    &mut self,
                    split: TuiSplit,
                    group: TuiGroup,
                    metric: MetricEntry,
                    name: Arc<String>,
                ) {
                    if let Some(existing) = self.data.get_mut(name.as_ref()) {
                        existing.update(split, group, metric);
                    } else {
                        let key = name.clone();
                        let value = MetricSplits::new(split, metric);
                        self.names.push(key.clone());
                        self.data
                            .insert(key.to_string(), MetricGroup::new(group, value));
                    }
                }
                pub(crate) fn view(&mut self) -> TextMetricView<'_> {
                    TextMetricView::new(&self.names, &self.data, &mut self.pane)
                }
                /// Updates hover state and reports what the event meant for the left pane.
                pub(crate) fn on_event(&mut self, event: &Event) -> TextEventOutcome {
                    let Event::Mouse(mouse) = event else {
                        return TextEventOutcome::Ignored;
                    };
                    let pos = Position::new(mouse.column, mouse.row);
                    let hit = if self.pane.rect.is_some_and(|pane| pane.contains(pos)) {
                        self.pane
                            .header_rows
                            .iter()
                            .find(|(_, rows)| rows.contains(&pos.y))
                            .map(|(name, _)| name.clone())
                    } else {
                        None
                    };
                    match mouse.kind {
                        MouseEventKind::Moved => {
                            if self.pane.hovered == hit {
                                TextEventOutcome::Ignored
                            } else {
                                self.pane.hovered = hit;
                                TextEventOutcome::HoverChanged
                            }
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            match hit {
                                Some(name) => TextEventOutcome::Clicked(name),
                                None => TextEventOutcome::Ignored,
                            }
                        }
                        _ => TextEventOutcome::Ignored,
                    }
                }
            }
            pub(crate) struct TextMetricView<'a> {
                lines: Vec<Vec<Span<'static>>>,
                /// Index into `lines` of each metric's header row, in display order.
                header_line_indices: Vec<(MetricName, usize)>,
                pane: &'a mut TextHitState,
            }
            impl<'a> TextMetricView<'a> {
                fn new(
                    names: &[MetricName],
                    data: &BTreeMap<String, MetricGroup>,
                    pane: &'a mut TextHitState,
                ) -> Self {
                    let mut lines = Vec::with_capacity(names.len() * 4);
                    let mut header_line_indices = Vec::with_capacity(names.len());
                    let hovered = pane.hovered.as_ref();
                    let start_line = |title: &str, is_hovered: bool| {
                        let span = Span::from(
                                ::alloc::__export::must_use({
                                    ::alloc::fmt::format(format_args!(" {0} ", title))
                                }),
                            )
                            .bold()
                            .yellow();
                        let span = if is_hovered { span.underlined() } else { span };
                        ::alloc::boxed::box_assume_init_into_vec_unsafe(
                            ::alloc::intrinsics::write_box_via_move(
                                ::alloc::boxed::Box::new_uninit(),
                                [span],
                            ),
                        )
                    };
                    let format_line = |
                        group: &TuiGroup,
                        split: &TuiSplit,
                        formatted: &str|
                    {
                        ::alloc::boxed::box_assume_init_into_vec_unsafe(
                            ::alloc::intrinsics::write_box_via_move(
                                ::alloc::boxed::Box::new_uninit(),
                                [
                                    Span::from(
                                            ::alloc::__export::must_use({
                                                ::alloc::fmt::format(format_args!(" {0}{1} ", group, split))
                                            }),
                                        )
                                        .bold(),
                                    Span::from(formatted.to_string()).italic(),
                                ],
                            ),
                        )
                    };
                    for name in names {
                        let is_hovered = hovered
                            .is_some_and(|h| h.as_ref() == name.as_ref());
                        header_line_indices.push((name.clone(), lines.len()));
                        lines.push(start_line(name, is_hovered));
                        let entry = data.get(name.as_ref()).unwrap();
                        for (name, group) in entry.groups.iter() {
                            for (split, entry) in group.splits.iter() {
                                lines
                                    .push(
                                        format_line(name, split, &entry.serialized_entry.formatted),
                                    );
                            }
                        }
                        lines
                            .push(
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [Span::from("")],
                                    ),
                                ),
                            );
                    }
                    Self {
                        lines,
                        header_line_indices,
                        pane,
                    }
                }
                pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
                    let Self { lines, header_line_indices, pane } = self;
                    let text_origin_y = size.y.saturating_add(1);
                    pane.rect = Some(size);
                    pane.header_rows = header_line_indices
                        .into_iter()
                        .map(|(name, line_idx)| {
                            let row = text_origin_y.saturating_add(line_idx as u16);
                            (name, row..row.saturating_add(1))
                        })
                        .collect();
                    let paragraph = Paragraph::new(
                            lines.into_iter().map(Line::from).collect::<Vec<_>>(),
                        )
                        .alignment(Alignment::Left)
                        .wrap(Wrap { trim: false })
                        .block(Block::default().borders(Borders::ALL).title("Metrics"))
                        .style(Style::default().fg(Color::Gray));
                    frame.render_widget(paragraph, size);
                }
            }
        }
        mod plot_utils {
            use crate::metric::format_float;
            const AXIS_TITLE_PRECISION: usize = 2;
            /// The data describing both X and Y axes.
            pub(crate) struct PlotAxes {
                pub(crate) labels_x: Vec<String>,
                pub(crate) labels_y: Vec<String>,
                pub(crate) bounds_x: [f64; 2],
                pub(crate) bounds_y: [f64; 2],
            }
            impl Default for PlotAxes {
                fn default() -> Self {
                    Self {
                        bounds_x: [f64::MAX, f64::MIN],
                        bounds_y: [f64::MAX, f64::MIN],
                        labels_x: Vec::new(),
                        labels_y: Vec::new(),
                    }
                }
            }
            impl PlotAxes {
                /// Update the bounds based on the min max of each X and Y axes with both train and valid data.
                pub(crate) fn update_bounds(
                    &mut self,
                    (x_min, x_max): (f64, f64),
                    (y_min, y_max): (f64, f64),
                ) {
                    self.bounds_x = [x_min, x_max];
                    self.bounds_y = [y_min, y_max];
                    self.labels_x = ::alloc::boxed::box_assume_init_into_vec_unsafe(
                        ::alloc::intrinsics::write_box_via_move(
                            ::alloc::boxed::Box::new_uninit(),
                            [
                                ::alloc::__export::must_use({
                                    ::alloc::fmt::format(format_args!("{0}", x_min))
                                }),
                                ::alloc::__export::must_use({
                                    ::alloc::fmt::format(format_args!("{0}", x_max))
                                }),
                            ],
                        ),
                    );
                    self.labels_y = ::alloc::boxed::box_assume_init_into_vec_unsafe(
                        ::alloc::intrinsics::write_box_via_move(
                            ::alloc::boxed::Box::new_uninit(),
                            [
                                format_float(y_min, AXIS_TITLE_PRECISION),
                                format_float(y_max, AXIS_TITLE_PRECISION),
                            ],
                        ),
                    );
                }
            }
        }
        mod popup {
            use ratatui::{
                crossterm::event::{Event, KeyCode},
                prelude::{Alignment, Constraint, Direction, Layout, Rect},
                style::{Color, Modifier, Style, Stylize},
                text::{Line, Span},
                widgets::{Block, Borders, Paragraph, Wrap},
            };
            use super::TerminalFrame;
            /// Popup callback function.
            pub(crate) trait CallbackFn: Send + Sync {
                /// Call the function and return if the popup state should be reset.
                fn call(&self) -> bool;
            }
            /// Popup callback.
            pub(crate) struct Callback {
                title: String,
                description: String,
                trigger: char,
                callback: Box<dyn CallbackFn>,
            }
            impl Callback {
                /// Create a new popup.
                pub(crate) fn new<T, D, C>(
                    title: T,
                    description: D,
                    trigger: char,
                    callback: C,
                ) -> Self
                where
                    T: Into<String>,
                    D: Into<String>,
                    C: CallbackFn + 'static,
                {
                    Self {
                        title: title.into(),
                        description: description.into(),
                        trigger,
                        callback: Box::new(callback),
                    }
                }
            }
            /// Popup state.
            pub(crate) enum PopupState {
                Empty,
                Full(String, Vec<Callback>),
            }
            impl PopupState {
                /// If the popup is empty.
                pub(crate) fn is_empty(&self) -> bool {
                    #[allow(non_exhaustive_omitted_patterns)]
                    match &self {
                        PopupState::Empty => true,
                        _ => false,
                    }
                }
                /// Handle popup events. Returns `true` when the popup state changed and a
                /// redraw is warranted, `false` for events that left the popup untouched.
                pub(crate) fn on_event(&mut self, event: &Event) -> bool {
                    let mut reset = false;
                    match self {
                        PopupState::Empty => {}
                        PopupState::Full(_, callbacks) => {
                            for callback in callbacks.iter() {
                                if let Event::Key(key) = event
                                    && let KeyCode::Char(key) = &key.code
                                    && &callback.trigger == key && callback.callback.call()
                                {
                                    reset = true;
                                }
                            }
                        }
                    };
                    if reset {
                        *self = Self::Empty;
                    }
                    reset
                }
                /// Create the popup view.
                pub(crate) fn view(&self) -> Option<PopupView<'_>> {
                    match self {
                        PopupState::Empty => None,
                        PopupState::Full(title, callbacks) => {
                            Some(PopupView::new(title, callbacks))
                        }
                    }
                }
            }
            pub(crate) struct PopupView<'a> {
                title: &'a String,
                callbacks: &'a [Callback],
            }
            impl<'a> PopupView<'a> {
                ///Constructs a new `PopupView`.
                pub fn new(title: &'a String, callbacks: &'a [Callback]) -> Self {
                    PopupView {
                        title: title,
                        callbacks: callbacks,
                    }
                }
            }
            impl<'a> PopupView<'a> {
                /// Render the view.
                pub(crate) fn render<'b>(
                    &'a self,
                    frame: &mut TerminalFrame<'b>,
                    size: Rect,
                ) {
                    let lines = self
                        .callbacks
                        .iter()
                        .flat_map(|callback| {
                            ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                ::alloc::intrinsics::write_box_via_move(
                                    ::alloc::boxed::Box::new_uninit(),
                                    [
                                        Line::from(
                                            ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                                ::alloc::intrinsics::write_box_via_move(
                                                    ::alloc::boxed::Box::new_uninit(),
                                                    [
                                                        Span::from(
                                                                ::alloc::__export::must_use({
                                                                    ::alloc::fmt::format(
                                                                        format_args!("[{0}] ", callback.trigger),
                                                                    )
                                                                }),
                                                            )
                                                            .bold(),
                                                        Span::from(
                                                                ::alloc::__export::must_use({
                                                                    ::alloc::fmt::format(format_args!("{0} ", callback.title))
                                                                }),
                                                            )
                                                            .yellow()
                                                            .bold(),
                                                    ],
                                                ),
                                            ),
                                        ),
                                        Line::from(Span::from("")),
                                        Line::from(
                                            Span::from(callback.description.to_string()).italic(),
                                        ),
                                        Line::from(Span::from("")),
                                    ],
                                ),
                            )
                        })
                        .collect::<Vec<_>>();
                    let paragraph = Paragraph::new(lines)
                        .alignment(Alignment::Left)
                        .wrap(Wrap { trim: false })
                        .style(Style::default().fg(Color::Gray))
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title_alignment(Alignment::Center)
                                .style(Style::default().fg(Color::Gray))
                                .title(
                                    Span::styled(
                                        self.title,
                                        Style::default().add_modifier(Modifier::BOLD),
                                    ),
                                ),
                        );
                    let area = centered_percent(20, size, Direction::Horizontal);
                    let area = centered_percent(20, area, Direction::Vertical);
                    frame.render_widget(paragraph, area);
                }
            }
            /// The percent represents the amount of space that will be taken by each side.
            fn centered_percent(percent: u16, size: Rect, direction: Direction) -> Rect {
                let center = 100 - (percent * 2);
                Layout::default()
                    .direction(direction)
                    .constraints([
                        Constraint::Percentage(percent),
                        Constraint::Percentage(center),
                        Constraint::Percentage(percent),
                    ])
                    .split(size)[1]
            }
        }
        mod progress {
            use super::TerminalFrame;
            use crate::{logger::ProgressSnapshot, renderer::tui::TuiSplit};
            use ratatui::{
                prelude::{Alignment, Constraint, Direction, Layout, Rect},
                style::{Color, Style, Stylize},
                text::{Line, Span},
                widgets::{Block, Borders, Gauge, Paragraph},
            };
            use std::time::{Duration, Instant};
            /// Simple progress bar for the training.
            ///
            /// We currently ignore the time taken for the validation part.
            pub(crate) struct ProgressBarState {
                progress_total: f64,
                progress_task: f64,
                split: TuiSplit,
                starting_epoch: usize,
                estimate: ProgressEstimate,
            }
            const MINUTE: u64 = 60;
            const HOUR: u64 = 60 * 60;
            const DAY: u64 = 24 * 60 * 60;
            impl ProgressBarState {
                pub fn new(checkpoint: Option<usize>) -> Self {
                    Self {
                        progress_total: 0.0,
                        progress_task: 0.0,
                        split: TuiSplit::Train,
                        estimate: ProgressEstimate::new(),
                        starting_epoch: checkpoint.unwrap_or(0),
                    }
                }
                /// Update the training progress.
                pub(crate) fn update_train(&mut self, progress: &ProgressSnapshot) {
                    self.progress_total = calculate_progress(progress, 0, 0);
                    self.progress_task = progress.split.items_processed as f64
                        / progress.split.items_total as f64;
                    self.estimate.update(progress, self.starting_epoch);
                    self.split = TuiSplit::Train;
                }
                /// Update the validation progress.
                pub(crate) fn update_valid(&mut self, progress: &ProgressSnapshot) {
                    self.progress_task = progress.split.items_processed as f64
                        / progress.split.items_total as f64;
                    self.split = TuiSplit::Valid;
                }
                /// Update the testing progress.
                pub(crate) fn update_test(&mut self, progress: &ProgressSnapshot) {
                    self.progress_task = progress.split.items_processed as f64
                        / progress.split.items_total as f64;
                    self.split = TuiSplit::Test;
                }
                /// Create a view for the current progress.
                pub(crate) fn view(&self) -> ProgressBarView {
                    const NO_ETA: &str = "---";
                    let eta = match self.estimate.secs() {
                        Some(eta) => format_eta(eta),
                        None => NO_ETA.to_string(),
                    };
                    ProgressBarView::new(
                        self.progress_total,
                        self.progress_task,
                        self.split.color(),
                        eta,
                    )
                }
            }
            pub(crate) struct ProgressBarView {
                progress: f64,
                progress_task: f64,
                color_task: Color,
                eta: String,
            }
            impl ProgressBarView {
                ///Constructs a new `ProgressBarView`.
                pub fn new(
                    progress: f64,
                    progress_task: f64,
                    color_task: Color,
                    eta: String,
                ) -> Self {
                    ProgressBarView {
                        progress: progress,
                        progress_task: progress_task,
                        color_task: color_task,
                        eta: eta,
                    }
                }
            }
            impl ProgressBarView {
                /// Render the view.
                pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
                    let block = Block::default()
                        .borders(Borders::ALL)
                        .title("Progress")
                        .title_alignment(Alignment::Left);
                    let size_new = block.inner(size);
                    frame.render_widget(block, size);
                    let size = size_new;
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints(
                            [Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)].as_ref(),
                        )
                        .split(size);
                    let size_task = chunks[0];
                    let size_total = chunks[1];
                    let calculate_size = |size: Rect| {
                        Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints(
                                [
                                    Constraint::Length(1),
                                    Constraint::Min(0),
                                    Constraint::Length(self.eta.len() as u16 + 4),
                                ]
                                    .as_ref(),
                            )
                            .split(size)
                    };
                    let chunks = calculate_size(size_total);
                    let size_gauge_total = chunks[1];
                    let size_eta = chunks[2];
                    let chunks = calculate_size(size_task);
                    let size_gauge_task = chunks[1];
                    let progress_total = Gauge::default()
                        .gauge_style(Style::default().fg(Color::Yellow))
                        .ratio(self.progress.min(1.0));
                    let progress_task = Gauge::default()
                        .gauge_style(Style::default().fg(self.color_task))
                        .ratio(self.progress_task.min(1.0));
                    let eta = Paragraph::new(
                        Line::from(
                            ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                ::alloc::intrinsics::write_box_via_move(
                                    ::alloc::boxed::Box::new_uninit(),
                                    [
                                        Span::from(" ("),
                                        Span::from(self.eta).italic(),
                                        Span::from(") "),
                                    ],
                                ),
                            ),
                        ),
                    );
                    frame.render_widget(progress_task, size_gauge_task);
                    frame.render_widget(progress_total, size_gauge_total);
                    frame.render_widget(eta, size_eta);
                }
            }
            struct ProgressEstimate {
                started: Instant,
                started_after_warmup: Option<Instant>,
                warmup_num_items: usize,
                progress: f64,
            }
            impl ProgressEstimate {
                fn new() -> Self {
                    Self {
                        started: Instant::now(),
                        started_after_warmup: None,
                        warmup_num_items: 0,
                        progress: 0.0,
                    }
                }
                fn secs(&self) -> Option<u64> {
                    let eta = self.started_after_warmup?.elapsed();
                    let total_estimated = (eta.as_secs() as f64) / self.progress;
                    if total_estimated.is_normal() {
                        let remaining = 1.0 - self.progress;
                        let eta = (total_estimated * remaining) as u64;
                        Some(eta)
                    } else {
                        None
                    }
                }
                fn update(
                    &mut self,
                    progress: &ProgressSnapshot,
                    starting_epoch: usize,
                ) {
                    if self.started_after_warmup.is_some() {
                        self.progress = calculate_progress(
                            progress,
                            starting_epoch,
                            self.warmup_num_items,
                        );
                        return;
                    }
                    const WARMUP_NUM_ITERATION: usize = 10;
                    if self.started.elapsed() > Duration::from_secs(30) {
                        self.init(progress, starting_epoch);
                        return;
                    }
                    if progress.split.items_processed >= WARMUP_NUM_ITERATION
                        && self.started.elapsed() > Duration::from_secs(10)
                    {
                        self.init(progress, starting_epoch);
                    }
                }
                fn init(&mut self, progress: &ProgressSnapshot, starting_epoch: usize) {
                    let epoch = progress.global.items_processed - starting_epoch;
                    let local = &progress.split;
                    let epoch_items = (epoch - 1) * local.items_total;
                    let iteration_items = local.items_processed;
                    self.warmup_num_items = epoch_items + iteration_items;
                    self.started_after_warmup = Some(Instant::now());
                    self.progress = calculate_progress(
                        progress,
                        starting_epoch,
                        self.warmup_num_items,
                    );
                }
            }
            fn calculate_progress(
                progress: &ProgressSnapshot,
                starting_epoch: usize,
                ignore_num_items: usize,
            ) -> f64 {
                let epoch_total = progress.global.items_total - starting_epoch;
                let epoch = progress.global.items_processed - starting_epoch;
                let local = &progress.split;
                let total_items = local.items_total * epoch_total;
                let epoch_items = (epoch - 1) * local.items_total;
                let iteration_items = local.items_processed;
                let num_items = epoch_items + iteration_items - ignore_num_items;
                num_items as f64 / total_items as f64
            }
            fn format_eta(eta_secs: u64) -> String {
                let seconds = eta_secs % 60;
                let minutes = eta_secs / MINUTE % 60;
                let hours = eta_secs / HOUR % 24;
                let days = eta_secs / DAY;
                if days > 1 {
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0} days", days))
                    })
                } else if days == 1 {
                    "1 day".to_string()
                } else if hours > 1 {
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0} hours", hours))
                    })
                } else if hours == 1 {
                    "1 hour".to_string()
                } else if minutes > 1 {
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0} mins", minutes))
                    })
                } else if minutes == 1 {
                    "1 min".to_string()
                } else if seconds > 1 {
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0} secs", seconds))
                    })
                } else {
                    "1 sec".to_string()
                }
            }
        }
        mod recent_history {
            use super::PlotAxes;
            use crate::renderer::tui::TuiTag;
            use ratatui::{
                style::{Color, Style},
                symbols, widgets::{Dataset, GraphType},
            };
            use std::collections::BTreeMap;
            const FACTOR_BEFORE_RESIZE: usize = 2;
            /// A plot that shows the recent history at full resolution.
            pub(crate) struct RecentHistoryPlot {
                pub(crate) axes: PlotAxes,
                points: BTreeMap<TuiTag, RecentHistoryPoints>,
                max_samples: usize,
            }
            struct RecentHistoryPoints {
                min_x: f64,
                max_x: f64,
                min_y: f64,
                max_y: f64,
                cursor: usize,
                points: Vec<(f64, f64)>,
                max_samples: usize,
                factor_before_resize: usize,
            }
            impl RecentHistoryPlot {
                pub(crate) fn new(max_samples: usize) -> Self {
                    Self {
                        axes: PlotAxes::default(),
                        points: BTreeMap::default(),
                        max_samples,
                    }
                }
                pub(crate) fn push(&mut self, tag: TuiTag, data: f64) {
                    if !self.points.contains_key(&tag) {
                        self.points
                            .insert(
                                tag.clone(),
                                RecentHistoryPoints::new(self.max_samples),
                            );
                    }
                    let (x_min, x_current) = self.point_x();
                    for (s, entry) in self.points.iter_mut() {
                        if s == &tag {
                            entry.push((x_current, data));
                        }
                        entry.update_cursor(x_min);
                    }
                    self.update_bounds();
                }
                pub(crate) fn datasets(&self) -> Vec<Dataset<'_>> {
                    let mut datasets = Vec::new();
                    for (tag, points) in self.points.iter() {
                        datasets
                            .push(
                                points
                                    .dataset(
                                        ::alloc::__export::must_use({
                                            ::alloc::fmt::format(format_args!("{0}", tag))
                                        }),
                                        tag.split.color(),
                                    ),
                            );
                    }
                    datasets
                }
                fn point_x(&mut self) -> (f64, f64) {
                    let mut x_current = f64::MIN;
                    let mut x_min = f64::MAX;
                    for point in self.points.values() {
                        x_current = f64::max(x_current, point.max_x);
                        x_min = f64::min(x_min, point.min_x);
                    }
                    if x_current - x_min >= self.max_samples as f64 {
                        x_min += 1.0;
                    }
                    (x_min, x_current + 1.0)
                }
                fn update_bounds(&mut self) {
                    let (mut x_min, mut x_max) = (f64::MAX, f64::MIN);
                    let (mut y_min, mut y_max) = (f64::MAX, f64::MIN);
                    for points in self.points.values() {
                        x_min = f64::min(x_min, points.min_x);
                        x_max = f64::max(x_max, points.max_x);
                        y_min = f64::min(y_min, points.min_y);
                        y_max = f64::max(y_max, points.max_y);
                    }
                    self.axes.update_bounds((x_min, x_max), (y_min, y_max));
                }
            }
            impl RecentHistoryPoints {
                fn new(max_samples: usize) -> Self {
                    let factor_before_resize = FACTOR_BEFORE_RESIZE;
                    Self {
                        min_x: 0.,
                        max_x: 0.,
                        min_y: f64::MAX,
                        max_y: f64::MIN,
                        points: Vec::with_capacity(factor_before_resize * max_samples),
                        cursor: 0,
                        max_samples,
                        factor_before_resize,
                    }
                }
                fn push(&mut self, (x, y): (f64, f64)) {
                    if x > self.max_x {
                        self.max_x = x;
                    }
                    if x < self.min_x {
                        self.min_x = x;
                    }
                    if y > self.max_y {
                        self.max_y = y;
                    }
                    if y < self.min_y {
                        self.min_y = y;
                    }
                    self.points.push((x, y));
                }
                fn update_cursor(&mut self, min_x: f64) {
                    if self.min_x >= min_x {
                        return;
                    }
                    self.min_x = min_x;
                    let mut update_y_max = false;
                    let mut update_y_min = false;
                    while let Some((x, y)) = self.points.get(self.cursor) {
                        if *x >= self.min_x {
                            break;
                        }
                        if *y == self.max_y {
                            update_y_max = true;
                        }
                        if *y == self.min_y {
                            update_y_min = true;
                        }
                        self.cursor += 1;
                    }
                    if update_y_max {
                        self.max_y = self.calculate_max_y();
                    }
                    if update_y_min {
                        self.min_y = self.calculate_min_y();
                    }
                    if self.points.len() >= self.max_samples * self.factor_before_resize
                    {
                        self.resize();
                    }
                }
                fn slice(&self) -> &[(f64, f64)] {
                    &self.points[self.cursor..self.points.len()]
                }
                fn calculate_max_y(&self) -> f64 {
                    let mut max_y = f64::MIN;
                    for (_x, y) in self.slice() {
                        max_y = f64::max(max_y, *y);
                    }
                    max_y
                }
                fn calculate_min_y(&self) -> f64 {
                    let mut min_y = f64::MAX;
                    for (_x, y) in self.slice() {
                        if *y < min_y {
                            min_y = *y;
                        }
                    }
                    min_y
                }
                fn resize(&mut self) {
                    let mut points = Vec::with_capacity(
                        self.max_samples * self.factor_before_resize,
                    );
                    for i in self.cursor..self.points.len() {
                        points.push(self.points[i]);
                    }
                    self.points = points;
                    self.cursor = 0;
                }
                fn dataset<'a>(&'a self, name: String, color: Color) -> Dataset<'a> {
                    let data = &self.points[self.cursor..self.points.len()];
                    Dataset::default()
                        .name(name)
                        .marker(symbols::Marker::Braille)
                        .style(Style::default().fg(color).bold())
                        .graph_type(GraphType::Scatter)
                        .data(data)
                }
            }
        }
        mod renderer {
            use crate::logger::{
                EvaluationProgressLogger, ProgressSnapshot, TrainingProgressLogger,
            };
            use crate::metric::{MetricDefinition, MetricId};
            use crate::renderer::tui::TuiSplit;
            use crate::renderer::{
                EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
            };
            use crate::renderer::{MetricsRendererTraining, tui::NumericMetricsState};
            use crate::{Interrupter, LearnerSummary};
            use burn_core::data::dataloader::Progress;
            use ratatui::{
                Terminal,
                crossterm::{
                    event::{
                        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode,
                    },
                    execute,
                    terminal::{
                        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
                        enable_raw_mode,
                    },
                },
                prelude::*,
            };
            use std::collections::HashMap;
            use std::panic::{set_hook, take_hook};
            use std::sync::mpsc::{Receiver, Sender};
            use std::sync::{Arc, Mutex, mpsc};
            use std::thread::JoinHandle;
            use std::{
                error::Error, io::{self, Stdout},
                time::{Duration, Instant},
            };
            use super::{
                Callback, CallbackFn, ControlsView, MetricsView, PopupState,
                ProgressBarState, StatusState, TextEventOutcome, TextMetricsState,
                TuiGroup, TuiTag,
            };
            /// The current terminal backend.
            pub(crate) type TerminalBackend = CrosstermBackend<Stdout>;
            /// The current terminal frame.
            pub(crate) type TerminalFrame<'a> = ratatui::Frame<'a>;
            type PanicHook = Box<
                dyn Fn(&std::panic::PanicHookInfo<'_>) + 'static + Sync + Send,
            >;
            const MAX_REFRESH_RATE_MILLIS: u64 = 100;
            enum TuiRendererEvent {
                MetricRegistration(MetricDefinition),
                MetricsUpdate((TuiSplit, TuiGroup, MetricState)),
                StatusUpdateTrain((TuiSplit, ProgressSnapshot)),
                StatusUpdateTest(ProgressSnapshot),
                ProcessEnd {
                    summary: Option<LearnerSummary>,
                    /// Interrupter reset.
                    reset: bool,
                },
                CounterUpdate(String),
                SplitEnd,
                ManualClose,
                Close,
                Persistent,
            }
            /// The terminal UI metrics renderer.
            pub struct TuiMetricsRendererWrapper {
                sender: mpsc::Sender<TuiRendererEvent>,
                interrupter: Interrupter,
                handle_join: Option<JoinHandle<()>>,
                kill_signal: Arc<Mutex<Receiver<()>>>,
                current_split: TuiSplit,
                training_progress: ProgressSnapshot,
                eval_progress: ProgressSnapshot,
            }
            impl TuiMetricsRendererWrapper {
                /// Create a new terminal UI renderer.
                pub fn new(interrupter: Interrupter, checkpoint: Option<usize>) -> Self {
                    let (sender, receiver) = mpsc::channel();
                    let (kill_signal_sender, kill_signal_receiver) = mpsc::channel();
                    let interrupter_clone = interrupter.clone();
                    let handle_join = std::thread::Builder::new()
                        .name("train-renderer".into())
                        .spawn(move || {
                            let mut renderer = TuiMetricsRenderer::new(
                                interrupter_clone,
                                checkpoint,
                                kill_signal_sender,
                            );
                            let tick_rate = Duration::from_millis(
                                MAX_REFRESH_RATE_MILLIS,
                            );
                            loop {
                                let remaining_time = tick_rate
                                    .saturating_sub(renderer.last_update.elapsed());
                                match receiver.recv_timeout(remaining_time) {
                                    Ok(event) => renderer.handle_event(event),
                                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                                        {
                                            {
                                                let lvl = ::log::Level::Error;
                                                if lvl <= ::log::STATIC_MAX_LEVEL
                                                    && lvl <= ::log::max_level()
                                                {
                                                    ::log::__private_api::log(
                                                        { ::log::__private_api::GlobalLogger },
                                                        format_args!("Renderer thread disconnected."),
                                                        lvl,
                                                        &(
                                                            "burn_train::renderer::tui::renderer",
                                                            "burn_train::renderer::tui::renderer",
                                                            ::log::__private_api::loc(),
                                                        ),
                                                        (),
                                                    );
                                                }
                                            }
                                        };
                                        break;
                                    }
                                }
                                if renderer.last_update.elapsed() >= tick_rate
                                    && let Err(err) = renderer.render()
                                {
                                    {
                                        {
                                            let lvl = ::log::Level::Error;
                                            if lvl <= ::log::STATIC_MAX_LEVEL
                                                && lvl <= ::log::max_level()
                                            {
                                                ::log::__private_api::log(
                                                    { ::log::__private_api::GlobalLogger },
                                                    format_args!("Render error: {0}", err),
                                                    lvl,
                                                    &(
                                                        "burn_train::renderer::tui::renderer",
                                                        "burn_train::renderer::tui::renderer",
                                                        ::log::__private_api::loc(),
                                                    ),
                                                    (),
                                                );
                                            }
                                        }
                                    };
                                    break;
                                }
                                if (renderer.manual_close
                                    && renderer.interrupter.should_stop()) || renderer.close
                                {
                                    break;
                                }
                            }
                        })
                        .unwrap();
                    let init = Progress::new(0, 0, None);
                    Self {
                        sender,
                        interrupter,
                        handle_join: Some(handle_join),
                        kill_signal: Arc::new(Mutex::new(kill_signal_receiver)),
                        current_split: TuiSplit::Train,
                        training_progress: ProgressSnapshot::new(
                            init.clone(),
                            init.clone(),
                        ),
                        eval_progress: ProgressSnapshot::new(init.clone(), init),
                    }
                }
                fn send_event(&self, event: TuiRendererEvent) {
                    if self.kill_signal.lock().unwrap().try_recv().is_ok() {
                        {
                            ::core::panicking::panic_fmt(
                                format_args!("Killing training from user input."),
                            );
                        }
                    }
                    if let Err(e) = self.sender.send(event) {
                        {
                            {
                                let lvl = ::log::Level::Warn;
                                if lvl <= ::log::STATIC_MAX_LEVEL
                                    && lvl <= ::log::max_level()
                                {
                                    ::log::__private_api::log(
                                        { ::log::__private_api::GlobalLogger },
                                        format_args!("Failed to send TUI event: {0}", e),
                                        lvl,
                                        &(
                                            "burn_train::renderer::tui::renderer",
                                            "burn_train::renderer::tui::renderer",
                                            ::log::__private_api::loc(),
                                        ),
                                        (),
                                    );
                                }
                            }
                        };
                    }
                }
                /// Set the renderer to persistent mode.
                pub fn persistent(self) -> Self {
                    self.send_event(TuiRendererEvent::Persistent);
                    self
                }
            }
            struct TuiMetricsRenderer {
                terminal: Terminal<TerminalBackend>,
                last_update: std::time::Instant,
                progress: ProgressBarState,
                metric_definitions: HashMap<MetricId, MetricDefinition>,
                metrics_numeric: NumericMetricsState,
                metrics_text: TextMetricsState,
                status: StatusState,
                interrupter: Interrupter,
                popup: PopupState,
                previous_panic_hook: Option<Arc<PanicHook>>,
                persistent: bool,
                manual_close: bool,
                close: bool,
                summary: Option<LearnerSummary>,
                kill_signal: Sender<()>,
            }
            impl MetricsRendererEvaluation for TuiMetricsRendererWrapper {
                fn update_test(&mut self, name: EvaluationName, state: MetricState) {
                    self.send_event(
                        TuiRendererEvent::MetricsUpdate((
                            TuiSplit::Test,
                            TuiGroup::Named(name.name),
                            state,
                        )),
                    );
                }
                fn on_test_end(
                    &mut self,
                    summary: Option<LearnerSummary>,
                ) -> Result<(), Box<dyn Error>> {
                    self.send_event(TuiRendererEvent::ProcessEnd {
                        summary,
                        reset: false,
                    });
                    Ok(())
                }
            }
            impl MetricsRenderer for TuiMetricsRendererWrapper {
                fn manual_close(&mut self) {
                    self.send_event(TuiRendererEvent::ManualClose);
                    let _ = self.handle_join.take().unwrap().join();
                }
                fn register_metric(&mut self, definition: MetricDefinition) {
                    self.send_event(TuiRendererEvent::MetricRegistration(definition));
                }
            }
            impl MetricsRendererTraining for TuiMetricsRendererWrapper {
                fn update_train(&mut self, state: MetricState) {
                    self.send_event(
                        TuiRendererEvent::MetricsUpdate((
                            TuiSplit::Train,
                            TuiGroup::Default,
                            state,
                        )),
                    );
                }
                fn update_valid(&mut self, state: MetricState) {
                    self.send_event(
                        TuiRendererEvent::MetricsUpdate((
                            TuiSplit::Valid,
                            TuiGroup::Default,
                            state,
                        )),
                    );
                }
                fn on_train_end(
                    &mut self,
                    summary: Option<LearnerSummary>,
                ) -> Result<(), Box<dyn Error>> {
                    self.interrupter.reset();
                    self.send_event(TuiRendererEvent::ProcessEnd {
                        summary,
                        reset: true,
                    });
                    Ok(())
                }
            }
            impl TrainingProgressLogger for TuiMetricsRendererWrapper {
                fn start(&mut self, total_epochs: usize, total_items: Option<usize>) {
                    self.training_progress.global = Progress::new(
                        1,
                        total_epochs,
                        Some("epochs".to_string()),
                    );
                    if let Some(items) = total_items {
                        self.training_progress.split = Progress::new(
                            0,
                            items,
                            Some("items".to_string()),
                        );
                    }
                }
                fn update_epoch(&mut self, epoch: usize) {
                    let total = self.training_progress.global.items_total;
                    let unit = self.training_progress.global.unit.clone();
                    self.training_progress.global = Progress::new(
                        epoch + 1,
                        total,
                        unit,
                    );
                }
                fn start_split(&mut self, split: &str, total_items: usize) {
                    self.training_progress.split = Progress::new(
                        0,
                        total_items,
                        Some("items".to_string()),
                    );
                    self.current_split = if split == "train" {
                        TuiSplit::Train
                    } else {
                        TuiSplit::Valid
                    };
                }
                fn update_split(&mut self, items_processed: usize) {
                    let total = self.training_progress.split.items_total;
                    let unit = self.training_progress.split.unit.clone();
                    self.training_progress.split = Progress::new(
                        items_processed,
                        total,
                        unit,
                    );
                    if self.training_progress.global.items_total == 0 {
                        self.training_progress.global = self
                            .training_progress
                            .split
                            .clone();
                    }
                    self.send_event(
                        TuiRendererEvent::StatusUpdateTrain((
                            self.current_split,
                            self.training_progress.clone(),
                        )),
                    );
                }
                fn end_split(&mut self) {
                    self.send_event(TuiRendererEvent::SplitEnd);
                    self.current_split = TuiSplit::Train;
                }
                fn end(&mut self) {}
                fn log_event_training(&mut self, event: String) {
                    self.send_event(TuiRendererEvent::CounterUpdate(event));
                }
            }
            impl EvaluationProgressLogger for TuiMetricsRendererWrapper {
                fn start_global_progress(&mut self, total_tests: usize) {
                    self.eval_progress.global = Progress::new(
                        0,
                        total_tests,
                        Some("tests".to_string()),
                    );
                }
                fn start_test(&mut self, _name: &str, total_items: usize) {
                    let current = self.eval_progress.global.items_processed + 1;
                    let total = self.eval_progress.global.items_total;
                    self.eval_progress.global = Progress::new(
                        current,
                        total,
                        Some("tests".to_string()),
                    );
                    self.eval_progress.split = Progress::new(
                        0,
                        total_items,
                        Some("items".to_string()),
                    );
                }
                fn update_test_progress(&mut self, items_processed: usize) {
                    let total = self.eval_progress.split.items_total;
                    let unit = self.eval_progress.split.unit.clone();
                    self.eval_progress.split = Progress::new(
                        items_processed,
                        total,
                        unit,
                    );
                    self.send_event(
                        TuiRendererEvent::StatusUpdateTest(self.eval_progress.clone()),
                    );
                }
                fn end_test(&mut self) {
                    self.send_event(TuiRendererEvent::SplitEnd);
                }
                fn end_global_progress(&mut self) {}
                fn log_event_evaluation(&mut self, event: String) {
                    self.send_event(TuiRendererEvent::CounterUpdate(event));
                }
            }
            impl Drop for TuiMetricsRendererWrapper {
                fn drop(&mut self) {
                    if !std::thread::panicking() {
                        self.send_event(TuiRendererEvent::Close);
                        if let Some(handle) = self.handle_join.take() {
                            let _ = handle.join();
                        }
                    }
                }
            }
            impl TuiMetricsRenderer {
                fn update_metric(
                    &mut self,
                    split: TuiSplit,
                    group: TuiGroup,
                    state: MetricState,
                ) {
                    match state {
                        MetricState::Generic(entry) => {
                            let name = self
                                .metric_definitions
                                .get(&entry.metric_id)
                                .unwrap()
                                .name
                                .clone()
                                .into();
                            self.metrics_text.update(split, group, entry, name);
                        }
                        MetricState::Numeric(entry, value) => {
                            let name: Arc<String> = self
                                .metric_definitions
                                .get(&entry.metric_id)
                                .unwrap()
                                .name
                                .clone()
                                .into();
                            self.metrics_numeric
                                .push(
                                    TuiTag::new(split, group.clone()),
                                    name.clone(),
                                    value,
                                );
                            self.metrics_text.update(split, group, entry, name);
                        }
                    };
                }
                pub fn new(
                    interrupter: Interrupter,
                    checkpoint: Option<usize>,
                    kill_signal: Sender<()>,
                ) -> Self {
                    let mut stdout = io::stdout();
                    {
                        use ::std::io::Write;
                        {
                            use ::std::io::Write;
                            Ok(stdout.by_ref())
                                .and_then(|writer| ::crossterm::QueueableCommand::queue(
                                    writer,
                                    EnterAlternateScreen,
                                ))
                                .and_then(|writer| ::crossterm::QueueableCommand::queue(
                                    writer,
                                    EnableMouseCapture,
                                ))
                                .map(|_| ())
                        }
                            .and_then(|()| { ::std::io::Write::flush(stdout.by_ref()) })
                    }
                        .unwrap();
                    enable_raw_mode().unwrap();
                    let terminal = Terminal::new(CrosstermBackend::new(stdout)).unwrap();
                    let previous_panic_hook = Arc::new(take_hook());
                    set_hook(
                        Box::new({
                            let previous_panic_hook = previous_panic_hook.clone();
                            move |panic_info| {
                                let _ = disable_raw_mode();
                                let _ = {
                                    use ::std::io::Write;
                                    {
                                        use ::std::io::Write;
                                        Ok(io::stdout().by_ref())
                                            .and_then(|writer| ::crossterm::QueueableCommand::queue(
                                                writer,
                                                DisableMouseCapture,
                                            ))
                                            .and_then(|writer| ::crossterm::QueueableCommand::queue(
                                                writer,
                                                LeaveAlternateScreen,
                                            ))
                                            .map(|_| ())
                                    }
                                        .and_then(|()| {
                                            ::std::io::Write::flush(io::stdout().by_ref())
                                        })
                                };
                                previous_panic_hook(panic_info);
                            }
                        }),
                    );
                    Self {
                        terminal,
                        last_update: Instant::now(),
                        progress: ProgressBarState::new(checkpoint),
                        metric_definitions: HashMap::default(),
                        metrics_numeric: NumericMetricsState::default(),
                        metrics_text: TextMetricsState::default(),
                        status: StatusState::default(),
                        interrupter,
                        popup: PopupState::Empty,
                        previous_panic_hook: Some(previous_panic_hook),
                        persistent: false,
                        manual_close: false,
                        close: false,
                        summary: None,
                        kill_signal,
                    }
                }
                fn handle_event(&mut self, event: TuiRendererEvent) {
                    match event {
                        TuiRendererEvent::MetricRegistration(definition) => {
                            self.metric_definitions
                                .insert(definition.metric_id.clone(), definition);
                        }
                        TuiRendererEvent::MetricsUpdate((split, group, state)) => {
                            self.update_metric(split, group, state);
                        }
                        TuiRendererEvent::StatusUpdateTrain((split, item)) => {
                            match split {
                                TuiSplit::Train => {
                                    self.progress.update_train(&item);
                                    self.metrics_numeric.update_progress_train(&item);
                                    self.status.update_train(&item);
                                }
                                TuiSplit::Valid => {
                                    self.progress.update_valid(&item);
                                    self.metrics_numeric.update_progress_valid(&item);
                                    self.status.update_valid(&item);
                                }
                                _ => {}
                            }
                        }
                        TuiRendererEvent::StatusUpdateTest(item) => {
                            self.progress.update_test(&item);
                            self.metrics_numeric.update_progress_test(&item);
                            self.status.update_test(&item);
                        }
                        TuiRendererEvent::ProcessEnd { summary, reset } => {
                            match (self.summary.take(), summary) {
                                (None, Some(summary)) => {
                                    self.summary = Some(summary);
                                }
                                (Some(current), Some(other)) => {
                                    self.summary = Some(current.merge(other));
                                }
                                (_, _) => {}
                            }
                            if reset {
                                self.interrupter.reset();
                            }
                        }
                        TuiRendererEvent::CounterUpdate(event) => {
                            self.status.update_counter(event);
                        }
                        TuiRendererEvent::SplitEnd => {
                            self.status.reset_counters();
                        }
                        TuiRendererEvent::ManualClose => self.manual_close = true,
                        TuiRendererEvent::Persistent => self.persistent = true,
                        TuiRendererEvent::Close => self.close = true,
                    }
                }
                fn render(&mut self) -> Result<(), Box<dyn Error>> {
                    self.draw()?;
                    self.handle_user_input()?;
                    self.last_update = Instant::now();
                    Ok(())
                }
                fn draw(&mut self) -> Result<(), Box<dyn Error>> {
                    self.terminal
                        .draw(|frame| {
                            let size = frame.area();
                            match self.popup.view() {
                                Some(view) => view.render(frame, size),
                                None => {
                                    let view = MetricsView::new(
                                        self.metrics_numeric.view(),
                                        self.metrics_text.view(),
                                        self.progress.view(),
                                        ControlsView,
                                        self.status.view(),
                                    );
                                    view.render(frame, size);
                                }
                            };
                        })?;
                    Ok(())
                }
                /// Dispatch a single user event to the popup / numeric / text components.
                /// Returns `true` when something visible changed and a redraw is warranted.
                /// The training loop ignores this (it is tick-gated); the post-training loop
                /// uses it to skip redraws on inert events like mouse jitter.
                fn dispatch_user_event(&mut self, event: &Event) -> bool {
                    let mut redraw = self.popup.on_event(event);
                    if self.popup.is_empty() {
                        redraw |= self.metrics_numeric.on_event(event);
                        redraw
                            |= match self.metrics_text.on_event(event) {
                                TextEventOutcome::Clicked(name) => {
                                    self.metrics_numeric.select_by_name(&name);
                                    true
                                }
                                TextEventOutcome::HoverChanged => true,
                                TextEventOutcome::Ignored => false,
                            };
                    }
                    redraw
                }
                fn handle_user_input(&mut self) -> Result<(), Box<dyn Error>> {
                    while event::poll(Duration::from_secs(0))? {
                        let event = event::read()?;
                        let _ = self.dispatch_user_event(&event);
                        if self.popup.is_empty() && let Event::Key(key) = event
                            && let KeyCode::Char('q') = key.code
                        {
                            self.popup = PopupState::Full(
                                "Quit".to_string(),
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            Callback::new(
                                                "Stop the training.",
                                                "Stop the training immediately. This will break from the \
                                 training loop, but any remaining code after the loop will be \
                                 executed.",
                                                's',
                                                QuitPopupAccept(self.interrupter.clone()),
                                            ),
                                            Callback::new(
                                                "Stop the training immediately.",
                                                "Kill the program. This will create a panic! which will make \
                                 the current training fails. Any code following the training \
                                 won't be executed.",
                                                'k',
                                                KillPopupAccept(self.kill_signal.clone()),
                                            ),
                                            Callback::new(
                                                "Cancel",
                                                "Cancel the action, continue the training.",
                                                'c',
                                                PopupCancel,
                                            ),
                                        ],
                                    ),
                                ),
                            );
                        }
                    }
                    Ok(())
                }
                fn handle_post_training(&mut self) -> Result<(), Box<dyn Error>> {
                    self.popup = PopupState::Full(
                        "Training is done".to_string(),
                        ::alloc::boxed::box_assume_init_into_vec_unsafe(
                            ::alloc::intrinsics::write_box_via_move(
                                ::alloc::boxed::Box::new_uninit(),
                                [
                                    Callback::new(
                                        "Training Done",
                                        "Press 'x' to close this popup.  Press 'q' to exit the application after the \
                popup is closed.",
                                        'x',
                                        PopupCancel,
                                    ),
                                ],
                            ),
                        ),
                    );
                    self.draw().ok();
                    loop {
                        if let Ok(true) = event::poll(
                            Duration::from_millis(MAX_REFRESH_RATE_MILLIS),
                        ) {
                            match event::read() {
                                Ok(event @ Event::Key(key)) => {
                                    let redraw = self.dispatch_user_event(&event);
                                    if self.popup.is_empty()
                                        && let KeyCode::Char('q') = key.code
                                    {
                                        break;
                                    }
                                    if redraw {
                                        self.draw().ok();
                                    }
                                }
                                Ok(event @ Event::Mouse(_)) => {
                                    if self.dispatch_user_event(&event) {
                                        self.draw().ok();
                                    }
                                }
                                Ok(Event::Resize(..)) => {
                                    self.draw().ok();
                                }
                                Err(err) => {
                                    {
                                        ::std::io::_eprint(
                                            format_args!("Error reading event: {0}\n", err),
                                        );
                                    };
                                    break;
                                }
                                _ => continue,
                            }
                        }
                    }
                    Ok(())
                }
                fn reset(&mut self) -> Result<(), Box<dyn Error>> {
                    if self.previous_panic_hook.is_some() {
                        if self.persistent && let Err(err) = self.handle_post_training()
                        {
                            {
                                ::std::io::_eprint(
                                    format_args!("Error in post-training handling: {0}\n", err),
                                );
                            };
                        }
                        disable_raw_mode()?;
                        {
                            use ::std::io::Write;
                            {
                                use ::std::io::Write;
                                Ok(self.terminal.backend_mut().by_ref())
                                    .and_then(|writer| ::crossterm::QueueableCommand::queue(
                                        writer,
                                        DisableMouseCapture,
                                    ))
                                    .and_then(|writer| ::crossterm::QueueableCommand::queue(
                                        writer,
                                        LeaveAlternateScreen,
                                    ))
                                    .map(|_| ())
                            }
                                .and_then(|()| {
                                    ::std::io::Write::flush(
                                        self.terminal.backend_mut().by_ref(),
                                    )
                                })
                        }?;
                        self.terminal.show_cursor()?;
                        let _ = take_hook();
                        if let Some(previous_panic_hook) = Arc::into_inner(
                            self.previous_panic_hook.take().unwrap(),
                        ) {
                            set_hook(previous_panic_hook);
                        }
                    }
                    Ok(())
                }
            }
            struct QuitPopupAccept(Interrupter);
            struct KillPopupAccept(Sender<()>);
            struct PopupCancel;
            impl CallbackFn for KillPopupAccept {
                fn call(&self) -> bool {
                    self.0.send(()).unwrap();
                    {
                        ::core::panicking::panic_fmt(
                            format_args!("Killing training from user input."),
                        );
                    };
                }
            }
            impl CallbackFn for QuitPopupAccept {
                fn call(&self) -> bool {
                    self.0.stop(Some("Stopping training from user input."));
                    true
                }
            }
            impl CallbackFn for PopupCancel {
                fn call(&self) -> bool {
                    true
                }
            }
            impl Drop for TuiMetricsRenderer {
                fn drop(&mut self) {
                    if !std::thread::panicking() {
                        self.reset().unwrap();
                        if let Some(summary) = &self.summary {
                            {
                                ::std::io::_print(format_args!("{0}\n", summary));
                            };
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!("{0}", summary),
                                            lvl,
                                            &(
                                                "burn_train::renderer::tui::renderer",
                                                "burn_train::renderer::tui::renderer",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                        }
                    }
                }
            }
        }
        mod status {
            use std::collections::BTreeMap;
            use crate::logger::ProgressSnapshot;
            use super::TerminalFrame;
            use ratatui::{
                prelude::{Alignment, Rect},
                style::{Color, Style, Stylize},
                text::{Line, Span},
                widgets::{Block, Borders, Paragraph, Wrap},
            };
            /// Show the training status with various information.
            pub(crate) struct StatusState {
                progress: Option<ProgressSnapshot>,
                mode: Mode,
                event_counters: BTreeMap<String, usize>,
            }
            enum Mode {
                Valid,
                Train,
                Evaluation,
            }
            impl Default for StatusState {
                fn default() -> Self {
                    Self {
                        progress: None,
                        mode: Mode::Train,
                        event_counters: BTreeMap::new(),
                    }
                }
            }
            impl StatusState {
                /// Update the training information.
                pub(crate) fn update_train(&mut self, progress: &ProgressSnapshot) {
                    self.progress = Some(progress.clone());
                    self.mode = Mode::Train;
                }
                /// Update the validation information.
                pub(crate) fn update_valid(&mut self, progress: &ProgressSnapshot) {
                    self.progress = Some(progress.clone());
                    self.mode = Mode::Valid;
                }
                /// Update the testing information.
                pub(crate) fn update_test(&mut self, progress: &ProgressSnapshot) {
                    self.progress = Some(progress.clone());
                    self.mode = Mode::Evaluation;
                }
                /// Update counters from a progress event.
                pub(crate) fn update_counter(&mut self, event: String) {
                    *self.event_counters.entry(event).or_insert(0) += 1;
                }
                /// Reset all counters at the end of a split.
                pub(crate) fn reset_counters(&mut self) {
                    for val in self.event_counters.values_mut() {
                        *val = 0;
                    }
                }
                /// Create a view.
                pub(crate) fn view(&self) -> StatusView {
                    StatusView::new(
                        self.progress.as_ref(),
                        &self.mode,
                        &self.event_counters,
                    )
                }
            }
            pub(crate) struct StatusView {
                lines: Vec<Vec<Span<'static>>>,
            }
            fn capitalize(s: &str) -> String {
                let mut chars = s.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => {
                        first.to_uppercase().collect::<String>() + chars.as_str()
                    }
                }
            }
            impl StatusView {
                fn new(
                    progress: Option<&ProgressSnapshot>,
                    mode: &Mode,
                    event_counters: &BTreeMap<String, usize>,
                ) -> Self {
                    let title = |title: &str| {
                        Span::from(
                                ::alloc::__export::must_use({
                                    ::alloc::fmt::format(format_args!(" {0} ", title))
                                }),
                            )
                            .bold()
                            .yellow()
                    };
                    let value = |value: String| Span::from(value).italic();
                    let mode_str = match mode {
                        Mode::Valid => "Validating",
                        Mode::Train => "Training",
                        Mode::Evaluation => "Evaluation",
                    };
                    let width = progress
                        .map(|p| {
                            p.global
                                .unit
                                .as_deref()
                                .map_or(0, |s| s.len())
                                .max(p.split.unit.as_deref().map_or(0, |s| s.len()))
                        })
                        .unwrap_or(0)
                        .max("Mode".len())
                        .max(event_counters.keys().map(|k| k.len()).max().unwrap_or(0));
                    let mut lines = ::alloc::boxed::box_assume_init_into_vec_unsafe(
                        ::alloc::intrinsics::write_box_via_move(
                            ::alloc::boxed::Box::new_uninit(),
                            [
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            title(
                                                &::alloc::__export::must_use({
                                                    ::alloc::fmt::format(
                                                        format_args!("{0: <1$} :", "Mode", width),
                                                    )
                                                }),
                                            ),
                                            value(mode_str.to_string()),
                                        ],
                                    ),
                                ),
                            ],
                        ),
                    );
                    if let Some(p) = progress {
                        let g = &p.global;
                        lines
                            .push(
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            title(
                                                &::alloc::__export::must_use({
                                                    ::alloc::fmt::format(
                                                        format_args!(
                                                            "{0: <1$} :",
                                                            capitalize(g.unit.as_deref().unwrap_or("")),
                                                            width,
                                                        ),
                                                    )
                                                }),
                                            ),
                                            value(
                                                ::alloc::__export::must_use({
                                                    ::alloc::fmt::format(
                                                        format_args!("{0}/{1}", g.items_processed, g.items_total),
                                                    )
                                                }),
                                            ),
                                        ],
                                    ),
                                ),
                            );
                        let s = &p.split;
                        lines
                            .push(
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            title(
                                                &::alloc::__export::must_use({
                                                    ::alloc::fmt::format(
                                                        format_args!(
                                                            "{0: <1$} :",
                                                            capitalize(s.unit.as_deref().unwrap_or("")),
                                                            width,
                                                        ),
                                                    )
                                                }),
                                            ),
                                            value(
                                                ::alloc::__export::must_use({
                                                    ::alloc::fmt::format(
                                                        format_args!("{0}/{1}", s.items_processed, s.items_total),
                                                    )
                                                }),
                                            ),
                                        ],
                                    ),
                                ),
                            );
                    }
                    for (key, val) in event_counters {
                        lines
                            .push(
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [
                                            title(
                                                &::alloc::__export::must_use({
                                                    ::alloc::fmt::format(format_args!("{0: <1$} :", key, width))
                                                }),
                                            ),
                                            value(
                                                ::alloc::__export::must_use({
                                                    ::alloc::fmt::format(format_args!("{0}", val))
                                                }),
                                            ),
                                        ],
                                    ),
                                ),
                            );
                    }
                    Self { lines }
                }
                pub(crate) fn render(self, frame: &mut TerminalFrame<'_>, size: Rect) {
                    let paragraph = Paragraph::new(
                            self.lines.into_iter().map(Line::from).collect::<Vec<_>>(),
                        )
                        .alignment(Alignment::Left)
                        .block(Block::default().borders(Borders::ALL).title("Status"))
                        .wrap(Wrap { trim: false })
                        .style(Style::default().fg(Color::Gray));
                    frame.render_widget(paragraph, size);
                }
            }
        }
        pub(crate) use base::*;
        pub(crate) use controls::*;
        pub(crate) use full_history::*;
        pub(crate) use metric_numeric::*;
        pub(crate) use metric_text::*;
        pub(crate) use plot_utils::*;
        pub(crate) use popup::*;
        pub(crate) use progress::*;
        pub(crate) use recent_history::*;
        pub use renderer::*;
        pub(crate) use status::*;
    }
    use crate::Interrupter;
    /// Return the default metrics renderer.
    ///
    /// This can be either:
    ///   - `TuiMetricsRenderer`, when the `tui` feature is enabled and `stdout` is
    ///     a terminal, or
    ///   - `CliMetricsRenderer`, when the `tui` feature is not enabled, or `stdout`
    ///     is not a terminal.
    #[allow(unused_variables)]
    pub(crate) fn default_renderer(
        interuptor: Interrupter,
        checkpoint: Option<usize>,
    ) -> Box<dyn MetricsRenderer> {
        if std::io::stdout().is_terminal() {
            return Box::new(tui::TuiMetricsRendererWrapper::new(interuptor, checkpoint));
        }
        Box::new(CliMetricsRenderer::new())
    }
}
/// The logger module.
pub mod logger {
    mod async_logger {
        use super::Logger;
        use std::sync::mpsc;
        enum Message<T> {
            Log(T),
            End,
            Sync(mpsc::Sender<()>),
        }
        /// Async logger.
        pub struct AsyncLogger<T> {
            sender: mpsc::Sender<Message<T>>,
            handler: Option<std::thread::JoinHandle<()>>,
        }
        struct LoggerThread<T, L: Logger<T>> {
            logger: L,
            receiver: mpsc::Receiver<Message<T>>,
        }
        impl<T, L: Logger<T>> LoggerThread<T, L> {
            ///Constructs a new `LoggerThread`.
            pub fn new(logger: L, receiver: mpsc::Receiver<Message<T>>) -> Self {
                LoggerThread {
                    logger: logger,
                    receiver: receiver,
                }
            }
        }
        impl<T, L> LoggerThread<T, L>
        where
            L: Logger<T>,
        {
            fn run(mut self) {
                for item in self.receiver.iter() {
                    match item {
                        Message::Log(item) => {
                            self.logger.log(item);
                        }
                        Message::End => {
                            return;
                        }
                        Message::Sync(callback) => {
                            callback
                                .send(())
                                .expect("Can return result with the callback channel.");
                        }
                    }
                }
            }
        }
        impl<T: Send + Sync + 'static> AsyncLogger<T> {
            /// Create a new async logger.
            pub fn new<L>(logger: L) -> Self
            where
                L: Logger<T> + 'static,
            {
                let (sender, receiver) = mpsc::channel();
                let thread = LoggerThread::new(logger, receiver);
                let handler = Some(std::thread::spawn(move || thread.run()));
                Self { sender, handler }
            }
            /// Sync the async logger.
            pub(crate) fn sync(&self) {
                let (sender, receiver) = mpsc::channel();
                self.sender
                    .send(Message::Sync(sender))
                    .expect("Can send message to logger thread.");
                receiver.recv().expect("Should sync, otherwise the thread is dead.");
            }
        }
        impl<T: Send> Logger<T> for AsyncLogger<T> {
            fn log(&mut self, item: T) {
                self.sender
                    .send(Message::Log(item))
                    .expect("Can log using the logger thread.");
            }
        }
        impl<T> Drop for AsyncLogger<T> {
            fn drop(&mut self) {
                self.sender
                    .send(Message::End)
                    .expect("Can send the end message to the logger thread.");
                let handler = self.handler.take();
                if let Some(handler) = handler {
                    handler.join().expect("The logger thread should stop.");
                }
            }
        }
    }
    mod base {
        /// The logger trait.
        pub trait Logger<T>: Send {
            /// Logs an item.
            ///
            /// # Arguments
            ///
            /// * `item` - The item.
            fn log(&mut self, item: T);
        }
    }
    mod file {
        use super::Logger;
        use std::{fs::File, io::Write, path::Path};
        /// File logger.
        pub struct FileLogger {
            file: File,
        }
        impl FileLogger {
            /// Create a new file logger.
            ///
            /// # Arguments
            ///
            /// * `path` - The path.
            ///
            /// # Returns
            ///
            /// The file logger.
            pub fn new(path: impl AsRef<Path>) -> Self {
                let path = path.as_ref();
                let mut options = std::fs::File::options();
                let file = options
                    .write(true)
                    .truncate(true)
                    .create(true)
                    .open(path)
                    .unwrap_or_else(|err| {
                        {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "Should be able to create the new file \'{0}\': {1}",
                                    path.display(),
                                    err,
                                ),
                            );
                        }
                    });
                Self { file }
            }
        }
        impl<T> Logger<T> for FileLogger
        where
            T: std::fmt::Display,
        {
            fn log(&mut self, item: T) {
                (&mut self.file)
                    .write_fmt(format_args!("{0}\n", item))
                    .expect("Can log an item.");
            }
        }
    }
    mod in_memory {
        use super::Logger;
        /// In memory logger.
        pub struct InMemoryLogger {
            pub(crate) values: Vec<String>,
        }
        #[automatically_derived]
        impl ::core::default::Default for InMemoryLogger {
            #[inline]
            fn default() -> InMemoryLogger {
                InMemoryLogger {
                    values: ::core::default::Default::default(),
                }
            }
        }
        impl<T> Logger<T> for InMemoryLogger
        where
            T: std::fmt::Display,
        {
            fn log(&mut self, item: T) {
                self.values.push(item.to_string());
            }
        }
    }
    mod metric {
        use super::{AsyncLogger, FileLogger, InMemoryLogger, Logger};
        use crate::metric::{
            MetricDefinition, MetricEntry, MetricId, NumericEntry,
            store::{EpochSummary, MetricsUpdate, Split},
        };
        use std::{collections::HashMap, fs, path::{Path, PathBuf}};
        const EPOCH_PREFIX: &str = "epoch-";
        /// Metric logger.
        pub trait MetricLogger: Send {
            /// Logs an item.
            ///
            /// # Arguments
            ///
            /// * `update` - Update information for all registered metrics.
            /// * `epoch` - Current epoch.
            /// * `split` - Current dataset split.
            fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split);
            /// Read the logs for an epoch.
            fn read_numeric(
                &mut self,
                name: &str,
                epoch: usize,
                split: &Split,
            ) -> Result<Vec<NumericEntry>, String>;
            /// Logs the metric definition information (name, description, unit, etc.)
            fn log_metric_definition(&mut self, definition: MetricDefinition);
            /// Logs summary at the end of the epoch.
            fn log_epoch_summary(&mut self, summary: EpochSummary);
        }
        /// The file metric logger.
        pub struct FileMetricLogger {
            loggers: HashMap<String, AsyncLogger<String>>,
            directory: PathBuf,
            metric_definitions: HashMap<MetricId, MetricDefinition>,
            is_eval: bool,
            last_epoch: Option<usize>,
        }
        impl FileMetricLogger {
            /// Create a new file metric logger.
            ///
            /// # Arguments
            ///
            /// * `directory` - The directory.
            ///
            /// # Returns
            ///
            /// The file metric logger.
            pub fn new(directory: impl AsRef<Path>) -> Self {
                Self {
                    loggers: HashMap::new(),
                    directory: directory.as_ref().to_path_buf(),
                    metric_definitions: HashMap::default(),
                    is_eval: false,
                    last_epoch: None,
                }
            }
            /// Create a new file metric logger.
            ///
            /// # Arguments
            ///
            /// * `directory` - The directory.
            ///
            /// # Returns
            ///
            /// The file metric logger.
            pub fn new_eval(directory: impl AsRef<Path>) -> Self {
                Self {
                    loggers: HashMap::new(),
                    directory: directory.as_ref().to_path_buf(),
                    metric_definitions: HashMap::default(),
                    is_eval: true,
                    last_epoch: None,
                }
            }
            pub(crate) fn split_exists(&self, split: &Split) -> bool {
                self.split_dir(split).is_some()
            }
            pub(crate) fn split_dir(&self, split: &Split) -> Option<PathBuf> {
                let split_path = match split {
                    Split::Test(Some(tag)) => {
                        self.directory.join(split.to_string()).join(tag.as_str())
                    }
                    other => self.directory.join(other.to_string()),
                };
                (split_path.exists() && split_path.is_dir()).then_some(split_path)
            }
            pub(crate) fn is_epoch_dir<P: AsRef<str>>(dirname: P) -> bool {
                dirname.as_ref().starts_with(EPOCH_PREFIX)
            }
            /// Number of epochs recorded.
            pub(crate) fn epochs(&self) -> usize {
                if self.is_eval {
                    {
                        {
                            let lvl = ::log::Level::Warn;
                            if lvl <= ::log::STATIC_MAX_LEVEL
                                && lvl <= ::log::max_level()
                            {
                                ::log::__private_api::log(
                                    { ::log::__private_api::GlobalLogger },
                                    format_args!(
                                        "Number of epochs not available when testing.",
                                    ),
                                    lvl,
                                    &(
                                        "burn_train::logger::metric",
                                        "burn_train::logger::metric",
                                        ::log::__private_api::loc(),
                                    ),
                                    (),
                                );
                            }
                        }
                    };
                    return 0;
                }
                let mut max_epoch = 0;
                for path in fs::read_dir(&self.directory).unwrap() {
                    let path = path.unwrap();
                    if fs::metadata(path.path()).unwrap().is_dir() {
                        for split_path in fs::read_dir(path.path()).unwrap() {
                            let split_path = split_path.unwrap();
                            if fs::metadata(split_path.path()).unwrap().is_dir() {
                                let dir_name = split_path
                                    .file_name()
                                    .into_string()
                                    .unwrap();
                                if !dir_name.starts_with(EPOCH_PREFIX) {
                                    continue;
                                }
                                let epoch = dir_name
                                    .replace(EPOCH_PREFIX, "")
                                    .parse::<usize>()
                                    .ok();
                                if let Some(epoch) = epoch && epoch > max_epoch {
                                    max_epoch = epoch;
                                }
                            }
                        }
                    }
                }
                max_epoch
            }
            fn train_directory(&self, epoch: usize, split: &Split) -> PathBuf {
                let name = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0}{1}", EPOCH_PREFIX, epoch))
                });
                match split {
                    Split::Train | Split::Valid | Split::Test(None) => {
                        self.directory.join(split.to_string()).join(name)
                    }
                    Split::Test(Some(tag)) => {
                        let tag = format_tag(tag);
                        self.directory.join(split.to_string()).join(tag).join(name)
                    }
                }
            }
            fn eval_directory(&self, split: &Split) -> PathBuf {
                match split {
                    Split::Train | Split::Valid | Split::Test(None) => {
                        self.directory.clone()
                    }
                    Split::Test(Some(tag)) => {
                        self.directory.join(split.to_string()).join(format_tag(tag))
                    }
                }
            }
            fn file_path(
                &self,
                name: &str,
                epoch: Option<usize>,
                split: &Split,
            ) -> PathBuf {
                let directory = match epoch {
                    Some(epoch) => self.train_directory(epoch, split),
                    None => self.eval_directory(split),
                };
                let name = name.replace(' ', "_");
                let name = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0}.log", name))
                });
                directory.join(name)
            }
            fn create_directory(&self, epoch: Option<usize>, split: &Split) {
                let directory = match epoch {
                    Some(epoch) => self.train_directory(epoch, split),
                    None => self.eval_directory(split),
                };
                std::fs::create_dir_all(directory).ok();
            }
        }
        impl FileMetricLogger {
            fn log_item(
                &mut self,
                item: &MetricEntry,
                epoch: Option<usize>,
                split: &Split,
            ) {
                let name = &self.metric_definitions.get(&item.metric_id).unwrap().name;
                let key = logger_key(name, split);
                let value = &item.serialized_entry.serialized;
                let logger = match self.loggers.get_mut(&key) {
                    Some(val) => val,
                    None => {
                        self.create_directory(epoch, split);
                        let file_path = self.file_path(name, epoch, split);
                        let logger = FileLogger::new(file_path);
                        let logger = AsyncLogger::new(logger);
                        self.loggers.insert(key.clone(), logger);
                        self.loggers
                            .get_mut(&key)
                            .expect("Can get the previously saved logger.")
                    }
                };
                logger.log(value.clone());
            }
        }
        fn format_tag(tag: &str) -> String {
            tag.trim().replace(' ', "-").to_lowercase()
        }
        impl MetricLogger for FileMetricLogger {
            fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split) {
                if !self.is_eval && self.last_epoch != Some(epoch) {
                    self.loggers.clear();
                    self.last_epoch = Some(epoch);
                }
                let entries: Vec<_> = update
                    .entries
                    .iter()
                    .chain(
                        update
                            .entries_numeric
                            .iter()
                            .map(|numeric_update| &numeric_update.entry),
                    )
                    .cloned()
                    .collect();
                for item in entries.iter() {
                    self.log_item(item, Some(epoch), split);
                }
            }
            fn read_numeric(
                &mut self,
                name: &str,
                epoch: usize,
                split: &Split,
            ) -> Result<Vec<NumericEntry>, String> {
                if let Some(value) = self.loggers.get(name) {
                    value.sync()
                }
                let file_path = self.file_path(name, Some(epoch), split);
                let mut errors = false;
                let data = std::fs::read_to_string(file_path)
                    .unwrap_or_default()
                    .split('\n')
                    .filter_map(|value| {
                        if value.is_empty() {
                            None
                        } else {
                            match NumericEntry::deserialize(value) {
                                Ok(value) => Some(value),
                                Err(err) => {
                                    {
                                        {
                                            let lvl = ::log::Level::Error;
                                            if lvl <= ::log::STATIC_MAX_LEVEL
                                                && lvl <= ::log::max_level()
                                            {
                                                ::log::__private_api::log(
                                                    { ::log::__private_api::GlobalLogger },
                                                    format_args!("{0}", err),
                                                    lvl,
                                                    &(
                                                        "burn_train::logger::metric",
                                                        "burn_train::logger::metric",
                                                        ::log::__private_api::loc(),
                                                    ),
                                                    (),
                                                );
                                            }
                                        }
                                    };
                                    errors = true;
                                    None
                                }
                            }
                        }
                    })
                    .collect();
                if errors {
                    Err("Parsing numeric entry errors".to_string())
                } else {
                    Ok(data)
                }
            }
            fn log_metric_definition(&mut self, definition: MetricDefinition) {
                self.metric_definitions.insert(definition.metric_id.clone(), definition);
            }
            fn log_epoch_summary(&mut self, _summary: EpochSummary) {
                if !self.is_eval {
                    self.loggers.clear();
                }
            }
        }
        fn logger_key(name: &str, split: &Split) -> String {
            ::alloc::__export::must_use({
                ::alloc::fmt::format(format_args!("{0}_{1}", name, split))
            })
        }
        /// In memory metric logger, useful when testing and debugging.
        pub struct InMemoryMetricLogger {
            values: HashMap<String, Vec<InMemoryLogger>>,
            last_epoch: Option<usize>,
            metric_definitions: HashMap<MetricId, MetricDefinition>,
        }
        #[automatically_derived]
        impl ::core::default::Default for InMemoryMetricLogger {
            #[inline]
            fn default() -> InMemoryMetricLogger {
                InMemoryMetricLogger {
                    values: ::core::default::Default::default(),
                    last_epoch: ::core::default::Default::default(),
                    metric_definitions: ::core::default::Default::default(),
                }
            }
        }
        impl InMemoryMetricLogger {
            /// Create a new in-memory metric logger.
            pub fn new() -> Self {
                Self::default()
            }
        }
        impl MetricLogger for InMemoryMetricLogger {
            fn log(&mut self, update: MetricsUpdate, epoch: usize, split: &Split) {
                if self.last_epoch != Some(epoch) {
                    self.values
                        .values_mut()
                        .for_each(|loggers| loggers.push(InMemoryLogger::default()));
                    self.last_epoch = Some(epoch);
                }
                let entries: Vec<_> = update
                    .entries
                    .iter()
                    .chain(
                        update
                            .entries_numeric
                            .iter()
                            .map(|numeric_update| &numeric_update.entry),
                    )
                    .cloned()
                    .collect();
                for item in entries.iter() {
                    let name = &self
                        .metric_definitions
                        .get(&item.metric_id)
                        .unwrap()
                        .name;
                    let key = logger_key(name, split);
                    if !self.values.contains_key(&key) {
                        self.values
                            .insert(
                                key.to_string(),
                                ::alloc::boxed::box_assume_init_into_vec_unsafe(
                                    ::alloc::intrinsics::write_box_via_move(
                                        ::alloc::boxed::Box::new_uninit(),
                                        [InMemoryLogger::default()],
                                    ),
                                ),
                            );
                    }
                    let values = self.values.get_mut(&key).unwrap();
                    values
                        .last_mut()
                        .unwrap()
                        .log(item.serialized_entry.serialized.clone());
                }
            }
            fn read_numeric(
                &mut self,
                name: &str,
                epoch: usize,
                split: &Split,
            ) -> Result<Vec<NumericEntry>, String> {
                let key = logger_key(name, split);
                let values = match self.values.get(&key) {
                    Some(values) => values,
                    None => return Ok(Vec::new()),
                };
                match values.get(epoch - 1) {
                    Some(logger) => {
                        Ok(
                            logger
                                .values
                                .iter()
                                .filter_map(|value| NumericEntry::deserialize(value).ok())
                                .collect(),
                        )
                    }
                    None => Ok(Vec::new()),
                }
            }
            fn log_metric_definition(&mut self, definition: MetricDefinition) {
                self.metric_definitions.insert(definition.metric_id.clone(), definition);
            }
            fn log_epoch_summary(&mut self, _summary: EpochSummary) {}
        }
    }
    mod progress {
        use burn_core::data::dataloader::Progress;
        /// Trait for logging training progress at each step and end of epoch.
        ///
        /// # Call sequence
        ///
        /// Implementors can expect the following sequence of calls for a complete training run:
        ///
        /// ```text
        /// start(total_epochs, total_items)
        ///   for each epoch:
        ///     start_split("train", total_items_train)
        ///       update_split(items_processed)  // called once per batch
        ///       ...
        ///     end_split()
        ///     start_split("valid", total_items_valid)
        ///       update_split(items_processed)  // called once per batch
        ///       ...
        ///     end_split()
        ///     update_epoch(epoch)
        /// end()
        /// ```
        ///
        /// `end()` is called whether training completes normally or is interrupted early.
        ///
        /// Implementors are responsible for tracking `total_items` and epoch state in order
        /// to reconstruct the full progress picture when `update_split` is called.
        pub trait TrainingProgressLogger: Send {
            /// Called once at the start of training, providing the total number of epochs.
            ///
            /// The total number of items of the training can optionally be provided if it is known.
            fn start(&mut self, total_epochs: usize, total_items: Option<usize>);
            /// Called at the end of each full epoch (after both train and valid splits complete).
            fn update_epoch(&mut self, epoch: usize);
            /// Called at the start of a training split, providing the split name and total number of items.
            fn start_split(&mut self, split: &str, total_items: usize);
            /// Log the progress of the current training step.
            fn update_split(&mut self, items_processed: usize);
            /// Called at the end of a training split.
            fn end_split(&mut self);
            /// Called at the end of training, whether it completed successfully or was interrupted.
            fn end(&mut self);
            /// Log a custom named event that falls outside the standard training lifecycle callbacks.
            fn log_event_training(&mut self, event: String);
        }
        /// Trait for logging evaluation progress at each step and end of evaluation.
        ///
        /// # Call sequence
        ///
        /// Implementors can expect the following sequence of calls for a complete evaluation run:
        ///
        /// ```text
        /// start_global_progress(total_tests)
        ///   for each test split:
        ///     start_test(name, total_items)
        ///       update_test_progress(items_processed)  // called once per batch
        ///       ...
        ///     end_test()
        /// end_global_progress()
        /// ```
        ///
        /// `end_global_progress()` is called whether evaluation completes normally or is interrupted early.
        ///
        /// Implementors are responsible for tracking `total_tests` and `total_items` (stored from
        /// `start_global_progress` and `start_test`) to reconstruct the full progress picture.
        pub trait EvaluationProgressLogger: Send {
            /// Called once at the start of evaluation, providing the total number of test splits.
            fn start_global_progress(&mut self, total_tests: usize);
            /// Called at the start of a test split, providing the split name and total number of items.
            fn start_test(&mut self, name: &str, total_items: usize);
            /// Log the progress of the current test step.
            fn update_test_progress(&mut self, items_processed: usize);
            /// Called at the end of a test split.
            fn end_test(&mut self);
            /// Called at the end of evaluation.
            fn end_global_progress(&mut self);
            /// Log a custom named event that falls outside the standard evaluation lifecycle callbacks.
            fn log_event_evaluation(&mut self, event: String);
        }
        /// Two-level progress snapshot combining run-level and phase-level tracking.
        ///
        /// `global_progress` spans the full training run (e.g., epochs completed out of total),
        /// while `split_progress` tracks the current phase (e.g., batches within the current epoch).
        pub struct ProgressSnapshot {
            /// Progress across the entire training run.
            pub global: Progress,
            /// Progress within the current phase (epoch or evaluation split).
            pub split: Progress,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for ProgressSnapshot {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "ProgressSnapshot",
                    "global",
                    &self.global,
                    "split",
                    &&self.split,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for ProgressSnapshot {
            #[inline]
            fn clone(&self) -> ProgressSnapshot {
                ProgressSnapshot {
                    global: ::core::clone::Clone::clone(&self.global),
                    split: ::core::clone::Clone::clone(&self.split),
                }
            }
        }
        impl ProgressSnapshot {
            /// Create a new overall progress snapshot.
            pub fn new(global: Progress, split: Progress) -> Self {
                Self { global, split }
            }
        }
    }
    pub use async_logger::*;
    pub use base::*;
    pub use file::*;
    pub use in_memory::*;
    pub use metric::*;
    pub use progress::*;
}
/// The metric module.
pub mod metric {
    /// State module.
    pub mod state {
        use std::sync::Arc;
        use burn_core::tensor::{Bool, Tensor};
        use crate::metric::{MetricName, NumericEntry, SerializedEntry, format_float};
        /// Useful utility to implement numeric metrics.
        ///
        /// # Notes
        ///
        /// The numeric metric store values inside floats.
        /// Even if some metric are integers, their mean are floats.
        pub struct NumericMetricState {
            sum: f64,
            count: usize,
            current: f64,
            current_count: usize,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for NumericMetricState {
            #[inline]
            fn clone(&self) -> NumericMetricState {
                NumericMetricState {
                    sum: ::core::clone::Clone::clone(&self.sum),
                    count: ::core::clone::Clone::clone(&self.count),
                    current: ::core::clone::Clone::clone(&self.current),
                    current_count: ::core::clone::Clone::clone(&self.current_count),
                }
            }
        }
        /// Accumulates raw predictions and targets across batches.
        ///
        /// Used by rank-based metrics (AUROC, AUC-PR) that must recompute over the
        /// whole epoch. Buffers are freed on [`reset`](Self::reset).
        pub struct PredictionAccumulatorState {
            predictions: Vec<Tensor<2>>,
            targets: Vec<Tensor<2, Bool>>,
            current: f64,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for PredictionAccumulatorState {
            #[inline]
            fn clone(&self) -> PredictionAccumulatorState {
                PredictionAccumulatorState {
                    predictions: ::core::clone::Clone::clone(&self.predictions),
                    targets: ::core::clone::Clone::clone(&self.targets),
                    current: ::core::clone::Clone::clone(&self.current),
                }
            }
        }
        /// Formatting options for the [numeric metric state](NumericMetricState).
        pub struct FormatOptions {
            name: Arc<String>,
            unit: Option<String>,
            precision: Option<usize>,
        }
        impl PredictionAccumulatorState {
            /// Create a new [prediction accumulator state](PredictionAccumulatorState).
            pub fn new() -> Self {
                Self {
                    predictions: ::alloc::vec::Vec::new(),
                    targets: ::alloc::vec::Vec::new(),
                    current: f64::NAN,
                }
            }
            /// Accumulate a batch of predictions and targets.
            pub fn accumulate(&mut self, preds: Tensor<2>, targets: Tensor<2, Bool>) {
                self.predictions.push(preds);
                self.targets.push(targets);
            }
            /// All accumulated predictions and targets, concatenated along the samples.
            pub fn tensors(&self) -> (Tensor<2>, Tensor<2, Bool>) {
                (
                    Tensor::cat(self.predictions.clone(), 0),
                    Tensor::cat(self.targets.clone(), 0),
                )
            }
            /// Record the value computed over the accumulated set and return the entry
            /// to log. Metrics using this state must declare
            /// [`NumericAggregation::Last`](crate::metric::NumericAggregation).
            pub fn update(
                &mut self,
                value: f64,
                format: FormatOptions,
            ) -> SerializedEntry {
                self.current = value;
                let serialized = NumericEntry::Value(value).serialize();
                let formatted_value = match format.precision {
                    Some(precision) => format_float(value, precision),
                    None => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0}", value))
                        })
                    }
                };
                let formatted = match format.unit {
                    Some(unit) => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("epoch {0} {1}", formatted_value, unit),
                            )
                        })
                    }
                    None => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("epoch {0}", formatted_value),
                            )
                        })
                    }
                };
                SerializedEntry::new(formatted, serialized)
            }
            /// Current value (computed over the accumulated set).
            pub fn value(&self) -> NumericEntry {
                NumericEntry::Value(self.current)
            }
            /// Reset the state, freeing the accumulated tensors.
            pub fn reset(&mut self) {
                self.predictions.clear();
                self.targets.clear();
                self.current = f64::NAN;
            }
        }
        impl Default for PredictionAccumulatorState {
            fn default() -> Self {
                Self::new()
            }
        }
        impl FormatOptions {
            /// Create the [formatting options](FormatOptions) with a name.
            pub fn new(name: MetricName) -> Self {
                Self {
                    name: name.clone(),
                    unit: None,
                    precision: None,
                }
            }
            /// Specify the metric unit.
            pub fn unit(mut self, unit: &str) -> Self {
                self.unit = Some(unit.to_string());
                self
            }
            /// Specify the floating point precision.
            pub fn precision(mut self, precision: usize) -> Self {
                self.precision = Some(precision);
                self
            }
            /// Get the metric name.
            pub fn name(&self) -> &Arc<String> {
                &self.name
            }
            /// Get the metric unit.
            pub fn unit_value(&self) -> &Option<String> {
                &self.unit
            }
            /// Get the precision.
            pub fn precision_value(&self) -> Option<usize> {
                self.precision
            }
        }
        impl NumericMetricState {
            /// Create a new [numeric metric state](NumericMetricState).
            pub fn new() -> Self {
                Self {
                    sum: 0.0,
                    count: 0,
                    current: f64::NAN,
                    current_count: 0,
                }
            }
            /// Reset the state.
            pub fn reset(&mut self) {
                self.sum = 0.0;
                self.count = 0;
                self.current = f64::NAN;
                self.current_count = 0;
            }
            /// Update the state.
            pub fn update(
                &mut self,
                value: f64,
                batch_size: usize,
                format: FormatOptions,
            ) -> SerializedEntry {
                self.sum += value * batch_size as f64;
                self.count += batch_size;
                self.current = value;
                self.current_count = batch_size;
                let value_current = value;
                let value_running = self.sum / self.count as f64;
                let serialized = NumericEntry::Aggregated {
                    aggregated_value: value_current,
                    count: batch_size,
                }
                    .serialize();
                let (formatted_current, formatted_running) = match format.precision {
                    Some(precision) => {
                        (
                            format_float(value_current, precision),
                            format_float(value_running, precision),
                        )
                    }
                    None => {
                        (
                            ::alloc::__export::must_use({
                                ::alloc::fmt::format(format_args!("{0}", value_current))
                            }),
                            ::alloc::__export::must_use({
                                ::alloc::fmt::format(format_args!("{0}", value_running))
                            }),
                        )
                    }
                };
                let formatted = match format.unit {
                    Some(unit) => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "epoch {0} {1} - batch {2} {1}",
                                    formatted_running,
                                    unit,
                                    formatted_current,
                                ),
                            )
                        })
                    }
                    None => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "epoch {0} - batch {1}",
                                    formatted_running,
                                    formatted_current,
                                ),
                            )
                        })
                    }
                };
                SerializedEntry::new(formatted, serialized)
            }
            /// Get the numeric value.
            pub fn current_value(&self) -> NumericEntry {
                NumericEntry::Aggregated {
                    aggregated_value: self.current,
                    count: self.current_count,
                }
            }
            /// Get the running aggregated value.
            pub fn running_value(&self) -> NumericEntry {
                NumericEntry::Aggregated {
                    aggregated_value: self.sum / self.count as f64,
                    count: self.count,
                }
            }
        }
        impl Default for NumericMetricState {
            fn default() -> Self {
                Self::new()
            }
        }
    }
    /// Module responsible to save and exposes data collected during training.
    pub mod store {
        pub(crate) mod aggregate {
            use crate::{
                logger::MetricLogger,
                metric::{
                    MetricAttributes, MetricDefinition, NumericAggregation, NumericEntry,
                    store::Split,
                },
            };
            use std::collections::HashMap;
            use super::{Aggregate, Direction};
            /// Type that can be used to fetch and use numeric metric aggregates.
            pub(crate) struct NumericMetricsAggregate {
                value_for_each_epoch: HashMap<Key, f64>,
                aggregations: HashMap<String, NumericAggregation>,
            }
            #[automatically_derived]
            impl ::core::default::Default for NumericMetricsAggregate {
                #[inline]
                fn default() -> NumericMetricsAggregate {
                    NumericMetricsAggregate {
                        value_for_each_epoch: ::core::default::Default::default(),
                        aggregations: ::core::default::Default::default(),
                    }
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for NumericMetricsAggregate {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    ::core::fmt::Formatter::debug_struct_field2_finish(
                        f,
                        "NumericMetricsAggregate",
                        "value_for_each_epoch",
                        &self.value_for_each_epoch,
                        "aggregations",
                        &&self.aggregations,
                    )
                }
            }
            struct Key {
                name: String,
                epoch: usize,
                split: Split,
                aggregate: Aggregate,
            }
            impl Key {
                ///Constructs a new `Key`.
                pub fn new(
                    name: String,
                    epoch: usize,
                    split: Split,
                    aggregate: Aggregate,
                ) -> Self {
                    Key {
                        name: name,
                        epoch: epoch,
                        split: split,
                        aggregate: aggregate,
                    }
                }
            }
            #[automatically_derived]
            impl ::core::hash::Hash for Key {
                #[inline]
                fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) {
                    ::core::hash::Hash::hash(&self.name, state);
                    ::core::hash::Hash::hash(&self.epoch, state);
                    ::core::hash::Hash::hash(&self.split, state);
                    ::core::hash::Hash::hash(&self.aggregate, state)
                }
            }
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for Key {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for Key {
                #[inline]
                fn eq(&self, other: &Key) -> bool {
                    self.name == other.name && self.epoch == other.epoch
                        && self.split == other.split && self.aggregate == other.aggregate
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for Key {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {
                    let _: ::core::cmp::AssertParamIsEq<String>;
                    let _: ::core::cmp::AssertParamIsEq<usize>;
                    let _: ::core::cmp::AssertParamIsEq<Split>;
                    let _: ::core::cmp::AssertParamIsEq<Aggregate>;
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for Key {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    ::core::fmt::Formatter::debug_struct_field4_finish(
                        f,
                        "Key",
                        "name",
                        &self.name,
                        "epoch",
                        &self.epoch,
                        "split",
                        &self.split,
                        "aggregate",
                        &&self.aggregate,
                    )
                }
            }
            impl NumericMetricsAggregate {
                /// Record each numeric metric's epoch-aggregation strategy from its definition.
                pub(crate) fn register_definitions(
                    &mut self,
                    definitions: &[MetricDefinition],
                ) {
                    for def in definitions {
                        if let MetricAttributes::Numeric(numeric) = &def.attributes {
                            self.aggregations
                                .insert(def.name.clone(), numeric.aggregation);
                        }
                    }
                }
                pub(crate) fn aggregate(
                    &mut self,
                    name: &str,
                    epoch: usize,
                    split: &Split,
                    aggregate: Aggregate,
                    loggers: &mut [Box<dyn MetricLogger>],
                ) -> Option<f64> {
                    let key = Key::new(
                        name.to_string(),
                        epoch,
                        split.clone(),
                        aggregate,
                    );
                    if let Some(value) = self.value_for_each_epoch.get(&key) {
                        return Some(*value);
                    }
                    let aggregation = self
                        .aggregations
                        .get(name)
                        .copied()
                        .unwrap_or_default();
                    let points = || {
                        let mut errors = Vec::new();
                        for logger in loggers {
                            match logger.read_numeric(name, epoch, split) {
                                Ok(points) => return Ok(points),
                                Err(err) => errors.push(err),
                            };
                        }
                        Err(errors.join(" "))
                    };
                    let points = points().expect("Can read values");
                    if points.is_empty() {
                        return None;
                    }
                    let value = match aggregation {
                        NumericAggregation::Last => {
                            points.last().expect("Points are not empty").current()
                        }
                        NumericAggregation::Mean => {
                            let (sum, num_points) = points
                                .into_iter()
                                .map(|entry| match entry {
                                    NumericEntry::Value(v) => (v, 1),
                                    NumericEntry::Aggregated { aggregated_value, count } => {
                                        (aggregated_value * count as f64, count)
                                    }
                                })
                                .reduce(|(acc_v, acc_n), (v, n)| (acc_v + v, acc_n + n))
                                .unwrap();
                            match aggregate {
                                Aggregate::Mean => sum / num_points as f64,
                            }
                        }
                    };
                    self.value_for_each_epoch.insert(key, value);
                    Some(value)
                }
                pub(crate) fn find_epoch(
                    &mut self,
                    name: &str,
                    split: &Split,
                    aggregate: Aggregate,
                    direction: Direction,
                    loggers: &mut [Box<dyn MetricLogger>],
                ) -> Option<usize> {
                    let mut data = Vec::new();
                    let mut current_epoch = 1;
                    while let Some(value) = self
                        .aggregate(name, current_epoch, split, aggregate, loggers)
                    {
                        data.push(value);
                        current_epoch += 1;
                    }
                    if data.is_empty() {
                        return None;
                    }
                    let mut current_value = match &direction {
                        Direction::Lowest => f64::MAX,
                        Direction::Highest => f64::MIN,
                    };
                    for (i, value) in data.into_iter().enumerate() {
                        match &direction {
                            Direction::Lowest => {
                                if value < current_value {
                                    current_value = value;
                                    current_epoch = i + 1;
                                }
                            }
                            Direction::Highest => {
                                if value > current_value {
                                    current_value = value;
                                    current_epoch = i + 1;
                                }
                            }
                        }
                    }
                    Some(current_epoch)
                }
            }
        }
        mod base {
            use std::sync::Arc;
            use crate::metric::{MetricDefinition, MetricEntry, NumericEntry};
            /// Event happening during the training/validation process.
            pub enum Event {
                /// Signal the iniialization of the metrics
                MetricsInit(Vec<MetricDefinition>),
                /// Signal that metrics have been updated.
                MetricsUpdate(MetricsUpdate),
                /// Signal the end of an epoch.
                EndEpoch(EpochSummary),
            }
            /// Contains all metric information.
            pub struct NumericMetricUpdate {
                /// Generic metric information.
                pub entry: MetricEntry,
                /// The numeric information.
                pub numeric_entry: NumericEntry,
                /// Numeric value averaged over the global step (epoch).
                pub running_entry: NumericEntry,
            }
            impl NumericMetricUpdate {
                ///Constructs a new `NumericMetricUpdate`.
                pub fn new(
                    entry: MetricEntry,
                    numeric_entry: NumericEntry,
                    running_entry: NumericEntry,
                ) -> Self {
                    NumericMetricUpdate {
                        entry: entry,
                        numeric_entry: numeric_entry,
                        running_entry: running_entry,
                    }
                }
            }
            #[automatically_derived]
            impl ::core::clone::Clone for NumericMetricUpdate {
                #[inline]
                fn clone(&self) -> NumericMetricUpdate {
                    NumericMetricUpdate {
                        entry: ::core::clone::Clone::clone(&self.entry),
                        numeric_entry: ::core::clone::Clone::clone(&self.numeric_entry),
                        running_entry: ::core::clone::Clone::clone(&self.running_entry),
                    }
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for NumericMetricUpdate {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    ::core::fmt::Formatter::debug_struct_field3_finish(
                        f,
                        "NumericMetricUpdate",
                        "entry",
                        &self.entry,
                        "numeric_entry",
                        &self.numeric_entry,
                        "running_entry",
                        &&self.running_entry,
                    )
                }
            }
            /// Contains all metric information.
            pub struct MetricsUpdate {
                /// Metrics information related to non-numeric metrics.
                pub entries: Vec<MetricEntry>,
                /// Metrics information related to numeric metrics.
                pub entries_numeric: Vec<NumericMetricUpdate>,
            }
            impl MetricsUpdate {
                ///Constructs a new `MetricsUpdate`.
                pub fn new(
                    entries: Vec<MetricEntry>,
                    entries_numeric: Vec<NumericMetricUpdate>,
                ) -> Self {
                    MetricsUpdate {
                        entries: entries,
                        entries_numeric: entries_numeric,
                    }
                }
            }
            #[automatically_derived]
            impl ::core::clone::Clone for MetricsUpdate {
                #[inline]
                fn clone(&self) -> MetricsUpdate {
                    MetricsUpdate {
                        entries: ::core::clone::Clone::clone(&self.entries),
                        entries_numeric: ::core::clone::Clone::clone(
                            &self.entries_numeric,
                        ),
                    }
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for MetricsUpdate {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    ::core::fmt::Formatter::debug_struct_field2_finish(
                        f,
                        "MetricsUpdate",
                        "entries",
                        &self.entries,
                        "entries_numeric",
                        &&self.entries_numeric,
                    )
                }
            }
            /// Summary information about a given epoch
            pub struct EpochSummary {
                /// Epoch number.
                pub epoch_number: usize,
                /// Dataset split (train, valid, test).
                pub split: Split,
            }
            impl EpochSummary {
                ///Constructs a new `EpochSummary`.
                pub fn new(epoch_number: usize, split: Split) -> Self {
                    EpochSummary {
                        epoch_number: epoch_number,
                        split: split,
                    }
                }
            }
            #[automatically_derived]
            impl ::core::clone::Clone for EpochSummary {
                #[inline]
                fn clone(&self) -> EpochSummary {
                    EpochSummary {
                        epoch_number: ::core::clone::Clone::clone(&self.epoch_number),
                        split: ::core::clone::Clone::clone(&self.split),
                    }
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for EpochSummary {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    ::core::fmt::Formatter::debug_struct_field2_finish(
                        f,
                        "EpochSummary",
                        "epoch_number",
                        &self.epoch_number,
                        "split",
                        &&self.split,
                    )
                }
            }
            /// Defines how training and validation events are collected and searched.
            ///
            /// This trait also exposes methods that uses the collected data to compute useful information.
            pub trait EventStore: Send {
                /// Collect a training/validation event.
                fn add_event(&mut self, event: Event, split: Split);
                /// Find the epoch following the given criteria from the collected data.
                fn find_epoch(
                    &mut self,
                    name: &str,
                    aggregate: Aggregate,
                    direction: Direction,
                    split: &Split,
                ) -> Option<usize>;
                /// Find the metric value for the current epoch following the given criteria.
                fn find_metric(
                    &mut self,
                    name: &str,
                    epoch: usize,
                    aggregate: Aggregate,
                    split: &Split,
                ) -> Option<f64>;
            }
            /// How to aggregate the metric.
            pub enum Aggregate {
                /// Compute the average.
                Mean,
            }
            #[automatically_derived]
            impl ::core::marker::Copy for Aggregate {}
            #[automatically_derived]
            #[doc(hidden)]
            unsafe impl ::core::clone::TrivialClone for Aggregate {}
            #[automatically_derived]
            impl ::core::clone::Clone for Aggregate {
                #[inline]
                fn clone(&self) -> Aggregate {
                    *self
                }
            }
            #[automatically_derived]
            impl ::core::hash::Hash for Aggregate {
                #[inline]
                fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) {}
            }
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for Aggregate {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for Aggregate {
                #[inline]
                fn eq(&self, other: &Aggregate) -> bool {
                    true
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for Aggregate {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {}
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for Aggregate {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    ::core::fmt::Formatter::write_str(f, "Mean")
                }
            }
            /// The split to use.
            pub enum Split {
                /// The training split.
                Train,
                /// The validation split.
                Valid,
                /// The testing split, which might be tagged.
                Test(Option<Arc<String>>),
            }
            #[automatically_derived]
            impl ::core::clone::Clone for Split {
                #[inline]
                fn clone(&self) -> Split {
                    match self {
                        Split::Train => Split::Train,
                        Split::Valid => Split::Valid,
                        Split::Test(__self_0) => {
                            Split::Test(::core::clone::Clone::clone(__self_0))
                        }
                    }
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for Split {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    match self {
                        Split::Train => ::core::fmt::Formatter::write_str(f, "Train"),
                        Split::Valid => ::core::fmt::Formatter::write_str(f, "Valid"),
                        Split::Test(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "Test",
                                &__self_0,
                            )
                        }
                    }
                }
            }
            #[automatically_derived]
            impl ::core::hash::Hash for Split {
                #[inline]
                fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    ::core::hash::Hash::hash(&__self_discr, state);
                    match self {
                        Split::Test(__self_0) => {
                            ::core::hash::Hash::hash(__self_0, state)
                        }
                        _ => {}
                    }
                }
            }
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for Split {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for Split {
                #[inline]
                fn eq(&self, other: &Split) -> bool {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    __self_discr == __arg1_discr
                        && match (self, other) {
                            (Split::Test(__self_0), Split::Test(__arg1_0)) => {
                                __self_0 == __arg1_0
                            }
                            _ => true,
                        }
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for Split {
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_fields_are_eq(&self) {
                    let _: ::core::cmp::AssertParamIsEq<Option<Arc<String>>>;
                }
            }
            impl std::fmt::Display for Split {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        Split::Train => f.write_fmt(format_args!("train")),
                        Split::Valid => f.write_fmt(format_args!("valid")),
                        Split::Test(_) => f.write_fmt(format_args!("test")),
                    }
                }
            }
            /// The direction of the query.
            pub enum Direction {
                /// Lower is better.
                Lowest,
                /// Higher is better.
                Highest,
            }
            #[automatically_derived]
            impl ::core::marker::Copy for Direction {}
            #[automatically_derived]
            #[doc(hidden)]
            unsafe impl ::core::clone::TrivialClone for Direction {}
            #[automatically_derived]
            impl ::core::clone::Clone for Direction {
                #[inline]
                fn clone(&self) -> Direction {
                    *self
                }
            }
        }
        mod client {
            use super::EventStore;
            use super::{Aggregate, Direction, Event, Split};
            use std::sync::Arc;
            use std::{sync::mpsc, thread::JoinHandle};
            /// Type that allows to communicate with an [event store](EventStore).
            pub struct EventStoreClient {
                sender: mpsc::Sender<Message>,
                handler: Option<JoinHandle<()>>,
            }
            impl EventStoreClient {
                /// Create a new [event store](EventStore) client.
                pub(crate) fn new<C>(store: C) -> Self
                where
                    C: EventStore + 'static,
                {
                    let (sender, receiver) = mpsc::channel();
                    let thread = WorkerThread::new(store, receiver);
                    let handler = std::thread::spawn(move || thread.run());
                    let handler = Some(handler);
                    Self { sender, handler }
                }
            }
            impl EventStoreClient {
                /// Add a training event to the [event store](EventStore).
                pub(crate) fn add_event_train(&self, event: Event) {
                    self.sender
                        .send(Message::OnEventTrain(event))
                        .expect("Can send event to event store thread.");
                }
                /// Add a validation event to the [event store](EventStore).
                pub(crate) fn add_event_valid(&self, event: Event) {
                    self.sender
                        .send(Message::OnEventValid(event))
                        .expect("Can send event to event store thread.");
                }
                /// Add a testing event to the [event store](EventStore).
                pub(crate) fn add_event_test(&self, event: Event, tag: Arc<String>) {
                    self.sender
                        .send(Message::OnEventTest(event, tag))
                        .expect("Can send event to event store thread.");
                }
                /// Find the epoch following the given criteria from the collected data.
                pub fn find_epoch(
                    &self,
                    name: &str,
                    aggregate: Aggregate,
                    direction: Direction,
                    split: &Split,
                ) -> Option<usize> {
                    let (sender, receiver) = mpsc::sync_channel(1);
                    self.sender
                        .send(
                            Message::FindEpoch(
                                name.to_string(),
                                aggregate,
                                direction,
                                split.clone(),
                                sender,
                            ),
                        )
                        .expect("Can send event to event store thread.");
                    match receiver.recv() {
                        Ok(value) => value,
                        Err(err) => {
                            ::core::panicking::panic_fmt(
                                format_args!("Event store thread crashed: {0:?}", err),
                            );
                        }
                    }
                }
                /// Find the metric value for the current epoch following the given criteria.
                pub fn find_metric(
                    &self,
                    name: &str,
                    epoch: usize,
                    aggregate: Aggregate,
                    split: &Split,
                ) -> Option<f64> {
                    let (sender, receiver) = mpsc::sync_channel(1);
                    self.sender
                        .send(
                            Message::FindMetric(
                                name.to_string(),
                                epoch,
                                aggregate,
                                split.clone(),
                                sender,
                            ),
                        )
                        .expect("Can send event to event store thread.");
                    match receiver.recv() {
                        Ok(value) => value,
                        Err(err) => {
                            ::core::panicking::panic_fmt(
                                format_args!("Event store thread crashed: {0:?}", err),
                            );
                        }
                    }
                }
            }
            struct WorkerThread<S> {
                store: S,
                receiver: mpsc::Receiver<Message>,
            }
            impl<S> WorkerThread<S> {
                ///Constructs a new `WorkerThread`.
                pub fn new(store: S, receiver: mpsc::Receiver<Message>) -> Self {
                    WorkerThread {
                        store: store,
                        receiver: receiver,
                    }
                }
            }
            impl<C> WorkerThread<C>
            where
                C: EventStore,
            {
                fn run(mut self) {
                    for item in self.receiver.iter() {
                        match item {
                            Message::End => {
                                return;
                            }
                            Message::FindEpoch(
                                name,
                                aggregate,
                                direction,
                                split,
                                callback,
                            ) => {
                                let response = self
                                    .store
                                    .find_epoch(&name, aggregate, direction, &split);
                                callback
                                    .send(response)
                                    .expect("Can send response using callback channel.");
                            }
                            Message::FindMetric(
                                name,
                                epoch,
                                aggregate,
                                split,
                                callback,
                            ) => {
                                let response = self
                                    .store
                                    .find_metric(&name, epoch, aggregate, &split);
                                callback
                                    .send(response)
                                    .expect("Can send response using callback channel.");
                            }
                            Message::OnEventTrain(event) => {
                                self.store.add_event(event, Split::Train)
                            }
                            Message::OnEventValid(event) => {
                                self.store.add_event(event, Split::Valid)
                            }
                            Message::OnEventTest(event, tag) => {
                                self.store.add_event(event, Split::Test(Some(tag)))
                            }
                        }
                    }
                }
            }
            enum Message {
                OnEventTest(Event, Arc<String>),
                OnEventTrain(Event),
                OnEventValid(Event),
                End,
                FindEpoch(
                    String,
                    Aggregate,
                    Direction,
                    Split,
                    mpsc::SyncSender<Option<usize>>,
                ),
                FindMetric(
                    String,
                    usize,
                    Aggregate,
                    Split,
                    mpsc::SyncSender<Option<f64>>,
                ),
            }
            impl Drop for EventStoreClient {
                fn drop(&mut self) {
                    self.sender
                        .send(Message::End)
                        .expect("Can send the end message to the event store thread.");
                    let handler = self.handler.take();
                    if let Some(handler) = handler {
                        handler.join().expect("The event store thread should stop.");
                    }
                }
            }
        }
        mod log {
            use std::collections::HashMap;
            use super::{
                Aggregate, Direction, Event, EventStore, Split,
                aggregate::NumericMetricsAggregate,
            };
            use crate::logger::MetricLogger;
            pub(crate) struct LogEventStore {
                loggers: Vec<Box<dyn MetricLogger>>,
                aggregate: NumericMetricsAggregate,
                epochs: HashMap<Split, usize>,
            }
            #[automatically_derived]
            impl ::core::default::Default for LogEventStore {
                #[inline]
                fn default() -> LogEventStore {
                    LogEventStore {
                        loggers: ::core::default::Default::default(),
                        aggregate: ::core::default::Default::default(),
                        epochs: ::core::default::Default::default(),
                    }
                }
            }
            impl EventStore for LogEventStore {
                fn add_event(&mut self, event: Event, split: Split) {
                    let epoch = *self.epochs.entry(split.clone()).or_insert(1);
                    match event {
                        Event::MetricsInit(definitions) => {
                            self.aggregate.register_definitions(&definitions);
                            definitions
                                .iter()
                                .for_each(|def| {
                                    self.loggers
                                        .iter_mut()
                                        .for_each(|logger| {
                                            logger.log_metric_definition(def.clone())
                                        });
                                });
                        }
                        Event::MetricsUpdate(update) => {
                            self.loggers
                                .iter_mut()
                                .for_each(|logger| {
                                    logger.log(update.clone(), epoch, &split)
                                });
                        }
                        Event::EndEpoch(summary) => {
                            self.epochs.insert(split, summary.epoch_number + 1);
                            self.loggers
                                .iter_mut()
                                .for_each(|logger| {
                                    logger.log_epoch_summary(summary.clone())
                                });
                        }
                    }
                }
                fn find_epoch(
                    &mut self,
                    name: &str,
                    aggregate: Aggregate,
                    direction: Direction,
                    split: &Split,
                ) -> Option<usize> {
                    self.aggregate
                        .find_epoch(name, split, aggregate, direction, &mut self.loggers)
                }
                fn find_metric(
                    &mut self,
                    name: &str,
                    epoch: usize,
                    aggregate: Aggregate,
                    split: &Split,
                ) -> Option<f64> {
                    self.aggregate
                        .aggregate(name, epoch, split, aggregate, &mut self.loggers)
                }
            }
            impl LogEventStore {
                /// Register a logger for metrics.
                pub(crate) fn register_logger<ML: MetricLogger + 'static>(
                    &mut self,
                    logger: ML,
                ) {
                    self.loggers.push(Box::new(logger));
                }
                /// Returns whether any loggers are registered.
                pub(crate) fn has_loggers(&self) -> bool {
                    !self.loggers.is_empty()
                }
            }
        }
        pub(crate) use self::log::*;
        pub use base::*;
        pub use client::*;
    }
    mod rl {
        mod cum_reward {
            use std::sync::Arc;
            use super::super::{
                MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
                state::{FormatOptions, NumericMetricState},
            };
            use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};
            /// Metric for the cumulative reward of the last completed episode.
            pub struct CumulativeRewardMetric {
                name: MetricName,
                state: NumericMetricState,
            }
            #[automatically_derived]
            impl ::core::clone::Clone for CumulativeRewardMetric {
                #[inline]
                fn clone(&self) -> CumulativeRewardMetric {
                    CumulativeRewardMetric {
                        name: ::core::clone::Clone::clone(&self.name),
                        state: ::core::clone::Clone::clone(&self.state),
                    }
                }
            }
            impl CumulativeRewardMetric {
                /// Creates a new episode length metric.
                pub fn new() -> Self {
                    Self {
                        name: Arc::new("Cum. Reward".to_string()),
                        state: NumericMetricState::new(),
                    }
                }
            }
            impl Default for CumulativeRewardMetric {
                fn default() -> Self {
                    Self::new()
                }
            }
            /// The [CumulativeRewardMetric](CumulativeRewardMetric) input type.
            pub struct CumulativeRewardInput {
                cum_reward: f64,
            }
            impl CumulativeRewardInput {
                ///Constructs a new `CumulativeRewardInput`.
                pub fn new(cum_reward: f64) -> Self {
                    CumulativeRewardInput {
                        cum_reward: cum_reward,
                    }
                }
            }
            impl Metric for CumulativeRewardMetric {
                type Input = CumulativeRewardInput;
                fn update(
                    &mut self,
                    item: &CumulativeRewardInput,
                    _metadata: &MetricMetadata,
                ) -> SerializedEntry {
                    self.state
                        .update(
                            item.cum_reward,
                            1,
                            FormatOptions::new(self.name()).precision(2),
                        )
                }
                fn clear(&mut self) {
                    self.state.reset()
                }
                fn name(&self) -> MetricName {
                    self.name.clone()
                }
                fn attributes(&self) -> MetricAttributes {
                    NumericAttributes {
                        unit: None,
                        higher_is_better: true,
                        ..Default::default()
                    }
                        .into()
                }
            }
            impl Numeric for CumulativeRewardMetric {
                fn value(&self) -> NumericEntry {
                    self.state.current_value()
                }
                fn running_value(&self) -> NumericEntry {
                    self.state.running_value()
                }
            }
        }
        mod ep_len {
            use std::sync::Arc;
            use super::super::{
                MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
                state::{FormatOptions, NumericMetricState},
            };
            use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};
            /// Metric for the length of the last completed episode.
            pub struct EpisodeLengthMetric {
                name: MetricName,
                state: NumericMetricState,
            }
            #[automatically_derived]
            impl ::core::clone::Clone for EpisodeLengthMetric {
                #[inline]
                fn clone(&self) -> EpisodeLengthMetric {
                    EpisodeLengthMetric {
                        name: ::core::clone::Clone::clone(&self.name),
                        state: ::core::clone::Clone::clone(&self.state),
                    }
                }
            }
            impl EpisodeLengthMetric {
                /// Creates a new episode length metric.
                pub fn new() -> Self {
                    Self {
                        name: Arc::new("Episode length".to_string()),
                        state: NumericMetricState::new(),
                    }
                }
            }
            impl Default for EpisodeLengthMetric {
                fn default() -> Self {
                    Self::new()
                }
            }
            /// The [EpisodeLengthMetric](EpisodeLengthMetric) input type.
            pub struct EpisodeLengthInput {
                ep_len: f64,
            }
            impl EpisodeLengthInput {
                ///Constructs a new `EpisodeLengthInput`.
                pub fn new(ep_len: f64) -> Self {
                    EpisodeLengthInput {
                        ep_len: ep_len,
                    }
                }
            }
            impl Metric for EpisodeLengthMetric {
                type Input = EpisodeLengthInput;
                fn update(
                    &mut self,
                    item: &EpisodeLengthInput,
                    _metadata: &MetricMetadata,
                ) -> SerializedEntry {
                    self.state
                        .update(
                            item.ep_len,
                            1,
                            FormatOptions::new(self.name()).precision(0),
                        )
                }
                fn clear(&mut self) {
                    self.state.reset()
                }
                fn name(&self) -> MetricName {
                    self.name.clone()
                }
                fn attributes(&self) -> MetricAttributes {
                    NumericAttributes {
                        unit: Some(String::from("steps")),
                        higher_is_better: true,
                        ..Default::default()
                    }
                        .into()
                }
            }
            impl Numeric for EpisodeLengthMetric {
                fn value(&self) -> NumericEntry {
                    self.state.current_value()
                }
                fn running_value(&self) -> NumericEntry {
                    self.state.running_value()
                }
            }
        }
        mod exploration_rate {
            use std::sync::Arc;
            use super::super::{
                MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
                state::{FormatOptions, NumericMetricState},
            };
            use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};
            /// Metric for the length of the last completed episode.
            pub struct ExplorationRateMetric {
                name: MetricName,
                state: NumericMetricState,
            }
            #[automatically_derived]
            impl ::core::clone::Clone for ExplorationRateMetric {
                #[inline]
                fn clone(&self) -> ExplorationRateMetric {
                    ExplorationRateMetric {
                        name: ::core::clone::Clone::clone(&self.name),
                        state: ::core::clone::Clone::clone(&self.state),
                    }
                }
            }
            impl ExplorationRateMetric {
                /// Creates a new episode length metric.
                pub fn new() -> Self {
                    Self {
                        name: Arc::new("Exploration rate".to_string()),
                        state: NumericMetricState::new(),
                    }
                }
            }
            impl Default for ExplorationRateMetric {
                fn default() -> Self {
                    Self::new()
                }
            }
            /// The [ExplorationRateMetric](ExplorationRateMetric) input type.
            pub struct ExplorationRateInput {
                exploration_rate: f64,
            }
            impl ExplorationRateInput {
                ///Constructs a new `ExplorationRateInput`.
                pub fn new(exploration_rate: f64) -> Self {
                    ExplorationRateInput {
                        exploration_rate: exploration_rate,
                    }
                }
            }
            impl Metric for ExplorationRateMetric {
                type Input = ExplorationRateInput;
                fn update(
                    &mut self,
                    item: &ExplorationRateInput,
                    _metadata: &MetricMetadata,
                ) -> SerializedEntry {
                    self.state
                        .update(
                            item.exploration_rate,
                            1,
                            FormatOptions::new(self.name()).precision(3),
                        )
                }
                fn clear(&mut self) {
                    self.state.reset()
                }
                fn name(&self) -> MetricName {
                    self.name.clone()
                }
                fn attributes(&self) -> MetricAttributes {
                    NumericAttributes {
                        unit: Some(String::from("%")),
                        higher_is_better: false,
                        ..Default::default()
                    }
                        .into()
                }
            }
            impl Numeric for ExplorationRateMetric {
                fn value(&self) -> NumericEntry {
                    self.state.current_value()
                }
                fn running_value(&self) -> NumericEntry {
                    self.state.running_value()
                }
            }
        }
        pub use cum_reward::*;
        pub use ep_len::*;
        pub use exploration_rate::*;
    }
    pub use rl::*;
    mod cpu_temp {
        use std::sync::Arc;
        /// CPU Temperature metric
        use super::MetricMetadata;
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericEntry, SerializedEntry,
        };
        use systemstat::{Platform, System};
        /// CPU Temperature in celsius degrees
        pub struct CpuTemperature {
            name: MetricName,
            temp_celsius: f32,
            sys: Arc<System>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for CpuTemperature {
            #[inline]
            fn clone(&self) -> CpuTemperature {
                CpuTemperature {
                    name: ::core::clone::Clone::clone(&self.name),
                    temp_celsius: ::core::clone::Clone::clone(&self.temp_celsius),
                    sys: ::core::clone::Clone::clone(&self.sys),
                }
            }
        }
        impl CpuTemperature {
            /// Creates a new CPU temp metric
            pub fn new() -> Self {
                let name = Arc::new("CPU Temperature".to_string());
                Self {
                    name,
                    temp_celsius: 0.,
                    sys: Arc::new(System::new()),
                }
            }
        }
        impl Default for CpuTemperature {
            fn default() -> Self {
                CpuTemperature::new()
            }
        }
        impl Metric for CpuTemperature {
            type Input = ();
            fn update(
                &mut self,
                _item: &Self::Input,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                match self.sys.cpu_temp() {
                    Ok(temp) => self.temp_celsius = temp,
                    Err(_) => self.temp_celsius = f32::NAN,
                }
                let formatted = match self.temp_celsius.is_nan() {
                    true => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0}: NaN °C", self.name()),
                            )
                        })
                    }
                    false => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "{0}: {1:.2} °C",
                                    self.name(),
                                    self.temp_celsius,
                                ),
                            )
                        })
                    }
                };
                let raw = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:.2}", self.temp_celsius))
                });
                SerializedEntry::new(formatted, raw)
            }
            fn clear(&mut self) {}
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                super::NumericAttributes {
                    unit: Some("°C".to_string()),
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for CpuTemperature {
            fn value(&self) -> NumericEntry {
                NumericEntry::Value(self.temp_celsius as f64)
            }
            fn running_value(&self) -> NumericEntry {
                NumericEntry::Value(self.temp_celsius as f64)
            }
        }
    }
    mod cpu_use {
        use super::MetricMetadata;
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericEntry, SerializedEntry,
        };
        use std::{sync::Arc, time::{Duration, Instant}};
        use sysinfo::{CpuRefreshKind, RefreshKind, System};
        /// General CPU Usage metric
        pub struct CpuUse {
            name: MetricName,
            last_refresh: Instant,
            refresh_frequency: Duration,
            sys: System,
            current: f64,
        }
        impl Clone for CpuUse {
            fn clone(&self) -> Self {
                Self {
                    name: self.name.clone(),
                    last_refresh: self.last_refresh,
                    refresh_frequency: self.refresh_frequency,
                    sys: System::new(),
                    current: self.current,
                }
            }
        }
        impl CpuUse {
            /// Creates a new CPU metric
            pub fn new() -> Self {
                let mut sys = System::new();
                let current = Self::refresh(&mut sys);
                let name = "CPU Usage".to_string();
                Self {
                    name: Arc::new(name),
                    last_refresh: Instant::now(),
                    refresh_frequency: Duration::from_millis(200),
                    sys,
                    current,
                }
            }
            fn refresh(sys: &mut System) -> f64 {
                sys.refresh_specifics(
                    RefreshKind::nothing()
                        .with_cpu(CpuRefreshKind::nothing().with_cpu_usage()),
                );
                let cpus = sys.cpus();
                let num_cpus = cpus.len();
                let use_percentage = cpus
                    .iter()
                    .fold(0.0, |acc, cpu| acc + cpu.cpu_usage()) as f64;
                use_percentage / num_cpus as f64
            }
        }
        impl Default for CpuUse {
            fn default() -> Self {
                CpuUse::new()
            }
        }
        impl Metric for CpuUse {
            type Input = ();
            fn update(
                &mut self,
                _item: &Self::Input,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                if self.last_refresh.elapsed() >= self.refresh_frequency {
                    self.current = Self::refresh(&mut self.sys);
                    self.last_refresh = Instant::now();
                }
                let formatted = ::alloc::__export::must_use({
                    ::alloc::fmt::format(
                        format_args!("{0}: {1:.2} %", self.name(), self.current),
                    )
                });
                let raw = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:.2}", self.current))
                });
                SerializedEntry::new(formatted, raw)
            }
            fn clear(&mut self) {}
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                super::NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for CpuUse {
            fn value(&self) -> NumericEntry {
                NumericEntry::Value(self.current)
            }
            fn running_value(&self) -> NumericEntry {
                NumericEntry::Value(self.current)
            }
        }
    }
    mod cuda {
        use std::sync::Arc;
        use super::MetricMetadata;
        use crate::metric::{Metric, MetricName, SerializedEntry};
        use nvml_wrapper::Nvml;
        /// Track basic cuda infos.
        pub struct CudaMetric {
            name: MetricName,
            nvml: Arc<Option<Nvml>>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for CudaMetric {
            #[inline]
            fn clone(&self) -> CudaMetric {
                CudaMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    nvml: ::core::clone::Clone::clone(&self.nvml),
                }
            }
        }
        impl CudaMetric {
            /// Creates a new metric for CUDA.
            pub fn new() -> Self {
                Self {
                    name: Arc::new("Cuda".to_string()),
                    nvml: Arc::new(
                        Nvml::init()
                            .map(Some)
                            .unwrap_or_else(|err| {
                                {
                                    {
                                        let lvl = ::log::Level::Warn;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api::log(
                                                { ::log::__private_api::GlobalLogger },
                                                format_args!("Unable to initialize CUDA Metric: {0}", err),
                                                lvl,
                                                &(
                                                    "burn_train::metric::cuda",
                                                    "burn_train::metric::cuda",
                                                    ::log::__private_api::loc(),
                                                ),
                                                (),
                                            );
                                        }
                                    }
                                };
                                None
                            }),
                    ),
                }
            }
        }
        impl Default for CudaMetric {
            fn default() -> Self {
                Self::new()
            }
        }
        impl Metric for CudaMetric {
            type Input = ();
            fn update(
                &mut self,
                _item: &(),
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let not_available = || SerializedEntry::new(
                    "Unavailable".to_string(),
                    "Unavailable".to_string(),
                );
                let available = |nvml: &Nvml| {
                    let mut formatted = String::new();
                    let mut raw_running = String::new();
                    let device_count = match nvml.device_count() {
                        Ok(val) => val,
                        Err(err) => {
                            {
                                {
                                    let lvl = ::log::Level::Warn;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Unable to get the number of cuda devices: {0}",
                                                err,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::metric::cuda",
                                                "burn_train::metric::cuda",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            return not_available();
                        }
                    };
                    for index in 0..device_count {
                        let device = match nvml.device_by_index(index) {
                            Ok(val) => val,
                            Err(err) => {
                                {
                                    {
                                        let lvl = ::log::Level::Warn;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api::log(
                                                { ::log::__private_api::GlobalLogger },
                                                format_args!("Unable to get device {0}: {1}", index, err),
                                                lvl,
                                                &(
                                                    "burn_train::metric::cuda",
                                                    "burn_train::metric::cuda",
                                                    ::log::__private_api::loc(),
                                                ),
                                                (),
                                            );
                                        }
                                    }
                                };
                                return not_available();
                            }
                        };
                        let memory_info = match device.memory_info() {
                            Ok(info) => info,
                            Err(err) => {
                                {
                                    {
                                        let lvl = ::log::Level::Warn;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api::log(
                                                { ::log::__private_api::GlobalLogger },
                                                format_args!(
                                                    "Unable to get memory info from device {0}: {1}",
                                                    index,
                                                    err,
                                                ),
                                                lvl,
                                                &(
                                                    "burn_train::metric::cuda",
                                                    "burn_train::metric::cuda",
                                                    ::log::__private_api::loc(),
                                                ),
                                                (),
                                            );
                                        }
                                    }
                                };
                                return not_available();
                            }
                        };
                        let used_gb = memory_info.used as f64 * 1e-9;
                        let total_gb = memory_info.total as f64 * 1e-9;
                        let memory_info_formatted = ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0:.2}/{1:.2} Gb", used_gb, total_gb),
                            )
                        });
                        let memory_info_raw = ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0}/{1}", used_gb, total_gb),
                            )
                        });
                        formatted = ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "{0} GPU #{1} - Memory {2}",
                                    formatted,
                                    index,
                                    memory_info_formatted,
                                ),
                            )
                        });
                        raw_running = ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0} ", memory_info_raw))
                        });
                        let utilization_rates = match device.utilization_rates() {
                            Ok(rate) => rate,
                            Err(err) => {
                                {
                                    {
                                        let lvl = ::log::Level::Warn;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api::log(
                                                { ::log::__private_api::GlobalLogger },
                                                format_args!(
                                                    "Unable to get utilization rates from device {0}: {1}",
                                                    index,
                                                    err,
                                                ),
                                                lvl,
                                                &(
                                                    "burn_train::metric::cuda",
                                                    "burn_train::metric::cuda",
                                                    ::log::__private_api::loc(),
                                                ),
                                                (),
                                            );
                                        }
                                    }
                                };
                                return not_available();
                            }
                        };
                        let utilization_rate_formatted = ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0}%", utilization_rates.gpu),
                            )
                        });
                        formatted = ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "{0} - Usage {1}",
                                    formatted,
                                    utilization_rate_formatted,
                                ),
                            )
                        });
                        if let Ok(power_mw) = device.power_usage() {
                            let power_w = power_mw as f64 / 1000.0;
                            formatted = ::alloc::__export::must_use({
                                ::alloc::fmt::format(
                                    format_args!("{0} - Power {1:.1} W", formatted, power_w),
                                )
                            });
                        }
                    }
                    SerializedEntry::new(formatted, raw_running)
                };
                match self.nvml.as_ref() {
                    Some(nvml) => available(nvml),
                    None => not_available(),
                }
            }
            fn clear(&mut self) {}
            fn name(&self) -> MetricName {
                self.name.clone()
            }
        }
    }
    mod memory_use {
        /// RAM use metric
        use super::{MetricAttributes, MetricMetadata, NumericAttributes};
        use crate::metric::{Metric, Numeric, NumericEntry, SerializedEntry};
        use std::{sync::Arc, time::{Duration, Instant}};
        use sysinfo::System;
        /// Memory information
        pub struct CpuMemory {
            name: Arc<String>,
            last_refresh: Instant,
            refresh_frequency: Duration,
            sys: System,
            ram_bytes_total: u64,
            ram_bytes_used: u64,
        }
        impl Clone for CpuMemory {
            fn clone(&self) -> Self {
                Self {
                    name: self.name.clone(),
                    last_refresh: self.last_refresh,
                    refresh_frequency: self.refresh_frequency,
                    sys: System::new(),
                    ram_bytes_total: self.ram_bytes_total,
                    ram_bytes_used: self.ram_bytes_used,
                }
            }
        }
        impl CpuMemory {
            /// Creates a new memory metric
            pub fn new() -> Self {
                let mut metric = Self {
                    name: Arc::new("CPU Memory".into()),
                    last_refresh: Instant::now(),
                    refresh_frequency: Duration::from_millis(200),
                    sys: System::new(),
                    ram_bytes_total: 0,
                    ram_bytes_used: 0,
                };
                metric.refresh();
                metric
            }
            fn refresh(&mut self) {
                self.sys.refresh_memory();
                self.last_refresh = Instant::now();
                self.ram_bytes_total = self.sys.total_memory();
                self.ram_bytes_used = self.sys.used_memory();
            }
        }
        impl Default for CpuMemory {
            fn default() -> Self {
                CpuMemory::new()
            }
        }
        impl Metric for CpuMemory {
            type Input = ();
            fn update(
                &mut self,
                _item: &Self::Input,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                if self.last_refresh.elapsed() >= self.refresh_frequency {
                    self.refresh();
                }
                let raw = bytes2gb(self.ram_bytes_used);
                let formatted = ::alloc::__export::must_use({
                    ::alloc::fmt::format(
                        format_args!(
                            "RAM Used: {0:.2} / {1:.2} Gb",
                            raw,
                            bytes2gb(self.ram_bytes_total),
                        ),
                    )
                });
                SerializedEntry::new(formatted, raw.to_string())
            }
            fn clear(&mut self) {}
            fn name(&self) -> Arc<String> {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("Gb".to_string()),
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for CpuMemory {
            fn value(&self) -> NumericEntry {
                NumericEntry::Value(bytes2gb(self.ram_bytes_used))
            }
            fn running_value(&self) -> NumericEntry {
                NumericEntry::Value(bytes2gb(self.ram_bytes_used))
            }
        }
        fn bytes2gb(bytes: u64) -> f64 {
            bytes as f64 / 1e9
        }
    }
    pub use cpu_temp::*;
    pub use cpu_use::*;
    pub use cuda::*;
    pub use memory_use::*;
    mod acc {
        use super::MetricMetadata;
        use super::state::{FormatOptions, NumericMetricState};
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, SerializedEntry,
        };
        use burn_core::tensor::{Int, Tensor};
        /// The accuracy metric.
        pub struct AccuracyMetric {
            name: MetricName,
            state: NumericMetricState,
            pad_token: Option<usize>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for AccuracyMetric {
            #[inline]
            fn clone(&self) -> AccuracyMetric {
                AccuracyMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    pad_token: ::core::clone::Clone::clone(&self.pad_token),
                }
            }
        }
        /// The [accuracy metric](AccuracyMetric) input type.
        pub struct AccuracyInput {
            outputs: Tensor<2>,
            targets: Tensor<1, Int>,
        }
        impl AccuracyInput {
            ///Constructs a new `AccuracyInput`.
            pub fn new(outputs: Tensor<2>, targets: Tensor<1, Int>) -> Self {
                AccuracyInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl Default for AccuracyMetric {
            fn default() -> Self {
                Self::new()
            }
        }
        impl AccuracyMetric {
            /// Creates the metric.
            pub fn new() -> Self {
                Self {
                    name: MetricName::new("Accuracy".to_string()),
                    state: Default::default(),
                    pad_token: Default::default(),
                }
            }
            /// Sets the pad token.
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
        }
        impl Metric for AccuracyMetric {
            type Input = AccuracyInput;
            fn update(
                &mut self,
                input: &AccuracyInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let targets = input.targets.clone();
                let outputs = input.outputs.clone();
                let [batch_size, _n_classes] = outputs.dims();
                let outputs = outputs.argmax(1).reshape([batch_size]);
                let accuracy = match self.pad_token {
                    Some(pad_token) => {
                        let mask = targets.clone().equal_elem(pad_token as i64);
                        let matches = outputs
                            .equal(targets)
                            .float()
                            .mask_fill(mask.clone(), 0);
                        let num_pad = mask.float().sum();
                        let acc = matches.sum() / (num_pad.neg() + batch_size as f32);
                        acc.into_scalar::<f64>()
                    }
                    None => {
                        outputs.equal(targets).int().sum().into_scalar::<f64>()
                            / batch_size as f64
                    }
                };
                self.state
                    .update(
                        100.0 * accuracy,
                        batch_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                super::NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for AccuracyMetric {
            fn value(&self) -> super::NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> super::NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod auc_pr {
        use super::MetricMetadata;
        use super::state::{FormatOptions, PredictionAccumulatorState};
        use crate::metric::{
            ClassReduction, ConfusionStatsInput, Metric, MetricAttributes, MetricName,
            Numeric, NumericAggregation, NumericAttributes, SerializedEntry,
        };
        use burn_core::tensor::{Int, Tensor};
        use std::sync::Arc;
        /// The Area Under the Precision-Recall Curve (AUC-PR).
        ///
        /// Computed as **Average Precision** — `AP = Σ (Rₙ − Rₙ₋₁) · Pₙ` — the
        /// standard non-interpolated estimator of the area under the
        /// precision-recall curve (equivalent to scikit-learn's
        /// `average_precision_score`), not the (biased) trapezoidal integration.
        ///
        /// Supports binary, multiclass and multi-label classification through a
        /// One-vs-Rest decomposition, aggregated with the configured
        /// [class reduction](ClassReduction).
        pub struct AucPrMetric {
            name: MetricName,
            state: PredictionAccumulatorState,
            class_reduction: ClassReduction,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for AucPrMetric {
            #[inline]
            fn clone(&self) -> AucPrMetric {
                AucPrMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    class_reduction: ::core::clone::Clone::clone(&self.class_reduction),
                }
            }
        }
        impl Default for AucPrMetric {
            fn default() -> Self {
                Self::new(Default::default())
            }
        }
        impl AucPrMetric {
            fn new(class_reduction: ClassReduction) -> Self {
                let state = Default::default();
                let name = Arc::new(
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(
                            format_args!("AUC-PR [{0:?}]", class_reduction),
                        )
                    }),
                );
                Self {
                    state,
                    class_reduction,
                    name,
                }
            }
            /// AUC-PR metric for binary classification.
            #[allow(dead_code)]
            pub fn binary() -> Self {
                Self::new(ClassReduction::default())
            }
            /// AUC-PR metric for multiclass classification.
            ///
            /// # Arguments
            ///
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multiclass(class_reduction: ClassReduction) -> Self {
                Self::new(class_reduction)
            }
            /// AUC-PR metric for multi-label classification.
            ///
            /// # Arguments
            ///
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multilabel(class_reduction: ClassReduction) -> Self {
                Self::new(class_reduction)
            }
            /// Per-column Average Precision via the step-wise estimator
            /// `AP = (1/P) · Σ_{positives, score desc} (cumulative positives / rank)`.
            ///
            /// `scores` and `targets` are `[n, c]` (`targets` as 0./1.); a column
            /// with no positive (`P = 0`) yields `NaN` (handled by the caller).
            fn average_precision(scores: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
                let [n, _c] = scores.dims();
                let device = scores.device();
                let order = scores.argsort_descending(0);
                let sorted_targets = targets.clone().gather(0, order);
                let tp = sorted_targets.clone().cumsum(0);
                let ranks = Tensor::<1, Int>::arange(1..n as i64 + 1, &device)
                    .float()
                    .reshape([n, 1]);
                let precision = tp / ranks;
                let p_total = targets.sum_dim(0);
                let delta_recall = sorted_targets / p_total;
                (precision * delta_recall).sum_dim(0).squeeze_dims::<1>(&[0])
            }
        }
        impl Metric for AucPrMetric {
            type Input = ConfusionStatsInput;
            fn update(
                &mut self,
                input: &ConfusionStatsInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                self.state.accumulate(input.predictions.clone(), input.targets.clone());
                let (predictions, targets) = self.state.tensors();
                let [n, c] = predictions.dims();
                let (scores, targets) = match self.class_reduction {
                    ClassReduction::Macro => (predictions, targets.float()),
                    ClassReduction::Micro => {
                        (
                            predictions.reshape([n * c, 1]),
                            targets.float().reshape([n * c, 1]),
                        )
                    }
                };
                let ap = Self::average_precision(scores, targets);
                let keep = ap.clone().is_nan().bool_not().argwhere().squeeze_dim::<1>(1);
                let metric = if keep.dims()[0] == 0 {
                    {
                        {
                            let lvl = ::log::Level::Warn;
                            if lvl <= ::log::STATIC_MAX_LEVEL
                                && lvl <= ::log::max_level()
                            {
                                ::log::__private_api::log(
                                    { ::log::__private_api::GlobalLogger },
                                    format_args!(
                                        "AUC-PR is undefined (no class has positive samples in the epoch); reporting 0.5 as a neutral fallback.",
                                    ),
                                    lvl,
                                    &(
                                        "burn_train::metric::auc_pr",
                                        "burn_train::metric::auc_pr",
                                        ::log::__private_api::loc(),
                                    ),
                                    (),
                                );
                            }
                        }
                    };
                    0.5
                } else {
                    ap.select(0, keep).mean().into_scalar()
                };
                self.state
                    .update(
                        100.0 * metric,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    aggregation: NumericAggregation::Last,
                }
                    .into()
            }
        }
        impl Numeric for AucPrMetric {
            fn value(&self) -> super::NumericEntry {
                self.state.value()
            }
            fn running_value(&self) -> super::NumericEntry {
                self.state.value()
            }
        }
    }
    mod auroc {
        use core::f64;
        use super::MetricMetadata;
        use super::state::{FormatOptions, NumericMetricState};
        use crate::metric::{
            ClassReduction, ConfusionStatsInput, Metric, MetricName, Numeric,
            SerializedEntry,
        };
        use burn_core::tensor::{Bool, Tensor};
        use std::sync::Arc;
        /// The Area Under the Receiver Operating Characteristic Curve (AUROC, also
        /// referred to as [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)).
        ///
        /// Supports binary, multiclass and multi-label classification through a
        /// One-vs-Rest decomposition, aggregated with the configured
        /// [class reduction](ClassReduction).
        pub struct AurocMetric {
            name: MetricName,
            state: NumericMetricState,
            class_reduction: ClassReduction,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for AurocMetric {
            #[inline]
            fn clone(&self) -> AurocMetric {
                AurocMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    class_reduction: ::core::clone::Clone::clone(&self.class_reduction),
                }
            }
        }
        impl Default for AurocMetric {
            fn default() -> Self {
                Self::new(Default::default())
            }
        }
        impl AurocMetric {
            fn new(class_reduction: ClassReduction) -> Self {
                let state = Default::default();
                let name = Arc::new(
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(
                            format_args!("AUROC [{0:?}]", class_reduction),
                        )
                    }),
                );
                Self {
                    state,
                    class_reduction,
                    name,
                }
            }
            /// AUROC metric for binary classification.
            #[allow(dead_code)]
            pub fn binary() -> Self {
                Self::new(ClassReduction::default())
            }
            /// AUROC metric for multiclass classification.
            ///
            /// # Arguments
            ///
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multiclass(class_reduction: ClassReduction) -> Self {
                Self::new(class_reduction)
            }
            /// AUROC metric for multi-label classification.
            ///
            /// # Arguments
            ///
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multilabel(class_reduction: ClassReduction) -> Self {
                Self::new(class_reduction)
            }
            fn pairwise_auc(scores: Tensor<2>, targets: Tensor<2>) -> Tensor<1> {
                let [n, c] = scores.dims();
                let si = scores.clone().reshape([n, 1, c]);
                let sj = scores.reshape([1, n, c]);
                let yi = targets.clone().reshape([n, 1, c]);
                let yj = targets.reshape([1, n, c]);
                let valid: Tensor<3> = yi * (1.0 - yj);
                let reduce = |t: Tensor<3>| {
                    t.sum_dim(0).sum_dim(1).squeeze_dims::<1>(&[0, 1])
                };
                let num_pairs = reduce(valid.clone());
                let correct_pairs = reduce(
                    si.clone().greater(sj.clone()).float() * valid.clone(),
                );
                let tied_pairs = reduce(si.equal(sj).float() * valid);
                (correct_pairs + 0.5 * tied_pairs) / num_pairs
            }
            fn compute_auc(
                &self,
                predictions: &Tensor<2>,
                targets: &Tensor<2, Bool>,
            ) -> f64 {
                let [n, c] = predictions.dims();
                let (scores, targets) = match self.class_reduction {
                    ClassReduction::Macro => {
                        (predictions.clone(), targets.clone().float())
                    }
                    ClassReduction::Micro => {
                        (
                            predictions.clone().reshape([n * c, 1]),
                            targets.clone().float().reshape([n * c, 1]),
                        )
                    }
                };
                let auc = Self::pairwise_auc(scores, targets);
                let keep = auc
                    .clone()
                    .is_nan()
                    .bool_not()
                    .argwhere()
                    .squeeze_dim::<1>(1);
                if keep.dims()[0] == 0 {
                    {
                        {
                            let lvl = ::log::Level::Warn;
                            if lvl <= ::log::STATIC_MAX_LEVEL
                                && lvl <= ::log::max_level()
                            {
                                ::log::__private_api::log(
                                    { ::log::__private_api::GlobalLogger },
                                    format_args!(
                                        "AUROC is undefined (no class has both positive and negative samples in the batch); reporting 0.5 (chance level).",
                                    ),
                                    lvl,
                                    &(
                                        "burn_train::metric::auroc",
                                        "burn_train::metric::auroc",
                                        ::log::__private_api::loc(),
                                    ),
                                    (),
                                );
                            }
                        }
                    };
                    return 0.5;
                }
                auc.select(0, keep).mean().into_scalar()
            }
        }
        impl Metric for AurocMetric {
            type Input = ConfusionStatsInput;
            fn update(
                &mut self,
                input: &ConfusionStatsInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let [sample_size, _] = input.predictions.dims();
                let metric = self.compute_auc(&input.predictions, &input.targets);
                self.state
                    .update(
                        100.0 * metric,
                        sample_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
        }
        impl Numeric for AurocMetric {
            fn value(&self) -> super::NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> super::NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod base {
        use std::sync::Arc;
        use burn_core::data::dataloader::Progress;
        use burn_optim::LearningRate;
        /// Metric metadata that can be used when computing metrics.
        pub struct MetricMetadata {
            /// The current progress.
            pub progress: Progress,
            /// The current iteration.
            pub iteration: Option<usize>,
            /// The current learning rate.
            pub lr: Option<LearningRate>,
        }
        impl MetricMetadata {}
        /// Metric id that can be used to compare metrics and retrieve entries of the same metric.
        /// For now we take the name as id to make sure that the same metric has the same id across different runs.
        pub struct MetricId {
            /// The metric id.
            id: Arc<String>,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for MetricId {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field1_finish(
                    f,
                    "MetricId",
                    "id",
                    &&self.id,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for MetricId {
            #[inline]
            fn clone(&self) -> MetricId {
                MetricId {
                    id: ::core::clone::Clone::clone(&self.id),
                }
            }
        }
        impl MetricId {
            ///Constructs a new `MetricId`.
            pub fn new(id: Arc<String>) -> Self {
                MetricId { id: id }
            }
        }
        #[automatically_derived]
        impl ::core::marker::StructuralPartialEq for MetricId {}
        #[automatically_derived]
        impl ::core::cmp::PartialEq for MetricId {
            #[inline]
            fn eq(&self, other: &MetricId) -> bool {
                self.id == other.id
            }
        }
        #[automatically_derived]
        impl ::core::cmp::Eq for MetricId {
            #[doc(hidden)]
            #[coverage(off)]
            fn assert_fields_are_eq(&self) {
                let _: ::core::cmp::AssertParamIsEq<Arc<String>>;
            }
        }
        #[automatically_derived]
        impl ::core::hash::Hash for MetricId {
            #[inline]
            fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) {
                ::core::hash::Hash::hash(&self.id, state)
            }
        }
        /// Metric attributes define the properties intrinsic to different types of metric.
        pub enum MetricAttributes {
            /// Numeric attributes.
            Numeric(NumericAttributes),
            /// No attributes.
            None,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for MetricAttributes {
            #[inline]
            fn clone(&self) -> MetricAttributes {
                match self {
                    MetricAttributes::Numeric(__self_0) => {
                        MetricAttributes::Numeric(::core::clone::Clone::clone(__self_0))
                    }
                    MetricAttributes::None => MetricAttributes::None,
                }
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for MetricAttributes {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    MetricAttributes::Numeric(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Numeric",
                            &__self_0,
                        )
                    }
                    MetricAttributes::None => {
                        ::core::fmt::Formatter::write_str(f, "None")
                    }
                }
            }
        }
        /// Definition of a metric.
        pub struct MetricDefinition {
            /// The metric's id.
            pub metric_id: MetricId,
            /// The name of the metric.
            pub name: String,
            /// The description of the metric.
            pub description: Option<String>,
            /// The attributes of the metric.
            pub attributes: MetricAttributes,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for MetricDefinition {
            #[inline]
            fn clone(&self) -> MetricDefinition {
                MetricDefinition {
                    metric_id: ::core::clone::Clone::clone(&self.metric_id),
                    name: ::core::clone::Clone::clone(&self.name),
                    description: ::core::clone::Clone::clone(&self.description),
                    attributes: ::core::clone::Clone::clone(&self.attributes),
                }
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for MetricDefinition {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field4_finish(
                    f,
                    "MetricDefinition",
                    "metric_id",
                    &self.metric_id,
                    "name",
                    &self.name,
                    "description",
                    &self.description,
                    "attributes",
                    &&self.attributes,
                )
            }
        }
        impl MetricDefinition {
            /// Create a new metric definition given the metric and a unique id.
            pub fn new<Me: Metric>(metric_id: MetricId, metric: &Me) -> Self {
                Self {
                    metric_id,
                    name: metric.name().to_string(),
                    description: metric.description(),
                    attributes: metric.attributes(),
                }
            }
        }
        /// Metric trait.
        ///
        /// # Notes
        ///
        /// Implementations should define their own input type only used by the metric.
        /// This is important since some conflict may happen when the model output is adapted for each
        /// metric's input type.
        pub trait Metric: Send + Sync + Clone {
            /// The input type of the metric.
            type Input;
            /// The parameterized name of the metric.
            ///
            /// This should be unique, so avoid using short generic names, prefer using the long name.
            ///
            /// For a metric that can exist at different parameters (e.g., top-k accuracy for different
            /// values of k), the name should be unique for each instance.
            fn name(&self) -> MetricName;
            /// A short description of the metric.
            fn description(&self) -> Option<String> {
                None
            }
            /// Attributes of the metric.
            ///
            /// By default, metrics have no attributes.
            fn attributes(&self) -> MetricAttributes {
                MetricAttributes::None
            }
            /// Update the metric state and returns the current metric entry.
            fn update(
                &mut self,
                item: &Self::Input,
                metadata: &MetricMetadata,
            ) -> SerializedEntry;
            /// Clear the metric state.
            fn clear(&mut self);
        }
        /// Type used to store metric names efficiently.
        pub type MetricName = Arc<String>;
        /// Adaptor are used to transform types so that they can be used by metrics.
        ///
        /// This should be implemented by a model's output type for all [metric inputs](Metric::Input) that are
        /// registered with the specific learning paradigm (i.e. [SupervisedTraining](crate::SupervisedTraining)).
        pub trait Adaptor<T> {
            /// Adapt the type to be passed to a [metric](Metric).
            fn adapt(&self) -> T;
        }
        impl<T> Adaptor<()> for T {
            fn adapt(&self) {}
        }
        /// How a numeric metric's per-batch values are reduced into an epoch value.
        pub enum NumericAggregation {
            /// Sample-weighted mean of the per-batch values.
            #[default]
            Mean,
            /// Last logged value, already computed over the whole epoch.
            Last,
        }
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for NumericAggregation {}
        #[automatically_derived]
        impl ::core::clone::Clone for NumericAggregation {
            #[inline]
            fn clone(&self) -> NumericAggregation {
                *self
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for NumericAggregation {}
        #[automatically_derived]
        impl ::core::fmt::Debug for NumericAggregation {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::write_str(
                    f,
                    match self {
                        NumericAggregation::Mean => "Mean",
                        NumericAggregation::Last => "Last",
                    },
                )
            }
        }
        #[automatically_derived]
        impl ::core::default::Default for NumericAggregation {
            #[inline]
            fn default() -> NumericAggregation {
                Self::Mean
            }
        }
        #[automatically_derived]
        impl ::core::marker::StructuralPartialEq for NumericAggregation {}
        #[automatically_derived]
        impl ::core::cmp::PartialEq for NumericAggregation {
            #[inline]
            fn eq(&self, other: &NumericAggregation) -> bool {
                let __self_discr = ::core::intrinsics::discriminant_value(self);
                let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                __self_discr == __arg1_discr
            }
        }
        #[automatically_derived]
        impl ::core::cmp::Eq for NumericAggregation {
            #[doc(hidden)]
            #[coverage(off)]
            fn assert_fields_are_eq(&self) {}
        }
        /// Attributes that describe intrinsic properties of a numeric metric.
        pub struct NumericAttributes {
            /// Optional unit (e.g. "%", "ms", "pixels")
            pub unit: Option<String>,
            /// Whether larger values are better (true) or smaller are better (false).
            pub higher_is_better: bool,
            /// How per-batch values are reduced into the epoch value.
            pub aggregation: NumericAggregation,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for NumericAttributes {
            #[inline]
            fn clone(&self) -> NumericAttributes {
                NumericAttributes {
                    unit: ::core::clone::Clone::clone(&self.unit),
                    higher_is_better: ::core::clone::Clone::clone(
                        &self.higher_is_better,
                    ),
                    aggregation: ::core::clone::Clone::clone(&self.aggregation),
                }
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for NumericAttributes {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field3_finish(
                    f,
                    "NumericAttributes",
                    "unit",
                    &self.unit,
                    "higher_is_better",
                    &self.higher_is_better,
                    "aggregation",
                    &&self.aggregation,
                )
            }
        }
        impl From<NumericAttributes> for MetricAttributes {
            fn from(attr: NumericAttributes) -> Self {
                MetricAttributes::Numeric(attr)
            }
        }
        impl Default for NumericAttributes {
            fn default() -> Self {
                Self {
                    unit: None,
                    higher_is_better: true,
                    aggregation: NumericAggregation::default(),
                }
            }
        }
        /// Declare a metric to be numeric.
        ///
        /// This is useful to plot the values of a metric during training.
        pub trait Numeric {
            /// Returns the numeric value of the metric.
            fn value(&self) -> NumericEntry;
            /// Returns the current aggregated value of the metric over the global step (epoch).
            fn running_value(&self) -> NumericEntry;
        }
        /// Serialized form of a metric entry.
        pub struct SerializedEntry {
            /// The string to be displayed.
            pub formatted: String,
            /// The string to be saved.
            pub serialized: String,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for SerializedEntry {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "SerializedEntry",
                    "formatted",
                    &self.formatted,
                    "serialized",
                    &&self.serialized,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for SerializedEntry {
            #[inline]
            fn clone(&self) -> SerializedEntry {
                SerializedEntry {
                    formatted: ::core::clone::Clone::clone(&self.formatted),
                    serialized: ::core::clone::Clone::clone(&self.serialized),
                }
            }
        }
        impl SerializedEntry {
            ///Constructs a new `SerializedEntry`.
            pub fn new(formatted: String, serialized: String) -> Self {
                SerializedEntry {
                    formatted: formatted,
                    serialized: serialized,
                }
            }
        }
        /// Data type that contains the current state of a metric at a given time.
        pub struct MetricEntry {
            /// Id of the entry's metric.
            pub metric_id: MetricId,
            /// The serialized form of the entry.
            pub serialized_entry: SerializedEntry,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for MetricEntry {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "MetricEntry",
                    "metric_id",
                    &self.metric_id,
                    "serialized_entry",
                    &&self.serialized_entry,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for MetricEntry {
            #[inline]
            fn clone(&self) -> MetricEntry {
                MetricEntry {
                    metric_id: ::core::clone::Clone::clone(&self.metric_id),
                    serialized_entry: ::core::clone::Clone::clone(&self.serialized_entry),
                }
            }
        }
        impl MetricEntry {
            /// Create a new metric.
            pub fn new(metric_id: MetricId, serialized_entry: SerializedEntry) -> Self {
                Self {
                    metric_id,
                    serialized_entry,
                }
            }
        }
        /// Numeric metric entry.
        pub enum NumericEntry {
            /// Single numeric value.
            Value(f64),
            /// Aggregated numeric (value, number of elements).
            Aggregated {
                /// The aggregated value of all entries.
                aggregated_value: f64,
                /// The number of entries present in the aggregated value.
                count: usize,
            },
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for NumericEntry {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    NumericEntry::Value(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Value",
                            &__self_0,
                        )
                    }
                    NumericEntry::Aggregated {
                        aggregated_value: __self_0,
                        count: __self_1,
                    } => {
                        ::core::fmt::Formatter::debug_struct_field2_finish(
                            f,
                            "Aggregated",
                            "aggregated_value",
                            __self_0,
                            "count",
                            &__self_1,
                        )
                    }
                }
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for NumericEntry {
            #[inline]
            fn clone(&self) -> NumericEntry {
                match self {
                    NumericEntry::Value(__self_0) => {
                        NumericEntry::Value(::core::clone::Clone::clone(__self_0))
                    }
                    NumericEntry::Aggregated {
                        aggregated_value: __self_0,
                        count: __self_1,
                    } => {
                        NumericEntry::Aggregated {
                            aggregated_value: ::core::clone::Clone::clone(__self_0),
                            count: ::core::clone::Clone::clone(__self_1),
                        }
                    }
                }
            }
        }
        impl NumericEntry {
            /// Gets the current aggregated value of the metric.
            pub fn current(&self) -> f64 {
                match self {
                    NumericEntry::Value(val) => *val,
                    NumericEntry::Aggregated { aggregated_value, .. } => {
                        *aggregated_value
                    }
                }
            }
            /// Returns a String representing the NumericEntry
            pub fn serialize(&self) -> String {
                match self {
                    Self::Value(v) => v.to_string(),
                    Self::Aggregated { aggregated_value, count } => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0},{1}", aggregated_value, count),
                            )
                        })
                    }
                }
            }
            /// De-serializes a string representing a NumericEntry and returns a Result containing the corresponding NumericEntry.
            pub fn deserialize(entry: &str) -> Result<Self, String> {
                let values = entry.split(',').collect::<Vec<_>>();
                let num_values = values.len();
                if num_values == 1 {
                    match values[0].parse::<f64>() {
                        Ok(value) => Ok(NumericEntry::Value(value)),
                        Err(err) => Err(err.to_string()),
                    }
                } else if num_values == 2 {
                    let (value, numel) = (values[0], values[1]);
                    match value.parse::<f64>() {
                        Ok(value) => {
                            match numel.parse::<usize>() {
                                Ok(numel) => {
                                    Ok(NumericEntry::Aggregated {
                                        aggregated_value: value,
                                        count: numel,
                                    })
                                }
                                Err(err) => Err(err.to_string()),
                            }
                        }
                        Err(err) => Err(err.to_string()),
                    }
                } else {
                    Err("Invalid number of values for numeric entry".to_string())
                }
            }
            /// Compare this numeric metric's value with another one using the specified direction.
            pub fn better_than(
                &self,
                other: &NumericEntry,
                higher_is_better: bool,
            ) -> bool {
                (self.current() > other.current()) == higher_is_better
            }
        }
        /// Format a float with the given precision. Will use scientific notation if necessary.
        pub fn format_float(float: f64, precision: usize) -> String {
            let scientific_notation_threshold = 0.1_f64.powf(precision as f64 - 1.0);
            match scientific_notation_threshold >= float {
                true => {
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0:.1$e}", float, precision))
                    })
                }
                false => {
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0:.1$}", float, precision))
                    })
                }
            }
        }
    }
    mod bleu {
        use super::state::{FormatOptions, NumericMetricState};
        use super::{MetricMetadata, SerializedEntry};
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericAttributes,
            NumericEntry,
        };
        use burn_core::tensor::{Int, Tensor};
        use std::collections::HashMap;
        use std::sync::Arc;
        /// Smoothing method for BLEU score computation.
        ///
        /// Sentence-level BLEU often produces zero scores when higher-order n-grams
        /// have no matches. Smoothing techniques address this by assigning small
        /// non-zero values to zero-count precisions.
        ///
        /// # References
        ///
        /// Chen & Cherry, "A Systematic Comparison of Smoothing Techniques for
        /// Sentence-Level BLEU", WMT 2014.
        pub enum BleuSmoothing {
            /// No smoothing. Zero precision at any n-gram order produces a zero
            /// overall score (standard corpus-level BLEU behaviour).
            #[default]
            None,
            /// Add a small constant (`epsilon`, default 0.1) to zero-count
            /// precisions. Corresponds to method 1 in Chen & Cherry (2014).
            AddEpsilon(f64),
            /// Exponential decay: for each n-gram order with zero matches, double a
            /// running multiplier `k` (starting at 1 and doubling on every zero) and
            /// replace the precision with `1 / (k * total_n)`. Corresponds to
            /// method 3 in Chen & Cherry (2014) and the default smoothing in
            /// SacreBLEU for sentence-level BLEU.
            Exponential,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for BleuSmoothing {
            #[inline]
            fn clone(&self) -> BleuSmoothing {
                match self {
                    BleuSmoothing::None => BleuSmoothing::None,
                    BleuSmoothing::AddEpsilon(__self_0) => {
                        BleuSmoothing::AddEpsilon(::core::clone::Clone::clone(__self_0))
                    }
                    BleuSmoothing::Exponential => BleuSmoothing::Exponential,
                }
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for BleuSmoothing {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    BleuSmoothing::None => ::core::fmt::Formatter::write_str(f, "None"),
                    BleuSmoothing::AddEpsilon(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "AddEpsilon",
                            &__self_0,
                        )
                    }
                    BleuSmoothing::Exponential => {
                        ::core::fmt::Formatter::write_str(f, "Exponential")
                    }
                }
            }
        }
        #[automatically_derived]
        impl ::core::default::Default for BleuSmoothing {
            #[inline]
            fn default() -> BleuSmoothing {
                Self::None
            }
        }
        /// Computes the BLEU (Bilingual Evaluation Understudy) score between predicted
        /// and reference token sequences.
        ///
        /// BLEU measures the quality of machine-translated text by comparing n-gram
        /// overlap between the candidate (prediction) and reference sequences. The
        /// score combines modified n-gram precision for n = 1..max_n with a brevity
        /// penalty that discourages overly short translations.
        ///
        /// The metric operates on integer token IDs (not raw text), matching the
        /// convention used by [`CharErrorRate`](super::CharErrorRate) and
        /// [`WordErrorRate`](super::WordErrorRate).
        ///
        /// # Batch-level scoring
        ///
        /// Within each batch the metric accumulates n-gram counts across all
        /// sentences and computes a single corpus-style BLEU score, following the
        /// same pattern CER/WER use for edit-distance aggregation.
        ///
        /// Epoch-level (running) aggregation averages these batch scores, which is
        /// slightly inaccurate compared to true corpus BLEU. Correct corpus-level
        /// accumulation would require a custom metric state; a TODO is left for
        /// future work.
        ///
        /// # References
        ///
        /// Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine
        /// Translation", ACL 2002.
        ///
        /// Chen & Cherry, "A Systematic Comparison of Smoothing Techniques for
        /// Sentence-Level BLEU", WMT 2014.
        pub struct BleuScore {
            name: MetricName,
            state: NumericMetricState,
            max_n: usize,
            pad_token: Option<usize>,
            smoothing: BleuSmoothing,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for BleuScore {
            #[inline]
            fn clone(&self) -> BleuScore {
                BleuScore {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    max_n: ::core::clone::Clone::clone(&self.max_n),
                    pad_token: ::core::clone::Clone::clone(&self.pad_token),
                    smoothing: ::core::clone::Clone::clone(&self.smoothing),
                }
            }
        }
        /// The [BLEU score metric](BleuScore) input type.
        pub struct BleuInput {
            /// The predicted token sequences (2-D tensor of token indices).
            pub outputs: Tensor<2, Int>,
            /// The reference token sequences (2-D tensor of token indices).
            pub targets: Tensor<2, Int>,
        }
        impl BleuInput {
            ///Constructs a new `BleuInput`.
            pub fn new(outputs: Tensor<2, Int>, targets: Tensor<2, Int>) -> Self {
                BleuInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl Default for BleuScore {
            fn default() -> Self {
                Self::with_max_n(4)
            }
        }
        impl BleuScore {
            /// Creates a BLEU metric with the given maximum n-gram order.
            ///
            /// Common values: 1 (BLEU-1), 2 (BLEU-2), 4 (BLEU-4).
            ///
            /// # Panics
            ///
            /// Panics if `max_n` is zero.
            pub fn with_max_n(max_n: usize) -> Self {
                if !(max_n >= 1) {
                    {
                        ::core::panicking::panic_fmt(
                            format_args!("max_n must be at least 1"),
                        );
                    }
                }
                Self {
                    name: Arc::new(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("BLEU-{0}", max_n))
                        }),
                    ),
                    state: NumericMetricState::default(),
                    max_n,
                    pad_token: None,
                    smoothing: BleuSmoothing::default(),
                }
            }
            /// Creates a BLEU-4 metric (the most common configuration).
            pub fn new() -> Self {
                Self::default()
            }
            /// Sets the pad token index. Tokens matching this value are stripped from
            /// the right of each sequence before scoring.
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
            /// Sets the smoothing method for handling zero-count n-gram precisions.
            ///
            /// Smoothing is recommended when evaluating short sentences where
            /// higher-order n-gram matches are sparse.
            pub fn with_smoothing(mut self, smoothing: BleuSmoothing) -> Self {
                self.smoothing = smoothing;
                self
            }
        }
        /// Extracts n-grams of order `n` from a slice and returns their counts.
        fn ngram_counts(tokens: &[i32], n: usize) -> HashMap<Vec<i32>, usize> {
            let mut counts = HashMap::new();
            if tokens.len() >= n {
                for window in tokens.windows(n) {
                    *counts.entry(window.to_vec()).or_insert(0) += 1;
                }
            }
            counts
        }
        /// Computes corpus-style BLEU score from accumulated n-gram statistics.
        ///
        /// `clipped_counts[n]` and `total_counts[n]` hold the clipped and total
        /// n-gram counts for order `n+1` across all sentences.
        /// `candidate_len` and `reference_len` are the total token counts.
        ///
        /// Returns a value in [0, 100].
        fn corpus_bleu(
            clipped_counts: &[usize],
            total_counts: &[usize],
            candidate_len: usize,
            reference_len: usize,
            max_n: usize,
            smoothing: &BleuSmoothing,
        ) -> f64 {
            if candidate_len == 0 {
                return 0.0;
            }
            let bp = if candidate_len < reference_len {
                (1.0 - reference_len as f64 / candidate_len as f64).exp()
            } else {
                1.0
            };
            let mut log_avg = 0.0;
            let mut counted_orders = 0;
            let mut smooth_mult = 1.0_f64;
            for n in 0..max_n {
                let total = total_counts[n];
                let clipped = clipped_counts[n];
                if total == 0 {
                    return 0.0;
                }
                let precision = if clipped == 0 {
                    match smoothing {
                        BleuSmoothing::None => return 0.0,
                        BleuSmoothing::AddEpsilon(eps) => *eps / total as f64,
                        BleuSmoothing::Exponential => {
                            smooth_mult *= 2.0;
                            1.0 / (smooth_mult * total as f64)
                        }
                    }
                } else {
                    clipped as f64 / total as f64
                };
                log_avg += precision.ln();
                counted_orders += 1;
            }
            if counted_orders == 0 {
                return 0.0;
            }
            let score = bp * (log_avg / counted_orders as f64).exp();
            score * 100.0
        }
        impl Metric for BleuScore {
            type Input = BleuInput;
            fn update(
                &mut self,
                input: &BleuInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let outputs = &input.outputs;
                let targets = &input.targets;
                let [batch_size, seq_len] = targets.dims();
                let outputs_data = outputs.to_data().iter::<i32>().collect::<Vec<_>>();
                let targets_data = targets.to_data().iter::<i32>().collect::<Vec<_>>();
                let pad_token = self.pad_token.map(|p| p as i32);
                let mut clipped_counts = ::alloc::vec::from_elem(0usize, self.max_n);
                let mut total_counts = ::alloc::vec::from_elem(0usize, self.max_n);
                let mut total_candidate_len = 0usize;
                let mut total_reference_len = 0usize;
                for i in 0..batch_size {
                    let start = i * seq_len;
                    let end = (i + 1) * seq_len;
                    let output_seq = &outputs_data[start..end];
                    let target_seq = &targets_data[start..end];
                    let output_seq = match pad_token {
                        Some(pad) => {
                            let len = output_seq
                                .iter()
                                .position(|&x| x == pad)
                                .unwrap_or(output_seq.len());
                            &output_seq[..len]
                        }
                        None => output_seq,
                    };
                    let target_seq = match pad_token {
                        Some(pad) => {
                            let len = target_seq
                                .iter()
                                .position(|&x| x == pad)
                                .unwrap_or(target_seq.len());
                            &target_seq[..len]
                        }
                        None => target_seq,
                    };
                    total_candidate_len += output_seq.len();
                    total_reference_len += target_seq.len();
                    for n in 1..=self.max_n {
                        let cand_ngrams = ngram_counts(output_seq, n);
                        let ref_ngrams = ngram_counts(target_seq, n);
                        for (ngram, &count) in &cand_ngrams {
                            let ref_count = ref_ngrams.get(ngram).copied().unwrap_or(0);
                            clipped_counts[n - 1] += count.min(ref_count);
                            total_counts[n - 1] += count;
                        }
                    }
                }
                let value = corpus_bleu(
                    &clipped_counts,
                    &total_counts,
                    total_candidate_len,
                    total_reference_len,
                    self.max_n,
                    &self.smoothing,
                );
                self.state
                    .update(
                        value,
                        batch_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset();
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for BleuScore {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod cer {
        use super::state::{FormatOptions, NumericMetricState};
        use super::{MetricMetadata, SerializedEntry};
        use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, NumericEntry};
        use burn_core::tensor::{Int, Tensor};
        use std::sync::Arc;
        /// Computes the edit distance (Levenshtein distance) between two sequences of integers.
        ///
        /// The edit distance is defined as the minimum number of single-element edits (insertions,
        /// deletions, or substitutions) required to change one sequence into the other. This
        /// implementation is optimized for space, using only two rows of the dynamic programming table.
        ///
        pub(crate) fn edit_distance(reference: &[i32], prediction: &[i32]) -> usize {
            let mut prev = (0..=prediction.len()).collect::<Vec<_>>();
            let mut curr = ::alloc::vec::from_elem(0, prediction.len() + 1);
            for (i, &r) in reference.iter().enumerate() {
                curr[0] = i + 1;
                for (j, &p) in prediction.iter().enumerate() {
                    curr[j + 1] = if r == p {
                        prev[j]
                    } else {
                        1 + prev[j].min(prev[j + 1]).min(curr[j])
                    };
                }
                core::mem::swap(&mut prev, &mut curr);
            }
            prev[prediction.len()]
        }
        /// Character error rate (CER) is defined as the edit distance (e.g. Levenshtein distance) between the predicted
        /// and reference character sequences, divided by the total number of characters in the reference.
        /// This metric is commonly used in tasks such as speech recognition, OCR, or text generation
        /// to quantify how closely the predicted output matches the ground truth at a character level.
        ///
        pub struct CharErrorRate {
            name: MetricName,
            state: NumericMetricState,
            pad_token: Option<usize>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for CharErrorRate {
            #[inline]
            fn clone(&self) -> CharErrorRate {
                CharErrorRate {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    pad_token: ::core::clone::Clone::clone(&self.pad_token),
                }
            }
        }
        /// The [character error rate metric](CharErrorRate) input type.
        pub struct CerInput {
            /// The predicted token sequences (as a 2-D tensor of token indices).
            pub outputs: Tensor<2, Int>,
            /// The target token sequences (as a 2-D tensor of token indices).
            pub targets: Tensor<2, Int>,
        }
        impl CerInput {
            ///Constructs a new `CerInput`.
            pub fn new(outputs: Tensor<2, Int>, targets: Tensor<2, Int>) -> Self {
                CerInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl Default for CharErrorRate {
            fn default() -> Self {
                Self::new()
            }
        }
        impl CharErrorRate {
            /// Creates the metric.
            pub fn new() -> Self {
                Self {
                    name: Arc::new("CER".to_string()),
                    state: NumericMetricState::default(),
                    pad_token: None,
                }
            }
            /// Sets the pad token.
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
        }
        /// The [character error rate metric](CharErrorRate) implementation.
        impl Metric for CharErrorRate {
            type Input = CerInput;
            fn update(
                &mut self,
                input: &CerInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let outputs = &input.outputs;
                let targets = &input.targets;
                let [batch_size, seq_len] = targets.dims();
                let (output_lengths, target_lengths) = if let Some(pad) = self.pad_token
                {
                    let output_mask = outputs.clone().not_equal_elem(pad as i64);
                    let target_mask = targets.clone().not_equal_elem(pad as i64);
                    let output_lengths_tensor = output_mask.int().sum_dim(1);
                    let target_lengths_tensor = target_mask.int().sum_dim(1);
                    (
                        output_lengths_tensor
                            .into_data()
                            .convert::<i32>()
                            .to_vec()
                            .unwrap(),
                        target_lengths_tensor
                            .into_data()
                            .convert::<i32>()
                            .to_vec()
                            .unwrap(),
                    )
                } else {
                    (
                        ::alloc::vec::from_elem(seq_len as i32, batch_size),
                        ::alloc::vec::from_elem(seq_len as i32, batch_size),
                    )
                };
                let outputs_data = outputs.to_data().convert::<i32>().to_vec().unwrap();
                let targets_data = targets.to_data().convert::<i32>().to_vec().unwrap();
                let total_edit_distance: usize = (0..batch_size)
                    .map(|i| {
                        let start = i * seq_len;
                        let output_len = output_lengths[i] as usize;
                        let target_len = target_lengths[i] as usize;
                        let output_seq_slice = &outputs_data[start..(start
                            + output_len)];
                        let target_seq_slice = &targets_data[start..(start
                            + target_len)];
                        edit_distance(target_seq_slice, output_seq_slice)
                    })
                    .sum();
                let total_target_length = target_lengths
                    .iter()
                    .map(|&x| x as f64)
                    .sum::<f64>();
                let value = if total_target_length > 0.0 {
                    100.0 * total_edit_distance as f64 / total_target_length
                } else {
                    0.0
                };
                self.state
                    .update(
                        value,
                        batch_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset();
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                super::NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for CharErrorRate {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod confusion_stats {
        use super::classification::{
            ClassReduction, ClassificationMetricConfig, DecisionRule,
        };
        use burn_core::{
            prelude::{Bool, Int, Tensor},
            tensor::IndexingUpdateOp,
        };
        use std::fmt::{self, Debug};
        /// Input for confusion statistics error types.
        pub struct ConfusionStatsInput {
            /// Sample x Class Non thresholded normalized predictions.
            pub predictions: Tensor<2>,
            /// Sample x Class one-hot encoded target.
            pub targets: Tensor<2, Bool>,
        }
        impl ConfusionStatsInput {
            ///Constructs a new `ConfusionStatsInput`.
            pub fn new(predictions: Tensor<2>, targets: Tensor<2, Bool>) -> Self {
                ConfusionStatsInput {
                    predictions: predictions,
                    targets: targets,
                }
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for ConfusionStatsInput {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "ConfusionStatsInput",
                    "predictions",
                    &self.predictions,
                    "targets",
                    &&self.targets,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for ConfusionStatsInput {
            #[inline]
            fn clone(&self) -> ConfusionStatsInput {
                ConfusionStatsInput {
                    predictions: ::core::clone::Clone::clone(&self.predictions),
                    targets: ::core::clone::Clone::clone(&self.targets),
                }
            }
        }
        impl From<ConfusionStatsInput> for (Tensor<2>, Tensor<2, Bool>) {
            fn from(input: ConfusionStatsInput) -> Self {
                (input.predictions, input.targets)
            }
        }
        impl From<(Tensor<2>, Tensor<2, Bool>)> for ConfusionStatsInput {
            fn from(value: (Tensor<2>, Tensor<2, Bool>)) -> Self {
                Self::new(value.0, value.1)
            }
        }
        pub struct ConfusionStats {
            confusion_classes: Tensor<2, Int>,
            class_reduction: ClassReduction,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for ConfusionStats {
            #[inline]
            fn clone(&self) -> ConfusionStats {
                ConfusionStats {
                    confusion_classes: ::core::clone::Clone::clone(
                        &self.confusion_classes,
                    ),
                    class_reduction: ::core::clone::Clone::clone(&self.class_reduction),
                }
            }
        }
        impl Debug for ConfusionStats {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let to_vec = |tensor_data: Tensor<1>| {
                    tensor_data
                        .to_data()
                        .to_vec::<f32>()
                        .expect(
                            "A vector representation of the input Tensor is expected",
                        )
                };
                let ratio_of_support_vec = |metric: Tensor<1>| to_vec(
                    self.clone().ratio_of_support(metric),
                );
                f.debug_struct("ConfusionStats")
                    .field("tp", &ratio_of_support_vec(self.clone().true_positive()))
                    .field("fp", &ratio_of_support_vec(self.clone().false_positive()))
                    .field("tn", &ratio_of_support_vec(self.clone().true_negative()))
                    .field("fn", &ratio_of_support_vec(self.clone().false_negative()))
                    .field("support", &to_vec(self.clone().support()))
                    .finish()
            }
        }
        impl ConfusionStats {
            /// Expects `predictions` to be normalized.
            pub fn new(
                input: &ConfusionStatsInput,
                config: &ClassificationMetricConfig,
            ) -> Self {
                let prediction_mask = match config.decision_rule {
                    DecisionRule::Threshold(threshold) => {
                        input.predictions.clone().greater_elem(threshold)
                    }
                    DecisionRule::TopK(top_k) => {
                        let mask = input.predictions.zeros_like();
                        let indexes = input
                            .predictions
                            .clone()
                            .argsort_descending(1)
                            .narrow(1, 0, top_k.get());
                        let values = indexes.ones_like().float();
                        mask.scatter(1, indexes, values, IndexingUpdateOp::Add).bool()
                    }
                };
                Self {
                    confusion_classes: prediction_mask.int()
                        + input.targets.clone().int() * 2,
                    class_reduction: config.class_reduction,
                }
            }
            /// sum over samples
            fn aggregate(
                sample_class_mask: Tensor<2, Bool>,
                class_reduction: ClassReduction,
            ) -> Tensor<1> {
                use ClassReduction::{Macro, Micro};
                match class_reduction {
                    Micro => sample_class_mask.float().sum(),
                    Macro => sample_class_mask.float().sum_dim(0).squeeze_dim(0),
                }
            }
            pub fn true_positive(self) -> Tensor<1> {
                Self::aggregate(
                    self.confusion_classes.equal_elem(3),
                    self.class_reduction,
                )
            }
            pub fn true_negative(self) -> Tensor<1> {
                Self::aggregate(
                    self.confusion_classes.equal_elem(0),
                    self.class_reduction,
                )
            }
            pub fn false_positive(self) -> Tensor<1> {
                Self::aggregate(
                    self.confusion_classes.equal_elem(1),
                    self.class_reduction,
                )
            }
            pub fn false_negative(self) -> Tensor<1> {
                Self::aggregate(
                    self.confusion_classes.equal_elem(2),
                    self.class_reduction,
                )
            }
            pub fn positive(self) -> Tensor<1> {
                self.clone().true_positive() + self.false_negative()
            }
            pub fn negative(self) -> Tensor<1> {
                self.clone().true_negative() + self.false_positive()
            }
            pub fn predicted_positive(self) -> Tensor<1> {
                self.clone().true_positive() + self.false_positive()
            }
            pub fn support(self) -> Tensor<1> {
                self.clone().positive() + self.negative()
            }
            pub fn ratio_of_support(self, metric: Tensor<1>) -> Tensor<1> {
                metric / self.clone().support()
            }
        }
    }
    mod fbetascore {
        use crate::metric::{MetricName, Numeric};
        use super::{
            Metric, MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
            SerializedEntry,
            classification::{ClassReduction, ClassificationMetricConfig, DecisionRule},
            confusion_stats::{ConfusionStats, ConfusionStatsInput},
            state::{FormatOptions, NumericMetricState},
        };
        use burn_core::prelude::Tensor;
        use std::{num::NonZeroUsize, sync::Arc};
        /// The [F-beta score](https://en.wikipedia.org/wiki/F-score) metric.
        ///
        /// The `beta` parameter represents the ratio of recall importance to precision importance.
        /// `beta > 1` gives more weight to recall, while `beta < 1` favors precision.
        pub struct FBetaScoreMetric {
            name: MetricName,
            state: NumericMetricState,
            config: ClassificationMetricConfig,
            beta: f64,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for FBetaScoreMetric {
            #[inline]
            fn clone(&self) -> FBetaScoreMetric {
                FBetaScoreMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    config: ::core::clone::Clone::clone(&self.config),
                    beta: ::core::clone::Clone::clone(&self.beta),
                }
            }
        }
        impl Default for FBetaScoreMetric {
            fn default() -> Self {
                Self::new(Default::default(), Default::default())
            }
        }
        impl FBetaScoreMetric {
            #[allow(dead_code)]
            fn new(config: ClassificationMetricConfig, beta: f64) -> Self {
                let name = Arc::new(
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(
                            format_args!(
                                "FBetaScore ({0}) @ {1:?} [{2:?}]",
                                beta,
                                config.decision_rule,
                                config.class_reduction,
                            ),
                        )
                    }),
                );
                Self {
                    name,
                    config,
                    beta,
                    state: Default::default(),
                }
            }
            /// F-beta score metric for binary classification.
            ///
            /// # Arguments
            ///
            /// * `beta` - Positive real factor to weight recall's importance.
            /// * `threshold` - The threshold to transform a probability into a binary prediction.
            #[allow(dead_code)]
            pub fn binary(beta: f64, threshold: f64) -> Self {
                Self::new(
                    ClassificationMetricConfig {
                        decision_rule: DecisionRule::Threshold(threshold),
                        ..Default::default()
                    },
                    beta,
                )
            }
            /// F-beta score metric for multiclass classification.
            ///
            /// # Arguments
            ///
            /// * `beta` - Positive real factor to weight recall's importance.
            /// * `top_k` - The number of highest predictions considered to find the correct label (typically `1`).
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multiclass(
                beta: f64,
                top_k: usize,
                class_reduction: ClassReduction,
            ) -> Self {
                Self::new(
                    ClassificationMetricConfig {
                        decision_rule: DecisionRule::TopK(
                            NonZeroUsize::new(top_k).expect("top_k must be non-zero"),
                        ),
                        class_reduction,
                    },
                    beta,
                )
            }
            /// F-beta score metric for multi-label classification.
            ///
            /// # Arguments
            ///
            /// * `beta` - Positive real factor to weight recall's importance.
            /// * `threshold` - The threshold to transform a probability into a binary prediction.
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multilabel(
                beta: f64,
                threshold: f64,
                class_reduction: ClassReduction,
            ) -> Self {
                Self::new(
                    ClassificationMetricConfig {
                        decision_rule: DecisionRule::Threshold(threshold),
                        class_reduction,
                    },
                    beta,
                )
            }
            fn class_average(&self, mut aggregated_metric: Tensor<1>) -> f64 {
                use ClassReduction::{Macro, Micro};
                let avg_tensor = match self.config.class_reduction {
                    Micro => aggregated_metric,
                    Macro => {
                        if aggregated_metric.clone().contains_nan().any().into_scalar() {
                            let nan_mask = aggregated_metric.clone().is_nan();
                            aggregated_metric = aggregated_metric
                                .clone()
                                .select(0, nan_mask.bool_not().argwhere().squeeze_dim(1));
                        }
                        aggregated_metric.mean()
                    }
                };
                avg_tensor.into_scalar()
            }
        }
        impl Metric for FBetaScoreMetric {
            type Input = ConfusionStatsInput;
            fn update(
                &mut self,
                input: &Self::Input,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let [sample_size, _] = input.predictions.dims();
                let cf_stats = ConfusionStats::new(input, &self.config);
                let scaled_true_positive = cf_stats.clone().true_positive()
                    * (1.0 + self.beta.powi(2));
                let metric = self
                    .class_average(
                        scaled_true_positive.clone()
                            / (scaled_true_positive
                                + cf_stats.clone().false_negative() * self.beta.powi(2)
                                + cf_stats.false_positive()),
                    );
                self.state
                    .update(
                        100.0 * metric,
                        sample_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for FBetaScoreMetric {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod hamming {
        use std::sync::Arc;
        use super::state::{FormatOptions, NumericMetricState};
        use super::{MetricMetadata, SerializedEntry};
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericAttributes,
            NumericEntry,
        };
        use burn_core::tensor::{Int, Tensor, activation::sigmoid};
        /// The hamming score, sometimes referred to as multi-label or label-based accuracy.
        pub struct HammingScore {
            name: MetricName,
            state: NumericMetricState,
            threshold: f32,
            sigmoid: bool,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for HammingScore {
            #[inline]
            fn clone(&self) -> HammingScore {
                HammingScore {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    threshold: ::core::clone::Clone::clone(&self.threshold),
                    sigmoid: ::core::clone::Clone::clone(&self.sigmoid),
                }
            }
        }
        /// The [hamming score](HammingScore) input type.
        pub struct HammingScoreInput {
            outputs: Tensor<2>,
            targets: Tensor<2, Int>,
        }
        impl HammingScoreInput {
            ///Constructs a new `HammingScoreInput`.
            pub fn new(outputs: Tensor<2>, targets: Tensor<2, Int>) -> Self {
                HammingScoreInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl HammingScore {
            /// Creates the metric.
            pub fn new() -> Self {
                Self::default()
            }
            fn update_name(&mut self) {
                self.name = Arc::new(
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(
                            format_args!(
                                "Hamming Score @ Threshold({0})",
                                self.threshold,
                            ),
                        )
                    }),
                );
            }
            /// Sets the threshold.
            pub fn with_threshold(mut self, threshold: f32) -> Self {
                self.threshold = threshold;
                self.update_name();
                self
            }
            /// Sets the sigmoid activation function usage.
            pub fn with_sigmoid(mut self, sigmoid: bool) -> Self {
                self.sigmoid = sigmoid;
                self.update_name();
                self
            }
        }
        impl Default for HammingScore {
            /// Creates a new metric instance with default values.
            fn default() -> Self {
                let threshold = 0.5;
                let name = Arc::new(
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(
                            format_args!("Hamming Score @ Threshold({0})", threshold),
                        )
                    }),
                );
                Self {
                    name,
                    state: NumericMetricState::default(),
                    threshold,
                    sigmoid: false,
                }
            }
        }
        impl Metric for HammingScore {
            type Input = HammingScoreInput;
            fn update(
                &mut self,
                input: &HammingScoreInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let [batch_size, _n_classes] = input.outputs.dims();
                let targets = input.targets.clone();
                let mut outputs = input.outputs.clone();
                if self.sigmoid {
                    outputs = sigmoid(outputs);
                }
                let score = outputs
                    .greater_elem(self.threshold)
                    .equal(targets.bool())
                    .float()
                    .mean()
                    .into_scalar::<f64>();
                self.state
                    .update(
                        100.0 * score,
                        batch_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for HammingScore {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod iteration {
        use std::sync::Arc;
        use super::MetricMetadata;
        use super::SerializedEntry;
        use super::state::FormatOptions;
        use super::state::NumericMetricState;
        use crate::metric::MetricName;
        use crate::metric::Numeric;
        use crate::metric::{Metric, MetricAttributes, NumericAttributes, NumericEntry};
        /// The loss metric.
        pub struct IterationSpeedMetric {
            name: MetricName,
            state: NumericMetricState,
            instant: Option<std::time::Instant>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for IterationSpeedMetric {
            #[inline]
            fn clone(&self) -> IterationSpeedMetric {
                IterationSpeedMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    instant: ::core::clone::Clone::clone(&self.instant),
                }
            }
        }
        impl Default for IterationSpeedMetric {
            fn default() -> Self {
                Self::new()
            }
        }
        impl IterationSpeedMetric {
            /// Create the metric.
            pub fn new() -> Self {
                Self {
                    name: Arc::new("Iteration Speed".to_string()),
                    state: Default::default(),
                    instant: Default::default(),
                }
            }
        }
        impl Metric for IterationSpeedMetric {
            type Input = ();
            fn update(
                &mut self,
                _: &Self::Input,
                metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let raw = match self.instant {
                    Some(val) => {
                        metadata.iteration.unwrap_or(metadata.progress.items_processed)
                            as f64 / val.elapsed().as_secs_f64()
                    }
                    None => {
                        self.instant = Some(std::time::Instant::now());
                        0.0
                    }
                };
                self.state
                    .update(
                        raw,
                        1,
                        FormatOptions::new(self.name()).unit("iter/sec").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.instant = None;
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("iter/sec".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for IterationSpeedMetric {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod learning_rate {
        use std::sync::Arc;
        use super::{
            MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
            state::{FormatOptions, NumericMetricState},
        };
        use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};
        /// Track the learning rate across iterations.
        pub struct LearningRateMetric {
            name: MetricName,
            state: NumericMetricState,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for LearningRateMetric {
            #[inline]
            fn clone(&self) -> LearningRateMetric {
                LearningRateMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                }
            }
        }
        impl LearningRateMetric {
            /// Creates a new learning rate metric.
            pub fn new() -> Self {
                Self {
                    name: Arc::new("Learning Rate".to_string()),
                    state: NumericMetricState::new(),
                }
            }
        }
        impl Default for LearningRateMetric {
            fn default() -> Self {
                Self::new()
            }
        }
        impl Metric for LearningRateMetric {
            type Input = ();
            fn update(
                &mut self,
                _item: &(),
                metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let lr = metadata.lr.unwrap_or(0.0);
                self.state.update(lr, 1, FormatOptions::new(self.name()).precision(2))
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: None,
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for LearningRateMetric {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod loss {
        use std::sync::Arc;
        use super::MetricMetadata;
        use super::SerializedEntry;
        use super::state::FormatOptions;
        use super::state::NumericMetricState;
        use crate::metric::MetricName;
        use crate::metric::{
            Metric, MetricAttributes, Numeric, NumericAttributes, NumericEntry,
        };
        use burn_core::tensor::Tensor;
        /// The loss metric.
        pub struct LossMetric {
            name: Arc<String>,
            state: NumericMetricState,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for LossMetric {
            #[inline]
            fn clone(&self) -> LossMetric {
                LossMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                }
            }
        }
        /// The [loss metric](LossMetric) input type.
        pub struct LossInput {
            tensor: Tensor<1>,
        }
        impl LossInput {
            ///Constructs a new `LossInput`.
            pub fn new(tensor: Tensor<1>) -> Self {
                LossInput { tensor: tensor }
            }
        }
        impl Default for LossMetric {
            fn default() -> Self {
                Self::new()
            }
        }
        impl LossMetric {
            /// Create the metric.
            pub fn new() -> Self {
                Self {
                    name: Arc::new("Loss".to_string()),
                    state: NumericMetricState::default(),
                }
            }
        }
        impl Metric for LossMetric {
            type Input = LossInput;
            fn update(
                &mut self,
                loss: &Self::Input,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let [batch_size] = loss.tensor.dims();
                let loss = loss
                    .tensor
                    .clone()
                    .mean()
                    .into_data()
                    .iter::<f64>()
                    .next()
                    .unwrap();
                self.state
                    .update(
                        loss,
                        batch_size,
                        FormatOptions::new(self.name()).precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: None,
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for LossMetric {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod perplexity {
        use super::state::FormatOptions;
        use super::{MetricMetadata, NumericEntry, SerializedEntry, format_float};
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericAttributes,
        };
        use burn_core::tensor::{Int, Tensor};
        /// Custom state for perplexity metric that correctly accumulates negative log-likelihood.
        ///
        /// Unlike other metrics that can be averaged, perplexity requires special handling:
        /// - Accumulate total negative log-likelihood across all tokens
        /// - Accumulate total number of effective tokens
        /// - Compute perplexity as exp(total_nll / total_tokens) only at the end
        struct PerplexityState {
            /// Sum of negative log-likelihood across all tokens
            sum_nll: f64,
            /// Total number of effective tokens (excluding padding)
            total_tokens: usize,
            /// Current batch perplexity (for display purposes)
            current: f64,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for PerplexityState {
            #[inline]
            fn clone(&self) -> PerplexityState {
                PerplexityState {
                    sum_nll: ::core::clone::Clone::clone(&self.sum_nll),
                    total_tokens: ::core::clone::Clone::clone(&self.total_tokens),
                    current: ::core::clone::Clone::clone(&self.current),
                }
            }
        }
        impl PerplexityState {
            fn new() -> Self {
                Self {
                    sum_nll: 0.0,
                    total_tokens: 0,
                    current: f64::NAN,
                }
            }
            fn reset(&mut self) {
                self.sum_nll = 0.0;
                self.total_tokens = 0;
                self.current = f64::NAN;
            }
            /// Update state with negative log-likelihood and token count from current batch
            fn update(
                &mut self,
                sum_log_prob: f64,
                effective_tokens: usize,
                format: FormatOptions,
            ) -> SerializedEntry {
                let batch_nll = -sum_log_prob;
                self.sum_nll += batch_nll;
                self.total_tokens += effective_tokens;
                let batch_perplexity = if effective_tokens > 0 {
                    (batch_nll / effective_tokens as f64).exp()
                } else {
                    f64::INFINITY
                };
                self.current = batch_perplexity;
                let epoch_perplexity = if self.total_tokens > 0 {
                    (self.sum_nll / self.total_tokens as f64).exp()
                } else {
                    f64::INFINITY
                };
                let (formatted_current, formatted_running) = match format
                    .precision_value()
                {
                    Some(precision) => {
                        (
                            format_float(batch_perplexity, precision),
                            format_float(epoch_perplexity, precision),
                        )
                    }
                    None => {
                        (
                            ::alloc::__export::must_use({
                                ::alloc::fmt::format(format_args!("{0}", batch_perplexity))
                            }),
                            ::alloc::__export::must_use({
                                ::alloc::fmt::format(format_args!("{0}", epoch_perplexity))
                            }),
                        )
                    }
                };
                let formatted = match format.unit_value() {
                    Some(unit) => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "epoch {0} {1} - batch {2} {1}",
                                    formatted_running,
                                    unit,
                                    formatted_current,
                                ),
                            )
                        })
                    }
                    None => {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "epoch {0} - batch {1}",
                                    formatted_running,
                                    formatted_current,
                                ),
                            )
                        })
                    }
                };
                let serialized = NumericEntry::Aggregated {
                    aggregated_value: epoch_perplexity,
                    count: self.total_tokens,
                }
                    .serialize();
                SerializedEntry::new(formatted, serialized)
            }
            fn value(&self) -> NumericEntry {
                let perplexity = if self.total_tokens > 0 {
                    (self.sum_nll / self.total_tokens as f64).exp()
                } else {
                    f64::INFINITY
                };
                NumericEntry::Aggregated {
                    aggregated_value: perplexity,
                    count: self.total_tokens,
                }
            }
            fn running_value(&self) -> NumericEntry {
                self.value()
            }
        }
        /// The perplexity metric.
        ///
        /// Perplexity is a measure of how well a probability distribution or probability model
        /// predicts a sample. It's commonly used to evaluate language models. A lower perplexity
        /// indicates that the model is more confident in its predictions.
        ///
        /// Mathematically, perplexity is defined as the exponentiation of the cross-entropy loss:
        /// PPL = exp(H(p, q)) = exp(-1/N * Σ log(p(x_i)))
        ///
        /// where:
        /// - H(p, q) is the cross-entropy between the true distribution p and predicted distribution q
        /// - N is the number of tokens
        /// - p(x_i) is the predicted probability of the i-th token
        ///
        /// # Aggregation
        /// Unlike other metrics, perplexity cannot be simply averaged across batches.
        /// This implementation correctly accumulates the total negative log-likelihood and
        /// total token count across batches, then computes perplexity as exp(total_nll / total_tokens).
        pub struct PerplexityMetric {
            name: MetricName,
            state: PerplexityState,
            pad_token: Option<usize>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for PerplexityMetric {
            #[inline]
            fn clone(&self) -> PerplexityMetric {
                PerplexityMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    pad_token: ::core::clone::Clone::clone(&self.pad_token),
                }
            }
        }
        /// The [perplexity metric](PerplexityMetric) input type.
        pub struct PerplexityInput {
            /// Logits tensor of shape [batch_size * sequence_length, vocab_size]
            outputs: Tensor<2>,
            /// Target tokens tensor of shape [batch_size * sequence_length]
            targets: Tensor<1, Int>,
        }
        impl PerplexityInput {
            ///Constructs a new `PerplexityInput`.
            pub fn new(outputs: Tensor<2>, targets: Tensor<1, Int>) -> Self {
                PerplexityInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl Default for PerplexityMetric {
            fn default() -> Self {
                Self::new()
            }
        }
        impl PerplexityMetric {
            /// Creates the metric.
            pub fn new() -> Self {
                Self {
                    name: MetricName::new("Perplexity".to_string()),
                    state: PerplexityState::new(),
                    pad_token: Default::default(),
                }
            }
            /// Sets the pad token to exclude from perplexity calculation.
            ///
            /// When a pad token is set, predictions for padding tokens are masked out
            /// and do not contribute to the perplexity calculation. This is important
            /// for variable-length sequences where padding is used.
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
        }
        impl Metric for PerplexityMetric {
            type Input = PerplexityInput;
            fn update(
                &mut self,
                input: &PerplexityInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let targets = input.targets.clone();
                let outputs = input.outputs.clone();
                let [total_tokens, _vocab_size] = outputs.dims();
                let log_probs = burn_core::tensor::activation::log_softmax(outputs, 1);
                let target_log_probs = log_probs
                    .gather(1, targets.clone().unsqueeze_dim(1))
                    .squeeze_dim(1);
                let (sum_log_prob, effective_tokens) = match self.pad_token {
                    Some(pad_token) => {
                        let mask = targets.clone().not_equal_elem(pad_token as i64);
                        let masked_log_probs = target_log_probs
                            .mask_fill(mask.clone().bool_not(), 0.0);
                        let sum_log_prob = masked_log_probs.sum().into_scalar::<f64>();
                        let effective_tokens = mask.int().sum().into_scalar::<i64>()
                            as usize;
                        (sum_log_prob, effective_tokens)
                    }
                    None => {
                        let sum_log_prob = target_log_probs.sum().into_scalar::<f64>();
                        (sum_log_prob, total_tokens)
                    }
                };
                self.state
                    .update(
                        sum_log_prob,
                        effective_tokens,
                        FormatOptions::new(self.name()).precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: None,
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for PerplexityMetric {
            fn value(&self) -> NumericEntry {
                self.state.value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod precision {
        use crate::metric::{MetricName, Numeric};
        use super::{
            Metric, MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
            SerializedEntry,
            classification::{ClassReduction, ClassificationMetricConfig, DecisionRule},
            confusion_stats::{ConfusionStats, ConfusionStatsInput},
            state::{FormatOptions, NumericMetricState},
        };
        use burn_core::prelude::Tensor;
        use std::{num::NonZeroUsize, sync::Arc};
        /// The Precision Metric
        pub struct PrecisionMetric {
            name: MetricName,
            state: NumericMetricState,
            config: ClassificationMetricConfig,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for PrecisionMetric {
            #[inline]
            fn clone(&self) -> PrecisionMetric {
                PrecisionMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    config: ::core::clone::Clone::clone(&self.config),
                }
            }
        }
        impl Default for PrecisionMetric {
            fn default() -> Self {
                Self::new(Default::default())
            }
        }
        impl PrecisionMetric {
            fn new(config: ClassificationMetricConfig) -> Self {
                let state = Default::default();
                let name = Arc::new(
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(
                            format_args!(
                                "Precision @ {0:?} [{1:?}]",
                                config.decision_rule,
                                config.class_reduction,
                            ),
                        )
                    }),
                );
                Self { state, config, name }
            }
            /// Precision metric for binary classification.
            ///
            /// # Arguments
            ///
            /// * `threshold` - The threshold to transform a probability into a binary prediction.
            #[allow(dead_code)]
            pub fn binary(threshold: f64) -> Self {
                Self::new(ClassificationMetricConfig {
                    decision_rule: DecisionRule::Threshold(threshold),
                    ..Default::default()
                })
            }
            /// Precision metric for multiclass classification.
            ///
            /// # Arguments
            ///
            /// * `top_k` - The number of highest predictions considered to find the correct label (typically `1`).
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multiclass(top_k: usize, class_reduction: ClassReduction) -> Self {
                Self::new(ClassificationMetricConfig {
                    decision_rule: DecisionRule::TopK(
                        NonZeroUsize::new(top_k).expect("top_k must be non-zero"),
                    ),
                    class_reduction,
                })
            }
            /// Precision metric for multi-label classification.
            ///
            /// # Arguments
            ///
            /// * `threshold` - The threshold to transform a probability into a binary value.
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multilabel(threshold: f64, class_reduction: ClassReduction) -> Self {
                Self {
                    config: ClassificationMetricConfig {
                        decision_rule: DecisionRule::Threshold(threshold),
                        class_reduction,
                    },
                    ..Default::default()
                }
            }
            fn class_average(&self, mut aggregated_metric: Tensor<1>) -> f64 {
                use ClassReduction::{Macro, Micro};
                let avg_tensor = match self.config.class_reduction {
                    Micro => aggregated_metric,
                    Macro => {
                        if aggregated_metric.clone().contains_nan().any().into_scalar() {
                            let nan_mask = aggregated_metric.clone().is_nan();
                            aggregated_metric = aggregated_metric
                                .clone()
                                .select(0, nan_mask.bool_not().argwhere().squeeze_dim(1));
                        }
                        aggregated_metric.mean()
                    }
                };
                avg_tensor.into_scalar()
            }
        }
        impl Metric for PrecisionMetric {
            type Input = ConfusionStatsInput;
            fn update(
                &mut self,
                input: &Self::Input,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let [sample_size, _] = input.predictions.dims();
                let cf_stats = ConfusionStats::new(input, &self.config);
                let metric = self
                    .class_average(
                        cf_stats.clone().true_positive() / cf_stats.predicted_positive(),
                    );
                self.state
                    .update(
                        100.0 * metric,
                        sample_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for PrecisionMetric {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod recall {
        use crate::metric::{MetricName, Numeric};
        use super::{
            Metric, MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
            SerializedEntry,
            classification::{ClassReduction, ClassificationMetricConfig, DecisionRule},
            confusion_stats::{ConfusionStats, ConfusionStatsInput},
            state::{FormatOptions, NumericMetricState},
        };
        use burn_core::prelude::Tensor;
        use std::{num::NonZeroUsize, sync::Arc};
        ///The Recall Metric
        pub struct RecallMetric {
            name: MetricName,
            state: NumericMetricState,
            config: ClassificationMetricConfig,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for RecallMetric {
            #[inline]
            fn clone(&self) -> RecallMetric {
                RecallMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    config: ::core::clone::Clone::clone(&self.config),
                }
            }
        }
        impl Default for RecallMetric {
            fn default() -> Self {
                Self::new(Default::default())
            }
        }
        impl RecallMetric {
            fn new(config: ClassificationMetricConfig) -> Self {
                let state = Default::default();
                let name = Arc::new(
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(
                            format_args!(
                                "Recall @ {0:?} [{1:?}]",
                                config.decision_rule,
                                config.class_reduction,
                            ),
                        )
                    }),
                );
                Self { state, config, name }
            }
            /// Recall metric for binary classification.
            ///
            /// # Arguments
            ///
            /// * `threshold` - The threshold to transform a probability into a binary prediction.
            #[allow(dead_code)]
            pub fn binary(threshold: f64) -> Self {
                Self::new(ClassificationMetricConfig {
                    decision_rule: DecisionRule::Threshold(threshold),
                    ..Default::default()
                })
            }
            /// Recall metric for multiclass classification.
            ///
            /// # Arguments
            ///
            /// * `top_k` - The number of highest predictions considered to find the correct label (typically `1`).
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multiclass(top_k: usize, class_reduction: ClassReduction) -> Self {
                Self::new(ClassificationMetricConfig {
                    decision_rule: DecisionRule::TopK(
                        NonZeroUsize::new(top_k).expect("top_k must be non-zero"),
                    ),
                    class_reduction,
                })
            }
            /// Recall metric for multi-label classification.
            ///
            /// # Arguments
            ///
            /// * `threshold` - The threshold to transform a probability into a binary prediction.
            /// * `class_reduction` - [Class reduction](ClassReduction) type.
            #[allow(dead_code)]
            pub fn multilabel(threshold: f64, class_reduction: ClassReduction) -> Self {
                Self::new(ClassificationMetricConfig {
                    decision_rule: DecisionRule::Threshold(threshold),
                    class_reduction,
                })
            }
            fn class_average(&self, mut aggregated_metric: Tensor<1>) -> f64 {
                use ClassReduction::{Macro, Micro};
                let avg_tensor = match self.config.class_reduction {
                    Micro => aggregated_metric,
                    Macro => {
                        if aggregated_metric.clone().contains_nan().any().into_scalar() {
                            let nan_mask = aggregated_metric.clone().is_nan();
                            aggregated_metric = aggregated_metric
                                .clone()
                                .select(0, nan_mask.bool_not().argwhere().squeeze_dim(1));
                        }
                        aggregated_metric.mean()
                    }
                };
                avg_tensor.into_scalar()
            }
        }
        impl Metric for RecallMetric {
            type Input = ConfusionStatsInput;
            fn update(
                &mut self,
                input: &Self::Input,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let [sample_size, _] = input.predictions.dims();
                let cf_stats = ConfusionStats::new(input, &self.config);
                let metric = self
                    .class_average(
                        cf_stats.clone().true_positive() / cf_stats.positive(),
                    );
                self.state
                    .update(
                        100.0 * metric,
                        sample_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for RecallMetric {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod rouge {
        use super::state::{FormatOptions, NumericMetricState};
        use super::{MetricMetadata, SerializedEntry};
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericAttributes,
            NumericEntry,
        };
        use burn_core::tensor::{Int, Tensor};
        use std::sync::Arc;
        fn lcs_length(a: &[i32], b: &[i32]) -> usize {
            let (shorter, longer) = if a.len() <= b.len() { (a, b) } else { (b, a) };
            let mut prev = ::alloc::vec::from_elem(0usize, shorter.len() + 1);
            let mut curr = ::alloc::vec::from_elem(0usize, shorter.len() + 1);
            for &x in longer {
                for (j, &y) in shorter.iter().enumerate() {
                    if x == y {
                        curr[j + 1] = prev[j] + 1;
                    } else {
                        curr[j + 1] = curr[j].max(prev[j + 1]);
                    }
                }
                core::mem::swap(&mut prev, &mut curr);
            }
            prev[shorter.len()]
        }
        /// ROUGE-L metric based on longest common subsequence.
        pub struct RougeLScore {
            name: MetricName,
            state: NumericMetricState,
            pad_token: Option<usize>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for RougeLScore {
            #[inline]
            fn clone(&self) -> RougeLScore {
                RougeLScore {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    pad_token: ::core::clone::Clone::clone(&self.pad_token),
                }
            }
        }
        /// Input for [RougeLScore].
        pub struct RougeLInput {
            /// Predicted token sequences.
            pub outputs: Tensor<2, Int>,
            /// Reference token sequences.
            pub targets: Tensor<2, Int>,
        }
        impl RougeLInput {
            ///Constructs a new `RougeLInput`.
            pub fn new(outputs: Tensor<2, Int>, targets: Tensor<2, Int>) -> Self {
                RougeLInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl Default for RougeLScore {
            fn default() -> Self {
                Self::new()
            }
        }
        impl RougeLScore {
            /// Creates a new ROUGE-L metric.
            pub fn new() -> Self {
                Self {
                    name: Arc::new("ROUGE-L".to_string()),
                    state: NumericMetricState::default(),
                    pad_token: None,
                }
            }
            /// Sets the pad token index. Tokens matching this value are stripped
            /// from the right of each sequence before scoring.
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
        }
        impl Metric for RougeLScore {
            type Input = RougeLInput;
            fn update(
                &mut self,
                input: &RougeLInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let outputs = &input.outputs;
                let targets = &input.targets;
                let [batch_size, seq_len] = targets.dims();
                let outputs_data = outputs.to_data().iter::<i32>().collect::<Vec<_>>();
                let targets_data = targets.to_data().iter::<i32>().collect::<Vec<_>>();
                let pad_token = self.pad_token.map(|p| p as i32);
                let mut total_f1 = 0.0_f64;
                for i in 0..batch_size {
                    let start = i * seq_len;
                    let end = (i + 1) * seq_len;
                    let output_seq = &outputs_data[start..end];
                    let target_seq = &targets_data[start..end];
                    let output_seq = match pad_token {
                        Some(pad) => {
                            let len = output_seq
                                .iter()
                                .position(|&x| x == pad)
                                .unwrap_or(output_seq.len());
                            &output_seq[..len]
                        }
                        None => output_seq,
                    };
                    let target_seq = match pad_token {
                        Some(pad) => {
                            let len = target_seq
                                .iter()
                                .position(|&x| x == pad)
                                .unwrap_or(target_seq.len());
                            &target_seq[..len]
                        }
                        None => target_seq,
                    };
                    let lcs_len = lcs_length(target_seq, output_seq) as f64;
                    let ref_len = target_seq.len() as f64;
                    let cand_len = output_seq.len() as f64;
                    if ref_len == 0.0 && cand_len == 0.0 {
                        total_f1 += 100.0;
                        continue;
                    }
                    if ref_len == 0.0 || cand_len == 0.0 {
                        continue;
                    }
                    let precision = lcs_len / cand_len;
                    let recall = lcs_len / ref_len;
                    let f1 = if precision + recall > 0.0 {
                        2.0 * precision * recall / (precision + recall)
                    } else {
                        0.0
                    };
                    total_f1 += f1 * 100.0;
                }
                let value = total_f1 / batch_size as f64;
                self.state
                    .update(
                        value,
                        batch_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset();
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for RougeLScore {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod top_k_acc {
        use std::sync::Arc;
        use super::state::{FormatOptions, NumericMetricState};
        use super::{MetricMetadata, SerializedEntry};
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericAttributes,
            NumericEntry,
        };
        use burn_core::tensor::{Int, Tensor};
        /// The Top-K accuracy metric.
        ///
        /// For K=1, this is equivalent to the [accuracy metric](`super::acc::AccuracyMetric`).
        pub struct TopKAccuracyMetric {
            name: Arc<String>,
            k: usize,
            state: NumericMetricState,
            /// If specified, targets equal to this value will be considered padding and will not count
            /// towards the metric
            pad_token: Option<usize>,
        }
        #[automatically_derived]
        impl ::core::default::Default for TopKAccuracyMetric {
            #[inline]
            fn default() -> TopKAccuracyMetric {
                TopKAccuracyMetric {
                    name: ::core::default::Default::default(),
                    k: ::core::default::Default::default(),
                    state: ::core::default::Default::default(),
                    pad_token: ::core::default::Default::default(),
                }
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for TopKAccuracyMetric {
            #[inline]
            fn clone(&self) -> TopKAccuracyMetric {
                TopKAccuracyMetric {
                    name: ::core::clone::Clone::clone(&self.name),
                    k: ::core::clone::Clone::clone(&self.k),
                    state: ::core::clone::Clone::clone(&self.state),
                    pad_token: ::core::clone::Clone::clone(&self.pad_token),
                }
            }
        }
        /// The [top-k accuracy metric](TopKAccuracyMetric) input type.
        pub struct TopKAccuracyInput {
            /// The outputs (batch_size, num_classes)
            outputs: Tensor<2>,
            /// The labels (batch_size)
            targets: Tensor<1, Int>,
        }
        impl TopKAccuracyInput {
            ///Constructs a new `TopKAccuracyInput`.
            pub fn new(outputs: Tensor<2>, targets: Tensor<1, Int>) -> Self {
                TopKAccuracyInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl TopKAccuracyMetric {
            /// Creates the metric.
            pub fn new(k: usize) -> Self {
                Self {
                    name: Arc::new(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("Top-K Accuracy @ TopK({0})", k),
                            )
                        }),
                    ),
                    k,
                    ..Default::default()
                }
            }
            /// Sets the pad token.
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
        }
        impl Metric for TopKAccuracyMetric {
            type Input = TopKAccuracyInput;
            fn update(
                &mut self,
                input: &TopKAccuracyInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let [batch_size, _n_classes] = input.outputs.dims();
                let targets = input.targets.clone();
                let outputs = input
                    .outputs
                    .clone()
                    .argsort_descending(1)
                    .narrow(1, 0, self.k)
                    .reshape([batch_size, self.k]);
                let (targets, num_pad) = match self.pad_token {
                    Some(pad_token) => {
                        let mask = targets.clone().equal_elem(pad_token as i64);
                        let num_pad = mask.clone().int().sum().into_scalar::<f64>();
                        (targets.clone().mask_fill(mask, -1_i64), num_pad)
                    }
                    None => (targets.clone(), 0_f64),
                };
                let accuracy = targets
                    .reshape([batch_size, 1])
                    .repeat_dim(1, self.k)
                    .equal(outputs)
                    .int()
                    .sum()
                    .into_scalar::<f64>() / (batch_size as f64 - num_pad);
                self.state
                    .update(
                        100.0 * accuracy,
                        batch_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn clear(&mut self) {
                self.state.reset()
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: true,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for TopKAccuracyMetric {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    mod wer {
        use super::cer::edit_distance;
        use super::state::{FormatOptions, NumericMetricState};
        use super::{MetricMetadata, SerializedEntry};
        use crate::metric::{
            Metric, MetricAttributes, MetricName, Numeric, NumericAttributes,
            NumericEntry,
        };
        use burn_core::tensor::{Int, Tensor};
        use std::sync::Arc;
        /// The word error rate (WER) metric, similar to the CER, is defined as the edit distance (e.g. Levenshtein distance) between the predicted
        /// and reference word sequences, divided by the total number of words in the reference. Here, the "units" within the sequences are words.
        ///
        pub struct WordErrorRate {
            name: MetricName,
            state: NumericMetricState,
            pad_token: Option<usize>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for WordErrorRate {
            #[inline]
            fn clone(&self) -> WordErrorRate {
                WordErrorRate {
                    name: ::core::clone::Clone::clone(&self.name),
                    state: ::core::clone::Clone::clone(&self.state),
                    pad_token: ::core::clone::Clone::clone(&self.pad_token),
                }
            }
        }
        /// The [word error rate metric](WordErrorRate) input type.
        pub struct WerInput {
            /// The predicted token sequences (as a 2-D tensor of token indices).
            pub outputs: Tensor<2, Int>,
            /// The target token sequences (as a 2-D tensor of token indices).
            pub targets: Tensor<2, Int>,
        }
        impl WerInput {
            ///Constructs a new `WerInput`.
            pub fn new(outputs: Tensor<2, Int>, targets: Tensor<2, Int>) -> Self {
                WerInput {
                    outputs: outputs,
                    targets: targets,
                }
            }
        }
        impl Default for WordErrorRate {
            fn default() -> Self {
                Self::new()
            }
        }
        impl WordErrorRate {
            /// Creates the metric.
            pub fn new() -> Self {
                Self {
                    name: Arc::new("WER".to_string()),
                    state: NumericMetricState::default(),
                    pad_token: None,
                }
            }
            /// Sets the pad token.
            pub fn with_pad_token(mut self, index: usize) -> Self {
                self.pad_token = Some(index);
                self
            }
        }
        impl Metric for WordErrorRate {
            type Input = WerInput;
            fn update(
                &mut self,
                input: &WerInput,
                _metadata: &MetricMetadata,
            ) -> SerializedEntry {
                let outputs = input.outputs.clone();
                let targets = input.targets.clone();
                let [batch_size, seq_len] = targets.dims();
                let outputs_data = outputs
                    .to_data()
                    .convert::<i32>()
                    .to_vec()
                    .expect("Failed to convert outputs to Vec");
                let targets_data = targets
                    .to_data()
                    .convert::<i32>()
                    .to_vec()
                    .expect("Failed to convert targets to Vec");
                let pad_token = self.pad_token.map(|p| p as i32);
                let mut total_edit_distance = 0.0;
                let mut total_target_length = 0.0;
                for i in 0..batch_size {
                    let start = i * seq_len;
                    let end = (i + 1) * seq_len;
                    let output_seq = &outputs_data[start..end];
                    let target_seq = &targets_data[start..end];
                    let (ed, target_len) = match pad_token {
                        Some(pad) => {
                            let output_seq_no_pad = output_seq
                                .iter()
                                .take_while(|&&x| x != pad)
                                .copied()
                                .collect::<Vec<_>>();
                            let target_seq_no_pad = target_seq
                                .iter()
                                .take_while(|&&x| x != pad)
                                .copied()
                                .collect::<Vec<_>>();
                            (
                                edit_distance(&target_seq_no_pad, &output_seq_no_pad),
                                target_seq_no_pad.len(),
                            )
                        }
                        None => (edit_distance(target_seq, output_seq), target_seq.len()),
                    };
                    total_edit_distance += ed as f64;
                    total_target_length += target_len as f64;
                }
                let value = if total_target_length > 0.0 {
                    100.0 * total_edit_distance / total_target_length
                } else {
                    0.0
                };
                self.state
                    .update(
                        value,
                        batch_size,
                        FormatOptions::new(self.name()).unit("%").precision(2),
                    )
            }
            fn name(&self) -> MetricName {
                self.name.clone()
            }
            fn clear(&mut self) {
                self.state.reset();
            }
            fn attributes(&self) -> MetricAttributes {
                NumericAttributes {
                    unit: Some("%".to_string()),
                    higher_is_better: false,
                    ..Default::default()
                }
                    .into()
            }
        }
        impl Numeric for WordErrorRate {
            fn value(&self) -> NumericEntry {
                self.state.current_value()
            }
            fn running_value(&self) -> NumericEntry {
                self.state.running_value()
            }
        }
    }
    pub use acc::*;
    pub use auc_pr::*;
    pub use auroc::*;
    pub use base::*;
    pub use bleu::*;
    pub use cer::*;
    pub use confusion_stats::ConfusionStatsInput;
    pub use fbetascore::*;
    pub use hamming::*;
    pub use iteration::*;
    pub use learning_rate::*;
    pub use loss::*;
    pub use perplexity::*;
    pub use precision::*;
    pub use recall::*;
    pub use rouge::*;
    pub use top_k_acc::*;
    pub use wer::*;
    pub(crate) mod classification {
        use std::num::NonZeroUsize;
        /// Necessary data for classification metrics.
        pub struct ClassificationMetricConfig {
            pub decision_rule: DecisionRule,
            pub class_reduction: ClassReduction,
        }
        #[automatically_derived]
        impl ::core::default::Default for ClassificationMetricConfig {
            #[inline]
            fn default() -> ClassificationMetricConfig {
                ClassificationMetricConfig {
                    decision_rule: ::core::default::Default::default(),
                    class_reduction: ::core::default::Default::default(),
                }
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for ClassificationMetricConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "ClassificationMetricConfig",
                    "decision_rule",
                    &self.decision_rule,
                    "class_reduction",
                    &&self.class_reduction,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for ClassificationMetricConfig {
            #[inline]
            fn clone(&self) -> ClassificationMetricConfig {
                ClassificationMetricConfig {
                    decision_rule: ::core::clone::Clone::clone(&self.decision_rule),
                    class_reduction: ::core::clone::Clone::clone(&self.class_reduction),
                }
            }
        }
        /// The prediction decision rule for classification metrics.
        pub enum DecisionRule {
            /// Consider a class predicted if its probability exceeds the threshold.
            Threshold(f64),
            /// Consider a class predicted correctly if it is within the top k predicted classes based on scores.
            TopK(NonZeroUsize),
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for DecisionRule {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    DecisionRule::Threshold(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Threshold",
                            &__self_0,
                        )
                    }
                    DecisionRule::TopK(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "TopK",
                            &__self_0,
                        )
                    }
                }
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for DecisionRule {
            #[inline]
            fn clone(&self) -> DecisionRule {
                match self {
                    DecisionRule::Threshold(__self_0) => {
                        DecisionRule::Threshold(::core::clone::Clone::clone(__self_0))
                    }
                    DecisionRule::TopK(__self_0) => {
                        DecisionRule::TopK(::core::clone::Clone::clone(__self_0))
                    }
                }
            }
        }
        impl Default for DecisionRule {
            fn default() -> Self {
                Self::Threshold(0.5)
            }
        }
        /// The reduction strategy for classification metrics.
        pub enum ClassReduction {
            /// Computes the statistics over all classes before averaging
            Micro,
            /// Computes the statistics independently for each class before averaging
            #[default]
            Macro,
        }
        #[automatically_derived]
        impl ::core::marker::Copy for ClassReduction {}
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for ClassReduction {}
        #[automatically_derived]
        impl ::core::clone::Clone for ClassReduction {
            #[inline]
            fn clone(&self) -> ClassReduction {
                *self
            }
        }
        #[automatically_derived]
        impl ::core::default::Default for ClassReduction {
            #[inline]
            fn default() -> ClassReduction {
                Self::Macro
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for ClassReduction {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::write_str(
                    f,
                    match self {
                        ClassReduction::Micro => "Micro",
                        ClassReduction::Macro => "Macro",
                    },
                )
            }
        }
    }
    pub(crate) mod processor {
        mod async_wrapper {
            use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation};
            use super::EventProcessorTraining;
            use async_channel::{Receiver, Sender};
            /// Event processor for the training process.
            pub struct AsyncProcessorTraining<ET, EV> {
                sender: Sender<Message<ET, EV>>,
            }
            /// Event processor for the model evaluation.
            pub struct AsyncProcessorEvaluation<P: EventProcessorEvaluation> {
                sender: Sender<EvalMessage<P>>,
            }
            struct WorkerTraining<ET, EV, P: EventProcessorTraining<ET, EV>> {
                processor: P,
                rec: Receiver<Message<ET, EV>>,
            }
            struct WorkerEvaluation<P: EventProcessorEvaluation> {
                processor: P,
                rec: Receiver<EvalMessage<P>>,
            }
            impl<
                ET: Send + 'static,
                EV: Send + 'static,
                P: EventProcessorTraining<ET, EV> + 'static,
            > WorkerTraining<ET, EV, P> {
                pub fn start(processor: P, rec: Receiver<Message<ET, EV>>) {
                    let mut worker = Self { processor, rec };
                    std::thread::Builder::new()
                        .name("train-worker".into())
                        .spawn(move || {
                            while let Ok(msg) = worker.rec.recv_blocking() {
                                match msg {
                                    Message::Train(event) => {
                                        worker.processor.process_train(event)
                                    }
                                    Message::Valid(event) => {
                                        worker.processor.process_valid(event)
                                    }
                                    Message::Renderer(callback) => {
                                        callback
                                            .send_blocking(worker.processor.renderer())
                                            .unwrap();
                                        return;
                                    }
                                }
                            }
                        })
                        .unwrap();
                }
            }
            impl<P: EventProcessorEvaluation + 'static> WorkerEvaluation<P> {
                pub fn start(processor: P, rec: Receiver<EvalMessage<P>>) {
                    let mut worker = Self { processor, rec };
                    std::thread::Builder::new()
                        .name("evel-worker".into())
                        .spawn(move || {
                            while let Ok(event) = worker.rec.recv_blocking() {
                                match event {
                                    EvalMessage::Test(event) => {
                                        worker.processor.process_test(event)
                                    }
                                    EvalMessage::Renderer(sender) => {
                                        sender.send_blocking(worker.processor.renderer()).unwrap();
                                        return;
                                    }
                                }
                            }
                        })
                        .unwrap();
                }
            }
            impl<ET: Send + 'static, EV: Send + 'static> AsyncProcessorTraining<ET, EV> {
                /// Create an event processor for training.
                pub fn new<P: EventProcessorTraining<ET, EV> + 'static>(
                    processor: P,
                ) -> Self {
                    let (sender, rec) = async_channel::bounded(1);
                    WorkerTraining::start(processor, rec);
                    Self { sender }
                }
            }
            impl<P: EventProcessorEvaluation + 'static> AsyncProcessorEvaluation<P> {
                /// Create an event processor for model evaluation.
                pub fn new(processor: P) -> Self {
                    let (sender, rec) = async_channel::bounded(1);
                    WorkerEvaluation::start(processor, rec);
                    Self { sender }
                }
            }
            enum Message<EventTrain, EventValid> {
                Train(EventTrain),
                Valid(EventValid),
                Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
            }
            enum EvalMessage<P: EventProcessorEvaluation> {
                Test(EvaluatorEvent<P::ItemTest>),
                Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
            }
            impl<ET: Send, EV: Send> EventProcessorTraining<ET, EV>
            for AsyncProcessorTraining<ET, EV> {
                fn process_train(&mut self, event: ET) {
                    self.sender.send_blocking(Message::Train(event)).unwrap();
                }
                fn process_valid(&mut self, event: EV) {
                    self.sender.send_blocking(Message::Valid(event)).unwrap();
                }
                fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
                    let (sender, rec) = async_channel::bounded(1);
                    self.sender.send_blocking(Message::Renderer(sender)).unwrap();
                    match rec.recv_blocking() {
                        Ok(value) => value,
                        Err(err) => {
                            ::core::panicking::panic_fmt(format_args!("{0:?}", err));
                        }
                    }
                }
            }
            impl<P: EventProcessorEvaluation> EventProcessorEvaluation
            for AsyncProcessorEvaluation<P> {
                type ItemTest = P::ItemTest;
                fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>) {
                    self.sender.send_blocking(EvalMessage::Test(event)).unwrap();
                }
                fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
                    let (sender, rec) = async_channel::bounded(1);
                    self.sender.send_blocking(EvalMessage::Renderer(sender)).unwrap();
                    match rec.recv_blocking() {
                        Ok(value) => value,
                        Err(err) => {
                            ::core::panicking::panic_fmt(format_args!("{0:?}", err));
                        }
                    }
                }
            }
        }
        mod base {
            use burn_core::data::dataloader::Progress;
            use burn_optim::LearningRate;
            use crate::{LearnerSummary, renderer::{EvaluationName, MetricsRenderer}};
            /// Event happening during the training/validation process.
            pub enum LearnerEvent<T> {
                /// Signal the start of the process (e.g., training start).
                Start {
                    /// The total number of training epochs.
                    total_epochs: usize,
                },
                /// Signal that an item have been processed.
                ProcessedItem(TrainingItem<T>),
                /// Signal the start of a split, carrying the total number of items in that split.
                StartSplit(usize),
                /// Signal the end of a split, carrying the current epoch number.
                EndSplit(usize),
                /// Signal the end of a full epoch.
                EndEpoch(usize),
                /// Signal the end of the process (e.g., training end).
                End(Option<LearnerSummary>),
            }
            /// Event happening during the evaluation process.
            pub enum EvaluatorEvent<T> {
                /// Signal the start of the process (e.g., evaluation start)
                Start {
                    /// The total number of items to evaluate.
                    total_tests: usize,
                },
                /// Signal the start of a test split, carrying the split name and total number of items.
                StartTest(EvaluationName, usize),
                /// Signal that an item have been processed.
                ProcessedItem(EvaluationName, EvaluationItem<T>),
                /// Signal the end of a single test split.
                EndTest,
                /// Signal the end of the process (e.g., evaluation end).
                End(Option<LearnerSummary>),
            }
            /// Items that are lazy are not ready to be processed by metrics.
            ///
            /// We want to sync them on a different thread to avoid blocking training.
            pub trait ItemLazy: Send {
                /// Sync the item.
                fn sync(self) -> Self;
            }
            /// Process events happening during training and validation.
            pub trait EventProcessorTraining<TrainEvent, ValidEvent>: Send {
                /// Collect a training event.
                fn process_train(&mut self, event: TrainEvent);
                /// Collect a validation event.
                fn process_valid(&mut self, event: ValidEvent);
                /// Returns the renderer used for training.
                fn renderer(self) -> Box<dyn MetricsRenderer>;
            }
            /// Process events happening during evaluation.
            pub trait EventProcessorEvaluation: Send {
                /// The test item.
                type ItemTest: ItemLazy;
                /// Collect a test event.
                fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>);
                /// Returns the renderer used for evaluation.
                fn renderer(self) -> Box<dyn MetricsRenderer>;
            }
            /// A learner item.
            pub struct TrainingItem<T> {
                /// The item.
                pub item: T,
                /// The progress.
                pub progress: Progress,
                /// The iteration, if it it different from the items processed.
                pub iteration: Option<usize>,
                /// The learning rate.
                pub lr: Option<LearningRate>,
            }
            impl<T> TrainingItem<T> {
                ///Constructs a new `TrainingItem`.
                pub fn new(
                    item: T,
                    progress: Progress,
                    iteration: Option<usize>,
                    lr: Option<LearningRate>,
                ) -> Self {
                    TrainingItem {
                        item: item,
                        progress: progress,
                        iteration: iteration,
                        lr: lr,
                    }
                }
            }
            impl<T: ItemLazy> ItemLazy for TrainingItem<T> {
                fn sync(self) -> Self {
                    TrainingItem {
                        item: self.item.sync(),
                        progress: self.progress,
                        iteration: self.iteration,
                        lr: self.lr,
                    }
                }
            }
            /// An evaluation item.
            pub struct EvaluationItem<T> {
                /// The item.
                pub item: T,
                /// The progress.
                pub progress: Progress,
                /// The iteration, if it it different from the items processed.
                pub iteration: Option<usize>,
            }
            impl<T> EvaluationItem<T> {
                ///Constructs a new `EvaluationItem`.
                pub fn new(
                    item: T,
                    progress: Progress,
                    iteration: Option<usize>,
                ) -> Self {
                    EvaluationItem {
                        item: item,
                        progress: progress,
                        iteration: iteration,
                    }
                }
            }
            impl<T: ItemLazy> ItemLazy for EvaluationItem<T> {
                fn sync(self) -> Self {
                    EvaluationItem {
                        item: self.item.sync(),
                        progress: self.progress,
                        iteration: self.iteration,
                    }
                }
            }
            impl ItemLazy for () {
                fn sync(self) -> Self {}
            }
        }
        mod full {
            use super::{EventProcessorTraining, ItemLazy, LearnerEvent, MetricsTraining};
            use crate::logger::{EvaluationProgressLogger, TrainingProgressLogger};
            use crate::metric::MetricMetadata;
            use crate::metric::processor::{
                EvaluatorEvent, EventProcessorEvaluation, MetricsEvaluation,
            };
            use crate::metric::store::{EpochSummary, EventStoreClient, Split};
            use crate::renderer::{MetricState, MetricsRenderer};
            use std::sync::Arc;
            /// An [event processor](EventProcessorTraining) that handles:
            ///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
            ///   - Render metrics using a [metrics renderer](MetricsRenderer).
            pub struct FullEventProcessorTraining<T: ItemLazy, V: ItemLazy> {
                metrics: MetricsTraining<T, V>,
                renderer: Box<dyn MetricsRenderer>,
                store: Arc<EventStoreClient>,
                progress_logger: Option<Box<dyn TrainingProgressLogger>>,
                current_epoch: usize,
                total_epochs: usize,
            }
            /// An [event processor](EventProcessorEvaluation) that handles:
            ///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
            ///   - Render metrics using a [metrics renderer](MetricsRenderer).
            pub struct FullEventProcessorEvaluation<T: ItemLazy> {
                metrics: MetricsEvaluation<T>,
                renderer: Box<dyn MetricsRenderer>,
                store: Arc<EventStoreClient>,
                progress_logger: Option<Box<dyn EvaluationProgressLogger>>,
                total_tests: usize,
                current_test: usize,
            }
            impl<T: ItemLazy, V: ItemLazy> FullEventProcessorTraining<T, V> {
                pub(crate) fn new(
                    metrics: MetricsTraining<T, V>,
                    renderer: Box<dyn MetricsRenderer>,
                    store: Arc<EventStoreClient>,
                ) -> Self {
                    Self {
                        metrics,
                        renderer,
                        store,
                        progress_logger: None,
                        current_epoch: 1,
                        total_epochs: 0,
                    }
                }
                pub(crate) fn with_progress_logger(
                    mut self,
                    logger: Box<dyn TrainingProgressLogger>,
                ) -> Self {
                    self.progress_logger = Some(logger);
                    self
                }
            }
            impl<T: ItemLazy> FullEventProcessorEvaluation<T> {
                pub(crate) fn new(
                    metrics: MetricsEvaluation<T>,
                    renderer: Box<dyn MetricsRenderer>,
                    store: Arc<EventStoreClient>,
                ) -> Self {
                    Self {
                        metrics,
                        renderer,
                        store,
                        progress_logger: None,
                        total_tests: 0,
                        current_test: 0,
                    }
                }
                pub(crate) fn with_progress_logger(
                    mut self,
                    logger: Box<dyn EvaluationProgressLogger>,
                ) -> Self {
                    self.progress_logger = Some(logger);
                    self
                }
            }
            impl<T: ItemLazy> EventProcessorEvaluation
            for FullEventProcessorEvaluation<T> {
                type ItemTest = T;
                fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>) {
                    match event {
                        EvaluatorEvent::Start { total_tests } => {
                            let definitions = self.metrics.metric_definitions();
                            self.store
                                .add_event_train(
                                    crate::metric::store::Event::MetricsInit(
                                        definitions.clone(),
                                    ),
                                );
                            definitions
                                .iter()
                                .for_each(|definition| {
                                    self.renderer.register_metric(definition.clone())
                                });
                            self.total_tests = total_tests;
                            self.current_test = 0;
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start_global_progress(total_tests);
                            }
                            self.renderer.start_global_progress(total_tests);
                        }
                        EvaluatorEvent::StartTest(name, total_items) => {
                            self.current_test += 1;
                            self.renderer.start_test(name.as_str(), total_items);
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start_test(name.as_str(), total_items);
                            }
                        }
                        EvaluatorEvent::ProcessedItem(name, item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self.metrics.update_test(&item, &metadata);
                            self.store
                                .add_event_test(
                                    crate::metric::store::Event::MetricsUpdate(update.clone()),
                                    name.name.clone(),
                                );
                            update
                                .entries
                                .into_iter()
                                .for_each(|entry| {
                                    self.renderer
                                        .update_test(name.clone(), MetricState::Generic(entry))
                                });
                            update
                                .entries_numeric
                                .into_iter()
                                .for_each(|numeric_update| {
                                    self.renderer
                                        .update_test(
                                            name.clone(),
                                            MetricState::Numeric(
                                                numeric_update.entry,
                                                numeric_update.numeric_entry,
                                            ),
                                        )
                                });
                            if let Some(logger) = &mut self.progress_logger {
                                logger.update_test_progress(item.progress.items_processed);
                                logger.log_event_evaluation("Iteration".to_string());
                            }
                            self.renderer
                                .update_test_progress(item.progress.items_processed);
                            self.renderer.log_event_evaluation("Iteration".to_string());
                        }
                        EvaluatorEvent::EndTest => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end_test();
                            }
                            self.renderer.end_test();
                        }
                        EvaluatorEvent::End(summary) => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end_global_progress();
                            }
                            self.renderer.end_global_progress();
                            self.renderer.on_test_end(summary).ok();
                        }
                    }
                }
                fn renderer(self) -> Box<dyn MetricsRenderer> {
                    self.renderer
                }
            }
            impl<
                T: ItemLazy,
                V: ItemLazy,
            > EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>
            for FullEventProcessorTraining<T, V> {
                fn process_train(&mut self, event: LearnerEvent<T>) {
                    match event {
                        LearnerEvent::Start { total_epochs } => {
                            self.total_epochs = total_epochs;
                            self.current_epoch = 1;
                            let definitions = self.metrics.metric_definitions();
                            self.store
                                .add_event_train(
                                    crate::metric::store::Event::MetricsInit(
                                        definitions.clone(),
                                    ),
                                );
                            definitions
                                .iter()
                                .for_each(|definition| {
                                    self.renderer.register_metric(definition.clone())
                                });
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start(total_epochs, None);
                            }
                            self.renderer.start(total_epochs, None);
                        }
                        LearnerEvent::StartSplit(total_items) => {
                            self.renderer.start_split("train", total_items);
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start_split("train", total_items);
                            }
                        }
                        LearnerEvent::ProcessedItem(item) => {
                            let item = item.sync();
                            let metadata = MetricMetadata {
                                progress: item.progress.clone(),
                                iteration: item.iteration,
                                lr: item.lr,
                            };
                            let update = self.metrics.update_train(&item, &metadata);
                            self.store
                                .add_event_train(
                                    crate::metric::store::Event::MetricsUpdate(update.clone()),
                                );
                            update
                                .entries
                                .into_iter()
                                .for_each(|entry| {
                                    self.renderer.update_train(MetricState::Generic(entry))
                                });
                            update
                                .entries_numeric
                                .into_iter()
                                .for_each(|numeric_update| {
                                    self.renderer
                                        .update_train(
                                            MetricState::Numeric(
                                                numeric_update.entry,
                                                numeric_update.numeric_entry,
                                            ),
                                        )
                                });
                            if let Some(logger) = &mut self.progress_logger {
                                logger.update_split(item.progress.items_processed);
                                logger.log_event_training("Iteration".to_string());
                            }
                            self.renderer.update_split(item.progress.items_processed);
                            self.renderer.log_event_training("Iteration".to_string());
                        }
                        LearnerEvent::EndSplit(epoch) => {
                            self.store
                                .add_event_train(
                                    crate::metric::store::Event::EndEpoch(
                                        EpochSummary::new(epoch, Split::Train),
                                    ),
                                );
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end_split();
                            }
                            self.renderer.end_split();
                            self.metrics.end_epoch_train();
                        }
                        LearnerEvent::EndEpoch(epoch) => {
                            self.current_epoch = epoch + 1;
                            if let Some(logger) = &mut self.progress_logger {
                                logger.update_epoch(epoch);
                            }
                            self.renderer.update_epoch(epoch)
                        }
                        LearnerEvent::End(summary) => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end();
                            }
                            self.renderer.end();
                            self.renderer.on_train_end(summary).ok();
                        }
                    }
                }
                fn process_valid(&mut self, event: LearnerEvent<V>) {
                    match event {
                        LearnerEvent::Start { .. } => {}
                        LearnerEvent::StartSplit(total_items) => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start_split("valid", total_items);
                            }
                            self.renderer.start_split("valid", total_items);
                        }
                        LearnerEvent::ProcessedItem(item) => {
                            let item = item.sync();
                            let metadata = MetricMetadata {
                                progress: item.progress.clone(),
                                iteration: item.iteration,
                                lr: item.lr,
                            };
                            let update = self.metrics.update_valid(&item, &metadata);
                            self.store
                                .add_event_valid(
                                    crate::metric::store::Event::MetricsUpdate(update.clone()),
                                );
                            update
                                .entries
                                .into_iter()
                                .for_each(|entry| {
                                    self.renderer.update_valid(MetricState::Generic(entry))
                                });
                            update
                                .entries_numeric
                                .into_iter()
                                .for_each(|numeric_update| {
                                    self.renderer
                                        .update_valid(
                                            MetricState::Numeric(
                                                numeric_update.entry,
                                                numeric_update.numeric_entry,
                                            ),
                                        )
                                });
                            if let Some(logger) = &mut self.progress_logger {
                                logger.update_split(item.progress.items_processed);
                                logger.log_event_training("Iteration".to_string());
                            }
                            self.renderer.update_split(item.progress.items_processed);
                            self.renderer.log_event_training("Iteration".to_string());
                        }
                        LearnerEvent::EndSplit(epoch) => {
                            self.store
                                .add_event_valid(
                                    crate::metric::store::Event::EndEpoch(
                                        EpochSummary::new(epoch, Split::Valid),
                                    ),
                                );
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end_split();
                            }
                            self.renderer.end_split();
                            self.metrics.end_epoch_valid();
                        }
                        LearnerEvent::EndEpoch(_) => {}
                        LearnerEvent::End(_) => {}
                    }
                }
                fn renderer(self) -> Box<dyn MetricsRenderer> {
                    self.renderer
                }
            }
        }
        mod metrics {
            use std::collections::HashMap;
            use super::{ItemLazy, TrainingItem};
            use crate::{
                EvaluationItem,
                metric::{
                    Adaptor, Metric, MetricDefinition, MetricEntry, MetricId,
                    MetricMetadata, Numeric, store::{MetricsUpdate, NumericMetricUpdate},
                },
            };
            pub(crate) struct MetricsTraining<T: ItemLazy, V: ItemLazy> {
                train: Vec<Box<dyn MetricUpdater<T>>>,
                valid: Vec<Box<dyn MetricUpdater<V>>>,
                train_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
                valid_numeric: Vec<Box<dyn NumericMetricUpdater<V>>>,
                metric_definitions: HashMap<MetricId, MetricDefinition>,
            }
            pub(crate) struct MetricsEvaluation<T: ItemLazy> {
                test: Vec<Box<dyn MetricUpdater<T>>>,
                test_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
                metric_definitions: HashMap<MetricId, MetricDefinition>,
            }
            impl<T: ItemLazy> Default for MetricsEvaluation<T> {
                fn default() -> Self {
                    Self {
                        test: Default::default(),
                        test_numeric: Default::default(),
                        metric_definitions: HashMap::default(),
                    }
                }
            }
            impl<T: ItemLazy, V: ItemLazy> Default for MetricsTraining<T, V> {
                fn default() -> Self {
                    Self {
                        train: Vec::default(),
                        valid: Vec::default(),
                        train_numeric: Vec::default(),
                        valid_numeric: Vec::default(),
                        metric_definitions: HashMap::default(),
                    }
                }
            }
            impl<T: ItemLazy> MetricsEvaluation<T> {
                /// Register a testing metric.
                pub(crate) fn register_test_metric<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    T: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.test.push(Box::new(metric))
                }
                /// Register a numeric testing metric.
                pub(crate) fn register_test_metric_numeric<
                    Me: Metric + Numeric + 'static,
                >(&mut self, metric: Me)
                where
                    T: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.test_numeric.push(Box::new(metric))
                }
                fn register_definition<Me: Metric>(
                    &mut self,
                    metric: &MetricWrapper<Me>,
                ) {
                    self.metric_definitions
                        .insert(
                            metric.id.clone(),
                            MetricDefinition::new(metric.id.clone(), &metric.metric),
                        );
                }
                /// Get metric definitions.
                pub(crate) fn metric_definitions(&mut self) -> Vec<MetricDefinition> {
                    self.metric_definitions.values().cloned().collect()
                }
                /// Update the testing information from the testing item.
                pub(crate) fn update_test(
                    &mut self,
                    item: &EvaluationItem<T>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.test.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.test_numeric.len(),
                    );
                    for metric in self.test.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.test_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
            }
            impl<T: ItemLazy, V: ItemLazy> MetricsTraining<T, V> {
                /// Register a training metric.
                pub(crate) fn register_train_metric<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    T: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.train.push(Box::new(metric))
                }
                /// Register a validation metric.
                pub(crate) fn register_valid_metric<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    V: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.valid.push(Box::new(metric))
                }
                /// Register a numeric training metric.
                pub(crate) fn register_train_metric_numeric<
                    Me: Metric + Numeric + 'static,
                >(&mut self, metric: Me)
                where
                    T: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.train_numeric.push(Box::new(metric))
                }
                /// Register a numeric validation metric.
                pub(crate) fn register_valid_metric_numeric<Me>(&mut self, metric: Me)
                where
                    V: Adaptor<Me::Input> + 'static,
                    Me: Metric + Numeric + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.valid_numeric.push(Box::new(metric))
                }
                fn register_definition<Me: Metric>(
                    &mut self,
                    metric: &MetricWrapper<Me>,
                ) {
                    self.metric_definitions
                        .insert(
                            metric.id.clone(),
                            MetricDefinition::new(metric.id.clone(), &metric.metric),
                        );
                }
                /// Get metric definitions for all splits
                pub(crate) fn metric_definitions(&mut self) -> Vec<MetricDefinition> {
                    self.metric_definitions.values().cloned().collect()
                }
                /// Update the training information from the training item.
                pub(crate) fn update_train(
                    &mut self,
                    item: &TrainingItem<T>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.train.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.train_numeric.len(),
                    );
                    for metric in self.train.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.train_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
                /// Update the training information from the validation item.
                pub(crate) fn update_valid(
                    &mut self,
                    item: &TrainingItem<V>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.valid.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.valid_numeric.len(),
                    );
                    for metric in self.valid.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.valid_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
                /// Signal the end of a training epoch.
                pub(crate) fn end_epoch_train(&mut self) {
                    for metric in self.train.iter_mut() {
                        metric.clear();
                    }
                    for metric in self.train_numeric.iter_mut() {
                        metric.clear();
                    }
                }
                /// Signal the end of a validation epoch.
                pub(crate) fn end_epoch_valid(&mut self) {
                    for metric in self.valid.iter_mut() {
                        metric.clear();
                    }
                    for metric in self.valid_numeric.iter_mut() {
                        metric.clear();
                    }
                }
            }
            impl<T> From<&TrainingItem<T>> for MetricMetadata {
                fn from(item: &TrainingItem<T>) -> Self {
                    Self {
                        progress: item.progress.clone(),
                        iteration: item.iteration,
                        lr: item.lr,
                    }
                }
            }
            impl<T> From<&EvaluationItem<T>> for MetricMetadata {
                fn from(item: &EvaluationItem<T>) -> Self {
                    Self {
                        progress: item.progress.clone(),
                        iteration: item.iteration,
                        lr: None,
                    }
                }
            }
            pub(crate) trait NumericMetricUpdater<T>: Send + Sync {
                fn update(
                    &mut self,
                    item: &T,
                    metadata: &MetricMetadata,
                ) -> NumericMetricUpdate;
                fn clear(&mut self);
            }
            pub(crate) trait MetricUpdater<T>: Send + Sync {
                fn update(&mut self, item: &T, metadata: &MetricMetadata) -> MetricEntry;
                fn clear(&mut self);
            }
            pub(crate) struct MetricWrapper<M> {
                pub id: MetricId,
                pub metric: M,
            }
            impl<M: Metric> MetricWrapper<M> {
                pub fn new(metric: M) -> Self {
                    Self {
                        id: MetricId::new(metric.name()),
                        metric,
                    }
                }
            }
            impl<T, M> NumericMetricUpdater<T> for MetricWrapper<M>
            where
                T: 'static,
                M: Metric + Numeric + 'static,
                T: Adaptor<M::Input>,
            {
                fn update(
                    &mut self,
                    item: &T,
                    metadata: &MetricMetadata,
                ) -> NumericMetricUpdate {
                    let serialized_entry = self.metric.update(&item.adapt(), metadata);
                    let update = MetricEntry::new(self.id.clone(), serialized_entry);
                    let numeric = self.metric.value();
                    let running = self.metric.running_value();
                    NumericMetricUpdate {
                        entry: update,
                        numeric_entry: numeric,
                        running_entry: running,
                    }
                }
                fn clear(&mut self) {
                    self.metric.clear()
                }
            }
            impl<T, M> MetricUpdater<T> for MetricWrapper<M>
            where
                T: 'static,
                M: Metric + 'static,
                T: Adaptor<M::Input>,
            {
                fn update(
                    &mut self,
                    item: &T,
                    metadata: &MetricMetadata,
                ) -> MetricEntry {
                    let serialized_entry = self.metric.update(&item.adapt(), metadata);
                    MetricEntry::new(self.id.clone(), serialized_entry)
                }
                fn clear(&mut self) {
                    self.metric.clear()
                }
            }
        }
        mod minimal {
            use super::{EventProcessorTraining, ItemLazy, LearnerEvent, MetricsTraining};
            use crate::{
                logger::TrainingProgressLogger,
                metric::store::{EpochSummary, EventStoreClient, Split},
                renderer::cli::CliMetricsRenderer,
            };
            use std::sync::Arc;
            /// An [event processor](EventProcessor) that handles:
            ///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
            ///   - Optionally logging training progress via a [TrainingProgressLogger].
            #[allow(dead_code)]
            pub(crate) struct MinimalEventProcessor<T: ItemLazy, V: ItemLazy> {
                metrics: MetricsTraining<T, V>,
                store: Arc<EventStoreClient>,
                progress_logger: Option<Box<dyn TrainingProgressLogger>>,
            }
            #[allow(dead_code)]
            impl<T: ItemLazy, V: ItemLazy> MinimalEventProcessor<T, V> {
                pub(crate) fn new(
                    metrics: MetricsTraining<T, V>,
                    store: Arc<EventStoreClient>,
                ) -> Self {
                    Self {
                        metrics,
                        store,
                        progress_logger: None,
                    }
                }
                pub(crate) fn with_progress_logger(
                    mut self,
                    logger: Box<dyn TrainingProgressLogger>,
                ) -> Self {
                    self.progress_logger = Some(logger);
                    self
                }
            }
            impl<
                T: ItemLazy,
                V: ItemLazy,
            > EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>
            for MinimalEventProcessor<T, V> {
                fn process_train(&mut self, event: LearnerEvent<T>) {
                    match event {
                        LearnerEvent::Start { total_epochs } => {
                            let definitions = self.metrics.metric_definitions();
                            self.store
                                .add_event_train(
                                    crate::metric::store::Event::MetricsInit(definitions),
                                );
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start(total_epochs, None);
                            }
                        }
                        LearnerEvent::StartSplit(total_items) => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start_split("train", total_items);
                            }
                        }
                        LearnerEvent::ProcessedItem(item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self.metrics.update_train(&item, &metadata);
                            self.store
                                .add_event_train(
                                    crate::metric::store::Event::MetricsUpdate(update),
                                );
                            if let Some(logger) = &mut self.progress_logger {
                                logger.update_split(item.progress.items_processed);
                            }
                        }
                        LearnerEvent::EndSplit(epoch) => {
                            self.metrics.end_epoch_train();
                            self.store
                                .add_event_train(
                                    crate::metric::store::Event::EndEpoch(
                                        EpochSummary::new(epoch, Split::Train),
                                    ),
                                );
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end_split();
                            }
                        }
                        LearnerEvent::EndEpoch(epoch) => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.update_epoch(epoch);
                            }
                        }
                        LearnerEvent::End(_summary) => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end();
                            }
                        }
                    }
                }
                fn process_valid(&mut self, event: LearnerEvent<V>) {
                    match event {
                        LearnerEvent::Start { .. } => {}
                        LearnerEvent::StartSplit(total_items) => {
                            if let Some(logger) = &mut self.progress_logger {
                                logger.start_split("valid", total_items);
                            }
                        }
                        LearnerEvent::ProcessedItem(item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self.metrics.update_valid(&item, &metadata);
                            self.store
                                .add_event_valid(
                                    crate::metric::store::Event::MetricsUpdate(update),
                                );
                            if let Some(logger) = &mut self.progress_logger {
                                logger.update_split(item.progress.items_processed);
                            }
                        }
                        LearnerEvent::EndSplit(epoch) => {
                            self.metrics.end_epoch_valid();
                            self.store
                                .add_event_valid(
                                    crate::metric::store::Event::EndEpoch(
                                        EpochSummary::new(epoch, Split::Valid),
                                    ),
                                );
                            if let Some(logger) = &mut self.progress_logger {
                                logger.end_split();
                            }
                        }
                        LearnerEvent::EndEpoch(_) => {}
                        LearnerEvent::End(_) => {}
                    }
                }
                fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
                    Box::new(CliMetricsRenderer::new())
                }
            }
        }
        mod rl_metrics {
            use std::collections::HashMap;
            use crate::{
                EpisodeSummary, EvaluationItem, ItemLazy, MetricUpdater, MetricWrapper,
                NumericMetricUpdater,
                metric::{
                    Adaptor, Metric, MetricDefinition, MetricId, MetricMetadata, Numeric,
                    store::MetricsUpdate,
                },
            };
            pub(crate) struct RLMetrics<TS: ItemLazy, ES: ItemLazy> {
                train_step: Vec<Box<dyn MetricUpdater<TS>>>,
                env_step: Vec<Box<dyn MetricUpdater<ES>>>,
                env_step_valid: Vec<Box<dyn MetricUpdater<ES>>>,
                episode_end: Vec<Box<dyn MetricUpdater<EpisodeSummary>>>,
                episode_end_valid: Vec<Box<dyn MetricUpdater<EpisodeSummary>>>,
                train_step_numeric: Vec<Box<dyn NumericMetricUpdater<TS>>>,
                env_step_numeric: Vec<Box<dyn NumericMetricUpdater<ES>>>,
                env_step_valid_numeric: Vec<Box<dyn NumericMetricUpdater<ES>>>,
                episode_end_numeric: Vec<Box<dyn NumericMetricUpdater<EpisodeSummary>>>,
                episode_end_valid_numeric: Vec<
                    Box<dyn NumericMetricUpdater<EpisodeSummary>>,
                >,
                metric_definitions: HashMap<MetricId, MetricDefinition>,
            }
            impl<TS: ItemLazy, ES: ItemLazy> Default for RLMetrics<TS, ES> {
                fn default() -> Self {
                    Self {
                        train_step: Vec::default(),
                        env_step: Vec::default(),
                        env_step_valid: Vec::default(),
                        episode_end: Vec::default(),
                        episode_end_valid: Vec::default(),
                        train_step_numeric: Vec::default(),
                        env_step_numeric: Vec::default(),
                        env_step_valid_numeric: Vec::default(),
                        episode_end_numeric: Vec::default(),
                        episode_end_valid_numeric: Vec::default(),
                        metric_definitions: HashMap::default(),
                    }
                }
            }
            impl<TS: ItemLazy, ES: ItemLazy> RLMetrics<TS, ES> {
                /// Register a training metric.
                pub(crate) fn register_text_metric_agent<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    ES: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.env_step.push(Box::new(metric))
                }
                /// Register a training metric.
                pub(crate) fn register_agent_metric<Me: Metric + Numeric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    ES: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.env_step_numeric.push(Box::new(metric))
                }
                /// Register a training metric.
                pub(crate) fn register_text_metric_train<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    TS: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.train_step.push(Box::new(metric))
                }
                /// Register a training metric.
                pub(crate) fn register_metric_train<Me: Metric + Numeric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    TS: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.train_step_numeric.push(Box::new(metric))
                }
                /// Register a validation env-step metric.
                pub(crate) fn register_text_metric_agent_valid<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    ES: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.env_step_valid.push(Box::new(metric))
                }
                /// Register a validation env-step numeric metric.
                pub(crate) fn register_agent_metric_valid<
                    Me: Metric + Numeric + 'static,
                >(&mut self, metric: Me)
                where
                    ES: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.env_step_valid_numeric.push(Box::new(metric))
                }
                /// Register an episode-end metric.
                pub(crate) fn register_text_metric_episode<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    EpisodeSummary: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.episode_end.push(Box::new(metric))
                }
                /// Register an episode-end numeric metric.
                pub(crate) fn register_episode_metric<Me: Metric + Numeric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    EpisodeSummary: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.episode_end_numeric.push(Box::new(metric))
                }
                /// Register an episode-end metric for validation.
                pub(crate) fn register_text_metric_episode_valid<Me: Metric + 'static>(
                    &mut self,
                    metric: Me,
                )
                where
                    EpisodeSummary: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.episode_end_valid.push(Box::new(metric))
                }
                /// Register an episode-end numeric metric for validation.
                pub(crate) fn register_episode_metric_valid<
                    Me: Metric + Numeric + 'static,
                >(&mut self, metric: Me)
                where
                    EpisodeSummary: Adaptor<Me::Input> + 'static,
                {
                    let metric = MetricWrapper::new(metric);
                    self.register_definition(&metric);
                    self.episode_end_valid_numeric.push(Box::new(metric))
                }
                fn register_definition<Me: Metric>(
                    &mut self,
                    metric: &MetricWrapper<Me>,
                ) {
                    self.metric_definitions
                        .insert(
                            metric.id.clone(),
                            MetricDefinition::new(metric.id.clone(), &metric.metric),
                        );
                }
                /// Get metric definitions for all splits
                pub(crate) fn metric_definitions(&mut self) -> Vec<MetricDefinition> {
                    self.metric_definitions.values().cloned().collect()
                }
                /// Update the training information from the training item.
                pub(crate) fn update_train_step(
                    &mut self,
                    item: &EvaluationItem<TS>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.train_step.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.train_step_numeric.len(),
                    );
                    for metric in self.train_step.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.train_step_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
                /// Update the env-step metrics from an environment step item.
                pub(crate) fn update_env_step(
                    &mut self,
                    item: &EvaluationItem<ES>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.env_step.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.env_step_numeric.len(),
                    );
                    for metric in self.env_step.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.env_step_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
                /// Update the env-step metrics for validation from an environment step item.
                pub(crate) fn update_env_step_valid(
                    &mut self,
                    item: &EvaluationItem<ES>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.env_step_valid.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.env_step_valid_numeric.len(),
                    );
                    for metric in self.env_step_valid.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.env_step_valid_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
                /// Update the episode-end metrics from an episode summary.
                pub(crate) fn update_episode_end(
                    &mut self,
                    item: &EvaluationItem<EpisodeSummary>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.episode_end.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.episode_end_numeric.len(),
                    );
                    for metric in self.episode_end.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.episode_end_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
                /// Update the episode-end metrics for validation from an episode summary.
                pub(crate) fn update_episode_end_valid(
                    &mut self,
                    item: &EvaluationItem<EpisodeSummary>,
                    metadata: &MetricMetadata,
                ) -> MetricsUpdate {
                    let mut entries = Vec::with_capacity(self.episode_end_valid.len());
                    let mut entries_numeric = Vec::with_capacity(
                        self.episode_end_valid_numeric.len(),
                    );
                    for metric in self.episode_end_valid.iter_mut() {
                        let state = metric.update(&item.item, metadata);
                        entries.push(state);
                    }
                    for metric in self.episode_end_valid_numeric.iter_mut() {
                        let numeric_update = metric.update(&item.item, metadata);
                        entries_numeric.push(numeric_update);
                    }
                    MetricsUpdate::new(entries, entries_numeric)
                }
            }
        }
        mod rl_processor {
            use std::sync::Arc;
            use crate::{
                EpisodeSummary, EvaluationItem, EventProcessorTraining, ItemLazy,
                LearnerSummary, RLMetrics, logger::TrainingProgressLogger,
                metric::store::{Event, EventStoreClient, MetricsUpdate},
                renderer::{MetricState, MetricsRenderer},
            };
            /// Event happening during reinforcement learning.
            pub enum RLEvent<TS, ES> {
                /// Signal the start of the process (e.g., learning starts).
                Start {
                    /// The total number of items to process during training (e.g., total number of environment steps).
                    total_items: usize,
                },
                /// Signal an agent's training step.
                TrainStep(EvaluationItem<TS>),
                /// Signal a timestep of the agent-environment interface.
                EnvStep(EvaluationItem<ES>),
                /// Signal an episode end.
                EpisodeEnd(EvaluationItem<EpisodeSummary>),
                /// Signal the end of the process (e.g., learning ends).
                End(Option<LearnerSummary>),
            }
            /// Event happening during evaluation of a reinforcement learning's agent.
            pub enum AgentEvaluationEvent<T> {
                /// Signal the start of the process (e.g., training start)
                Start(usize),
                /// Signal a timestep of the agent-environment interface.
                EnvStep(EvaluationItem<T>),
                /// Signal an episode end.
                EpisodeEnd(EvaluationItem<EpisodeSummary>),
                /// Signal the end of the process (e.g., training end).
                End,
            }
            /// An [event processor](EventProcessorTraining) that handles:
            ///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
            ///   - Render metrics using a [metrics renderer](MetricsRenderer).
            pub struct RLEventProcessor<TS: ItemLazy, ES: ItemLazy> {
                metrics: RLMetrics<TS, ES>,
                renderer: Box<dyn MetricsRenderer>,
                store: Arc<EventStoreClient>,
                training_progress_logger: Option<Box<dyn TrainingProgressLogger>>,
            }
            impl<TS: ItemLazy, ES: ItemLazy> RLEventProcessor<TS, ES> {
                pub(crate) fn new(
                    metrics: RLMetrics<TS, ES>,
                    renderer: Box<dyn MetricsRenderer>,
                    store: Arc<EventStoreClient>,
                ) -> Self {
                    Self {
                        metrics,
                        renderer,
                        store,
                        training_progress_logger: None,
                    }
                }
                fn process_update_train(&mut self, update: MetricsUpdate) {
                    self.store
                        .add_event_train(
                            crate::metric::store::Event::MetricsUpdate(update.clone()),
                        );
                    update
                        .entries
                        .into_iter()
                        .for_each(|entry| {
                            self.renderer.update_train(MetricState::Generic(entry))
                        });
                    update
                        .entries_numeric
                        .into_iter()
                        .for_each(|numeric_update| {
                            self.renderer
                                .update_train(
                                    MetricState::Numeric(
                                        numeric_update.entry,
                                        numeric_update.numeric_entry,
                                    ),
                                )
                        });
                }
                fn process_update_valid(&mut self, update: MetricsUpdate) {
                    self.store
                        .add_event_valid(
                            crate::metric::store::Event::MetricsUpdate(update.clone()),
                        );
                    update
                        .entries
                        .into_iter()
                        .for_each(|entry| {
                            self.renderer.update_valid(MetricState::Generic(entry))
                        });
                    update
                        .entries_numeric
                        .into_iter()
                        .for_each(|numeric_update| {
                            self.renderer
                                .update_valid(
                                    MetricState::Numeric(
                                        numeric_update.entry,
                                        numeric_update.numeric_entry,
                                    ),
                                )
                        });
                }
            }
            impl<
                TS: ItemLazy,
                ES: ItemLazy,
            > EventProcessorTraining<RLEvent<TS, ES>, AgentEvaluationEvent<ES>>
            for RLEventProcessor<TS, ES> {
                fn process_train(&mut self, event: RLEvent<TS, ES>) {
                    match event {
                        RLEvent::Start { total_items } => {
                            let definitions = self.metrics.metric_definitions();
                            self.store
                                .add_event_train(Event::MetricsInit(definitions.clone()));
                            definitions
                                .iter()
                                .for_each(|definition| {
                                    self.renderer.register_metric(definition.clone())
                                });
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.start(0, Some(total_items));
                            }
                            self.renderer.start(0, Some(total_items));
                        }
                        RLEvent::TrainStep(item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self
                                .metrics
                                .update_train_step(&item, &metadata);
                            self.process_update_train(update);
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.log_event_training("TrainStep".to_string());
                            }
                            self.renderer.log_event_training("TrainStep".to_string());
                        }
                        RLEvent::EnvStep(item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self.metrics.update_env_step(&item, &metadata);
                            self.process_update_train(update);
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.update_split(item.progress.items_processed);
                                logger.log_event_training("EnvStep".to_string());
                            }
                            self.renderer.update_split(item.progress.items_processed);
                            self.renderer.log_event_training("EnvStep".to_string());
                        }
                        RLEvent::EpisodeEnd(item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self
                                .metrics
                                .update_episode_end(&item, &metadata);
                            self.process_update_train(update);
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.log_event_training("EpisodeEnd".to_string());
                            }
                            self.renderer.log_event_training("EpisodeEnd".to_string());
                        }
                        RLEvent::End(learner_summary) => {
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.end();
                            }
                            self.renderer.end();
                            self.renderer.on_train_end(learner_summary).ok();
                        }
                    }
                }
                fn process_valid(&mut self, event: AgentEvaluationEvent<ES>) {
                    match event {
                        AgentEvaluationEvent::Start(num_episodes) => {
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.start_split("valid", num_episodes);
                            }
                            self.renderer.start_split("valid", num_episodes);
                        }
                        AgentEvaluationEvent::EnvStep(item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self
                                .metrics
                                .update_env_step_valid(&item, &metadata);
                            self.process_update_valid(update);
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.log_event_training("EnvStep".to_string());
                            }
                            self.renderer.log_event_training("EnvStep".to_string());
                        }
                        AgentEvaluationEvent::EpisodeEnd(item) => {
                            let item = item.sync();
                            let metadata = (&item).into();
                            let update = self
                                .metrics
                                .update_episode_end_valid(&item, &metadata);
                            self.process_update_valid(update);
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.update_split(item.progress.items_processed);
                                logger.log_event_training("EpisodeEnd".to_string());
                            }
                            self.renderer.update_split(item.progress.items_processed);
                            self.renderer.log_event_training("EpisodeEnd".to_string());
                        }
                        AgentEvaluationEvent::End => {
                            if let Some(logger) = &mut self.training_progress_logger {
                                logger.end_split();
                            }
                            self.renderer.end_split();
                        }
                    }
                }
                fn renderer(self) -> Box<dyn MetricsRenderer> {
                    self.renderer
                }
            }
        }
        pub use base::*;
        pub(crate) use full::*;
        pub(crate) use metrics::*;
        pub(crate) use rl_metrics::*;
        pub use rl_processor::*;
        pub use async_wrapper::{AsyncProcessorEvaluation, AsyncProcessorTraining};
    }
    pub use crate::metric::classification::ClassReduction;
    pub use processor::ItemLazy;
}
pub use metric::processor::*;
mod learner {
    mod rl {
        mod checkpointer {
            use burn_core::tensor::Device;
            use burn_rl::{Policy, PolicyLearner, PolicyState};
            use crate::RLAgentRecord;
            use crate::{
                RLComponentsTypes, RLPolicyRecord, checkpoint::Checkpointer,
                checkpoint::{
                    AsyncCheckpointer, CheckpointingAction, CheckpointingStrategy,
                },
                metric::store::EventStoreClient,
            };
            /// Used to create, delete, or load checkpoints of the training process.
            pub struct RLCheckpointer<RLC: RLComponentsTypes> {
                policy: AsyncCheckpointer<RLPolicyRecord<RLC>>,
                learning_agent: AsyncCheckpointer<RLAgentRecord<RLC>>,
                strategy: Box<dyn CheckpointingStrategy>,
            }
            impl<RLC: RLComponentsTypes> RLCheckpointer<RLC> {
                ///Constructs a new `RLCheckpointer`.
                pub fn new(
                    policy: AsyncCheckpointer<RLPolicyRecord<RLC>>,
                    learning_agent: AsyncCheckpointer<RLAgentRecord<RLC>>,
                    strategy: Box<dyn CheckpointingStrategy>,
                ) -> Self {
                    RLCheckpointer {
                        policy: policy,
                        learning_agent: learning_agent,
                        strategy: strategy,
                    }
                }
            }
            impl<RLC: RLComponentsTypes> RLCheckpointer<RLC> {
                /// Create checkpoint for the training process.
                pub fn checkpoint(
                    &mut self,
                    policy: &RLC::PolicyState,
                    learning_agent: &RLC::LearningAgent,
                    epoch: usize,
                    store: &EventStoreClient,
                ) {
                    let actions = self.strategy.checkpointing(epoch, store);
                    for action in actions {
                        match action {
                            CheckpointingAction::Delete(epoch) => {
                                self.policy
                                    .delete(epoch)
                                    .expect("Can delete policy checkpoint.");
                                self.learning_agent
                                    .delete(epoch)
                                    .expect("Can delete learning agent checkpoint.")
                            }
                            CheckpointingAction::Save => {
                                self.policy
                                    .save(epoch, policy.clone().into_record())
                                    .expect("Can save policy checkpoint.");
                                self.learning_agent
                                    .save(epoch, learning_agent.record())
                                    .expect("Can save learning agent checkpoint.");
                            }
                        }
                    }
                }
                /// Load a training checkpoint.
                pub fn load_checkpoint(
                    &self,
                    learning_agent: RLC::LearningAgent,
                    device: &Device,
                    epoch: usize,
                ) -> RLC::LearningAgent {
                    let record = self
                        .policy
                        .restore(epoch, device)
                        .expect("Can load model checkpoint.");
                    let policy = learning_agent.policy().load_record(record);
                    let record = self
                        .learning_agent
                        .restore(epoch, device)
                        .expect("Can load learning agent checkpoint.");
                    let mut learning_agent = learning_agent.load_record(record);
                    learning_agent.update_policy(policy);
                    learning_agent
                }
            }
        }
        mod components {
            use std::marker::PhantomData;
            use burn_rl::{
                Batchable, Environment, EnvironmentInit, Policy, PolicyLearner,
                PolicyState, ToAction, ToObservation,
            };
            use crate::{AgentEvaluationEvent, AsyncProcessorTraining, ItemLazy, RLEvent};
            /// All components used by the reinforcement learning paradigm, grouped in one trait.
            pub trait RLComponentsTypes {
                /// The learning environment.
                type Env: Environment<State = Self::State, Action = Self::Action>
                    + 'static;
                /// Specifies how to initialize the environment.
                type EnvInit: EnvironmentInit<Self::Env> + Send + 'static;
                /// The type of the environment state.
                type State: ToObservation<<Self::Policy as Policy>::Observation>
                    + Clone
                    + Send
                    + 'static;
                /// The type of the environment action.
                type Action: From<<Self::Policy as Policy>::Action>
                    + ToAction<<Self::Policy as Policy>::Action>
                    + Clone
                    + Send
                    + 'static;
                /// The policy used to take actions in the environment.
                type Policy: Policy<
                        Observation = Self::PolicyObs,
                        ActionDistribution = Self::PolicyAD,
                        Action = Self::PolicyAction,
                        ActionContext = Self::ActionContext,
                        PolicyState = Self::PolicyState,
                    >
                    + Send
                    + 'static;
                /// The policy's observation type.
                type PolicyObs: Clone + Send + Batchable + 'static;
                /// The policy's action distribution type.
                type PolicyAD: Clone + Send + Batchable;
                /// The policy's action type.
                type PolicyAction: Clone + Send + Batchable;
                /// Additional data as context for an agent's action.
                type ActionContext: ItemLazy + Clone + Send + 'static;
                /// The state of the parameterized policy.
                type PolicyState: Clone + Send + PolicyState + 'static;
                /// The learning agent.
                type LearningAgent: PolicyLearner<
                        TrainContext = Self::TrainingOutput,
                        InnerPolicy = Self::Policy,
                    >
                    + Send
                    + 'static;
                /// The output data of a training step.
                type TrainingOutput: ItemLazy + Clone + Send;
            }
            /// Concrete type that implements the [RLComponentsTypes](RLComponentsTypes) trait.
            pub struct RLComponentsMarker<E, EI, A> {
                _env: PhantomData<E>,
                _env_init: PhantomData<EI>,
                _agent: PhantomData<A>,
            }
            impl<E, EI, A> RLComponentsTypes for RLComponentsMarker<E, EI, A>
            where
                E: Environment + 'static,
                EI: EnvironmentInit<E> + Send + 'static,
                A: PolicyLearner + Send + 'static,
                A::TrainContext: ItemLazy + Clone + Send,
                A::InnerPolicy: Policy + Send,
                <A::InnerPolicy as Policy>::Observation: Batchable + Clone + Send,
                <A::InnerPolicy as Policy>::ActionDistribution: Batchable + Clone + Send,
                <A::InnerPolicy as Policy>::Action: Batchable + Clone + Send,
                <A::InnerPolicy as Policy>::ActionContext: ItemLazy + Clone + Send
                    + 'static,
                <A::InnerPolicy as Policy>::PolicyState: Clone + Send,
                E::State: ToObservation<<A::InnerPolicy as Policy>::Observation> + Clone
                    + Send + 'static,
                E::Action: From<<A::InnerPolicy as Policy>::Action>
                    + ToAction<<A::InnerPolicy as Policy>::Action> + Clone + Send
                    + 'static,
            {
                type Env = E;
                type EnvInit = EI;
                type LearningAgent = A;
                type Policy = A::InnerPolicy;
                type PolicyObs = <A::InnerPolicy as Policy>::Observation;
                type PolicyAD = <A::InnerPolicy as Policy>::ActionDistribution;
                type PolicyAction = <A::InnerPolicy as Policy>::Action;
                type ActionContext = <A::InnerPolicy as Policy>::ActionContext;
                type PolicyState = <A::InnerPolicy as Policy>::PolicyState;
                type TrainingOutput = A::TrainContext;
                type State = E::State;
                type Action = E::Action;
            }
            pub(crate) type RlPolicy<RLC> = <<RLC as RLComponentsTypes>::LearningAgent as PolicyLearner>::InnerPolicy;
            /// The event processor type for reinforcement learning.
            pub type RLEventProcessorType<RLC> = AsyncProcessorTraining<
                RLEvent<
                    <RLC as RLComponentsTypes>::TrainingOutput,
                    <RLC as RLComponentsTypes>::ActionContext,
                >,
                AgentEvaluationEvent<<RLC as RLComponentsTypes>::ActionContext>,
            >;
            /// The record of the policy.
            pub type RLPolicyRecord<RLC> = <<<RLC as RLComponentsTypes>::Policy as Policy>::PolicyState as PolicyState>::Record;
            /// The record of the learning agent.
            pub type RLAgentRecord<RLC> = <<RLC as RLComponentsTypes>::LearningAgent as PolicyLearner>::Record;
        }
        mod env_runner {
            mod async_runner {
                use rand::prelude::SliceRandom;
                use std::{
                    sync::mpsc::{Receiver, Sender},
                    thread::spawn,
                };
                use burn_core::{Tensor, data::dataloader::Progress, tensor::Device};
                use burn_rl::Policy;
                use burn_rl::Transition;
                use burn_rl::{AsyncPolicy, Environment};
                use burn_rl::{EnvironmentInit, ToObservation};
                use crate::{
                    AgentEnvLoop, AgentEvaluationEvent, EpisodeSummary, EvaluationItem,
                    EventProcessorTraining, Interrupter, RLComponentsTypes, RLEvent,
                    RLEventProcessorType, RLTimeStep, RLTrajectory, RlPolicy, TimeStep,
                    Trajectory,
                };
                enum RequestMessage {
                    Step(),
                    Episode(),
                }
                /// Configuration for an async agent/environment loop.
                pub struct AsyncAgentEnvLoopConfig {
                    /// If the loop is used for evaluation (as opposed to training).
                    pub eval: bool,
                    /// If the agent should take action deterministically.
                    pub deterministic: bool,
                    /// An arbitrary ID for the loop.
                    pub id: usize,
                }
                /// An asynchronous agent/environement interface.
                pub struct AgentEnvAsyncLoop<RLC: RLComponentsTypes> {
                    eval: bool,
                    agent: AsyncPolicy<RlPolicy<RLC>>,
                    transition_receiver: Receiver<RLTimeStep<RLC>>,
                    trajectory_receiver: Receiver<RLTrajectory<RLC>>,
                    request_sender: Sender<RequestMessage>,
                    device: Device,
                }
                impl<RLC: RLComponentsTypes> AgentEnvAsyncLoop<RLC> {
                    /// Create a new asynchronous runner.
                    ///
                    /// # Arguments
                    ///
                    /// * `env_init` - A function returning an environment instance.
                    /// * `agent` - An [AsyncPolicy](AsyncPolicy) taking actions in the loop.
                    /// * `config` - An [AsyncAgentEnvLoopConfig](AsyncAgentEnvLoopConfig).
                    /// * `transition_sender` - Optional sender for transitions if you want to drive the requests from outside of the loop instance.
                    /// * `trajectory_sender` - Optional sender for trajectories if you want to drive the requests from outside of the loop instance.
                    ///
                    /// # Returns
                    ///
                    /// An async Agent/Environement loop.
                    pub fn new(
                        env_init: RLC::EnvInit,
                        agent: AsyncPolicy<RlPolicy<RLC>>,
                        config: AsyncAgentEnvLoopConfig,
                        transition_device: &Device,
                        transition_sender: Option<Sender<RLTimeStep<RLC>>>,
                        trajectory_sender: Option<Sender<RLTrajectory<RLC>>>,
                    ) -> Self {
                        let (loop_transition_sender, transition_receiver) = std::sync::mpsc::channel();
                        let (loop_trajectory_sender, trajectory_receiver) = std::sync::mpsc::channel();
                        let (request_sender, request_receiver) = std::sync::mpsc::channel();
                        let loop_transition_sender = transition_sender
                            .unwrap_or(loop_transition_sender);
                        let loop_trajectory_sender = trajectory_sender
                            .unwrap_or(loop_trajectory_sender);
                        let device = transition_device.clone();
                        let mut loop_agent = agent.clone().to_device(transition_device);
                        let eval = config.eval;
                        let mut current_steps = ::alloc::vec::Vec::new();
                        let mut current_reward = 0.0;
                        let mut step_num = 0;
                        spawn(move || {
                            let mut env = env_init.init();
                            env.reset();
                            let mut request_episode = false;
                            loop {
                                let state = env.state();
                                let (action, context) = loop_agent
                                    .action(
                                        state.clone().to_observation(&device),
                                        config.deterministic,
                                    );
                                let env_action = RLC::Action::from(action);
                                let step_result = env.step(env_action.clone());
                                current_reward += step_result.reward;
                                step_num += 1;
                                let transition = Transition::new(
                                    state.clone(),
                                    step_result.next_state,
                                    env_action,
                                    Tensor::from_data([step_result.reward], &device),
                                    Tensor::from_data(
                                        [(step_result.done || step_result.truncated) as i32 as f64],
                                        &device,
                                    ),
                                );
                                if !request_episode {
                                    loop_agent.decrement_agents(1);
                                    let request = match request_receiver.recv() {
                                        Ok(req) => req,
                                        Err(err) => {
                                            {
                                                {
                                                    let lvl = ::log::Level::Error;
                                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                                        && lvl <= ::log::max_level()
                                                    {
                                                        ::log::__private_api::log(
                                                            { ::log::__private_api::GlobalLogger },
                                                            format_args!("Error in env runner : {0}", err),
                                                            lvl,
                                                            &(
                                                                "burn_train::learner::rl::env_runner::async_runner",
                                                                "burn_train::learner::rl::env_runner::async_runner",
                                                                ::log::__private_api::loc(),
                                                            ),
                                                            (),
                                                        );
                                                    }
                                                }
                                            };
                                            break;
                                        }
                                    };
                                    loop_agent.increment_agents(1);
                                    match request {
                                        RequestMessage::Step() => {}
                                        RequestMessage::Episode() => request_episode = true,
                                    }
                                }
                                let time_step = TimeStep {
                                    env_id: config.id,
                                    transition,
                                    done: step_result.done,
                                    ep_len: step_num,
                                    cum_reward: current_reward,
                                    action_context: context[0].clone(),
                                };
                                current_steps.push(time_step.clone());
                                if !request_episode
                                    && let Err(err) = loop_transition_sender.send(time_step)
                                {
                                    {
                                        {
                                            let lvl = ::log::Level::Error;
                                            if lvl <= ::log::STATIC_MAX_LEVEL
                                                && lvl <= ::log::max_level()
                                            {
                                                ::log::__private_api::log(
                                                    { ::log::__private_api::GlobalLogger },
                                                    format_args!("Error in env runner : {0}", err),
                                                    lvl,
                                                    &(
                                                        "burn_train::learner::rl::env_runner::async_runner",
                                                        "burn_train::learner::rl::env_runner::async_runner",
                                                        ::log::__private_api::loc(),
                                                    ),
                                                    (),
                                                );
                                            }
                                        }
                                    };
                                    break;
                                }
                                if step_result.done || step_result.truncated {
                                    if request_episode {
                                        request_episode = false;
                                        loop_trajectory_sender
                                            .send(Trajectory {
                                                timesteps: current_steps.clone(),
                                            })
                                            .expect("Can send trajectory to main thread.");
                                    }
                                    current_steps.clear();
                                    env.reset();
                                    current_reward = 0.;
                                    step_num = 0;
                                }
                            }
                        });
                        Self {
                            eval,
                            agent,
                            transition_receiver,
                            trajectory_receiver,
                            request_sender,
                            device: transition_device.clone(),
                        }
                    }
                }
                impl<RLC> AgentEnvLoop<RLC> for AgentEnvAsyncLoop<RLC>
                where
                    RLC: RLComponentsTypes,
                {
                    fn run_steps(
                        &mut self,
                        num_steps: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        progress: &mut Progress,
                    ) -> Vec<RLTimeStep<RLC>> {
                        let mut items = ::alloc::vec::Vec::new();
                        for _ in 0..num_steps {
                            self.request_sender
                                .send(RequestMessage::Step())
                                .expect("Can request transitions.");
                            let transition = self
                                .transition_receiver
                                .recv()
                                .expect("Can receive transitions.");
                            items.push(transition.clone());
                            if !self.eval {
                                progress.items_processed += 1;
                                processor
                                    .process_train(
                                        RLEvent::EnvStep(
                                            EvaluationItem::new(
                                                transition.action_context,
                                                progress.clone(),
                                                None,
                                            ),
                                        ),
                                    );
                                if transition.done {
                                    processor
                                        .process_train(
                                            RLEvent::EpisodeEnd(
                                                EvaluationItem::new(
                                                    EpisodeSummary {
                                                        episode_length: transition.ep_len,
                                                        cum_reward: transition.cum_reward,
                                                    },
                                                    progress.clone(),
                                                    None,
                                                ),
                                            ),
                                        );
                                }
                            }
                            if interrupter.should_stop() {
                                break;
                            }
                        }
                        items
                    }
                    fn run_episodes(
                        &mut self,
                        num_episodes: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        _progress: &mut Progress,
                    ) -> Vec<RLTrajectory<RLC>> {
                        let mut items = ::alloc::vec::Vec::new();
                        self.agent.increment_agents(1);
                        for episode_num in 0..num_episodes {
                            self.request_sender
                                .send(RequestMessage::Episode())
                                .expect("Can request episodes.");
                            let trajectory = self
                                .trajectory_receiver
                                .recv()
                                .expect("Main thread can receive trajectory.");
                            for (i, step) in trajectory.timesteps.iter().enumerate() {
                                if self.eval {
                                    processor
                                        .process_valid(
                                            AgentEvaluationEvent::EnvStep(
                                                EvaluationItem::new(
                                                    step.action_context.clone(),
                                                    Progress::new(i, i, Some("steps".to_string())),
                                                    None,
                                                ),
                                            ),
                                        );
                                    if step.done {
                                        processor
                                            .process_valid(
                                                AgentEvaluationEvent::EpisodeEnd(
                                                    EvaluationItem::new(
                                                        EpisodeSummary {
                                                            episode_length: step.ep_len,
                                                            cum_reward: step.cum_reward,
                                                        },
                                                        Progress::new(
                                                            episode_num + 1,
                                                            num_episodes,
                                                            Some("episodes".to_string()),
                                                        ),
                                                        None,
                                                    ),
                                                ),
                                            );
                                    }
                                } else {
                                    processor
                                        .process_train(
                                            RLEvent::EnvStep(
                                                EvaluationItem::new(
                                                    step.action_context.clone(),
                                                    Progress::new(i, i, Some("steps".to_string())),
                                                    None,
                                                ),
                                            ),
                                        );
                                    if step.done {
                                        processor
                                            .process_train(
                                                RLEvent::EpisodeEnd(
                                                    EvaluationItem::new(
                                                        EpisodeSummary {
                                                            episode_length: step.ep_len,
                                                            cum_reward: step.cum_reward,
                                                        },
                                                        Progress::new(
                                                            episode_num + 1,
                                                            num_episodes,
                                                            Some("episodes".to_string()),
                                                        ),
                                                        None,
                                                    ),
                                                ),
                                            );
                                    }
                                }
                            }
                            items.push(trajectory);
                            if interrupter.should_stop() {
                                break;
                            }
                        }
                        self.agent.decrement_agents(1);
                        items
                    }
                    fn update_policy(&mut self, update: RLC::PolicyState) {
                        self.agent.update(update);
                        self.agent.clone().to_device(&self.device);
                    }
                    fn policy(&self) -> RLC::PolicyState {
                        self.agent.state()
                    }
                    fn device(&self) -> Device {
                        self.device.clone()
                    }
                }
                /// An asynchronous runner for multiple agent/environement interfaces.
                pub struct MultiAgentEnvLoop<RLC: RLComponentsTypes> {
                    num_envs: usize,
                    eval: bool,
                    agent: AsyncPolicy<RLC::Policy>,
                    transition_receiver: Receiver<RLTimeStep<RLC>>,
                    trajectory_receiver: Receiver<RLTrajectory<RLC>>,
                    request_senders: Vec<Sender<RequestMessage>>,
                    device: Device,
                }
                impl<RLC: RLComponentsTypes> MultiAgentEnvLoop<RLC> {
                    /// Create a new asynchronous runner for multiple agent/environement interfaces.
                    pub fn new(
                        num_envs: usize,
                        env_init: RLC::EnvInit,
                        agent: AsyncPolicy<RLC::Policy>,
                        eval: bool,
                        deterministic: bool,
                        device: &Device,
                    ) -> Self {
                        let (transition_sender, transition_receiver) = std::sync::mpsc::channel();
                        let (trajectory_sender, trajectory_receiver) = std::sync::mpsc::channel();
                        let mut request_senders = ::alloc::vec::Vec::new();
                        agent.increment_agents(num_envs);
                        for i in 0..num_envs {
                            let config = AsyncAgentEnvLoopConfig {
                                eval,
                                deterministic,
                                id: i,
                            };
                            let runner = AgentEnvAsyncLoop::<
                                RLC,
                            >::new(
                                env_init.clone(),
                                agent.clone(),
                                config,
                                &device.clone(),
                                Some(transition_sender.clone()),
                                Some(trajectory_sender.clone()),
                            );
                            request_senders.push(runner.request_sender.clone());
                        }
                        request_senders
                            .iter()
                            .for_each(|s| {
                                s.send(RequestMessage::Step())
                                    .expect("Main thread can send step requests.")
                            });
                        Self {
                            num_envs,
                            eval,
                            agent: agent.clone(),
                            transition_receiver,
                            trajectory_receiver,
                            request_senders,
                            device: device.clone(),
                        }
                    }
                }
                impl<RLC> AgentEnvLoop<RLC> for MultiAgentEnvLoop<RLC>
                where
                    RLC: RLComponentsTypes,
                {
                    fn run_steps(
                        &mut self,
                        num_steps: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        progress: &mut Progress,
                    ) -> Vec<RLTimeStep<RLC>> {
                        let mut items = ::alloc::vec::Vec::new();
                        for _ in 0..num_steps {
                            let transition = self
                                .transition_receiver
                                .recv()
                                .expect("Can receive transitions.");
                            items.push(transition.clone());
                            self.request_senders[transition.env_id]
                                .send(RequestMessage::Step())
                                .expect("Main thread can request steps.");
                            if !self.eval {
                                progress.items_processed += 1;
                                processor
                                    .process_train(
                                        RLEvent::EnvStep(
                                            EvaluationItem::new(
                                                transition.action_context,
                                                progress.clone(),
                                                None,
                                            ),
                                        ),
                                    );
                                if transition.done {
                                    processor
                                        .process_train(
                                            RLEvent::EpisodeEnd(
                                                EvaluationItem::new(
                                                    EpisodeSummary {
                                                        episode_length: transition.ep_len,
                                                        cum_reward: transition.cum_reward,
                                                    },
                                                    progress.clone(),
                                                    None,
                                                ),
                                            ),
                                        );
                                }
                            }
                            if interrupter.should_stop() {
                                break;
                            }
                        }
                        items
                    }
                    fn update_policy(&mut self, update: RLC::PolicyState) {
                        self.agent.update(update);
                        self.agent.clone().to_device(&self.device);
                    }
                    fn run_episodes(
                        &mut self,
                        num_episodes: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        _progress: &mut Progress,
                    ) -> Vec<RLTrajectory<RLC>> {
                        let mut idx = ::alloc::vec::Vec::new();
                        if num_episodes < self.num_envs {
                            let mut rng = rand::rng();
                            let mut vec: Vec<usize> = (0..self.num_envs).collect();
                            vec.shuffle(&mut rng);
                            idx = vec.into_iter().take(num_episodes).collect();
                        } else {
                            idx = (0..self.num_envs).collect();
                        }
                        let num_requests = self.num_envs.min(num_episodes);
                        idx.into_iter()
                            .for_each(|i| {
                                self.request_senders[i]
                                    .send(RequestMessage::Episode())
                                    .expect("Main thread can request steps.");
                            });
                        let mut items = ::alloc::vec::Vec::new();
                        for episode_num in 0..num_episodes {
                            let trajectory = self
                                .trajectory_receiver
                                .recv()
                                .expect("Can receive trajectory.");
                            items.push(trajectory.clone());
                            if items.len() + num_requests <= num_episodes {
                                self.request_senders[trajectory.timesteps[0].env_id]
                                    .send(RequestMessage::Episode())
                                    .expect("Main thread can request steps.");
                            }
                            for (i, step) in trajectory.timesteps.iter().enumerate() {
                                if self.eval {
                                    processor
                                        .process_valid(
                                            AgentEvaluationEvent::EnvStep(
                                                EvaluationItem::new(
                                                    step.action_context.clone(),
                                                    Progress::new(i, i, Some("steps".to_string())),
                                                    None,
                                                ),
                                            ),
                                        );
                                    if step.done {
                                        processor
                                            .process_valid(
                                                AgentEvaluationEvent::EpisodeEnd(
                                                    EvaluationItem::new(
                                                        EpisodeSummary {
                                                            episode_length: step.ep_len,
                                                            cum_reward: step.cum_reward,
                                                        },
                                                        Progress::new(
                                                            episode_num + 1,
                                                            num_episodes,
                                                            Some("episodes".to_string()),
                                                        ),
                                                        None,
                                                    ),
                                                ),
                                            );
                                    }
                                } else {
                                    processor
                                        .process_train(
                                            RLEvent::EnvStep(
                                                EvaluationItem::new(
                                                    step.action_context.clone(),
                                                    Progress::new(i, i, Some("steps".to_string())),
                                                    None,
                                                ),
                                            ),
                                        );
                                    if step.done {
                                        processor
                                            .process_train(
                                                RLEvent::EpisodeEnd(
                                                    EvaluationItem::new(
                                                        EpisodeSummary {
                                                            episode_length: step.ep_len,
                                                            cum_reward: step.cum_reward,
                                                        },
                                                        Progress::new(
                                                            episode_num + 1,
                                                            num_episodes,
                                                            Some("episodes".to_string()),
                                                        ),
                                                        None,
                                                    ),
                                                ),
                                            );
                                    }
                                }
                            }
                            if interrupter.should_stop() {
                                break;
                            }
                        }
                        items
                    }
                    fn policy(&self) -> RLC::PolicyState {
                        self.agent.state()
                    }
                    fn device(&self) -> Device {
                        self.device.clone()
                    }
                }
            }
            mod base {
                use burn_core::Tensor;
                use burn_core::data::dataloader::Progress;
                use burn_core::tensor::Device;
                use burn_rl::Policy;
                use burn_rl::ToObservation;
                use burn_rl::Transition;
                use burn_rl::{Environment, EnvironmentInit};
                use crate::RLEvent;
                use crate::{
                    AgentEvaluationEvent, EpisodeSummary, EvaluationItem,
                    EventProcessorTraining, RLEventProcessorType,
                };
                use crate::{Interrupter, RLComponentsTypes};
                /// A trajectory, i.e. a list of ordered [TimeStep](TimeStep).
                pub struct Trajectory<S, A, C> {
                    /// A list of ordered [TimeStep](TimeStep)s.
                    pub timesteps: Vec<TimeStep<S, A, C>>,
                }
                #[automatically_derived]
                impl<
                    S: ::core::clone::Clone,
                    A: ::core::clone::Clone,
                    C: ::core::clone::Clone,
                > ::core::clone::Clone for Trajectory<S, A, C> {
                    #[inline]
                    fn clone(&self) -> Trajectory<S, A, C> {
                        Trajectory {
                            timesteps: ::core::clone::Clone::clone(&self.timesteps),
                        }
                    }
                }
                impl<S, A, C> Trajectory<S, A, C> {
                    ///Constructs a new `Trajectory`.
                    pub fn new(timesteps: Vec<TimeStep<S, A, C>>) -> Self {
                        Trajectory { timesteps: timesteps }
                    }
                }
                /// A timestep debscribing an iteration of the state/decision process.
                pub struct TimeStep<S, A, C> {
                    /// The environment id.
                    pub env_id: usize,
                    /// The [burn_rl::Transition](burn_rl::Transition).
                    pub transition: Transition<S, A>,
                    /// True if the environment reaches a terminal state.
                    pub done: bool,
                    /// The running length of the current episode.
                    pub ep_len: usize,
                    /// The running cumulative reward.
                    pub cum_reward: f64,
                    /// The action's context for this timestep.
                    pub action_context: C,
                }
                #[automatically_derived]
                impl<
                    S: ::core::clone::Clone,
                    A: ::core::clone::Clone,
                    C: ::core::clone::Clone,
                > ::core::clone::Clone for TimeStep<S, A, C> {
                    #[inline]
                    fn clone(&self) -> TimeStep<S, A, C> {
                        TimeStep {
                            env_id: ::core::clone::Clone::clone(&self.env_id),
                            transition: ::core::clone::Clone::clone(&self.transition),
                            done: ::core::clone::Clone::clone(&self.done),
                            ep_len: ::core::clone::Clone::clone(&self.ep_len),
                            cum_reward: ::core::clone::Clone::clone(&self.cum_reward),
                            action_context: ::core::clone::Clone::clone(
                                &self.action_context,
                            ),
                        }
                    }
                }
                pub(crate) type RLTimeStep<RLC> = TimeStep<
                    <RLC as RLComponentsTypes>::State,
                    <RLC as RLComponentsTypes>::Action,
                    <RLC as RLComponentsTypes>::ActionContext,
                >;
                pub(crate) type RLTrajectory<RLC> = Trajectory<
                    <RLC as RLComponentsTypes>::State,
                    <RLC as RLComponentsTypes>::Action,
                    <RLC as RLComponentsTypes>::ActionContext,
                >;
                /// Trait for a structure that implements an agent/environement interface.
                pub trait AgentEnvLoop<RLC: RLComponentsTypes> {
                    /// Run a certain number of timesteps.
                    ///
                    /// # Arguments
                    ///
                    /// * `num_steps` - The number of time_steps to run.
                    /// * `processor` - An [crate::EventProcessorTraining](crate::EventProcessorTraining).
                    /// * `interrupter` - An [crate::Interrupter](crate::Interrupter).
                    /// * `num_steps` - The number of time_steps to run.
                    /// * `progress` - A mutable reference to the learning progress.
                    ///
                    /// # Returns
                    ///
                    /// A list of ordered timesteps.
                    fn run_steps(
                        &mut self,
                        num_steps: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        progress: &mut Progress,
                    ) -> Vec<RLTimeStep<RLC>>;
                    /// Run a certain number of episodes.
                    ///
                    /// # Arguments
                    ///
                    /// * `num_episodes` - The number of episodes to run.
                    /// * `processor` - An [crate::EventProcessorTraining](crate::EventProcessorTraining).
                    /// * `interrupter` - An [crate::Interrupter](crate::Interrupter).
                    /// * `progress` - A mutable reference to the learning progress.
                    ///
                    /// # Returns
                    ///
                    /// A list of ordered timesteps.
                    fn run_episodes(
                        &mut self,
                        num_episodes: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        progress: &mut Progress,
                    ) -> Vec<RLTrajectory<RLC>>;
                    /// Update the runner's agent.
                    fn update_policy(&mut self, update: RLC::PolicyState);
                    /// Get the state of the runner's agent.
                    fn policy(&self) -> RLC::PolicyState;
                    /// Returns the device on which the runner's agent runs.
                    fn device(&self) -> Device;
                }
                /// A simple, synchronized agent/environement interface.
                pub struct AgentEnvBaseLoop<RLC: RLComponentsTypes> {
                    env: RLC::Env,
                    eval: bool,
                    agent: RLC::Policy,
                    deterministic: bool,
                    current_reward: f64,
                    run_num: usize,
                    step_num: usize,
                    device: Device,
                }
                impl<RLC: RLComponentsTypes> AgentEnvBaseLoop<RLC> {
                    /// Create a new base runner.
                    pub fn new(
                        env_init: RLC::EnvInit,
                        agent: RLC::Policy,
                        eval: bool,
                        deterministic: bool,
                        device: &Device,
                    ) -> Self {
                        let agent = agent.to_device(device);
                        let mut env = env_init.init();
                        env.reset();
                        Self {
                            env,
                            eval,
                            agent: agent.clone(),
                            deterministic,
                            current_reward: 0.0,
                            run_num: 0,
                            step_num: 0,
                            device: device.clone(),
                        }
                    }
                }
                impl<RLC> AgentEnvLoop<RLC> for AgentEnvBaseLoop<RLC>
                where
                    RLC: RLComponentsTypes,
                {
                    fn run_steps(
                        &mut self,
                        num_steps: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        progress: &mut Progress,
                    ) -> Vec<RLTimeStep<RLC>> {
                        let mut items = ::alloc::vec::Vec::new();
                        let device = Default::default();
                        for _ in 0..num_steps {
                            let state = self.env.state();
                            let (action, context) = self
                                .agent
                                .action(
                                    state.clone().to_observation(&self.device),
                                    self.deterministic,
                                );
                            let step_result = self
                                .env
                                .step(RLC::Action::from(action.clone()));
                            self.current_reward += step_result.reward;
                            self.step_num += 1;
                            let transition = Transition::new(
                                state.clone(),
                                step_result.next_state,
                                RLC::Action::from(action),
                                Tensor::from_data([step_result.reward], &device),
                                Tensor::from_data(
                                    [(step_result.done || step_result.truncated) as i32 as f64],
                                    &device,
                                ),
                            );
                            items
                                .push(TimeStep {
                                    env_id: 0,
                                    transition,
                                    done: step_result.done,
                                    ep_len: self.step_num,
                                    cum_reward: self.current_reward,
                                    action_context: context[0].clone(),
                                });
                            if !self.eval {
                                progress.items_processed += 1;
                                processor
                                    .process_train(
                                        RLEvent::EnvStep(
                                            EvaluationItem::new(
                                                context[0].clone(),
                                                progress.clone(),
                                                None,
                                            ),
                                        ),
                                    );
                                if step_result.done {
                                    processor
                                        .process_train(
                                            RLEvent::EpisodeEnd(
                                                EvaluationItem::new(
                                                    EpisodeSummary {
                                                        episode_length: self.step_num,
                                                        cum_reward: self.current_reward,
                                                    },
                                                    progress.clone(),
                                                    None,
                                                ),
                                            ),
                                        );
                                }
                            }
                            if interrupter.should_stop() {
                                break;
                            }
                            if step_result.done || step_result.truncated {
                                self.env.reset();
                                self.current_reward = 0.;
                                self.step_num = 0;
                                self.run_num += 1;
                            }
                        }
                        items
                    }
                    fn update_policy(&mut self, update: RLC::PolicyState) {
                        self.agent.update(update);
                    }
                    fn run_episodes(
                        &mut self,
                        num_episodes: usize,
                        processor: &mut RLEventProcessorType<RLC>,
                        interrupter: &Interrupter,
                        progress: &mut Progress,
                    ) -> Vec<RLTrajectory<RLC>> {
                        self.env.reset();
                        let mut items = ::alloc::vec::Vec::new();
                        for ep in 0..num_episodes {
                            let mut steps = ::alloc::vec::Vec::new();
                            loop {
                                let step = self
                                    .run_steps(1, processor, interrupter, progress)[0]
                                    .clone();
                                steps.push(step.clone());
                                if self.eval {
                                    processor
                                        .process_valid(
                                            AgentEvaluationEvent::EnvStep(
                                                EvaluationItem::new(
                                                    step.action_context.clone(),
                                                    Progress::new(
                                                        steps.len() + 1,
                                                        steps.len() + 1,
                                                        Some("steps".to_string()),
                                                    ),
                                                    None,
                                                ),
                                            ),
                                        );
                                    if step.done {
                                        processor
                                            .process_valid(
                                                AgentEvaluationEvent::EpisodeEnd(
                                                    EvaluationItem::new(
                                                        EpisodeSummary {
                                                            episode_length: step.ep_len,
                                                            cum_reward: step.cum_reward,
                                                        },
                                                        Progress::new(
                                                            ep + 1,
                                                            num_episodes,
                                                            Some("episodes".to_string()),
                                                        ),
                                                        None,
                                                    ),
                                                ),
                                            );
                                    }
                                }
                                if interrupter.should_stop() || step.done {
                                    break;
                                }
                            }
                            items.push(Trajectory::new(steps));
                            if interrupter.should_stop() {
                                break;
                            }
                        }
                        items
                    }
                    fn policy(&self) -> RLC::PolicyState {
                        self.agent.state()
                    }
                    fn device(&self) -> Device {
                        self.device.clone()
                    }
                }
            }
            pub use async_runner::*;
            pub use base::*;
        }
        mod off_policy {
            use crate::{
                AgentEnvAsyncLoop, AgentEnvLoop, AsyncAgentEnvLoopConfig, EvaluationItem,
                EventProcessorTraining, MultiAgentEnvLoop, RLComponents,
                RLComponentsTypes, RLEvent, RLEventProcessorType, RLStrategy,
            };
            use burn_core::self as burn;
            use burn_core::{config::Config, data::dataloader::Progress};
            use burn_rl::{
                AsyncPolicy, Policy, PolicyLearner, SliceAccess, ToAction, ToObservation,
                TransitionBuffer,
            };
            /// Parameters of an on policy training with multi environments and double-batching.
            pub struct OffPolicyConfig {
                /// The number of environments to run simultaneously for experience collection.
                #[config(default = 1)]
                pub num_envs: usize,
                /// Number of environment state to accumulate before running one step of inference with the policy.
                /// Must be equal or less than the number of simultaneous environments.
                #[config(default = 1)]
                pub autobatch_size: usize,
                /// Max number of transitions stored in the replay buffer.
                #[config(default = 1024)]
                pub replay_buffer_size: usize,
                /// The number of steps to collect between each step of training.
                #[config(default = 1)]
                pub train_interval: usize,
                /// Number of optimization steps done each `train_interval`.
                #[config(default = 1)]
                pub train_steps: usize,
                /// The number of steps to collect between each evaluation.
                #[config(default = 10_000)]
                pub eval_interval: usize,
                /// The number of episodes to run for each evaluation.
                #[config(default = 1)]
                pub eval_episodes: usize,
                /// The number of transition to train on.
                #[config(default = 32)]
                pub train_batch_size: usize,
                /// Number of steps to collect before starting to train.
                #[config(default = 0)]
                pub warmup_steps: usize,
            }
            impl burn::config::Config for OffPolicyConfig {}
            impl OffPolicyConfig {
                ///Create a new instance of the config.
                ///# Arguments
                ///###### Default Arguments
                /**###### `num_envs`
*/
                /// The number of environments to run simultaneously for experience collection.
                ///- Defaults to `1`
                /**###### `autobatch_size`
*/
                /// Number of environment state to accumulate before running one step of inference with the policy.
                /// Must be equal or less than the number of simultaneous environments.
                ///- Defaults to `1`
                /**###### `replay_buffer_size`
*/
                /// Max number of transitions stored in the replay buffer.
                ///- Defaults to `1024`
                /**###### `train_interval`
*/
                /// The number of steps to collect between each step of training.
                ///- Defaults to `1`
                /**###### `train_steps`
*/
                /// Number of optimization steps done each `train_interval`.
                ///- Defaults to `1`
                /**###### `eval_interval`
*/
                /// The number of steps to collect between each evaluation.
                ///- Defaults to `10_000`
                /**###### `eval_episodes`
*/
                /// The number of episodes to run for each evaluation.
                ///- Defaults to `1`
                /**###### `train_batch_size`
*/
                /// The number of transition to train on.
                ///- Defaults to `32`
                /**###### `warmup_steps`
*/
                /// Number of steps to collect before starting to train.
                ///- Defaults to `0`
                #[allow(clippy::too_many_arguments)]
                pub fn new() -> Self {
                    Self {
                        num_envs: 1,
                        autobatch_size: 1,
                        replay_buffer_size: 1024,
                        train_interval: 1,
                        train_steps: 1,
                        eval_interval: 10_000,
                        eval_episodes: 1,
                        train_batch_size: 32,
                        warmup_steps: 0,
                    }
                }
            }
            impl OffPolicyConfig {
                /**Sets the value for the field [`num_envs`](Self::num_envs).

*/
                /// The number of environments to run simultaneously for experience collection.
                ///- Defaults to `1`
                pub fn with_num_envs(mut self, num_envs: usize) -> Self {
                    self.num_envs = num_envs;
                    self
                }
                /**Sets the value for the field [`autobatch_size`](Self::autobatch_size).

*/
                /// Number of environment state to accumulate before running one step of inference with the policy.
                /// Must be equal or less than the number of simultaneous environments.
                ///- Defaults to `1`
                pub fn with_autobatch_size(mut self, autobatch_size: usize) -> Self {
                    self.autobatch_size = autobatch_size;
                    self
                }
                /**Sets the value for the field [`replay_buffer_size`](Self::replay_buffer_size).

*/
                /// Max number of transitions stored in the replay buffer.
                ///- Defaults to `1024`
                pub fn with_replay_buffer_size(
                    mut self,
                    replay_buffer_size: usize,
                ) -> Self {
                    self.replay_buffer_size = replay_buffer_size;
                    self
                }
                /**Sets the value for the field [`train_interval`](Self::train_interval).

*/
                /// The number of steps to collect between each step of training.
                ///- Defaults to `1`
                pub fn with_train_interval(mut self, train_interval: usize) -> Self {
                    self.train_interval = train_interval;
                    self
                }
                /**Sets the value for the field [`train_steps`](Self::train_steps).

*/
                /// Number of optimization steps done each `train_interval`.
                ///- Defaults to `1`
                pub fn with_train_steps(mut self, train_steps: usize) -> Self {
                    self.train_steps = train_steps;
                    self
                }
                /**Sets the value for the field [`eval_interval`](Self::eval_interval).

*/
                /// The number of steps to collect between each evaluation.
                ///- Defaults to `10_000`
                pub fn with_eval_interval(mut self, eval_interval: usize) -> Self {
                    self.eval_interval = eval_interval;
                    self
                }
                /**Sets the value for the field [`eval_episodes`](Self::eval_episodes).

*/
                /// The number of episodes to run for each evaluation.
                ///- Defaults to `1`
                pub fn with_eval_episodes(mut self, eval_episodes: usize) -> Self {
                    self.eval_episodes = eval_episodes;
                    self
                }
                /**Sets the value for the field [`train_batch_size`](Self::train_batch_size).

*/
                /// The number of transition to train on.
                ///- Defaults to `32`
                pub fn with_train_batch_size(mut self, train_batch_size: usize) -> Self {
                    self.train_batch_size = train_batch_size;
                    self
                }
                /**Sets the value for the field [`warmup_steps`](Self::warmup_steps).

*/
                /// Number of steps to collect before starting to train.
                ///- Defaults to `0`
                pub fn with_warmup_steps(mut self, warmup_steps: usize) -> Self {
                    self.warmup_steps = warmup_steps;
                    self
                }
            }
            impl burn::serde::Serialize for OffPolicyConfig {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where
                    S: burn::serde::Serializer,
                {
                    #[serde(crate = "burn::serde")]
                    struct OffPolicyConfigSerde {
                        num_envs: usize,
                        autobatch_size: usize,
                        replay_buffer_size: usize,
                        train_interval: usize,
                        train_steps: usize,
                        eval_interval: usize,
                        eval_episodes: usize,
                        train_batch_size: usize,
                        warmup_steps: usize,
                    }
                    #[doc(hidden)]
                    #[allow(
                        non_upper_case_globals,
                        unused_attributes,
                        unused_qualifications,
                        clippy::absolute_paths,
                    )]
                    const _: () = {
                        use burn::serde as _serde;
                        #[automatically_derived]
                        impl _serde::Serialize for OffPolicyConfigSerde {
                            fn serialize<__S>(
                                &self,
                                __serializer: __S,
                            ) -> _serde::__private228::Result<__S::Ok, __S::Error>
                            where
                                __S: _serde::Serializer,
                            {
                                let mut __serde_state = _serde::Serializer::serialize_struct(
                                    __serializer,
                                    "OffPolicyConfigSerde",
                                    false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "num_envs",
                                    &self.num_envs,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "autobatch_size",
                                    &self.autobatch_size,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "replay_buffer_size",
                                    &self.replay_buffer_size,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "train_interval",
                                    &self.train_interval,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "train_steps",
                                    &self.train_steps,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "eval_interval",
                                    &self.eval_interval,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "eval_episodes",
                                    &self.eval_episodes,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "train_batch_size",
                                    &self.train_batch_size,
                                )?;
                                _serde::ser::SerializeStruct::serialize_field(
                                    &mut __serde_state,
                                    "warmup_steps",
                                    &self.warmup_steps,
                                )?;
                                _serde::ser::SerializeStruct::end(__serde_state)
                            }
                        }
                    };
                    let serde_state = OffPolicyConfigSerde {
                        num_envs: self.num_envs.clone(),
                        autobatch_size: self.autobatch_size.clone(),
                        replay_buffer_size: self.replay_buffer_size.clone(),
                        train_interval: self.train_interval.clone(),
                        train_steps: self.train_steps.clone(),
                        eval_interval: self.eval_interval.clone(),
                        eval_episodes: self.eval_episodes.clone(),
                        train_batch_size: self.train_batch_size.clone(),
                        warmup_steps: self.warmup_steps.clone(),
                    };
                    serde_state.serialize(serializer)
                }
            }
            impl<'de> burn::serde::Deserialize<'de> for OffPolicyConfig {
                fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where
                    D: burn::serde::Deserializer<'de>,
                {
                    #[serde(crate = "burn::serde")]
                    struct OffPolicyConfigSerde {
                        num_envs: usize,
                        autobatch_size: usize,
                        replay_buffer_size: usize,
                        train_interval: usize,
                        train_steps: usize,
                        eval_interval: usize,
                        eval_episodes: usize,
                        train_batch_size: usize,
                        warmup_steps: usize,
                    }
                    #[doc(hidden)]
                    #[allow(
                        non_upper_case_globals,
                        unused_attributes,
                        unused_qualifications,
                        clippy::absolute_paths,
                    )]
                    const _: () = {
                        use burn::serde as _serde;
                        #[automatically_derived]
                        impl<'de> _serde::Deserialize<'de> for OffPolicyConfigSerde {
                            fn deserialize<__D>(
                                __deserializer: __D,
                            ) -> _serde::__private228::Result<Self, __D::Error>
                            where
                                __D: _serde::Deserializer<'de>,
                            {
                                #[allow(non_camel_case_types)]
                                #[doc(hidden)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __field2,
                                    __field3,
                                    __field4,
                                    __field5,
                                    __field6,
                                    __field7,
                                    __field8,
                                    __ignore,
                                }
                                #[doc(hidden)]
                                struct __FieldVisitor;
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private228::Ok(__Field::__field0),
                                            1u64 => _serde::__private228::Ok(__Field::__field1),
                                            2u64 => _serde::__private228::Ok(__Field::__field2),
                                            3u64 => _serde::__private228::Ok(__Field::__field3),
                                            4u64 => _serde::__private228::Ok(__Field::__field4),
                                            5u64 => _serde::__private228::Ok(__Field::__field5),
                                            6u64 => _serde::__private228::Ok(__Field::__field6),
                                            7u64 => _serde::__private228::Ok(__Field::__field7),
                                            8u64 => _serde::__private228::Ok(__Field::__field8),
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "num_envs" => _serde::__private228::Ok(__Field::__field0),
                                            "autobatch_size" => {
                                                _serde::__private228::Ok(__Field::__field1)
                                            }
                                            "replay_buffer_size" => {
                                                _serde::__private228::Ok(__Field::__field2)
                                            }
                                            "train_interval" => {
                                                _serde::__private228::Ok(__Field::__field3)
                                            }
                                            "train_steps" => _serde::__private228::Ok(__Field::__field4),
                                            "eval_interval" => {
                                                _serde::__private228::Ok(__Field::__field5)
                                            }
                                            "eval_episodes" => {
                                                _serde::__private228::Ok(__Field::__field6)
                                            }
                                            "train_batch_size" => {
                                                _serde::__private228::Ok(__Field::__field7)
                                            }
                                            "warmup_steps" => {
                                                _serde::__private228::Ok(__Field::__field8)
                                            }
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private228::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"num_envs" => _serde::__private228::Ok(__Field::__field0),
                                            b"autobatch_size" => {
                                                _serde::__private228::Ok(__Field::__field1)
                                            }
                                            b"replay_buffer_size" => {
                                                _serde::__private228::Ok(__Field::__field2)
                                            }
                                            b"train_interval" => {
                                                _serde::__private228::Ok(__Field::__field3)
                                            }
                                            b"train_steps" => {
                                                _serde::__private228::Ok(__Field::__field4)
                                            }
                                            b"eval_interval" => {
                                                _serde::__private228::Ok(__Field::__field5)
                                            }
                                            b"eval_episodes" => {
                                                _serde::__private228::Ok(__Field::__field6)
                                            }
                                            b"train_batch_size" => {
                                                _serde::__private228::Ok(__Field::__field7)
                                            }
                                            b"warmup_steps" => {
                                                _serde::__private228::Ok(__Field::__field8)
                                            }
                                            _ => _serde::__private228::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                #[automatically_derived]
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private228::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                #[doc(hidden)]
                                struct __Visitor<'de> {
                                    marker: _serde::__private228::PhantomData<
                                        OffPolicyConfigSerde,
                                    >,
                                    lifetime: _serde::__private228::PhantomData<&'de ()>,
                                }
                                #[automatically_derived]
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = OffPolicyConfigSerde;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private228::Formatter,
                                    ) -> _serde::__private228::fmt::Result {
                                        _serde::__private228::Formatter::write_str(
                                            __formatter,
                                            "struct OffPolicyConfigSerde",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        0usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field1 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        1usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field2 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        2usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field3 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        3usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field4 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        4usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field5 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        5usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field6 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        6usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field7 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        7usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        let __field8 = match _serde::de::SeqAccess::next_element::<
                                            usize,
                                        >(&mut __seq)? {
                                            _serde::__private228::Some(__value) => __value,
                                            _serde::__private228::None => {
                                                return _serde::__private228::Err(
                                                    _serde::de::Error::invalid_length(
                                                        8usize,
                                                        &"struct OffPolicyConfigSerde with 9 elements",
                                                    ),
                                                );
                                            }
                                        };
                                        _serde::__private228::Ok(OffPolicyConfigSerde {
                                            num_envs: __field0,
                                            autobatch_size: __field1,
                                            replay_buffer_size: __field2,
                                            train_interval: __field3,
                                            train_steps: __field4,
                                            eval_interval: __field5,
                                            eval_episodes: __field6,
                                            train_batch_size: __field7,
                                            warmup_steps: __field8,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private228::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field1: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field2: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field3: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field4: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field5: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field6: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field7: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        let mut __field8: _serde::__private228::Option<usize> = _serde::__private228::None;
                                        while let _serde::__private228::Some(__key) = _serde::de::MapAccess::next_key::<
                                            __Field,
                                        >(&mut __map)? {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private228::Option::is_some(&__field0) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "num_envs",
                                                            ),
                                                        );
                                                    }
                                                    __field0 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private228::Option::is_some(&__field1) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "autobatch_size",
                                                            ),
                                                        );
                                                    }
                                                    __field1 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field2 => {
                                                    if _serde::__private228::Option::is_some(&__field2) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "replay_buffer_size",
                                                            ),
                                                        );
                                                    }
                                                    __field2 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field3 => {
                                                    if _serde::__private228::Option::is_some(&__field3) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "train_interval",
                                                            ),
                                                        );
                                                    }
                                                    __field3 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field4 => {
                                                    if _serde::__private228::Option::is_some(&__field4) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "train_steps",
                                                            ),
                                                        );
                                                    }
                                                    __field4 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field5 => {
                                                    if _serde::__private228::Option::is_some(&__field5) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "eval_interval",
                                                            ),
                                                        );
                                                    }
                                                    __field5 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field6 => {
                                                    if _serde::__private228::Option::is_some(&__field6) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "eval_episodes",
                                                            ),
                                                        );
                                                    }
                                                    __field6 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field7 => {
                                                    if _serde::__private228::Option::is_some(&__field7) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "train_batch_size",
                                                            ),
                                                        );
                                                    }
                                                    __field7 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                __Field::__field8 => {
                                                    if _serde::__private228::Option::is_some(&__field8) {
                                                        return _serde::__private228::Err(
                                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                                "warmup_steps",
                                                            ),
                                                        );
                                                    }
                                                    __field8 = _serde::__private228::Some(
                                                        _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                    );
                                                }
                                                _ => {
                                                    let _ = _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(&mut __map)?;
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private228::Some(__field0) => __field0,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("num_envs")?
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private228::Some(__field1) => __field1,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("autobatch_size")?
                                            }
                                        };
                                        let __field2 = match __field2 {
                                            _serde::__private228::Some(__field2) => __field2,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field(
                                                    "replay_buffer_size",
                                                )?
                                            }
                                        };
                                        let __field3 = match __field3 {
                                            _serde::__private228::Some(__field3) => __field3,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("train_interval")?
                                            }
                                        };
                                        let __field4 = match __field4 {
                                            _serde::__private228::Some(__field4) => __field4,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("train_steps")?
                                            }
                                        };
                                        let __field5 = match __field5 {
                                            _serde::__private228::Some(__field5) => __field5,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("eval_interval")?
                                            }
                                        };
                                        let __field6 = match __field6 {
                                            _serde::__private228::Some(__field6) => __field6,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("eval_episodes")?
                                            }
                                        };
                                        let __field7 = match __field7 {
                                            _serde::__private228::Some(__field7) => __field7,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("train_batch_size")?
                                            }
                                        };
                                        let __field8 = match __field8 {
                                            _serde::__private228::Some(__field8) => __field8,
                                            _serde::__private228::None => {
                                                _serde::__private228::de::missing_field("warmup_steps")?
                                            }
                                        };
                                        _serde::__private228::Ok(OffPolicyConfigSerde {
                                            num_envs: __field0,
                                            autobatch_size: __field1,
                                            replay_buffer_size: __field2,
                                            train_interval: __field3,
                                            train_steps: __field4,
                                            eval_interval: __field5,
                                            eval_episodes: __field6,
                                            train_batch_size: __field7,
                                            warmup_steps: __field8,
                                        })
                                    }
                                }
                                #[doc(hidden)]
                                const FIELDS: &'static [&'static str] = &[
                                    "num_envs",
                                    "autobatch_size",
                                    "replay_buffer_size",
                                    "train_interval",
                                    "train_steps",
                                    "eval_interval",
                                    "eval_episodes",
                                    "train_batch_size",
                                    "warmup_steps",
                                ];
                                _serde::Deserializer::deserialize_struct(
                                    __deserializer,
                                    "OffPolicyConfigSerde",
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private228::PhantomData::<
                                            OffPolicyConfigSerde,
                                        >,
                                        lifetime: _serde::__private228::PhantomData,
                                    },
                                )
                            }
                        }
                    };
                    let serde_state = OffPolicyConfigSerde::deserialize(deserializer)?;
                    Ok(OffPolicyConfig {
                        num_envs: serde_state.num_envs,
                        autobatch_size: serde_state.autobatch_size,
                        replay_buffer_size: serde_state.replay_buffer_size,
                        train_interval: serde_state.train_interval,
                        train_steps: serde_state.train_steps,
                        eval_interval: serde_state.eval_interval,
                        eval_episodes: serde_state.eval_episodes,
                        train_batch_size: serde_state.train_batch_size,
                        warmup_steps: serde_state.warmup_steps,
                    })
                }
            }
            impl Clone for OffPolicyConfig {
                fn clone(&self) -> Self {
                    Self {
                        num_envs: self.num_envs.clone(),
                        autobatch_size: self.autobatch_size.clone(),
                        replay_buffer_size: self.replay_buffer_size.clone(),
                        train_interval: self.train_interval.clone(),
                        train_steps: self.train_steps.clone(),
                        eval_interval: self.eval_interval.clone(),
                        eval_episodes: self.eval_episodes.clone(),
                        train_batch_size: self.train_batch_size.clone(),
                        warmup_steps: self.warmup_steps.clone(),
                    }
                }
            }
            impl core::fmt::Display for OffPolicyConfig {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.write_str(&burn::config::config_to_json(self))
                }
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for OffPolicyConfig {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    let names: &'static _ = &[
                        "num_envs",
                        "autobatch_size",
                        "replay_buffer_size",
                        "train_interval",
                        "train_steps",
                        "eval_interval",
                        "eval_episodes",
                        "train_batch_size",
                        "warmup_steps",
                    ];
                    let values: &[&dyn ::core::fmt::Debug] = &[
                        &self.num_envs,
                        &self.autobatch_size,
                        &self.replay_buffer_size,
                        &self.train_interval,
                        &self.train_steps,
                        &self.eval_interval,
                        &self.eval_episodes,
                        &self.train_batch_size,
                        &&self.warmup_steps,
                    ];
                    ::core::fmt::Formatter::debug_struct_fields_finish(
                        f,
                        "OffPolicyConfig",
                        names,
                        values,
                    )
                }
            }
            /// Off-policy reinforcement learning strategy with multi-env experience collection and double-batching.
            pub struct OffPolicyStrategy {
                config: OffPolicyConfig,
            }
            impl OffPolicyStrategy {
                /// Create a new off-policy base strategy.
                pub fn new(config: OffPolicyConfig) -> Self {
                    Self { config }
                }
            }
            impl<RLC> RLStrategy<RLC> for OffPolicyStrategy
            where
                RLC: RLComponentsTypes,
                RLC::PolicyObs: SliceAccess,
                RLC::PolicyAction: SliceAccess,
            {
                fn train_loop(
                    &self,
                    training_components: RLComponents<RLC>,
                    learner_agent: &mut RLC::LearningAgent,
                    starting_epoch: usize,
                    env_init: RLC::EnvInit,
                ) -> (RLC::Policy, RLEventProcessorType<RLC>) {
                    let mut event_processor = training_components.event_processor;
                    let mut checkpointer = training_components.checkpointer;
                    let num_steps_total = training_components.num_steps;
                    let inference_device = training_components.inference_device;
                    let mut env_runner = MultiAgentEnvLoop::<
                        RLC,
                    >::new(
                        self.config.num_envs,
                        env_init.clone(),
                        AsyncPolicy::new(
                            self.config.num_envs.min(self.config.autobatch_size),
                            learner_agent.policy(),
                        ),
                        false,
                        false,
                        &inference_device,
                    );
                    let runner_config = AsyncAgentEnvLoopConfig {
                        eval: true,
                        deterministic: true,
                        id: 0,
                    };
                    let mut env_runner_valid = AgentEnvAsyncLoop::<
                        RLC,
                    >::new(
                        env_init,
                        AsyncPolicy::new(1, learner_agent.policy()),
                        runner_config,
                        &inference_device,
                        None,
                        None,
                    );
                    let mut transition_buffer = TransitionBuffer::<
                        RLC::PolicyObs,
                        RLC::PolicyAction,
                    >::new(self.config.replay_buffer_size, &learner_agent.device());
                    let mut valid_next = self.config.eval_interval + starting_epoch - 1;
                    let mut progress = Progress {
                        items_processed: starting_epoch,
                        items_total: num_steps_total,
                        unit: Some("steps".to_string()),
                    };
                    let mut intermediary_update: Option<
                        <RLC::Policy as Policy>::PolicyState,
                    > = None;
                    while progress.items_processed < num_steps_total {
                        if training_components.interrupter.should_stop() {
                            let reason = training_components
                                .interrupter
                                .get_message()
                                .unwrap_or(String::from("Reason unknown"));
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!("Training interrupted: {0}", reason),
                                            lvl,
                                            &(
                                                "burn_train::learner::rl::off_policy",
                                                "burn_train::learner::rl::off_policy",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            break;
                        }
                        let previous_steps = progress.items_processed;
                        let items = env_runner
                            .run_steps(
                                self.config.train_interval,
                                &mut event_processor,
                                &training_components.interrupter,
                                &mut progress,
                            );
                        for item in &items {
                            let t = &item.transition;
                            let state: RLC::PolicyObs = t
                                .state
                                .clone()
                                .to_observation(&env_runner.device());
                            let next_state: RLC::PolicyObs = t
                                .next_state
                                .clone()
                                .to_observation(&env_runner.device());
                            let action: RLC::PolicyAction = t
                                .action
                                .clone()
                                .to_action(&env_runner.device());
                            let reward = t.reward.to_data().to_vec::<f32>().unwrap()[0];
                            let done = t.done.to_data().to_vec::<f32>().unwrap()[0]
                                > 0.5;
                            transition_buffer
                                .push(state, next_state, action, reward, done);
                        }
                        if transition_buffer.len() >= self.config.train_batch_size
                            && progress.items_processed >= self.config.warmup_steps
                        {
                            if let Some(ref u) = intermediary_update {
                                env_runner.update_policy(u.clone());
                            }
                            for _ in 0..self.config.train_steps {
                                let batch = transition_buffer
                                    .sample(self.config.train_batch_size);
                                let train_item = learner_agent.train(batch);
                                intermediary_update = Some(learner_agent.policy().state());
                                event_processor
                                    .process_train(
                                        RLEvent::TrainStep(
                                            EvaluationItem::new(train_item.item, progress.clone(), None),
                                        ),
                                    );
                            }
                        }
                        if valid_next > previous_steps
                            && valid_next <= progress.items_processed
                        {
                            event_processor
                                .process_valid(
                                    crate::AgentEvaluationEvent::Start(
                                        self.config.eval_episodes,
                                    ),
                                );
                            env_runner_valid
                                .update_policy(learner_agent.policy().state());
                            env_runner_valid
                                .run_episodes(
                                    self.config.eval_episodes,
                                    &mut event_processor,
                                    &training_components.interrupter,
                                    &mut progress,
                                );
                            if let Some(checkpointer) = &mut checkpointer {
                                checkpointer
                                    .checkpoint(
                                        &env_runner.policy(),
                                        learner_agent,
                                        valid_next,
                                        &training_components.event_store,
                                    );
                            }
                            valid_next += self.config.eval_interval;
                            event_processor
                                .process_valid(crate::AgentEvaluationEvent::End);
                        }
                    }
                    (learner_agent.policy(), event_processor)
                }
            }
        }
        mod output {
            use crate::{
                ItemLazy, metric::{Adaptor, CumulativeRewardInput, EpisodeLengthInput},
            };
            /// Summary of an episode.
            pub struct EpisodeSummary {
                /// The total length of the episode.
                pub episode_length: usize,
                /// The final cumulative reward.
                pub cum_reward: f64,
            }
            impl ItemLazy for EpisodeSummary {
                fn sync(self) -> Self {
                    self
                }
            }
            impl Adaptor<EpisodeLengthInput> for EpisodeSummary {
                fn adapt(&self) -> EpisodeLengthInput {
                    EpisodeLengthInput::new(self.episode_length as f64)
                }
            }
            impl Adaptor<CumulativeRewardInput> for EpisodeSummary {
                fn adapt(&self) -> CumulativeRewardInput {
                    CumulativeRewardInput::new(self.cum_reward)
                }
            }
        }
        mod paradigm {
            use crate::checkpoint::{
                AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy,
                FileCheckpointer, KeepLastNCheckpoints, MetricCheckpointingStrategy,
            };
            use crate::learner::base::Interrupter;
            use crate::logger::{FileMetricLogger, MetricLogger};
            use crate::metric::store::{
                Aggregate, Direction, EventStoreClient, LogEventStore, Split,
            };
            use crate::metric::{Adaptor, EpisodeLengthMetric, Metric, Numeric};
            use crate::renderer::{MetricsRenderer, default_renderer};
            use crate::{
                ApplicationLoggerInstaller, AsyncProcessorTraining,
                FileApplicationLoggerInstaller, ItemLazy, LearnerSummaryConfig,
                OffPolicyConfig, OffPolicyStrategy, RLAgentRecord, RLCheckpointer,
                RLComponents, RLComponentsMarker, RLComponentsTypes, RLEventProcessor,
                RLMetrics, RLPolicyRecord, RLStrategy,
            };
            use crate::{EpisodeSummary, RLStrategies};
            use burn_core::record::FileRecorder;
            use burn_core::tensor::Device;
            use burn_rl::{
                Batchable, Environment, EnvironmentInit, Policy, PolicyLearner,
                SliceAccess, ToAction, ToObservation,
            };
            use std::collections::BTreeSet;
            use std::path::{Path, PathBuf};
            use std::sync::Arc;
            /// Structure to configure and launch reinforcement learning trainings.
            pub struct RLTraining<RLC: RLComponentsTypes> {
                #[allow(clippy::type_complexity)]
                checkpointers: Option<
                    (
                        AsyncCheckpointer<RLPolicyRecord<RLC>>,
                        AsyncCheckpointer<RLAgentRecord<RLC>>,
                    ),
                >,
                num_steps: usize,
                checkpoint: Option<usize>,
                directory: PathBuf,
                grad_accumulation: Option<usize>,
                renderer: Option<Box<dyn MetricsRenderer + 'static>>,
                metrics: RLMetrics<RLC::TrainingOutput, RLC::ActionContext>,
                event_store: LogEventStore,
                interrupter: Interrupter,
                tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
                checkpointer_strategy: Box<dyn CheckpointingStrategy>,
                learning_strategy: RLStrategies<RLC>,
                summary_metrics: BTreeSet<String>,
                summary: bool,
                env_initializer: RLC::EnvInit,
                inference_device: Device,
            }
            impl<E, EI, A> RLTraining<RLComponentsMarker<E, EI, A>>
            where
                E: Environment + 'static,
                EI: EnvironmentInit<E> + Send + 'static,
                A: PolicyLearner + Send + 'static,
                A::TrainContext: ItemLazy + Clone + Send,
                A::InnerPolicy: Policy + Send,
                <A::InnerPolicy as Policy>::Observation: Batchable + Clone + Send,
                <A::InnerPolicy as Policy>::ActionDistribution: Batchable + Clone + Send,
                <A::InnerPolicy as Policy>::Action: Batchable + Clone + Send,
                <A::InnerPolicy as Policy>::ActionContext: ItemLazy + Clone + Send
                    + 'static,
                <A::InnerPolicy as Policy>::PolicyState: Clone + Send,
                E::State: ToObservation<<A::InnerPolicy as Policy>::Observation> + Clone
                    + Send + 'static,
                E::Action: From<<A::InnerPolicy as Policy>::Action>
                    + ToAction<<A::InnerPolicy as Policy>::Action> + Clone + Send
                    + 'static,
            {
                /// Creates a new runner for reinforcement learning.
                ///
                /// # Arguments
                ///
                /// * `directory` - The directory to save the checkpoints.
                /// * `env_init` - Specifies how to initialize the environment.
                pub fn new(directory: impl AsRef<Path>, env_initializer: EI) -> Self {
                    let directory = directory.as_ref().to_path_buf();
                    let experiment_log_file = directory.join("experiment.log");
                    Self {
                        num_steps: 1,
                        checkpoint: None,
                        checkpointers: None,
                        directory,
                        grad_accumulation: None,
                        metrics: RLMetrics::default(),
                        event_store: LogEventStore::default(),
                        renderer: None,
                        interrupter: Interrupter::new(),
                        tracing_logger: Some(
                            Box::new(
                                FileApplicationLoggerInstaller::new(experiment_log_file),
                            ),
                        ),
                        checkpointer_strategy: Box::new(
                            ComposedCheckpointingStrategy::builder()
                                .add(KeepLastNCheckpoints::new(2))
                                .add(
                                    MetricCheckpointingStrategy::new(
                                        &EpisodeLengthMetric::new(),
                                        Aggregate::Mean,
                                        Direction::Lowest,
                                        Split::Valid,
                                    ),
                                )
                                .build(),
                        ),
                        learning_strategy: RLStrategies::OffPolicyStrategy(
                            OffPolicyConfig::new(),
                        ),
                        summary_metrics: BTreeSet::new(),
                        summary: false,
                        env_initializer,
                        inference_device: Default::default(),
                    }
                }
            }
            impl<RLC: RLComponentsTypes + 'static> RLTraining<RLC> {
                /// Replace the default learning strategy (Off Policy learning) with the provided one.
                ///
                /// # Arguments
                ///
                /// * `training_strategy` - The training strategy.
                pub fn with_learning_strategy(
                    mut self,
                    learning_strategy: RLStrategies<RLC>,
                ) -> Self {
                    self.learning_strategy = learning_strategy;
                    self
                }
                /// Replace the default metric loggers with the provided ones.
                ///
                /// # Arguments
                ///
                /// * `logger` - The training logger.
                pub fn with_metric_logger<ML>(mut self, logger: ML) -> Self
                where
                    ML: MetricLogger + 'static,
                {
                    self.event_store.register_logger(logger);
                    self
                }
                /// Update the checkpointing_strategy.
                pub fn with_checkpointing_strategy<CS: CheckpointingStrategy + 'static>(
                    mut self,
                    strategy: CS,
                ) -> Self {
                    self.checkpointer_strategy = Box::new(strategy);
                    self
                }
                /// Replace the default CLI renderer with a custom one.
                ///
                /// # Arguments
                ///
                /// * `renderer` - The custom renderer.
                pub fn renderer<MR>(mut self, renderer: MR) -> Self
                where
                    MR: MetricsRenderer + 'static,
                {
                    self.renderer = Some(Box::new(renderer));
                    self
                }
                /// Register numerical metrics for a training step of the agent.
                pub fn metrics_train<Me: TrainMetricRegistration<RLC>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register textual metrics for a training step of the agent.
                pub fn text_metrics_train<Me: TrainTextMetricRegistration<RLC>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register numerical metrics for each action of the agent.
                pub fn metrics_agent<Me: AgentMetricRegistration<RLC>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register textual metrics for each action of the agent.
                pub fn text_metrics_agent<Me: AgentTextMetricRegistration<RLC>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register numerical metrics for a completed episode.
                pub fn metrics_episode<Me: EpisodeMetricRegistration<RLC>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register textual metrics for a completed episode.
                pub fn text_metrics_episode<Me: EpisodeTextMetricRegistration<RLC>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register a textual metric for a training step.
                pub fn text_metric_train<Me: Metric + 'static>(
                    mut self,
                    metric: Me,
                ) -> Self
                where
                    RLC::TrainingOutput: Adaptor<Me::Input>,
                {
                    self.metrics.register_text_metric_train(metric);
                    self
                }
                /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for a training step.
                pub fn metric_train<Me>(mut self, metric: Me) -> Self
                where
                    Me: Metric + Numeric + 'static,
                    RLC::TrainingOutput: Adaptor<Me::Input>,
                {
                    self.summary_metrics.insert(metric.name().to_string());
                    self.metrics.register_metric_train(metric);
                    self
                }
                /// Register a textual metric for each action taken by the agent.
                pub fn text_metric_agent<Me: Metric + 'static>(
                    mut self,
                    metric: Me,
                ) -> Self
                where
                    RLC::ActionContext: Adaptor<Me::Input>,
                {
                    self.metrics.register_text_metric_agent(metric.clone());
                    self.metrics.register_text_metric_agent_valid(metric);
                    self
                }
                /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for each action taken by the agent.
                pub fn metric_agent<Me>(mut self, metric: Me) -> Self
                where
                    Me: Metric + Numeric + 'static,
                    RLC::ActionContext: Adaptor<Me::Input>,
                {
                    self.summary_metrics.insert(metric.name().to_string());
                    self.metrics.register_agent_metric(metric.clone());
                    self.metrics.register_agent_metric_valid(metric);
                    self
                }
                /// Register a textual metric for a completed episode.
                pub fn text_metric_episode<Me: Metric + 'static>(
                    mut self,
                    metric: Me,
                ) -> Self
                where
                    EpisodeSummary: Adaptor<Me::Input> + 'static,
                {
                    self.metrics.register_text_metric_episode(metric.clone());
                    self.metrics.register_text_metric_episode_valid(metric);
                    self
                }
                /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for a completed episode.
                pub fn metric_episode<Me>(mut self, metric: Me) -> Self
                where
                    Me: Metric + Numeric + 'static,
                    EpisodeSummary: Adaptor<Me::Input> + 'static,
                {
                    self.summary_metrics.insert(metric.name().to_string());
                    self.metrics.register_episode_metric(metric.clone());
                    self.metrics.register_episode_metric_valid(metric);
                    self
                }
                /// The number of environment steps to train for.
                pub fn num_steps(mut self, num_steps: usize) -> Self {
                    self.num_steps = num_steps;
                    self
                }
                /// The step from which the training must resume.
                pub fn checkpoint(mut self, checkpoint: usize) -> Self {
                    self.checkpoint = Some(checkpoint);
                    self
                }
                /// Provides a handle that can be used to interrupt training.
                pub fn interrupter(&self) -> Interrupter {
                    self.interrupter.clone()
                }
                /// Override the handle for stopping training with an externally provided handle
                pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
                    self.interrupter = interrupter;
                    self
                }
                /// By default, Rust logs are captured and written into
                /// `experiment.log`. If disabled, standard Rust log handling
                /// will apply.
                pub fn with_application_logger(
                    mut self,
                    logger: Option<Box<dyn ApplicationLoggerInstaller>>,
                ) -> Self {
                    self.tracing_logger = logger;
                    self
                }
                /// Register a checkpointer that will save the environment runner's [policy](Policy)
                /// and the [PolicyLearner](PolicyLearner) state to different files.
                pub fn with_file_checkpointer<FR>(mut self, recorder: FR) -> Self
                where
                    FR: FileRecorder + 'static,
                    FR: FileRecorder + 'static,
                {
                    let checkpoint_dir = self.directory.join("checkpoint");
                    let checkpointer_policy = FileCheckpointer::new(
                        recorder.clone(),
                        &checkpoint_dir,
                        "policy",
                    );
                    let checkpointer_learning = FileCheckpointer::new(
                        recorder.clone(),
                        &checkpoint_dir,
                        "learning-agent",
                    );
                    self.checkpointers = Some((
                        AsyncCheckpointer::new(checkpointer_policy),
                        AsyncCheckpointer::new(checkpointer_learning),
                    ));
                    self
                }
                /// The device on which to run inference during rollout collection and validation.
                pub fn with_inference_device(mut self, device: Device) -> Self {
                    self.inference_device = device;
                    self
                }
                /// Enable the training summary report.
                ///
                /// The summary will be displayed after `.launch()`, when the renderer is dropped.
                pub fn summary(mut self) -> Self {
                    self.summary = true;
                    self
                }
                /// Launch the training with the specified [PolicyLearner](PolicyLearner) on the specified environment.
                pub fn launch(
                    mut self,
                    learner_agent: RLC::LearningAgent,
                ) -> RLResult<RLC::Policy>
                where
                    RLC::PolicyObs: SliceAccess,
                    RLC::PolicyAction: SliceAccess,
                {
                    if self.tracing_logger.is_some()
                        && let Err(e) = self.tracing_logger.as_ref().unwrap().install()
                    {
                        {
                            {
                                let lvl = ::log::Level::Warn;
                                if lvl <= ::log::STATIC_MAX_LEVEL
                                    && lvl <= ::log::max_level()
                                {
                                    ::log::__private_api::log(
                                        { ::log::__private_api::GlobalLogger },
                                        format_args!(
                                            "Failed to install the experiment logger: {0}",
                                            e,
                                        ),
                                        lvl,
                                        &(
                                            "burn_train::learner::rl::paradigm",
                                            "burn_train::learner::rl::paradigm",
                                            ::log::__private_api::loc(),
                                        ),
                                        (),
                                    );
                                }
                            }
                        };
                    }
                    let renderer = self
                        .renderer
                        .unwrap_or_else(|| default_renderer(
                            self.interrupter.clone(),
                            self.checkpoint,
                        ));
                    if !self.event_store.has_loggers() {
                        self.event_store
                            .register_logger(
                                FileMetricLogger::new(self.directory.clone()),
                            );
                    }
                    let event_store = Arc::new(EventStoreClient::new(self.event_store));
                    let event_processor = AsyncProcessorTraining::new(
                        RLEventProcessor::new(
                            self.metrics,
                            renderer,
                            event_store.clone(),
                        ),
                    );
                    let checkpointer = self
                        .checkpointers
                        .map(|(policy, learning_agent)| {
                            RLCheckpointer::new(
                                policy,
                                learning_agent,
                                self.checkpointer_strategy,
                            )
                        });
                    let summary = if self.summary {
                        Some(LearnerSummaryConfig {
                            directory: self.directory,
                            metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
                        })
                    } else {
                        None
                    };
                    let components = RLComponents::<RLC> {
                        checkpoint: self.checkpoint,
                        checkpointer,
                        interrupter: self.interrupter,
                        event_processor,
                        event_store,
                        num_steps: self.num_steps,
                        grad_accumulation: self.grad_accumulation,
                        summary,
                        inference_device: self.inference_device,
                    };
                    match self.learning_strategy {
                        RLStrategies::OffPolicyStrategy(config) => {
                            let strategy = OffPolicyStrategy::new(config);
                            strategy
                                .train(learner_agent, components, self.env_initializer)
                        }
                        RLStrategies::Custom(strategy) => {
                            strategy
                                .train(learner_agent, components, self.env_initializer)
                        }
                    }
                }
            }
            /// The result of reinforcement learning, containing the final policy along with the [renderer](MetricsRenderer).
            pub struct RLResult<P> {
                /// The learned policy.
                pub policy: P,
                /// The renderer that can be used for follow up training and evaluation.
                pub renderer: Box<dyn MetricsRenderer>,
            }
            /// Trait to fake variadic generics for train step metrics.
            pub trait AgentMetricRegistration<RLC: RLComponentsTypes>: Sized {
                /// Register the metrics.
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
            }
            /// Trait to fake variadic generics for train step text metrics.
            pub trait AgentTextMetricRegistration<RLC: RLComponentsTypes>: Sized {
                /// Register the metrics.
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
            }
            /// Trait to fake variadic generics for env step metrics.
            pub trait TrainMetricRegistration<RLC: RLComponentsTypes>: Sized {
                /// Register the metrics.
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
            }
            /// Trait to fake variadic generics for env step text metrics.
            pub trait TrainTextMetricRegistration<RLC: RLComponentsTypes>: Sized {
                /// Register the metrics.
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
            }
            /// Trait to fake variadic generics for episode metrics.
            pub trait EpisodeMetricRegistration<RLC: RLComponentsTypes>: Sized {
                /// Register the metrics.
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
            }
            /// Trait to fake variadic generics for episode text metrics.
            pub trait EpisodeTextMetricRegistration<RLC: RLComponentsTypes>: Sized {
                /// Register the metrics.
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
            }
            impl<M1, RLC: RLComponentsTypes + 'static> TrainTextMetricRegistration<RLC>
            for (M1,)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                M1: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1,) = self;
                    let builder = builder.text_metric_train(M1.clone());
                    builder
                }
            }
            impl<M1, RLC: RLComponentsTypes + 'static> TrainMetricRegistration<RLC>
            for (M1,)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                M1: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1,) = self;
                    let builder = builder.metric_train(M1.clone());
                    builder
                }
            }
            impl<M1, RLC: RLComponentsTypes + 'static> AgentTextMetricRegistration<RLC>
            for (M1,)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                M1: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1,) = self;
                    let builder = builder.text_metric_agent(M1.clone());
                    builder
                }
            }
            impl<M1, RLC: RLComponentsTypes + 'static> AgentMetricRegistration<RLC>
            for (M1,)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                M1: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1,) = self;
                    let builder = builder.metric_agent(M1.clone());
                    builder
                }
            }
            impl<M1, RLC: RLComponentsTypes + 'static> EpisodeTextMetricRegistration<RLC>
            for (M1,)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                M1: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1,) = self;
                    let builder = builder.text_metric_episode(M1.clone());
                    builder
                }
            }
            impl<M1, RLC: RLComponentsTypes + 'static> EpisodeMetricRegistration<RLC>
            for (M1,)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                M1: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1,) = self;
                    let builder = builder.metric_episode(M1.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                RLC: RLComponentsTypes + 'static,
            > TrainTextMetricRegistration<RLC> for (M1, M2)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2) = self;
                    let builder = builder.text_metric_train(M1.clone());
                    let builder = builder.text_metric_train(M2.clone());
                    builder
                }
            }
            impl<M1, M2, RLC: RLComponentsTypes + 'static> TrainMetricRegistration<RLC>
            for (M1, M2)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                RLC: RLComponentsTypes + 'static,
            > AgentTextMetricRegistration<RLC> for (M1, M2)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2) = self;
                    let builder = builder.text_metric_agent(M1.clone());
                    let builder = builder.text_metric_agent(M2.clone());
                    builder
                }
            }
            impl<M1, M2, RLC: RLComponentsTypes + 'static> AgentMetricRegistration<RLC>
            for (M1, M2)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2) = self;
                    let builder = builder.metric_agent(M1.clone());
                    let builder = builder.metric_agent(M2.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                RLC: RLComponentsTypes + 'static,
            > EpisodeTextMetricRegistration<RLC> for (M1, M2)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                M1: Metric + 'static,
                M2: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2) = self;
                    let builder = builder.text_metric_episode(M1.clone());
                    let builder = builder.text_metric_episode(M2.clone());
                    builder
                }
            }
            impl<M1, M2, RLC: RLComponentsTypes + 'static> EpisodeMetricRegistration<RLC>
            for (M1, M2)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2) = self;
                    let builder = builder.metric_episode(M1.clone());
                    let builder = builder.metric_episode(M2.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                RLC: RLComponentsTypes + 'static,
            > TrainTextMetricRegistration<RLC> for (M1, M2, M3)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3) = self;
                    let builder = builder.text_metric_train(M1.clone());
                    let builder = builder.text_metric_train(M2.clone());
                    let builder = builder.text_metric_train(M3.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                RLC: RLComponentsTypes + 'static,
            > TrainMetricRegistration<RLC> for (M1, M2, M3)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                RLC: RLComponentsTypes + 'static,
            > AgentTextMetricRegistration<RLC> for (M1, M2, M3)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3) = self;
                    let builder = builder.text_metric_agent(M1.clone());
                    let builder = builder.text_metric_agent(M2.clone());
                    let builder = builder.text_metric_agent(M3.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                RLC: RLComponentsTypes + 'static,
            > AgentMetricRegistration<RLC> for (M1, M2, M3)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3) = self;
                    let builder = builder.metric_agent(M1.clone());
                    let builder = builder.metric_agent(M2.clone());
                    let builder = builder.metric_agent(M3.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                RLC: RLComponentsTypes + 'static,
            > EpisodeTextMetricRegistration<RLC> for (M1, M2, M3)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3) = self;
                    let builder = builder.text_metric_episode(M1.clone());
                    let builder = builder.text_metric_episode(M2.clone());
                    let builder = builder.text_metric_episode(M3.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                RLC: RLComponentsTypes + 'static,
            > EpisodeMetricRegistration<RLC> for (M1, M2, M3)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3) = self;
                    let builder = builder.metric_episode(M1.clone());
                    let builder = builder.metric_episode(M2.clone());
                    let builder = builder.metric_episode(M3.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                RLC: RLComponentsTypes + 'static,
            > TrainTextMetricRegistration<RLC> for (M1, M2, M3, M4)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                RLC::TrainingOutput: Adaptor<M4::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.text_metric_train(M1.clone());
                    let builder = builder.text_metric_train(M2.clone());
                    let builder = builder.text_metric_train(M3.clone());
                    let builder = builder.text_metric_train(M4.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                RLC: RLComponentsTypes + 'static,
            > TrainMetricRegistration<RLC> for (M1, M2, M3, M4)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                RLC::TrainingOutput: Adaptor<M4::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    let builder = builder.metric_train(M4.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                RLC: RLComponentsTypes + 'static,
            > AgentTextMetricRegistration<RLC> for (M1, M2, M3, M4)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                RLC::ActionContext: Adaptor<M4::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.text_metric_agent(M1.clone());
                    let builder = builder.text_metric_agent(M2.clone());
                    let builder = builder.text_metric_agent(M3.clone());
                    let builder = builder.text_metric_agent(M4.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                RLC: RLComponentsTypes + 'static,
            > AgentMetricRegistration<RLC> for (M1, M2, M3, M4)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                RLC::ActionContext: Adaptor<M4::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.metric_agent(M1.clone());
                    let builder = builder.metric_agent(M2.clone());
                    let builder = builder.metric_agent(M3.clone());
                    let builder = builder.metric_agent(M4.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                RLC: RLComponentsTypes + 'static,
            > EpisodeTextMetricRegistration<RLC> for (M1, M2, M3, M4)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                EpisodeSummary: Adaptor<M4::Input> + 'static,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.text_metric_episode(M1.clone());
                    let builder = builder.text_metric_episode(M2.clone());
                    let builder = builder.text_metric_episode(M3.clone());
                    let builder = builder.text_metric_episode(M4.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                RLC: RLComponentsTypes + 'static,
            > EpisodeMetricRegistration<RLC> for (M1, M2, M3, M4)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                EpisodeSummary: Adaptor<M4::Input> + 'static,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.metric_episode(M1.clone());
                    let builder = builder.metric_episode(M2.clone());
                    let builder = builder.metric_episode(M3.clone());
                    let builder = builder.metric_episode(M4.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                RLC: RLComponentsTypes + 'static,
            > TrainTextMetricRegistration<RLC> for (M1, M2, M3, M4, M5)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                RLC::TrainingOutput: Adaptor<M4::Input>,
                RLC::TrainingOutput: Adaptor<M5::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.text_metric_train(M1.clone());
                    let builder = builder.text_metric_train(M2.clone());
                    let builder = builder.text_metric_train(M3.clone());
                    let builder = builder.text_metric_train(M4.clone());
                    let builder = builder.text_metric_train(M5.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                RLC: RLComponentsTypes + 'static,
            > TrainMetricRegistration<RLC> for (M1, M2, M3, M4, M5)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                RLC::TrainingOutput: Adaptor<M4::Input>,
                RLC::TrainingOutput: Adaptor<M5::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    let builder = builder.metric_train(M4.clone());
                    let builder = builder.metric_train(M5.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                RLC: RLComponentsTypes + 'static,
            > AgentTextMetricRegistration<RLC> for (M1, M2, M3, M4, M5)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                RLC::ActionContext: Adaptor<M4::Input>,
                RLC::ActionContext: Adaptor<M5::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.text_metric_agent(M1.clone());
                    let builder = builder.text_metric_agent(M2.clone());
                    let builder = builder.text_metric_agent(M3.clone());
                    let builder = builder.text_metric_agent(M4.clone());
                    let builder = builder.text_metric_agent(M5.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                RLC: RLComponentsTypes + 'static,
            > AgentMetricRegistration<RLC> for (M1, M2, M3, M4, M5)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                RLC::ActionContext: Adaptor<M4::Input>,
                RLC::ActionContext: Adaptor<M5::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.metric_agent(M1.clone());
                    let builder = builder.metric_agent(M2.clone());
                    let builder = builder.metric_agent(M3.clone());
                    let builder = builder.metric_agent(M4.clone());
                    let builder = builder.metric_agent(M5.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                RLC: RLComponentsTypes + 'static,
            > EpisodeTextMetricRegistration<RLC> for (M1, M2, M3, M4, M5)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                EpisodeSummary: Adaptor<M4::Input> + 'static,
                EpisodeSummary: Adaptor<M5::Input> + 'static,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.text_metric_episode(M1.clone());
                    let builder = builder.text_metric_episode(M2.clone());
                    let builder = builder.text_metric_episode(M3.clone());
                    let builder = builder.text_metric_episode(M4.clone());
                    let builder = builder.text_metric_episode(M5.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                RLC: RLComponentsTypes + 'static,
            > EpisodeMetricRegistration<RLC> for (M1, M2, M3, M4, M5)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                EpisodeSummary: Adaptor<M4::Input> + 'static,
                EpisodeSummary: Adaptor<M5::Input> + 'static,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.metric_episode(M1.clone());
                    let builder = builder.metric_episode(M2.clone());
                    let builder = builder.metric_episode(M3.clone());
                    let builder = builder.metric_episode(M4.clone());
                    let builder = builder.metric_episode(M5.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                RLC: RLComponentsTypes + 'static,
            > TrainTextMetricRegistration<RLC> for (M1, M2, M3, M4, M5, M6)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                RLC::TrainingOutput: Adaptor<M4::Input>,
                RLC::TrainingOutput: Adaptor<M5::Input>,
                RLC::TrainingOutput: Adaptor<M6::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
                M6: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.text_metric_train(M1.clone());
                    let builder = builder.text_metric_train(M2.clone());
                    let builder = builder.text_metric_train(M3.clone());
                    let builder = builder.text_metric_train(M4.clone());
                    let builder = builder.text_metric_train(M5.clone());
                    let builder = builder.text_metric_train(M6.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                RLC: RLComponentsTypes + 'static,
            > TrainMetricRegistration<RLC> for (M1, M2, M3, M4, M5, M6)
            where
                RLC::TrainingOutput: Adaptor<M1::Input>,
                RLC::TrainingOutput: Adaptor<M2::Input>,
                RLC::TrainingOutput: Adaptor<M3::Input>,
                RLC::TrainingOutput: Adaptor<M4::Input>,
                RLC::TrainingOutput: Adaptor<M5::Input>,
                RLC::TrainingOutput: Adaptor<M6::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
                M6: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    let builder = builder.metric_train(M4.clone());
                    let builder = builder.metric_train(M5.clone());
                    let builder = builder.metric_train(M6.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                RLC: RLComponentsTypes + 'static,
            > AgentTextMetricRegistration<RLC> for (M1, M2, M3, M4, M5, M6)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                RLC::ActionContext: Adaptor<M4::Input>,
                RLC::ActionContext: Adaptor<M5::Input>,
                RLC::ActionContext: Adaptor<M6::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
                M6: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.text_metric_agent(M1.clone());
                    let builder = builder.text_metric_agent(M2.clone());
                    let builder = builder.text_metric_agent(M3.clone());
                    let builder = builder.text_metric_agent(M4.clone());
                    let builder = builder.text_metric_agent(M5.clone());
                    let builder = builder.text_metric_agent(M6.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                RLC: RLComponentsTypes + 'static,
            > AgentMetricRegistration<RLC> for (M1, M2, M3, M4, M5, M6)
            where
                RLC::ActionContext: Adaptor<M1::Input>,
                RLC::ActionContext: Adaptor<M2::Input>,
                RLC::ActionContext: Adaptor<M3::Input>,
                RLC::ActionContext: Adaptor<M4::Input>,
                RLC::ActionContext: Adaptor<M5::Input>,
                RLC::ActionContext: Adaptor<M6::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
                M6: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.metric_agent(M1.clone());
                    let builder = builder.metric_agent(M2.clone());
                    let builder = builder.metric_agent(M3.clone());
                    let builder = builder.metric_agent(M4.clone());
                    let builder = builder.metric_agent(M5.clone());
                    let builder = builder.metric_agent(M6.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                RLC: RLComponentsTypes + 'static,
            > EpisodeTextMetricRegistration<RLC> for (M1, M2, M3, M4, M5, M6)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                EpisodeSummary: Adaptor<M4::Input> + 'static,
                EpisodeSummary: Adaptor<M5::Input> + 'static,
                EpisodeSummary: Adaptor<M6::Input> + 'static,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
                M6: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.text_metric_episode(M1.clone());
                    let builder = builder.text_metric_episode(M2.clone());
                    let builder = builder.text_metric_episode(M3.clone());
                    let builder = builder.text_metric_episode(M4.clone());
                    let builder = builder.text_metric_episode(M5.clone());
                    let builder = builder.text_metric_episode(M6.clone());
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                RLC: RLComponentsTypes + 'static,
            > EpisodeMetricRegistration<RLC> for (M1, M2, M3, M4, M5, M6)
            where
                EpisodeSummary: Adaptor<M1::Input> + 'static,
                EpisodeSummary: Adaptor<M2::Input> + 'static,
                EpisodeSummary: Adaptor<M3::Input> + 'static,
                EpisodeSummary: Adaptor<M4::Input> + 'static,
                EpisodeSummary: Adaptor<M5::Input> + 'static,
                EpisodeSummary: Adaptor<M6::Input> + 'static,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
                M6: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.metric_episode(M1.clone());
                    let builder = builder.metric_episode(M2.clone());
                    let builder = builder.metric_episode(M3.clone());
                    let builder = builder.metric_episode(M4.clone());
                    let builder = builder.metric_episode(M5.clone());
                    let builder = builder.metric_episode(M6.clone());
                    builder
                }
            }
        }
        mod strategy {
            use std::sync::Arc;
            use burn_core::tensor::Device;
            use crate::{
                Interrupter, LearnerSummaryConfig, OffPolicyConfig, RLCheckpointer,
                RLComponentsTypes, RLEvent, RLEventProcessorType, RLResult,
                metric::{processor::EventProcessorTraining, store::EventStoreClient},
            };
            /// Struct to minimise parameters passed to [RLStrategy::train].
            pub struct RLComponents<RLC: RLComponentsTypes> {
                /// The total number of environment steps.
                pub num_steps: usize,
                /// The step number from which to continue the training.
                pub checkpoint: Option<usize>,
                /// A checkpointer used to load and save learning checkpoints.
                pub checkpointer: Option<RLCheckpointer<RLC>>,
                /// Enables gradients accumulation.
                pub grad_accumulation: Option<usize>,
                /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
                pub interrupter: Interrupter,
                /// An [EventProcessor](crate::EventProcessorTraining) that processes events happening during training and evaluation.
                pub event_processor: RLEventProcessorType<RLC>,
                /// A reference to an [EventStoreClient](EventStoreClient).
                pub event_store: Arc<EventStoreClient>,
                /// Config for creating a summary of the learning
                pub summary: Option<LearnerSummaryConfig>,
                /// Device used for running inference during environmment sampling or validation.
                pub inference_device: Device,
            }
            /// The strategy for reinforcement learning.
            pub enum RLStrategies<RLC: RLComponentsTypes> {
                /// Training on one device
                OffPolicyStrategy(OffPolicyConfig),
                /// Training using a custom learning strategy
                Custom(CustomRLStrategy<RLC>),
            }
            #[automatically_derived]
            impl<RLC: ::core::clone::Clone + RLComponentsTypes> ::core::clone::Clone
            for RLStrategies<RLC> {
                #[inline]
                fn clone(&self) -> RLStrategies<RLC> {
                    match self {
                        RLStrategies::OffPolicyStrategy(__self_0) => {
                            RLStrategies::OffPolicyStrategy(
                                ::core::clone::Clone::clone(__self_0),
                            )
                        }
                        RLStrategies::Custom(__self_0) => {
                            RLStrategies::Custom(::core::clone::Clone::clone(__self_0))
                        }
                    }
                }
            }
            /// A reference to an implementation of [RLStrategy].
            pub type CustomRLStrategy<LC> = Arc<dyn RLStrategy<LC>>;
            /// Provides the `fit` function for any learning strategy
            pub trait RLStrategy<RLC: RLComponentsTypes> {
                /// Train the learner agent with this strategy.
                fn train(
                    &self,
                    mut learner_agent: RLC::LearningAgent,
                    mut training_components: RLComponents<RLC>,
                    env_init: RLC::EnvInit,
                ) -> RLResult<RLC::Policy> {
                    let starting_epoch = match training_components.checkpoint {
                        Some(checkpoint) => {
                            if let Some(checkpointer) = &mut training_components
                                .checkpointer
                            {
                                learner_agent = checkpointer
                                    .load_checkpoint(
                                        learner_agent,
                                        &Default::default(),
                                        checkpoint,
                                    );
                            }
                            checkpoint + 1
                        }
                        None => 1,
                    };
                    let summary_config = training_components.summary.clone();
                    training_components
                        .event_processor
                        .process_train(RLEvent::Start {
                            total_items: training_components.num_steps,
                        });
                    let (policy, mut event_processor) = self
                        .train_loop(
                            training_components,
                            &mut learner_agent,
                            starting_epoch,
                            env_init,
                        );
                    let summary = summary_config.and_then(|summary| summary.init().ok());
                    event_processor.process_train(RLEvent::End(summary));
                    let renderer = event_processor.renderer();
                    RLResult { policy, renderer }
                }
                /// Training loop for this strategy
                fn train_loop(
                    &self,
                    training_components: RLComponents<RLC>,
                    learner_agent: &mut RLC::LearningAgent,
                    starting_epoch: usize,
                    env_init: RLC::EnvInit,
                ) -> (RLC::Policy, RLEventProcessorType<RLC>);
            }
        }
        pub use checkpointer::*;
        pub use components::*;
        pub use env_runner::*;
        pub use off_policy::*;
        pub use output::*;
        pub use paradigm::*;
        pub use strategy::*;
    }
    pub use rl::*;
    mod application_logger {
        use std::path::{Path, PathBuf};
        use tracing_core::{Level, LevelFilter};
        use tracing_subscriber::filter::filter_fn;
        use tracing_subscriber::prelude::*;
        use tracing_subscriber::{Layer, registry};
        /// This trait is used to install an application logger.
        pub trait ApplicationLoggerInstaller {
            /// Install the application logger.
            fn install(&self) -> Result<(), String>;
        }
        /// This struct is used to install a local file application logger to output logs to a given file path.
        pub struct FileApplicationLoggerInstaller {
            path: PathBuf,
        }
        impl FileApplicationLoggerInstaller {
            /// Create a new file application logger.
            pub fn new(path: impl AsRef<Path>) -> Self {
                Self {
                    path: path.as_ref().to_path_buf(),
                }
            }
        }
        impl ApplicationLoggerInstaller for FileApplicationLoggerInstaller {
            fn install(&self) -> Result<(), String> {
                let path = Path::new(&self.path);
                let writer = tracing_appender::rolling::never(
                    path.parent().unwrap_or_else(|| Path::new(".")),
                    path
                        .file_name()
                        .unwrap_or_else(|| {
                            {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "The path \'{0}\' to point to a file.",
                                        self.path.display(),
                                    ),
                                );
                            }
                        }),
                );
                let layer = tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(writer)
                    .with_filter(LevelFilter::INFO)
                    .with_filter(
                        filter_fn(|m| {
                            if let Some(path) = m.module_path() {
                                if path.starts_with("wgpu") && *m.level() >= Level::INFO {
                                    return false;
                                }
                            }
                            true
                        }),
                    );
                if registry().with(layer).try_init().is_err() {
                    return Err("Failed to install the file logger.".to_string());
                }
                let hook = std::panic::take_hook();
                let file_path = self.path.to_owned();
                std::panic::set_hook(
                    Box::new(move |info| {
                        {
                            {
                                let lvl = ::log::Level::Error;
                                if lvl <= ::log::STATIC_MAX_LEVEL
                                    && lvl <= ::log::max_level()
                                {
                                    ::log::__private_api::log(
                                        { ::log::__private_api::GlobalLogger },
                                        format_args!("PANIC => {0}", info),
                                        lvl,
                                        &(
                                            "burn_train::learner::application_logger",
                                            "burn_train::learner::application_logger",
                                            ::log::__private_api::loc(),
                                        ),
                                        (),
                                    );
                                }
                            }
                        };
                        {
                            ::std::io::_eprint(
                                format_args!(
                                    "=== PANIC ===\nA fatal error happened, you can check the experiment logs here => \'{0}\'\n=============\n",
                                    file_path.display(),
                                ),
                            );
                        };
                        hook(info);
                    }),
                );
                Ok(())
            }
        }
    }
    mod base {
        use crate::LearningComponentsMarker;
        use crate::checkpoint::{
            AsyncCheckpointer, Checkpointer, CheckpointingAction, CheckpointingStrategy,
        };
        use crate::components::LearningComponentsTypes;
        use crate::metric::store::EventStoreClient;
        use crate::{
            CloneEarlyStoppingStrategy, InferenceStep, TrainOutput, TrainStep,
            TrainingModelInput, TrainingModelOutput,
        };
        use burn_core::module::{AutodiffModule, Module};
        use burn_core::tensor::Device;
        use burn_optim::lr_scheduler::LrScheduler;
        use burn_optim::{GradientsParams, MultiGradientsParams, Optimizer};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        /// The record of the learner's model.
        pub type LearnerModelRecord<LC> = <<LC as LearningComponentsTypes>::Model as Module>::Record;
        /// The record of the optimizer.
        pub type LearnerOptimizerRecord<LC> = <<LC as LearningComponentsTypes>::Optimizer as Optimizer<
            <LC as LearningComponentsTypes>::Model,
        >>::Record;
        /// The record of the LR scheduler.
        pub type LearnerSchedulerRecord<LC> = <<LC as LearningComponentsTypes>::LrScheduler as LrScheduler>::Record;
        /// Learner struct encapsulating all components necessary to train a Neural Network model.
        pub struct Learner<LC: LearningComponentsTypes> {
            pub(crate) model: LC::Model,
            optim: LC::Optimizer,
            lr_scheduler: LC::LrScheduler,
            lr: f64,
        }
        impl<LC: LearningComponentsTypes> Clone for Learner<LC> {
            fn clone(&self) -> Self {
                Self {
                    model: self.model.clone(),
                    optim: self.optim.clone(),
                    lr_scheduler: self.lr_scheduler.clone(),
                    lr: self.lr,
                }
            }
        }
        impl<LR, M, O> Learner<LearningComponentsMarker<LR, M, O>>
        where
            LR: LrScheduler + 'static,
            M: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static,
            O: Optimizer<M> + 'static,
        {
            /// Create a learner.
            pub fn new(model: M, optim: O, lr_scheduler: LR) -> Self {
                Self {
                    model,
                    optim,
                    lr_scheduler,
                    lr: 0.0,
                }
            }
        }
        impl<LC: LearningComponentsTypes> Learner<LC> {
            /// Fork the learner's model to the given device.
            pub fn fork(&mut self, device: &Device) {
                self.model = self.model().fork(device);
            }
            /// Returns the current model.
            pub fn model(&self) -> LC::Model {
                self.model.clone()
            }
            /// Returns the current learning rate.
            pub fn lr_current(&self) -> f64 {
                self.lr
            }
            /// Executes a step of the learning rate scheduler.
            pub fn lr_step(&mut self) {
                self.lr = self.lr_scheduler.step();
            }
            /// Runs a step of the model for training, which executes the forward and backward passes.
            ///
            /// # Arguments
            ///
            /// * `item` - The input for the model.
            ///
            /// # Returns
            ///
            /// The output containing the model output and the gradients.
            pub fn train_step(
                &self,
                item: TrainingModelInput<LC>,
            ) -> TrainOutput<TrainingModelOutput<LC>> {
                TrainStep::step(&self.model, item)
            }
            /// Optimize the current module with the provided gradients and learning rate.
            ///
            /// # Arguments
            ///
            /// * `optim`: Optimizer used for learning.
            /// * `lr`: The learning rate used for this step.
            /// * `grads`: The gradients of each parameter in the current model.
            pub fn optimizer_step(&mut self, grads: GradientsParams) {
                self.model = self.model().optimize(&mut self.optim, self.lr, grads);
            }
            /// Optimize the current module with the provided gradients and learning rate.
            ///
            /// # Arguments
            ///
            /// * `optim`: Optimizer used for learning.
            /// * `lr`: The learning rate used for this step.
            /// * `grads`: Multiple gradients associated to each parameter in the current model.
            pub fn optimizer_step_multi(&mut self, grads: MultiGradientsParams) {
                self.model = self
                    .model()
                    .optimize_multi(&mut self.optim, self.lr, grads);
            }
            /// Load the module state from a [record](LearnerModelRecord<LC>).
            pub fn load_model(&mut self, record: LearnerModelRecord<LC>) {
                self.model = self.model.clone().load_record(record);
            }
            /// Load the state of the learner's optimizer as a [record](LearnerOptimizerRecord<LC>).
            pub fn load_optim(&mut self, record: LearnerOptimizerRecord<LC>) {
                self.optim = self.optim.clone().load_record(record);
            }
            /// Load the state of the learner's scheduler as a [record](LearnerSchedulerRecord<LC>).
            pub fn load_scheduler(&mut self, record: LearnerSchedulerRecord<LC>) {
                self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
            }
        }
        /// Used to create, delete, or load checkpoints of the training process.
        pub struct LearningCheckpointer<LC: LearningComponentsTypes> {
            model: AsyncCheckpointer<LearnerModelRecord<LC>>,
            optim: AsyncCheckpointer<LearnerOptimizerRecord<LC>>,
            lr_scheduler: AsyncCheckpointer<LearnerSchedulerRecord<LC>>,
            strategy: Box<dyn CheckpointingStrategy>,
        }
        impl<LC: LearningComponentsTypes> LearningCheckpointer<LC> {
            ///Constructs a new `LearningCheckpointer`.
            pub fn new(
                model: AsyncCheckpointer<LearnerModelRecord<LC>>,
                optim: AsyncCheckpointer<LearnerOptimizerRecord<LC>>,
                lr_scheduler: AsyncCheckpointer<LearnerSchedulerRecord<LC>>,
                strategy: Box<dyn CheckpointingStrategy>,
            ) -> Self {
                LearningCheckpointer {
                    model: model,
                    optim: optim,
                    lr_scheduler: lr_scheduler,
                    strategy: strategy,
                }
            }
        }
        impl<LC: LearningComponentsTypes> LearningCheckpointer<LC> {
            /// Create checkpoint for the training process.
            pub fn checkpoint(
                &mut self,
                learner: &Learner<LC>,
                epoch: usize,
                store: &EventStoreClient,
            ) {
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
                                .save(epoch, learner.model.clone().into_record())
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
            pub fn load_checkpoint(
                &self,
                mut learner: Learner<LC>,
                device: &Device,
                epoch: usize,
            ) -> Learner<LC> {
                let record = self
                    .model
                    .restore(epoch, device)
                    .expect("Can load model checkpoint.");
                learner.load_model(record);
                let record = self
                    .optim
                    .restore(epoch, device)
                    .expect("Can load optimizer checkpoint.");
                learner.load_optim(record);
                let record = self
                    .lr_scheduler
                    .restore(epoch, device)
                    .expect("Can load learning rate scheduler checkpoint.");
                learner.load_scheduler(record);
                learner
            }
        }
        /// Cloneable reference to an early stopping strategy
        pub(crate) type EarlyStoppingStrategyRef = Box<dyn CloneEarlyStoppingStrategy>;
        /// A handle that allows aborting the training/evaluation process early.
        pub struct Interrupter {
            state: Arc<AtomicBool>,
            message: Arc<Mutex<Option<String>>>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for Interrupter {
            #[inline]
            fn clone(&self) -> Interrupter {
                Interrupter {
                    state: ::core::clone::Clone::clone(&self.state),
                    message: ::core::clone::Clone::clone(&self.message),
                }
            }
        }
        #[automatically_derived]
        impl ::core::default::Default for Interrupter {
            #[inline]
            fn default() -> Interrupter {
                Interrupter {
                    state: ::core::default::Default::default(),
                    message: ::core::default::Default::default(),
                }
            }
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
                reason
                    .inspect(|r| {
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
    }
    mod classification {
        use crate::metric::{
            AccuracyInput, Adaptor, ConfusionStatsInput, HammingScoreInput, LossInput,
            PerplexityInput, TopKAccuracyInput, processor::ItemLazy,
        };
        use burn_core::tensor::{Device, Int, Tensor, Transaction};
        /// Simple classification output adapted for multiple metrics.
        ///
        /// Supported metrics:
        /// - Accuracy
        /// - AUROC
        /// - TopKAccuracy
        /// - Perplexity
        /// - Precision (via ConfusionStatsInput)
        /// - Recall (via ConfusionStatsInput)
        /// - FBetaScore (via ConfusionStatsInput)
        /// - Loss.
        pub struct ClassificationOutput {
            /// The loss.
            pub loss: Tensor<1>,
            /// The class logits or probabilities. Shape: \[batch_size, num_classes\].
            pub output: Tensor<2>,
            /// The ground truth class index for each sample. Shape: \[batch_size\].
            pub targets: Tensor<1, Int>,
        }
        impl ClassificationOutput {
            ///Constructs a new `ClassificationOutput`.
            pub fn new(
                loss: Tensor<1>,
                output: Tensor<2>,
                targets: Tensor<1, Int>,
            ) -> Self {
                ClassificationOutput {
                    loss: loss,
                    output: output,
                    targets: targets,
                }
            }
        }
        impl ItemLazy for ClassificationOutput {
            fn sync(self) -> Self {
                let [output, loss, targets] = Transaction::default()
                    .register(self.output)
                    .register(self.loss)
                    .register(self.targets)
                    .execute()
                    .try_into()
                    .expect("Correct amount of tensor data");
                let device: Device = Device::flex();
                ClassificationOutput {
                    output: Tensor::from_data(output, &device),
                    loss: Tensor::from_data(loss, &device),
                    targets: Tensor::from_data(targets, &device),
                }
            }
        }
        impl Adaptor<AccuracyInput> for ClassificationOutput {
            fn adapt(&self) -> AccuracyInput {
                AccuracyInput::new(self.output.clone(), self.targets.clone())
            }
        }
        impl Adaptor<LossInput> for ClassificationOutput {
            fn adapt(&self) -> LossInput {
                LossInput::new(self.loss.clone())
            }
        }
        impl Adaptor<TopKAccuracyInput> for ClassificationOutput {
            fn adapt(&self) -> TopKAccuracyInput {
                TopKAccuracyInput::new(self.output.clone(), self.targets.clone())
            }
        }
        impl Adaptor<PerplexityInput> for ClassificationOutput {
            fn adapt(&self) -> PerplexityInput {
                PerplexityInput::new(self.output.clone(), self.targets.clone())
            }
        }
        impl Adaptor<ConfusionStatsInput> for ClassificationOutput {
            fn adapt(&self) -> ConfusionStatsInput {
                let [_, num_classes] = self.output.dims();
                if num_classes > 1 {
                    ConfusionStatsInput::new(
                        self.output.clone(),
                        self.targets.clone().one_hot(num_classes).bool(),
                    )
                } else {
                    ConfusionStatsInput::new(
                        self.output.clone(),
                        self.targets.clone().unsqueeze_dim(1).bool(),
                    )
                }
            }
        }
        /// Multi-label classification output adapted for multiple metrics.
        ///
        /// Supported metrics:
        /// - HammingScore
        /// - Precision (via ConfusionStatsInput)
        /// - Recall (via ConfusionStatsInput)
        /// - FBetaScore (via ConfusionStatsInput)
        /// - Loss
        pub struct MultiLabelClassificationOutput {
            /// The loss.
            pub loss: Tensor<1>,
            /// The label logits or probabilities. Shape: \[batch_size, num_classes\].
            pub output: Tensor<2>,
            /// The ground truth labels. Shape: \[batch_size, num_classes\].
            pub targets: Tensor<2, Int>,
        }
        impl MultiLabelClassificationOutput {
            ///Constructs a new `MultiLabelClassificationOutput`.
            pub fn new(
                loss: Tensor<1>,
                output: Tensor<2>,
                targets: Tensor<2, Int>,
            ) -> Self {
                MultiLabelClassificationOutput {
                    loss: loss,
                    output: output,
                    targets: targets,
                }
            }
        }
        impl ItemLazy for MultiLabelClassificationOutput {
            fn sync(self) -> Self {
                let [output, loss, targets] = Transaction::default()
                    .register(self.output)
                    .register(self.loss)
                    .register(self.targets)
                    .execute()
                    .try_into()
                    .expect("Correct amount of tensor data");
                let device: Device = Device::flex();
                MultiLabelClassificationOutput {
                    output: Tensor::from_data(output, &device),
                    loss: Tensor::from_data(loss, &device),
                    targets: Tensor::from_data(targets, &device),
                }
            }
        }
        impl Adaptor<HammingScoreInput> for MultiLabelClassificationOutput {
            fn adapt(&self) -> HammingScoreInput {
                HammingScoreInput::new(self.output.clone(), self.targets.clone())
            }
        }
        impl Adaptor<LossInput> for MultiLabelClassificationOutput {
            fn adapt(&self) -> LossInput {
                LossInput::new(self.loss.clone())
            }
        }
        impl Adaptor<ConfusionStatsInput> for MultiLabelClassificationOutput {
            fn adapt(&self) -> ConfusionStatsInput {
                ConfusionStatsInput::new(
                    self.output.clone(),
                    self.targets.clone().bool(),
                )
            }
        }
    }
    mod early_stopping {
        use crate::metric::{
            Metric, MetricName, store::{Aggregate, Direction, EventStoreClient, Split},
        };
        /// The condition that [early stopping strategies](EarlyStoppingStrategy) should follow.
        pub enum StoppingCondition {
            /// When no improvement has happened since the given number of epochs.
            NoImprovementSince {
                /// The number of epochs allowed to worsen before it gets better.
                n_epochs: usize,
            },
        }
        #[automatically_derived]
        impl ::core::clone::Clone for StoppingCondition {
            #[inline]
            fn clone(&self) -> StoppingCondition {
                match self {
                    StoppingCondition::NoImprovementSince { n_epochs: __self_0 } => {
                        StoppingCondition::NoImprovementSince {
                            n_epochs: ::core::clone::Clone::clone(__self_0),
                        }
                    }
                }
            }
        }
        /// A strategy that checks if the training should be stopped.
        pub trait EarlyStoppingStrategy: Send {
            /// Update its current state and returns if the training should be stopped.
            fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool;
        }
        /// A helper trait to provide type-erased cloning.
        pub trait CloneEarlyStoppingStrategy: EarlyStoppingStrategy + Send {
            /// Clone into a boxed trait object.
            fn clone_box(&self) -> Box<dyn CloneEarlyStoppingStrategy>;
        }
        /// Blanket-implement `CloneEarlyStoppingStrategy` for any `T` that
        /// already implements your strategy + `Clone` + `Send` + `'static`.
        impl<T> CloneEarlyStoppingStrategy for T
        where
            T: EarlyStoppingStrategy + Clone + Send + 'static,
        {
            fn clone_box(&self) -> Box<dyn CloneEarlyStoppingStrategy> {
                Box::new(self.clone())
            }
        }
        /// Now you can `impl Clone` for the boxed trait object.
        impl Clone for Box<dyn CloneEarlyStoppingStrategy> {
            fn clone(&self) -> Box<dyn CloneEarlyStoppingStrategy> {
                self.clone_box()
            }
        }
        /// An [early stopping strategy](EarlyStoppingStrategy) based on a metrics collected
        /// during training or validation.
        pub struct MetricEarlyStoppingStrategy {
            condition: StoppingCondition,
            metric_name: MetricName,
            aggregate: Aggregate,
            direction: Direction,
            split: Split,
            best_epoch: usize,
            best_value: f64,
            warmup_epochs: Option<usize>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for MetricEarlyStoppingStrategy {
            #[inline]
            fn clone(&self) -> MetricEarlyStoppingStrategy {
                MetricEarlyStoppingStrategy {
                    condition: ::core::clone::Clone::clone(&self.condition),
                    metric_name: ::core::clone::Clone::clone(&self.metric_name),
                    aggregate: ::core::clone::Clone::clone(&self.aggregate),
                    direction: ::core::clone::Clone::clone(&self.direction),
                    split: ::core::clone::Clone::clone(&self.split),
                    best_epoch: ::core::clone::Clone::clone(&self.best_epoch),
                    best_value: ::core::clone::Clone::clone(&self.best_value),
                    warmup_epochs: ::core::clone::Clone::clone(&self.warmup_epochs),
                }
            }
        }
        impl EarlyStoppingStrategy for MetricEarlyStoppingStrategy {
            fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool {
                let current_value = match store
                    .find_metric(&self.metric_name, epoch, self.aggregate, &self.split)
                {
                    Some(value) => value,
                    None => {
                        {
                            {
                                let lvl = ::log::Level::Warn;
                                if lvl <= ::log::STATIC_MAX_LEVEL
                                    && lvl <= ::log::max_level()
                                {
                                    ::log::__private_api::log(
                                        { ::log::__private_api::GlobalLogger },
                                        format_args!("Can\'t find metric for early stopping."),
                                        lvl,
                                        &(
                                            "burn_train::learner::early_stopping",
                                            "burn_train::learner::early_stopping",
                                            ::log::__private_api::loc(),
                                        ),
                                        (),
                                    );
                                }
                            }
                        };
                        return false;
                    }
                };
                let is_best = match self.direction {
                    Direction::Lowest => current_value < self.best_value,
                    Direction::Highest => current_value > self.best_value,
                };
                if is_best {
                    {
                        {
                            let lvl = ::log::Level::Info;
                            if lvl <= ::log::STATIC_MAX_LEVEL
                                && lvl <= ::log::max_level()
                            {
                                ::log::__private_api::log(
                                    { ::log::__private_api::GlobalLogger },
                                    format_args!(
                                        "New best epoch found {0} {1}: {2}",
                                        epoch,
                                        self.metric_name,
                                        current_value,
                                    ),
                                    lvl,
                                    &(
                                        "burn_train::learner::early_stopping",
                                        "burn_train::learner::early_stopping",
                                        ::log::__private_api::loc(),
                                    ),
                                    (),
                                );
                            }
                        }
                    };
                    self.best_value = current_value;
                    self.best_epoch = epoch;
                    return false;
                }
                if let Some(warmup_epochs) = self.warmup_epochs && epoch <= warmup_epochs
                {
                    return false;
                }
                match self.condition {
                    StoppingCondition::NoImprovementSince { n_epochs } => {
                        let should_stop = epoch - self.best_epoch >= n_epochs;
                        if should_stop {
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Stopping training loop, no improvement since epoch {0}, {1}: {2},  current epoch {3}, {4}: {5}",
                                                self.best_epoch,
                                                self.metric_name,
                                                self.best_value,
                                                epoch,
                                                self.metric_name,
                                                current_value,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::learner::early_stopping",
                                                "burn_train::learner::early_stopping",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                        }
                        should_stop
                    }
                }
            }
        }
        impl MetricEarlyStoppingStrategy {
            /// Create a new [early stopping strategy](EarlyStoppingStrategy) based on a metrics collected
            /// during training or validation.
            ///
            /// # Notes
            ///
            /// The metric should be registered for early stopping to work, otherwise no data is collected.
            pub fn new<Me: Metric>(
                metric: &Me,
                aggregate: Aggregate,
                direction: Direction,
                split: Split,
                condition: StoppingCondition,
            ) -> Self {
                let init_value = match direction {
                    Direction::Lowest => f64::MAX,
                    Direction::Highest => f64::MIN,
                };
                Self {
                    metric_name: metric.name(),
                    condition,
                    aggregate,
                    direction,
                    split,
                    best_epoch: 1,
                    best_value: init_value,
                    warmup_epochs: None,
                }
            }
            /// Get the warmup period.
            ///
            /// Early stopping will not trigger during the warmup epochs.
            pub fn warmup_epochs(&self) -> Option<usize> {
                self.warmup_epochs
            }
            /// Set the warmup epochs.
            ///
            /// Early stopping will not trigger during the warmup epochs.
            ///
            /// # Arguments
            /// - `warmup`: the number of warmup epochs, or None.
            pub fn with_warmup_epochs(self, warmup: Option<usize>) -> Self {
                Self {
                    warmup_epochs: warmup,
                    ..self
                }
            }
        }
    }
    mod regression {
        use crate::metric::processor::ItemLazy;
        use crate::metric::{Adaptor, LossInput};
        use burn_core::tensor::{Device, Tensor, Transaction};
        /// Regression output adapted for the loss metric.
        pub struct RegressionOutput {
            /// The loss.
            pub loss: Tensor<1>,
            /// The predicted values. Shape: \[batch_size, num_targets\].
            pub output: Tensor<2>,
            /// The ground truth values. Shape: \[batch_size, num_targets\].
            pub targets: Tensor<2>,
        }
        impl RegressionOutput {
            ///Constructs a new `RegressionOutput`.
            pub fn new(loss: Tensor<1>, output: Tensor<2>, targets: Tensor<2>) -> Self {
                RegressionOutput {
                    loss: loss,
                    output: output,
                    targets: targets,
                }
            }
        }
        impl Adaptor<LossInput> for RegressionOutput {
            fn adapt(&self) -> LossInput {
                LossInput::new(self.loss.clone())
            }
        }
        impl ItemLazy for RegressionOutput {
            fn sync(self) -> Self {
                let [output, loss, targets] = Transaction::default()
                    .register(self.output)
                    .register(self.loss)
                    .register(self.targets)
                    .execute()
                    .try_into()
                    .expect("Correct amount of tensor data");
                let device: Device = Device::flex();
                RegressionOutput {
                    output: Tensor::from_data(output, &device),
                    loss: Tensor::from_data(loss, &device),
                    targets: Tensor::from_data(targets, &device),
                }
            }
        }
    }
    mod sequence {
        use crate::metric::{AccuracyInput, PerplexityInput, TopKAccuracyInput};
        use crate::metric::{Adaptor, CerInput, LossInput, WerInput, processor::ItemLazy};
        use burn_core::tensor::{Device, Int, Tensor, Transaction};
        /// Sequence prediction output adapted for multiple metrics.
        ///
        /// Supported metrics:
        /// - Accuracy
        /// - TopKAccuracy
        /// - Perplexity
        /// - Loss
        /// - CER
        /// - WER
        pub struct SequenceOutput {
            /// The loss.
            pub loss: Tensor<1>,
            /// Raw logits. Shape: `[batch_size, seq_len, vocab_size]`
            pub logits: Tensor<3>,
            /// Optional predicted token indices. Shape: `[batch_size, seq_length]`.
            /// If not provided, predictions default to argmax of `logits` along the last dimension.
            pub predictions: Option<Tensor<2, Int>>,
            /// The target token indices. Shape: `[batch_size, seq_length]`
            pub targets: Tensor<2, Int>,
        }
        impl SequenceOutput {
            ///Constructs a new `SequenceOutput`.
            pub fn new(
                loss: Tensor<1>,
                logits: Tensor<3>,
                predictions: Option<Tensor<2, Int>>,
                targets: Tensor<2, Int>,
            ) -> Self {
                SequenceOutput {
                    loss: loss,
                    logits: logits,
                    predictions: predictions,
                    targets: targets,
                }
            }
        }
        impl SequenceOutput {
            fn predicted_tokens(&self) -> Tensor<2, Int> {
                match &self.predictions {
                    Some(preds) => preds.clone(),
                    None => self.logits.clone().argmax(2).squeeze_dim::<2>(2),
                }
            }
            fn flat_logits(&self) -> Tensor<2> {
                let [batch_size, seq_len, vocab_size] = self.logits.dims();
                self.logits.clone().reshape([batch_size * seq_len, vocab_size])
            }
            fn flat_targets(&self) -> Tensor<1, Int> {
                let [batch_size, seq_len] = self.targets.dims();
                self.targets.clone().reshape([batch_size * seq_len])
            }
        }
        impl ItemLazy for SequenceOutput {
            fn sync(self) -> Self {
                let device: Device = Device::flex();
                match self.predictions {
                    Some(preds) => {
                        let [logits, loss, targets, predictions] = Transaction::default()
                            .register(self.logits)
                            .register(self.loss)
                            .register(self.targets)
                            .register(preds)
                            .execute()
                            .try_into()
                            .expect("Correct amount of tensor data");
                        SequenceOutput {
                            logits: Tensor::from_data(logits, &device),
                            loss: Tensor::from_data(loss, &device),
                            targets: Tensor::from_data(targets, &device),
                            predictions: Some(Tensor::from_data(predictions, &device)),
                        }
                    }
                    None => {
                        let [logits, loss, targets] = Transaction::default()
                            .register(self.logits)
                            .register(self.loss)
                            .register(self.targets)
                            .execute()
                            .try_into()
                            .expect("Correct amount of tensor data");
                        SequenceOutput {
                            logits: Tensor::from_data(logits, &device),
                            loss: Tensor::from_data(loss, &device),
                            targets: Tensor::from_data(targets, &device),
                            predictions: None,
                        }
                    }
                }
            }
        }
        impl Adaptor<LossInput> for SequenceOutput {
            fn adapt(&self) -> LossInput {
                LossInput::new(self.loss.clone())
            }
        }
        impl Adaptor<CerInput> for SequenceOutput {
            fn adapt(&self) -> CerInput {
                CerInput::new(self.predicted_tokens(), self.targets.clone())
            }
        }
        impl Adaptor<WerInput> for SequenceOutput {
            fn adapt(&self) -> WerInput {
                WerInput::new(self.predicted_tokens(), self.targets.clone())
            }
        }
        impl Adaptor<AccuracyInput> for SequenceOutput {
            fn adapt(&self) -> AccuracyInput {
                AccuracyInput::new(self.flat_logits(), self.flat_targets())
            }
        }
        impl Adaptor<TopKAccuracyInput> for SequenceOutput {
            fn adapt(&self) -> TopKAccuracyInput {
                TopKAccuracyInput::new(self.flat_logits(), self.flat_targets())
            }
        }
        impl Adaptor<PerplexityInput> for SequenceOutput {
            fn adapt(&self) -> PerplexityInput {
                PerplexityInput::new(self.flat_logits(), self.flat_targets())
            }
        }
    }
    mod sharder {
        use burn_core::{Tensor, module::{Module, ModuleMapper, Param}};
        use crate::{Learner, LearningComponentsTypes};
        /// Describes how the module is distributed across multiple devices.
        pub struct ModuleSharder;
        impl ModuleMapper for ModuleSharder {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<D>>,
            ) -> Param<Tensor<D>> {
                let (id, tensor, mapper) = param.consume();
                let tensor = tensor.set_distributed(id);
                Param::from_mapped_value(id, tensor, mapper)
            }
        }
        impl<LC: LearningComponentsTypes> Learner<LC> {
            /// Mark the model as sharded across multiple devices.
            pub fn grad_sharded(&mut self) {
                self.model = self.model.clone().map(&mut ModuleSharder);
            }
        }
    }
    mod summary {
        use core::cmp::Ordering;
        use std::{
            collections::{HashMap, hash_map::Entry},
            fmt::Display, path::{Path, PathBuf},
        };
        use crate::{
            logger::FileMetricLogger,
            metric::store::{Aggregate, EventStore, LogEventStore, Split},
        };
        /// Contains the metric value at a given time.
        pub struct MetricEntry {
            /// The step at which the metric was recorded (i.e., epoch).
            pub step: usize,
            /// The metric value.
            pub value: f64,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for MetricEntry {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "MetricEntry",
                    "step",
                    &self.step,
                    "value",
                    &&self.value,
                )
            }
        }
        /// Contains the summary of recorded values for a given metric.
        pub struct MetricSummary {
            /// The metric name.
            pub name: String,
            /// The metric entries.
            pub entries: Vec<MetricEntry>,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for MetricSummary {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "MetricSummary",
                    "name",
                    &self.name,
                    "entries",
                    &&self.entries,
                )
            }
        }
        impl MetricSummary {
            fn collect<E: EventStore>(
                event_store: &mut E,
                metric: &str,
                split: &Split,
                num_epochs: usize,
            ) -> Option<Self> {
                let entries = (1..=num_epochs)
                    .filter_map(|epoch| {
                        event_store
                            .find_metric(metric, epoch, Aggregate::Mean, split)
                            .map(|value| MetricEntry { step: epoch, value })
                    })
                    .collect::<Vec<_>>();
                if entries.is_empty() {
                    None
                } else {
                    Some(Self {
                        name: metric.to_string(),
                        entries,
                    })
                }
            }
        }
        /// Contains the summary of recorded metrics for the training and validation steps.
        pub struct SummaryMetrics {
            /// Training metrics summary.
            pub train: Vec<MetricSummary>,
            /// Validation metrics summary.
            pub valid: Vec<MetricSummary>,
            /// Test metrics summary per test split tag.
            ///
            /// Each key corresponds to a `Split::Test(Some(tag))`.
            /// The empty string represents `Split::Test(None)`.
            pub test: HashMap<String, Vec<MetricSummary>>,
        }
        /// Detailed training summary.
        pub struct LearnerSummary {
            /// The number of epochs completed.
            pub epochs: usize,
            /// The summary of recorded metrics during training.
            pub metrics: SummaryMetrics,
            /// The model name (only recorded within the learner).
            pub(crate) model: Option<String>,
        }
        impl LearnerSummary {
            /// Creates a new learner summary for the specified metrics.
            ///
            /// # Arguments
            ///
            /// * `directory` - The directory containing the training artifacts (checkpoints and logs).
            /// * `metrics` - The list of metrics to collect for the summary.
            pub fn new<S: AsRef<str>>(
                directory: impl AsRef<Path>,
                metrics: &[S],
            ) -> Result<Self, String> {
                let directory = directory.as_ref();
                if !directory.exists() {
                    return Err(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "Artifact directory does not exist at: {0}",
                                    directory.display(),
                                ),
                            )
                        }),
                    );
                }
                let mut event_store = LogEventStore::default();
                let train_split = Split::Train;
                let valid_split = Split::Valid;
                let logger = FileMetricLogger::new(directory);
                let test_split_root = logger.split_dir(&Split::Test(None));
                if !logger.split_exists(&train_split)
                    && !logger.split_exists(&valid_split) && test_split_root.is_none()
                {
                    return Err(
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!(
                                    "No training, validation or test artifacts found at: {0}",
                                    directory.display(),
                                ),
                            )
                        }),
                    );
                }
                let epochs = logger.epochs();
                event_store.register_logger(logger);
                let train_summary = metrics
                    .iter()
                    .filter_map(|metric| {
                        MetricSummary::collect(
                            &mut event_store,
                            metric.as_ref(),
                            &train_split,
                            epochs,
                        )
                    })
                    .collect::<Vec<_>>();
                let valid_summary = metrics
                    .iter()
                    .filter_map(|metric| {
                        MetricSummary::collect(
                            &mut event_store,
                            metric.as_ref(),
                            &valid_split,
                            epochs,
                        )
                    })
                    .collect::<Vec<_>>();
                let test_summary = match test_split_root {
                    Some(root) => {
                        collect_test_split_metrics(
                            root,
                            metrics,
                            &mut event_store,
                            epochs,
                        )
                    }
                    None => Default::default(),
                };
                Ok(Self {
                    epochs,
                    metrics: SummaryMetrics {
                        train: train_summary,
                        valid: valid_summary,
                        test: test_summary,
                    },
                    model: None,
                })
            }
            pub(crate) fn with_model(mut self, name: String) -> Self {
                self.model = Some(name);
                self
            }
            /// Merges another summary into this one, combining all metric entries.
            pub(crate) fn merge(mut self, other: LearnerSummary) -> Self {
                fn merge_metrics(
                    base: Vec<MetricSummary>,
                    incoming: Vec<MetricSummary>,
                ) -> Vec<MetricSummary> {
                    let mut map: HashMap<String, MetricSummary> = base
                        .into_iter()
                        .map(|m| (m.name.clone(), m))
                        .collect();
                    for metric in incoming {
                        match map.entry(metric.name.clone()) {
                            Entry::Occupied(mut entry) => {
                                entry.get_mut().entries.extend(metric.entries);
                            }
                            Entry::Vacant(entry) => {
                                entry.insert(metric);
                            }
                        }
                    }
                    map.into_values().collect()
                }
                self.metrics.train = merge_metrics(
                    self.metrics.train,
                    other.metrics.train,
                );
                self.metrics.valid = merge_metrics(
                    self.metrics.valid,
                    other.metrics.valid,
                );
                for (tag, metrics) in other.metrics.test {
                    match self.metrics.test.entry(tag) {
                        Entry::Occupied(mut entry) => {
                            let current = std::mem::take(entry.get_mut());
                            let merged = merge_metrics(current, metrics);
                            *entry.get_mut() = merged;
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(metrics);
                        }
                    }
                }
                if self.model != other.model {
                    self.model = None;
                }
                self
            }
        }
        fn collect_test_split_metrics<P: AsRef<Path>, S: AsRef<str>>(
            root: P,
            metrics: &[S],
            event_store: &mut LogEventStore,
            epochs: usize,
        ) -> HashMap<String, Vec<MetricSummary>> {
            let dirs = match std::fs::read_dir(root) {
                Ok(entries) => {
                    entries
                        .filter_map(|entry| {
                            let entry = entry.ok()?;
                            let file_type = entry.file_type().ok()?;
                            if file_type.is_dir() {
                                Some(entry.file_name().to_string_lossy().to_string())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                }
                Err(_) => Vec::new(),
            };
            let mut map = HashMap::new();
            if dirs.is_empty() {
                return map;
            }
            let all_epochs = dirs.iter().all(FileMetricLogger::is_epoch_dir);
            if all_epochs {
                let split = Split::Test(None);
                let summaries = metrics
                    .iter()
                    .filter_map(|metric| {
                        MetricSummary::collect(
                            event_store,
                            metric.as_ref(),
                            &split,
                            epochs,
                        )
                    })
                    .collect::<Vec<_>>();
                map.insert("".to_string(), summaries);
            } else {
                for tag in dirs {
                    let split = Split::Test(Some(tag.clone().into()));
                    let summaries = metrics
                        .iter()
                        .filter_map(|metric| {
                            MetricSummary::collect(
                                event_store,
                                metric.as_ref(),
                                &split,
                                epochs,
                            )
                        })
                        .collect::<Vec<_>>();
                    map.insert(tag, summaries);
                }
            }
            map
        }
        impl Display for LearnerSummary {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut max_split_len = 5;
                let mut max_metric_len = "Metric".len();
                for metric in self.metrics.train.iter() {
                    max_metric_len = max_metric_len.max(metric.name.len());
                }
                for metric in self.metrics.valid.iter() {
                    max_metric_len = max_metric_len.max(metric.name.len());
                }
                for (tag, metrics) in self.metrics.test.iter() {
                    let split_name = if tag.is_empty() {
                        "Test".to_string()
                    } else {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("Test ({0})", tag))
                        })
                    };
                    max_split_len = max_split_len.max(split_name.len());
                    for metric in metrics {
                        max_metric_len = max_metric_len.max(metric.name.len());
                    }
                }
                f.write_fmt(
                    format_args!("{0:=>2$} Learner Summary {1:=>2$}\n", "", "", 24),
                )?;
                if let Some(model) = &self.model {
                    f.write_fmt(format_args!("Model:\n{0}\n", model))?;
                }
                f.write_fmt(format_args!("Total Epochs: {0}\n\n\n", self.epochs))?;
                f.write_fmt(
                    format_args!(
                        "| {0:<4$} | {1:<5$} | Min.     | Epoch    | Max.     | Epoch    |\n|{2:->4$}--|{3:->5$}--|----------|----------|----------|----------|\n",
                        "Split",
                        "Metric",
                        "",
                        "",
                        max_split_len,
                        max_metric_len,
                    ),
                )?;
                fn cmp_f64(a: &f64, b: &f64) -> Ordering {
                    match (a.is_nan(), b.is_nan()) {
                        (true, true) => Ordering::Equal,
                        (true, false) => Ordering::Greater,
                        (false, true) => Ordering::Less,
                        _ => a.partial_cmp(b).unwrap(),
                    }
                }
                fn fmt_val(val: f64) -> String {
                    if val < 1e-2 {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:<9.3e}", val))
                        })
                    } else {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:<9.3}", val))
                        })
                    }
                }
                let mut write_metrics_summary = |
                    metrics: &[MetricSummary],
                    split: String,
                | -> std::fmt::Result {
                    for metric in metrics.iter() {
                        if metric.entries.is_empty() {
                            continue;
                        }
                        let metric_min = metric
                            .entries
                            .iter()
                            .min_by(|a, b| cmp_f64(&a.value, &b.value))
                            .unwrap();
                        let metric_max = metric
                            .entries
                            .iter()
                            .max_by(|a, b| cmp_f64(&a.value, &b.value))
                            .unwrap();
                        f.write_fmt(
                            format_args!(
                                "| {0:<6$} | {1:<7$} | {2}| {3:<9?}| {4}| {5:<9?}|\n",
                                split,
                                metric.name,
                                fmt_val(metric_min.value),
                                metric_min.step,
                                fmt_val(metric_max.value),
                                metric_max.step,
                                max_split_len,
                                max_metric_len,
                            ),
                        )?;
                    }
                    Ok(())
                };
                write_metrics_summary(
                    &self.metrics.train,
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0:?}", Split::Train))
                    }),
                )?;
                write_metrics_summary(
                    &self.metrics.valid,
                    ::alloc::__export::must_use({
                        ::alloc::fmt::format(format_args!("{0:?}", Split::Valid))
                    }),
                )?;
                for (tag, metrics) in &self.metrics.test {
                    let split_name = if tag.is_empty() {
                        "Test".to_string()
                    } else {
                        ::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("Test ({0})", tag))
                        })
                    };
                    write_metrics_summary(metrics, split_name)?;
                }
                Ok(())
            }
        }
        /// Learning summary config.
        pub struct LearnerSummaryConfig {
            pub(crate) directory: PathBuf,
            pub(crate) metrics: Vec<String>,
        }
        #[automatically_derived]
        impl ::core::clone::Clone for LearnerSummaryConfig {
            #[inline]
            fn clone(&self) -> LearnerSummaryConfig {
                LearnerSummaryConfig {
                    directory: ::core::clone::Clone::clone(&self.directory),
                    metrics: ::core::clone::Clone::clone(&self.metrics),
                }
            }
        }
        impl LearnerSummaryConfig {
            /// Create the learning summary.
            pub fn init(&self) -> Result<LearnerSummary, String> {
                LearnerSummary::new(&self.directory, &self.metrics[..])
            }
        }
    }
    mod supervised {
        mod paradigm {
            use crate::checkpoint::{
                AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy,
                FileCheckpointer, KeepLastNCheckpoints, MetricCheckpointingStrategy,
            };
            use crate::components::{InferenceModelOutput, TrainingModelOutput};
            use crate::learner::EarlyStoppingStrategy;
            use crate::learner::base::Interrupter;
            use crate::logger::{FileMetricLogger, MetricLogger, TrainingProgressLogger};
            use crate::metric::processor::{
                AsyncProcessorTraining, FullEventProcessorTraining, MetricsTraining,
            };
            use crate::metric::store::{
                Aggregate, Direction, EventStoreClient, LogEventStore, Split,
            };
            use crate::metric::{Adaptor, LossMetric, Metric, Numeric};
            use crate::multi::MultiDeviceLearningStrategy;
            use crate::renderer::{MetricsRenderer, default_renderer};
            use crate::single::SingleDeviceTrainingStrategy;
            use crate::{
                ApplicationLoggerInstaller, EarlyStoppingStrategyRef, ExecutionStrategy,
                FileApplicationLoggerInstaller, InferenceModel, InferenceModelInput,
                InferenceStep, LearnerEvent, LearnerModelRecord, LearnerOptimizerRecord,
                LearnerSchedulerRecord, LearnerSummaryConfig, LearningCheckpointer,
                LearningComponentsMarker, LearningComponentsTypes, LearningResult,
                TrainStep, TrainingComponents, TrainingModelInput, TrainingStrategy,
            };
            use crate::{Learner, SupervisedLearningStrategy};
            use burn_core::data::dataloader::DataLoader;
            use burn_core::module::{AutodiffModule, Module};
            use burn_core::record::FileRecorder;
            use burn_core::tensor::Device;
            use burn_optim::Optimizer;
            use burn_optim::lr_scheduler::LrScheduler;
            use std::collections::BTreeSet;
            use std::path::{Path, PathBuf};
            use std::sync::Arc;
            use typing_rules::*;
            /// A reference to the training split [DataLoader](DataLoader).
            pub type TrainLoader<LC, L: Label> = Arc<
                dyn DataLoader<TrainingModelInput<LC>, L>,
            >;
            /// A reference to the validation split [DataLoader](DataLoader).
            pub type ValidLoader<LC, L: Label> = Arc<
                dyn DataLoader<InferenceModelInput<LC>, L>,
            >;
            /// The event processor type for supervised learning.
            pub type SupervisedTrainingEventProcessor<LC> = AsyncProcessorTraining<
                LearnerEvent<TrainingModelOutput<LC>>,
                LearnerEvent<InferenceModelOutput<LC>>,
            >;
            /// Structure to configure and launch supervised learning trainings.
            pub struct SupervisedTraining<LC, L>
            where
                LC: LearningComponentsTypes,
                L: Label,
            {
                #[allow(clippy::type_complexity)]
                checkpointers: Option<
                    (
                        AsyncCheckpointer<LearnerModelRecord<LC>>,
                        AsyncCheckpointer<LearnerOptimizerRecord<LC>>,
                        AsyncCheckpointer<LearnerSchedulerRecord<LC>>,
                    ),
                >,
                num_epochs: usize,
                checkpoint: Option<usize>,
                directory: PathBuf,
                grad_accumulation: Option<usize>,
                grad_checkpointing: bool,
                renderer: Option<Box<dyn MetricsRenderer + 'static>>,
                metrics: MetricsTraining<
                    TrainingModelOutput<LC>,
                    InferenceModelOutput<LC>,
                >,
                event_store: LogEventStore,
                interrupter: Interrupter,
                tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
                checkpointer_strategy: Box<dyn CheckpointingStrategy>,
                early_stopping: Option<EarlyStoppingStrategyRef>,
                training_strategy: Option<TrainingStrategy<LC, L>>,
                dataloader_train: TrainLoader<LC, L>,
                dataloader_valid: ValidLoader<LC, L>,
                summary_metrics: BTreeSet<String>,
                summary: bool,
                progress_logger: Option<Box<dyn TrainingProgressLogger>>,
            }
            impl<LR, M, O, L> SupervisedTraining<LearningComponentsMarker<LR, M, O>, L>
            where
                LR: LrScheduler + 'static,
                M: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display
                    + 'static,
                O: Optimizer<M> + 'static,
                L: Label,
            {
                /// Creates a new runner for a supervised training.
                ///
                /// # Arguments
                ///
                /// * `directory` - The directory to save the checkpoints.
                /// * `dataloader_train` - The dataloader for the training split.
                /// * `dataloader_valid` - The dataloader for the validation split.
                pub fn new(
                    directory: impl AsRef<Path>,
                    dataloader_train: Arc<dyn DataLoader<<M as TrainStep>::Input, L>>,
                    dataloader_valid: Arc<dyn DataLoader<<M as InferenceStep>::Input, L>>,
                ) -> Self {
                    let directory = directory.as_ref().to_path_buf();
                    let experiment_log_file = directory.join("experiment.log");
                    Self {
                        num_epochs: 1,
                        checkpoint: None,
                        checkpointers: None,
                        directory,
                        grad_accumulation: None,
                        grad_checkpointing: false,
                        metrics: MetricsTraining::default(),
                        event_store: LogEventStore::default(),
                        renderer: None,
                        interrupter: Interrupter::new(),
                        tracing_logger: Some(
                            Box::new(
                                FileApplicationLoggerInstaller::new(experiment_log_file),
                            ),
                        ),
                        checkpointer_strategy: Box::new(
                            ComposedCheckpointingStrategy::builder()
                                .add(KeepLastNCheckpoints::new(2))
                                .add(
                                    MetricCheckpointingStrategy::new(
                                        &LossMetric::new(),
                                        Aggregate::Mean,
                                        Direction::Lowest,
                                        Split::Valid,
                                    ),
                                )
                                .build(),
                        ),
                        early_stopping: None,
                        training_strategy: None,
                        summary_metrics: BTreeSet::new(),
                        summary: false,
                        dataloader_train,
                        dataloader_valid,
                        progress_logger: None,
                    }
                }
            }
            impl<LC: LearningComponentsTypes, L: Label> SupervisedTraining<LC, L> {
                /// Replace the default training strategy (SingleDeviceTrainingStrategy) with the provided one.
                ///
                /// # Arguments
                ///
                /// * `training_strategy` - The training strategy.
                pub fn with_training_strategy(
                    mut self,
                    training_strategy: TrainingStrategy<LC, L>,
                ) -> Self {
                    self.training_strategy = Some(training_strategy);
                    self
                }
                /// Replace the default metric loggers with the provided ones.
                ///
                /// # Arguments
                ///
                /// * `logger` - The training logger.
                pub fn with_metric_logger<ML>(mut self, logger: ML) -> Self
                where
                    ML: MetricLogger + 'static,
                {
                    self.event_store.register_logger(logger);
                    self
                }
                /// Register a progress logger to track and store training progress.
                ///
                /// # Example
                ///
                /// ```ignore
                /// // `MyTrainingProgressLogger` is a user-defined type that implements
                /// // `burn_train::logger::TrainingProgressLogger`.
                /// let learner = SupervisedTraining::new(...)
                ///     .with_progress_logger(MyTrainingProgressLogger);
                /// ```
                pub fn with_progress_logger<PL>(mut self, logger: PL) -> Self
                where
                    PL: TrainingProgressLogger + 'static,
                {
                    self.progress_logger = Some(Box::new(logger));
                    self
                }
                /// Update the checkpointing_strategy.
                pub fn with_checkpointing_strategy<CS: CheckpointingStrategy + 'static>(
                    mut self,
                    strategy: CS,
                ) -> Self {
                    self.checkpointer_strategy = Box::new(strategy);
                    self
                }
                /// Replace the default CLI renderer with a custom one.
                ///
                /// # Arguments
                ///
                /// * `renderer` - The custom renderer.
                pub fn renderer<MR>(mut self, renderer: MR) -> Self
                where
                    MR: MetricsRenderer + 'static,
                {
                    self.renderer = Some(Box::new(renderer));
                    self
                }
                /// Register all metrics as numeric for the training and validation set.
                pub fn metrics<Me: MetricRegistration<LC, L>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register all metrics as text for the training and validation set.
                pub fn metrics_text<Me: TextMetricRegistration<LC, L>>(
                    self,
                    metrics: Me,
                ) -> Self {
                    metrics.register(self)
                }
                /// Register a training metric.
                pub fn metric_train<Me: Metric + 'static>(mut self, metric: Me) -> Self
                where
                    TrainingModelOutput<LC>: Adaptor<Me::Input>,
                {
                    self.metrics.register_train_metric(metric);
                    self
                }
                /// Register a validation metric.
                pub fn metric_valid<Me: Metric + 'static>(mut self, metric: Me) -> Self
                where
                    InferenceModelOutput<LC>: Adaptor<Me::Input>,
                {
                    self.metrics.register_valid_metric(metric);
                    self
                }
                /// Enable gradients accumulation.
                ///
                /// # Notes
                ///
                /// When you enable gradients accumulation, the gradients object used by the optimizer will be
                /// the sum of all gradients generated by each backward pass. It might be a good idea to
                /// reduce the learning to compensate.
                ///
                /// The effect is similar to increasing the `batch size` and the `learning rate` by the `accumulation`
                /// amount.
                pub fn grads_accumulation(mut self, accumulation: usize) -> Self {
                    self.grad_accumulation = Some(accumulation);
                    self
                }
                /// Enables autodiff checkpointing.
                ///
                /// # Notes
                /// Gradient checkpointing recomputes activations during backpropagation for operations
                /// marked as memory-bound, while compute-bound operations still cache their
                /// output. This reduces peak memory usage at the cost of additional computation
                /// for memory-bound ops.
                pub fn gradient_checkpointing(mut self) -> Self {
                    self.grad_checkpointing = true;
                    self
                }
                /// Register a [numeric](crate::metric::Numeric) training [metric](Metric).
                pub fn metric_train_numeric<Me>(mut self, metric: Me) -> Self
                where
                    Me: Metric + Numeric + 'static,
                    TrainingModelOutput<LC>: Adaptor<Me::Input>,
                {
                    self.summary_metrics.insert(metric.name().to_string());
                    self.metrics.register_train_metric_numeric(metric);
                    self
                }
                /// Register a [numeric](crate::metric::Numeric) validation [metric](Metric).
                pub fn metric_valid_numeric<Me: Metric + Numeric + 'static>(
                    mut self,
                    metric: Me,
                ) -> Self
                where
                    InferenceModelOutput<LC>: Adaptor<Me::Input>,
                {
                    self.summary_metrics.insert(metric.name().to_string());
                    self.metrics.register_valid_metric_numeric(metric);
                    self
                }
                /// The number of epochs the training should last.
                pub fn num_epochs(mut self, num_epochs: usize) -> Self {
                    self.num_epochs = num_epochs;
                    self
                }
                /// The epoch from which the training must resume.
                pub fn checkpoint(mut self, checkpoint: usize) -> Self {
                    self.checkpoint = Some(checkpoint);
                    self
                }
                /// Provides a handle that can be used to interrupt training.
                pub fn interrupter(&self) -> Interrupter {
                    self.interrupter.clone()
                }
                /// Override the handle for stopping training with an externally provided handle
                pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
                    self.interrupter = interrupter;
                    self
                }
                /// Register an [early stopping strategy](EarlyStoppingStrategy) to stop the training when the
                /// conditions are meet.
                pub fn early_stopping<Strategy>(mut self, strategy: Strategy) -> Self
                where
                    Strategy: EarlyStoppingStrategy + Clone + Send + Sync + 'static,
                {
                    self.early_stopping = Some(Box::new(strategy));
                    self
                }
                /// By default, Rust logs are captured and written into
                /// `experiment.log`. If disabled, standard Rust log handling
                /// will apply.
                pub fn with_application_logger(
                    mut self,
                    logger: Option<Box<dyn ApplicationLoggerInstaller>>,
                ) -> Self {
                    self.tracing_logger = logger;
                    self
                }
                /// Register a checkpointer that will save the [optimizer](Optimizer), the
                /// [model](AutodiffModule) and the [scheduler](LrScheduler) to different files.
                pub fn with_file_checkpointer<FR>(mut self, recorder: FR) -> Self
                where
                    FR: FileRecorder + 'static,
                    FR: FileRecorder + 'static,
                {
                    let checkpoint_dir = self.directory.join("checkpoint");
                    let checkpointer_model = FileCheckpointer::new(
                        recorder.clone(),
                        &checkpoint_dir,
                        "model",
                    );
                    let checkpointer_optimizer = FileCheckpointer::new(
                        recorder.clone(),
                        &checkpoint_dir,
                        "optim",
                    );
                    let checkpointer_scheduler: FileCheckpointer<FR> = FileCheckpointer::new(
                        recorder,
                        &checkpoint_dir,
                        "scheduler",
                    );
                    self.checkpointers = Some((
                        AsyncCheckpointer::new(checkpointer_model),
                        AsyncCheckpointer::new(checkpointer_optimizer),
                        AsyncCheckpointer::new(checkpointer_scheduler),
                    ));
                    self
                }
                /// Enable the training summary report.
                ///
                /// The summary will be displayed after `.fit()`, when the renderer is dropped.
                pub fn summary(mut self) -> Self {
                    self.summary = true;
                    self
                }
            }
            impl<LC, L> SupervisedTraining<LC, L>
            where
                LC: LearningComponentsTypes + Send + 'static,
                L: Label,
            {
                /// Launch this training with the given [Learner](Learner).
                pub fn launch(
                    mut self,
                    learner: Learner<LC>,
                ) -> LearningResult<InferenceModel<LC>> {
                    if self.tracing_logger.is_some()
                        && let Err(e) = self.tracing_logger.as_ref().unwrap().install()
                    {
                        {
                            {
                                let lvl = ::log::Level::Warn;
                                if lvl <= ::log::STATIC_MAX_LEVEL
                                    && lvl <= ::log::max_level()
                                {
                                    ::log::__private_api::log(
                                        { ::log::__private_api::GlobalLogger },
                                        format_args!(
                                            "Failed to install the experiment logger: {0}",
                                            e,
                                        ),
                                        lvl,
                                        &(
                                            "burn_train::learner::supervised::paradigm",
                                            "burn_train::learner::supervised::paradigm",
                                            ::log::__private_api::loc(),
                                        ),
                                        (),
                                    );
                                }
                            }
                        };
                    }
                    let renderer = self
                        .renderer
                        .unwrap_or_else(|| default_renderer(
                            self.interrupter.clone(),
                            self.checkpoint,
                        ));
                    if !self.event_store.has_loggers() {
                        self.event_store
                            .register_logger(
                                FileMetricLogger::new(self.directory.clone()),
                            );
                    }
                    let event_store = Arc::new(EventStoreClient::new(self.event_store));
                    let full_processor = FullEventProcessorTraining::new(
                        self.metrics,
                        renderer,
                        event_store.clone(),
                    );
                    let full_processor = match self.progress_logger {
                        Some(logger) => full_processor.with_progress_logger(logger),
                        None => full_processor,
                    };
                    let event_processor = AsyncProcessorTraining::new(full_processor);
                    let checkpointer = self
                        .checkpointers
                        .map(|(model, optim, scheduler)| {
                            LearningCheckpointer::new(
                                model.with_interrupter(self.interrupter.clone()),
                                optim.with_interrupter(self.interrupter.clone()),
                                scheduler.with_interrupter(self.interrupter.clone()),
                                self.checkpointer_strategy,
                            )
                        });
                    let summary = if self.summary {
                        Some(LearnerSummaryConfig {
                            directory: self.directory,
                            metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
                        })
                    } else {
                        None
                    };
                    let components = TrainingComponents {
                        checkpoint: self.checkpoint,
                        checkpointer,
                        interrupter: self.interrupter,
                        early_stopping: self.early_stopping,
                        event_processor,
                        event_store,
                        num_epochs: self.num_epochs,
                        grad_accumulation: self.grad_accumulation,
                        summary,
                    };
                    let training_strategy = self
                        .training_strategy
                        .unwrap_or(
                            TrainingStrategy::Default(
                                ExecutionStrategy::SingleDevice(
                                    autodiff_device(
                                        learner.model.devices()[0].clone(),
                                        self.grad_checkpointing,
                                    ),
                                ),
                            ),
                        );
                    match training_strategy {
                        TrainingStrategy::Custom(learning_paradigm) => {
                            learning_paradigm
                                .train(
                                    learner,
                                    self.dataloader_train,
                                    self.dataloader_valid,
                                    components,
                                )
                        }
                        TrainingStrategy::Default(strategy) => {
                            match strategy {
                                ExecutionStrategy::SingleDevice(device) => {
                                    let single_device = SingleDeviceTrainingStrategy::new(
                                        autodiff_device(device, self.grad_checkpointing),
                                    );
                                    single_device
                                        .train(
                                            learner,
                                            self.dataloader_train,
                                            self.dataloader_valid,
                                            components,
                                        )
                                }
                                ExecutionStrategy::MultiDevice(
                                    devices,
                                    multi_device_optim,
                                ) => {
                                    let strategy: Box<dyn SupervisedLearningStrategy<LC, L>> = match devices
                                        .len() == 1
                                    {
                                        true => {
                                            Box::new(
                                                SingleDeviceTrainingStrategy::new(
                                                    autodiff_device(devices[0].clone(), self.grad_checkpointing),
                                                ),
                                            )
                                        }
                                        false => {
                                            Box::new(
                                                MultiDeviceLearningStrategy::new(
                                                    devices
                                                        .into_iter()
                                                        .map(|d| autodiff_device(d, self.grad_checkpointing))
                                                        .collect(),
                                                    multi_device_optim,
                                                ),
                                            )
                                        }
                                    };
                                    strategy
                                        .train(
                                            learner,
                                            self.dataloader_train,
                                            self.dataloader_valid,
                                            components,
                                        )
                                }
                                ExecutionStrategy::DistributedDataParallel {
                                    devices,
                                    context,
                                } => {
                                    use crate::ddp::DdpTrainingStrategy;
                                    let ddp = DdpTrainingStrategy::new(
                                        devices
                                            .into_iter()
                                            .map(|d| autodiff_device(d, self.grad_checkpointing))
                                            .collect(),
                                        context,
                                    );
                                    ddp.train(
                                        learner,
                                        self.dataloader_train,
                                        self.dataloader_valid,
                                        components,
                                    )
                                }
                            }
                        }
                    }
                }
            }
            fn autodiff_device(mut device: Device, grad_checkpointing: bool) -> Device {
                if !device.is_autodiff() {
                    device = device.autodiff();
                }
                if grad_checkpointing {
                    device = device.gradient_checkpointing();
                }
                device
            }
            /// Trait to fake variadic generics.
            pub trait MetricRegistration<LC: LearningComponentsTypes, L: Label>: Sized {
                /// Register the metrics.
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L>;
            }
            /// Trait to fake variadic generics.
            pub trait TextMetricRegistration<
                LC: LearningComponentsTypes,
                L: Label,
            >: Sized {
                /// Register the metrics.
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L>;
            }
            impl<M1, LC: LearningComponentsTypes, L: Label> TextMetricRegistration<LC, L>
            for (M1,)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                M1: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1,) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_valid(M1);
                    builder
                }
            }
            impl<M1, LC: LearningComponentsTypes, L: Label> MetricRegistration<LC, L>
            for (M1,)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                M1: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1,) = self;
                    let builder = builder.metric_train_numeric(M1.clone());
                    let builder = builder.metric_valid_numeric(M1);
                    builder
                }
            }
            impl<
                M1,
                M2,
                LC: LearningComponentsTypes,
                L: Label,
            > TextMetricRegistration<LC, L> for (M1, M2)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_valid(M1);
                    let builder = builder.metric_valid(M2);
                    builder
                }
            }
            impl<M1, M2, LC: LearningComponentsTypes, L: Label> MetricRegistration<LC, L>
            for (M1, M2)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2) = self;
                    let builder = builder.metric_train_numeric(M1.clone());
                    let builder = builder.metric_train_numeric(M2.clone());
                    let builder = builder.metric_valid_numeric(M1);
                    let builder = builder.metric_valid_numeric(M2);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                LC: LearningComponentsTypes,
                L: Label,
            > TextMetricRegistration<LC, L> for (M1, M2, M3)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    let builder = builder.metric_valid(M1);
                    let builder = builder.metric_valid(M2);
                    let builder = builder.metric_valid(M3);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                LC: LearningComponentsTypes,
                L: Label,
            > MetricRegistration<LC, L> for (M1, M2, M3)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3) = self;
                    let builder = builder.metric_train_numeric(M1.clone());
                    let builder = builder.metric_train_numeric(M2.clone());
                    let builder = builder.metric_train_numeric(M3.clone());
                    let builder = builder.metric_valid_numeric(M1);
                    let builder = builder.metric_valid_numeric(M2);
                    let builder = builder.metric_valid_numeric(M3);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                LC: LearningComponentsTypes,
                L: Label,
            > TextMetricRegistration<LC, L> for (M1, M2, M3, M4)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                TrainingModelOutput<LC>: Adaptor<M4::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M4::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    let builder = builder.metric_train(M4.clone());
                    let builder = builder.metric_valid(M1);
                    let builder = builder.metric_valid(M2);
                    let builder = builder.metric_valid(M3);
                    let builder = builder.metric_valid(M4);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                LC: LearningComponentsTypes,
                L: Label,
            > MetricRegistration<LC, L> for (M1, M2, M3, M4)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                TrainingModelOutput<LC>: Adaptor<M4::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M4::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3, M4) = self;
                    let builder = builder.metric_train_numeric(M1.clone());
                    let builder = builder.metric_train_numeric(M2.clone());
                    let builder = builder.metric_train_numeric(M3.clone());
                    let builder = builder.metric_train_numeric(M4.clone());
                    let builder = builder.metric_valid_numeric(M1);
                    let builder = builder.metric_valid_numeric(M2);
                    let builder = builder.metric_valid_numeric(M3);
                    let builder = builder.metric_valid_numeric(M4);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                LC: LearningComponentsTypes,
                L: Label,
            > TextMetricRegistration<LC, L> for (M1, M2, M3, M4, M5)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                TrainingModelOutput<LC>: Adaptor<M4::Input>,
                TrainingModelOutput<LC>: Adaptor<M5::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M4::Input>,
                InferenceModelOutput<LC>: Adaptor<M5::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    let builder = builder.metric_train(M4.clone());
                    let builder = builder.metric_train(M5.clone());
                    let builder = builder.metric_valid(M1);
                    let builder = builder.metric_valid(M2);
                    let builder = builder.metric_valid(M3);
                    let builder = builder.metric_valid(M4);
                    let builder = builder.metric_valid(M5);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                LC: LearningComponentsTypes,
                L: Label,
            > MetricRegistration<LC, L> for (M1, M2, M3, M4, M5)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                TrainingModelOutput<LC>: Adaptor<M4::Input>,
                TrainingModelOutput<LC>: Adaptor<M5::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M4::Input>,
                InferenceModelOutput<LC>: Adaptor<M5::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3, M4, M5) = self;
                    let builder = builder.metric_train_numeric(M1.clone());
                    let builder = builder.metric_train_numeric(M2.clone());
                    let builder = builder.metric_train_numeric(M3.clone());
                    let builder = builder.metric_train_numeric(M4.clone());
                    let builder = builder.metric_train_numeric(M5.clone());
                    let builder = builder.metric_valid_numeric(M1);
                    let builder = builder.metric_valid_numeric(M2);
                    let builder = builder.metric_valid_numeric(M3);
                    let builder = builder.metric_valid_numeric(M4);
                    let builder = builder.metric_valid_numeric(M5);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                LC: LearningComponentsTypes,
                L: Label,
            > TextMetricRegistration<LC, L> for (M1, M2, M3, M4, M5, M6)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                TrainingModelOutput<LC>: Adaptor<M4::Input>,
                TrainingModelOutput<LC>: Adaptor<M5::Input>,
                TrainingModelOutput<LC>: Adaptor<M6::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M4::Input>,
                InferenceModelOutput<LC>: Adaptor<M5::Input>,
                InferenceModelOutput<LC>: Adaptor<M6::Input>,
                M1: Metric + 'static,
                M2: Metric + 'static,
                M3: Metric + 'static,
                M4: Metric + 'static,
                M5: Metric + 'static,
                M6: Metric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.metric_train(M1.clone());
                    let builder = builder.metric_train(M2.clone());
                    let builder = builder.metric_train(M3.clone());
                    let builder = builder.metric_train(M4.clone());
                    let builder = builder.metric_train(M5.clone());
                    let builder = builder.metric_train(M6.clone());
                    let builder = builder.metric_valid(M1);
                    let builder = builder.metric_valid(M2);
                    let builder = builder.metric_valid(M3);
                    let builder = builder.metric_valid(M4);
                    let builder = builder.metric_valid(M5);
                    let builder = builder.metric_valid(M6);
                    builder
                }
            }
            impl<
                M1,
                M2,
                M3,
                M4,
                M5,
                M6,
                LC: LearningComponentsTypes,
                L: Label,
            > MetricRegistration<LC, L> for (M1, M2, M3, M4, M5, M6)
            where
                TrainingModelOutput<LC>: Adaptor<M1::Input>,
                TrainingModelOutput<LC>: Adaptor<M2::Input>,
                TrainingModelOutput<LC>: Adaptor<M3::Input>,
                TrainingModelOutput<LC>: Adaptor<M4::Input>,
                TrainingModelOutput<LC>: Adaptor<M5::Input>,
                TrainingModelOutput<LC>: Adaptor<M6::Input>,
                InferenceModelOutput<LC>: Adaptor<M1::Input>,
                InferenceModelOutput<LC>: Adaptor<M2::Input>,
                InferenceModelOutput<LC>: Adaptor<M3::Input>,
                InferenceModelOutput<LC>: Adaptor<M4::Input>,
                InferenceModelOutput<LC>: Adaptor<M5::Input>,
                InferenceModelOutput<LC>: Adaptor<M6::Input>,
                M1: Metric + Numeric + 'static,
                M2: Metric + Numeric + 'static,
                M3: Metric + Numeric + 'static,
                M4: Metric + Numeric + 'static,
                M5: Metric + Numeric + 'static,
                M6: Metric + Numeric + 'static,
            {
                #[allow(non_snake_case)]
                fn register(
                    self,
                    builder: SupervisedTraining<LC, L>,
                ) -> SupervisedTraining<LC, L> {
                    let (M1, M2, M3, M4, M5, M6) = self;
                    let builder = builder.metric_train_numeric(M1.clone());
                    let builder = builder.metric_train_numeric(M2.clone());
                    let builder = builder.metric_train_numeric(M3.clone());
                    let builder = builder.metric_train_numeric(M4.clone());
                    let builder = builder.metric_train_numeric(M5.clone());
                    let builder = builder.metric_train_numeric(M6.clone());
                    let builder = builder.metric_valid_numeric(M1);
                    let builder = builder.metric_valid_numeric(M2);
                    let builder = builder.metric_valid_numeric(M3);
                    let builder = builder.metric_valid_numeric(M4);
                    let builder = builder.metric_valid_numeric(M5);
                    let builder = builder.metric_valid_numeric(M6);
                    builder
                }
            }
        }
        mod step {
            /// The trainer module.
            pub mod train {
                use crate::{LearningComponentsTypes, TrainingModel};
                use crate::{
                    TrainOutput, TrainStep, TrainingModelInput, TrainingModelOutput,
                };
                use burn_core::data::dataloader::DataLoaderIterator;
                use burn_core::data::dataloader::Progress;
                use burn_core::module::Module;
                use burn_core::tensor::Device;
                use std::sync::mpsc::{Receiver, Sender};
                use std::thread::spawn;
                use typing_rules::*;
                use macros::{fcall, mcall};
                /// Multi devices train step.
                pub struct MultiDevicesTrainStep<LC: LearningComponentsTypes, L: Label> {
                    workers: Vec<Worker<LC, L>>,
                    receiver: Receiver<MultiTrainOutput<TrainingModelOutput<LC>>>,
                }
                struct Message<M, TI> {
                    item: TI,
                    model: M,
                }
                struct Worker<LC: LearningComponentsTypes, L: Label> {
                    sender_input: Sender<
                        Message<TrainingModel<LC>, Labeled<TrainingModelInput<LC>, L>>,
                    >,
                    device: Device,
                    device_id: usize,
                }
                impl<LC: LearningComponentsTypes, L: Label> Worker<LC, L> {
                    fn register(
                        &self,
                        item: Labeled<TrainingModelInput<LC>, L>,
                        model: &TrainingModel<LC>,
                    ) {
                        let message = Message {
                            item,
                            model: model.clone(),
                        };
                        self.sender_input.send(message).unwrap();
                    }
                    fn start(
                        &self,
                        sender_output: Sender<MultiTrainOutput<TrainingModelOutput<LC>>>,
                        receiver_input: Receiver<
                            Message<
                                TrainingModel<LC>,
                                Labeled<TrainingModelInput<LC>, L>,
                            >,
                        >,
                    ) {
                        let device = self.device.clone();
                        let device_id = self.device_id;
                        spawn(move || {
                            loop {
                                match receiver_input.recv() {
                                    Ok(item) => {
                                        let model = item.model.fork(&device);
                                        let _: Labeled<TrainingModelInput<LC>, L> = item.item;
                                        let output = {
                                            use ::typing_rules::function_rewrite::SecureChain;
                                            use ::typing_rules::function_rewrite::SecureChainRef;
                                            let __fcall_prev_hook = ::std::panic::take_hook();
                                            ::std::panic::set_hook(::std::boxed::Box::new(|_| {}));
                                            struct __FcallPanicGuard(
                                                ::std::option::Option<
                                                    ::std::boxed::Box<dyn ::std::ops::FnMut()>,
                                                >,
                                            );
                                            impl ::std::ops::Drop for __FcallPanicGuard {
                                                fn drop(&mut self) {
                                                    if !::std::thread::panicking() {
                                                        if let ::std::option::Option::Some(mut f) = self.0.take() {
                                                            f();
                                                        }
                                                    }
                                                }
                                            }
                                            let mut __fcall_hook_opt = ::std::option::Option::Some(
                                                __fcall_prev_hook,
                                            );
                                            let __fcall_panic_guard = __FcallPanicGuard(
                                                ::std::option::Option::Some(
                                                    ::std::boxed::Box::new(move || {
                                                        if let ::std::option::Option::Some(hook) = __fcall_hook_opt
                                                            .take()
                                                        {
                                                            ::std::panic::set_hook(hook);
                                                        }
                                                    }),
                                                ),
                                            );
                                            let __fcall_result = {
                                                (model)
                                                    .__chain_ref(|__v0| {
                                                        ((item.item))
                                                            .__chain(|__v1| {
                                                                ::typing_rules::lattice::Labeled::<
                                                                    _,
                                                                    ::typing_rules::lattice::Public,
                                                                >::new(TrainStep::step(__v0, __v1))
                                                            })
                                                    })
                                            };
                                            drop(__fcall_panic_guard);
                                            __fcall_result
                                        };
                                        let output = {
                                            use ::typing_rules::function_rewrite::SecureChain;
                                            use ::typing_rules::function_rewrite::SecureChainRef;
                                            let __fcall_prev_hook = ::std::panic::take_hook();
                                            ::std::panic::set_hook(::std::boxed::Box::new(|_| {}));
                                            struct __FcallPanicGuard(
                                                ::std::option::Option<
                                                    ::std::boxed::Box<dyn ::std::ops::FnMut()>,
                                                >,
                                            );
                                            impl ::std::ops::Drop for __FcallPanicGuard {
                                                fn drop(&mut self) {
                                                    if !::std::thread::panicking() {
                                                        if let ::std::option::Option::Some(mut f) = self.0.take() {
                                                            f();
                                                        }
                                                    }
                                                }
                                            }
                                            let mut __fcall_hook_opt = ::std::option::Option::Some(
                                                __fcall_prev_hook,
                                            );
                                            let __fcall_panic_guard = __FcallPanicGuard(
                                                ::std::option::Option::Some(
                                                    ::std::boxed::Box::new(move || {
                                                        if let ::std::option::Option::Some(hook) = __fcall_hook_opt
                                                            .take()
                                                        {
                                                            ::std::panic::set_hook(hook);
                                                        }
                                                    }),
                                                ),
                                            );
                                            let __fcall_result = {
                                                (model)
                                                    .__chain_ref(|__v0| {
                                                        ((item.item))
                                                            .__chain(|__v1| {
                                                                ::typing_rules::lattice::Labeled::<
                                                                    _,
                                                                    ::typing_rules::lattice::Public,
                                                                >::new(TrainingModel::<LC>::step(__v0, __v1))
                                                            })
                                                    })
                                            };
                                            drop(__fcall_panic_guard);
                                            __fcall_result
                                        };
                                        let item = MultiTrainOutput {
                                            output,
                                            device_id,
                                        };
                                        sender_output.send(item).unwrap();
                                    }
                                    Err(_err) => {
                                        {
                                            {
                                                let lvl = ::log::Level::Info;
                                                if lvl <= ::log::STATIC_MAX_LEVEL
                                                    && lvl <= ::log::max_level()
                                                {
                                                    ::log::__private_api::log(
                                                        { ::log::__private_api::GlobalLogger },
                                                        format_args!("Closing thread on device {0:?}", device),
                                                        lvl,
                                                        &(
                                                            "burn_train::learner::supervised::step::train",
                                                            "burn_train::learner::supervised::step::train",
                                                            ::log::__private_api::loc(),
                                                        ),
                                                        (),
                                                    );
                                                }
                                            }
                                        };
                                        break;
                                    }
                                }
                            }
                        });
                    }
                }
                /// Multiple output items.
                pub struct MultiTrainOutput<TO> {
                    /// The training output.
                    pub output: TrainOutput<TO>,
                    /// The worker/device on which the computing happened.
                    pub(crate) device_id: usize,
                }
                impl<
                    LC: LearningComponentsTypes,
                    L: Label,
                > MultiDevicesTrainStep<LC, L> {
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
                        dataloaders: &mut [Box<
                            dyn DataLoaderIterator<TrainingModelInput<LC>, L> + 'a,
                        >],
                        model: &TrainingModel<LC>,
                    ) -> (Vec<MultiTrainOutput<TrainingModelOutput<LC>>>, Progress) {
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
            }
        }
        mod strategies {
            mod base {
                use crate::{
                    EarlyStoppingStrategyRef, InferenceModel, Interrupter, Learner,
                    LearnerSummaryConfig, LearningCheckpointer, LearningResult,
                    SupervisedTrainingEventProcessor, TrainLoader, TrainingModel,
                    ValidLoader, components::LearningComponentsTypes,
                    metric::{
                        processor::{EventProcessorTraining, LearnerEvent},
                        store::EventStoreClient,
                    },
                };
                use burn_core::tensor::distributed::{
                    DistributedConfig, DistributedContext,
                };
                use burn_core::{module::AutodiffModule, prelude::Device};
                use std::sync::Arc;
                use typing_rules::*;
                /// A reference to an implementation of SupervisedLearningStrategy.
                pub type CustomLearningStrategy<LC, L: Label> = Arc<
                    dyn SupervisedLearningStrategy<LC, L>,
                >;
                /// Determine how the optimization is performed when training with multiple devices.
                pub enum MultiDeviceOptim {
                    /// The optimization is done on an elected device.
                    OptimMainDevice,
                    /// The optimization is sharded across all devices.
                    OptimSharded,
                }
                #[automatically_derived]
                #[doc(hidden)]
                unsafe impl ::core::clone::TrivialClone for MultiDeviceOptim {}
                #[automatically_derived]
                impl ::core::clone::Clone for MultiDeviceOptim {
                    #[inline]
                    fn clone(&self) -> MultiDeviceOptim {
                        *self
                    }
                }
                #[automatically_derived]
                impl ::core::marker::Copy for MultiDeviceOptim {}
                #[automatically_derived]
                impl ::core::fmt::Debug for MultiDeviceOptim {
                    #[inline]
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter,
                    ) -> ::core::fmt::Result {
                        ::core::fmt::Formatter::write_str(
                            f,
                            match self {
                                MultiDeviceOptim::OptimMainDevice => "OptimMainDevice",
                                MultiDeviceOptim::OptimSharded => "OptimSharded",
                            },
                        )
                    }
                }
                /// Describes where training runs.
                pub enum ExecutionStrategy {
                    /// Training on one device
                    SingleDevice(Device),
                    /// Performs data-parallel distributed training where the optimization is
                    /// done on an elected master device.
                    MultiDevice(Vec<Device>, MultiDeviceOptim),
                    /// Training with input distributed across devices, each device has its own copy of the model.
                    /// Collective ops are used to sync the gradients after each pass.
                    DistributedDataParallel {
                        /// Devices on this node for the DDP
                        devices: Vec<Device>,
                        /// The distributed runtime.
                        context: DistributedContext,
                    },
                }
                impl ExecutionStrategy {
                    /// Returns the primary device responsible for coordination.
                    pub fn main_device(&self) -> &Device {
                        match self {
                            ExecutionStrategy::SingleDevice(device) => device,
                            ExecutionStrategy::MultiDevice(devices, _optim) => {
                                &devices[0]
                            }
                            ExecutionStrategy::DistributedDataParallel {
                                devices,
                                context: _,
                            } => &devices[0],
                        }
                    }
                    /// Creates a strategy for a single device.
                    pub fn single(device: Device) -> Self {
                        Self::SingleDevice(device)
                    }
                    /// Creates a multi-device strategy.
                    pub fn multi(devices: Vec<Device>, optim: MultiDeviceOptim) -> Self {
                        Self::MultiDevice(devices, optim)
                    }
                }
                impl ExecutionStrategy {
                    /// Creates a distributed data parallel (DDP) strategy.
                    pub fn ddp(devices: Vec<Device>, config: DistributedConfig) -> Self {
                        let context = DistributedContext::init(devices.clone(), config);
                        Self::DistributedDataParallel {
                            devices,
                            context,
                        }
                    }
                }
                /// How should the learner run the learning for the model
                pub enum TrainingStrategy<LC: LearningComponentsTypes, L: Label> {
                    /// Default training loop with specified device strategy.
                    Default(ExecutionStrategy),
                    /// Training using a custom learning strategy
                    Custom(CustomLearningStrategy<LC, L>),
                }
                impl<LC: LearningComponentsTypes, L: Label> From<ExecutionStrategy>
                for TrainingStrategy<LC, L> {
                    fn from(value: ExecutionStrategy) -> Self {
                        Self::Default(value)
                    }
                }
                impl<LC: LearningComponentsTypes, L: Label> Default
                for TrainingStrategy<LC, L> {
                    fn default() -> Self {
                        Self::Default(
                            ExecutionStrategy::SingleDevice(Default::default()),
                        )
                    }
                }
                /// Struct to minimise parameters passed to [SupervisedLearningStrategy::train].
                /// These components are used during training.
                pub struct TrainingComponents<LC: LearningComponentsTypes> {
                    /// The total number of epochs
                    pub num_epochs: usize,
                    /// The epoch number from which to continue the training.
                    pub checkpoint: Option<usize>,
                    /// A checkpointer used to load and save learner checkpoints.
                    pub checkpointer: Option<LearningCheckpointer<LC>>,
                    /// Enables gradients accumulation.
                    pub grad_accumulation: Option<usize>,
                    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
                    pub interrupter: Interrupter,
                    /// Cloneable reference to an early stopping strategy.
                    pub early_stopping: Option<EarlyStoppingStrategyRef>,
                    /// An [EventProcessor](crate::EventProcessorTraining) that processes events happening during training and validation.
                    pub event_processor: SupervisedTrainingEventProcessor<LC>,
                    /// A reference to an [EventStoreClient](EventStoreClient).
                    pub event_store: Arc<EventStoreClient>,
                    /// Config for creating a summary of the learning
                    pub summary: Option<LearnerSummaryConfig>,
                }
                /// Provides the `fit` function for any learning strategy
                pub trait SupervisedLearningStrategy<
                    LC: LearningComponentsTypes,
                    L: Label,
                > {
                    /// Train the learner's model with this strategy.
                    fn train(
                        &self,
                        mut learner: Learner<LC>,
                        dataloader_train: TrainLoader<LC, L>,
                        dataloader_valid: ValidLoader<LC, L>,
                        mut training_components: TrainingComponents<LC>,
                    ) -> LearningResult<InferenceModel<LC>> {
                        let starting_epoch = match training_components.checkpoint {
                            Some(checkpoint) => {
                                if let Some(checkpointer) = &mut training_components
                                    .checkpointer
                                {
                                    learner = checkpointer
                                        .load_checkpoint(learner, &Default::default(), checkpoint);
                                }
                                checkpoint + 1
                            }
                            None => 1,
                        };
                        let summary_config = training_components.summary.clone();
                        training_components
                            .event_processor
                            .process_train(LearnerEvent::Start {
                                total_epochs: training_components.num_epochs,
                            });
                        let (model, mut event_processor) = self
                            .fit(
                                training_components,
                                learner,
                                dataloader_train,
                                dataloader_valid,
                                starting_epoch,
                            );
                        let summary = summary_config
                            .and_then(|summary| {
                                summary
                                    .init()
                                    .map(|summary| summary.with_model(model.to_string()))
                                    .ok()
                            });
                        event_processor.process_train(LearnerEvent::End(summary));
                        let model = model.valid();
                        let renderer = event_processor.renderer();
                        LearningResult::<InferenceModel<LC>> {
                            model,
                            renderer,
                        }
                    }
                    /// Training loop for this strategy
                    fn fit(
                        &self,
                        training_components: TrainingComponents<LC>,
                        learner: Learner<LC>,
                        dataloader_train: TrainLoader<LC, L>,
                        dataloader_valid: ValidLoader<LC, L>,
                        starting_epoch: usize,
                    ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>);
                }
            }
            pub(crate) mod ddp {
                mod epoch {
                    use burn_core::data::dataloader::Progress;
                    use burn_core::module::AutodiffModule;
                    use burn_optim::GradientsAccumulator;
                    use std::sync::{Arc, Mutex};
                    use typing_rules::*;
                    use macros::{fcall, mcall};
                    use crate::SupervisedTrainingEventProcessor;
                    use crate::learner::base::Interrupter;
                    use crate::metric::processor::{
                        EventProcessorTraining, LearnerEvent, TrainingItem,
                    };
                    use crate::{
                        InferenceStep, Learner, LearningComponentsTypes, TrainLoader,
                        ValidLoader,
                    };
                    /// A validation epoch.
                    pub struct DdpValidEpoch<LC: LearningComponentsTypes, L: Label> {
                        dataloader: ValidLoader<LC, L>,
                    }
                    impl<LC: LearningComponentsTypes, L: Label> DdpValidEpoch<LC, L> {
                        ///Constructs a new `DdpValidEpoch`.
                        pub fn new(dataloader: ValidLoader<LC, L>) -> Self {
                            DdpValidEpoch {
                                dataloader: dataloader,
                            }
                        }
                    }
                    /// A training epoch.
                    pub struct DdpTrainEpoch<LC: LearningComponentsTypes, L: Label> {
                        dataloader: TrainLoader<LC, L>,
                        grad_accumulation: Option<usize>,
                    }
                    impl<LC: LearningComponentsTypes, L: Label> DdpTrainEpoch<LC, L> {
                        ///Constructs a new `DdpTrainEpoch`.
                        pub fn new(
                            dataloader: TrainLoader<LC, L>,
                            grad_accumulation: Option<usize>,
                        ) -> Self {
                            DdpTrainEpoch {
                                dataloader: dataloader,
                                grad_accumulation: grad_accumulation,
                            }
                        }
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
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Executing validation step for epoch {0}",
                                                epoch,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::learner::supervised::strategies::ddp::epoch",
                                                "burn_train::learner::supervised::strategies::ddp::epoch",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            let model = model.valid();
                            let mut iterator = self.dataloader.iter();
                            let mut iteration = 0;
                            while let Some(item) = iterator.next() {
                                let progress = iterator.progress();
                                iteration += 1;
                                let item = (learner)
                                    .__chain_ref(|__v0| {
                                        (item)
                                            .__chain(|__v1| {
                                                let _: TrainingModelInput<LC> = __v1;
                                                Labeled::<_, Public>::new(Learner::train_step(__v0, __v1))
                                            })
                                    });
                                let item = TrainingItem::new(
                                    item,
                                    progress,
                                    Some(iteration),
                                    None,
                                );
                                processor.process_valid(LearnerEvent::ProcessedItem(item));
                                if interrupter.should_stop() {
                                    {
                                        {
                                            let lvl = ::log::Level::Info;
                                            if lvl <= ::log::STATIC_MAX_LEVEL
                                                && lvl <= ::log::max_level()
                                            {
                                                ::log::__private_api::log(
                                                    { ::log::__private_api::GlobalLogger },
                                                    format_args!("Training interrupted."),
                                                    lvl,
                                                    &(
                                                        "burn_train::learner::supervised::strategies::ddp::epoch",
                                                        "burn_train::learner::supervised::strategies::ddp::epoch",
                                                        ::log::__private_api::loc(),
                                                    ),
                                                    (),
                                                );
                                            }
                                        }
                                    };
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
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Executing training step for epoch {0}",
                                                epoch,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::learner::supervised::strategies::ddp::epoch",
                                                "burn_train::learner::supervised::strategies::ddp::epoch",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            let mut iterator = self.dataloader.iter();
                            let mut iteration = 0;
                            let mut accumulator = GradientsAccumulator::new();
                            let mut accumulation_current = 0;
                            while let Some(item) = iterator.next() {
                                for _ in 0..peer_count {
                                    iteration += 1;
                                    learner.lr_step();
                                }
                                {
                                    {
                                        let lvl = ::log::Level::Info;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api::log(
                                                { ::log::__private_api::GlobalLogger },
                                                format_args!("Iteration {0}", iteration),
                                                lvl,
                                                &(
                                                    "burn_train::learner::supervised::strategies::ddp::epoch",
                                                    "burn_train::learner::supervised::strategies::ddp::epoch",
                                                    ::log::__private_api::loc(),
                                                ),
                                                (),
                                            );
                                        }
                                    }
                                };
                                let mut progress = iterator.progress();
                                progress.items_processed *= peer_count;
                                progress.items_total *= peer_count;
                                let item = {
                                    use ::typing_rules::function_rewrite::SecureChain;
                                    use ::typing_rules::function_rewrite::SecureChainRef;
                                    let __fcall_prev_hook = ::std::panic::take_hook();
                                    ::std::panic::set_hook(::std::boxed::Box::new(|_| {}));
                                    struct __FcallPanicGuard(
                                        ::std::option::Option<
                                            ::std::boxed::Box<dyn ::std::ops::FnMut()>,
                                        >,
                                    );
                                    impl ::std::ops::Drop for __FcallPanicGuard {
                                        fn drop(&mut self) {
                                            if !::std::thread::panicking() {
                                                if let ::std::option::Option::Some(mut f) = self.0.take() {
                                                    f();
                                                }
                                            }
                                        }
                                    }
                                    let mut __fcall_hook_opt = ::std::option::Option::Some(
                                        __fcall_prev_hook,
                                    );
                                    let __fcall_panic_guard = __FcallPanicGuard(
                                        ::std::option::Option::Some(
                                            ::std::boxed::Box::new(move || {
                                                if let ::std::option::Option::Some(hook) = __fcall_hook_opt
                                                    .take()
                                                {
                                                    ::std::panic::set_hook(hook);
                                                }
                                            }),
                                        ),
                                    );
                                    let __fcall_result = {
                                        (learner)
                                            .__chain_ref(|__v0| {
                                                ((item))
                                                    .__chain(|__v1| {
                                                        ::typing_rules::lattice::Labeled::<
                                                            _,
                                                            ::typing_rules::lattice::Public,
                                                        >::new(Learner::train_step(__v0, __v1))
                                                    })
                                            })
                                    };
                                    drop(__fcall_panic_guard);
                                    __fcall_result
                                };
                                match self.grad_accumulation {
                                    Some(accumulation) => {
                                        accumulator.accumulate(&learner.model(), item.grads);
                                        accumulation_current += 1;
                                        if accumulation <= accumulation_current {
                                            let grads = accumulator.grads();
                                            learner.optimizer_step(grads);
                                            accumulation_current = 0;
                                        }
                                    }
                                    None => {
                                        learner.optimizer_step(item.grads);
                                    }
                                }
                                let item = TrainingItem::new(
                                    item.item,
                                    progress,
                                    Some(iteration),
                                    Some(learner.lr_current()),
                                );
                                {
                                    let mut processor = processor.lock().unwrap();
                                    processor.process_train(LearnerEvent::ProcessedItem(item));
                                }
                                if interrupter.should_stop() {
                                    {
                                        {
                                            let lvl = ::log::Level::Info;
                                            if lvl <= ::log::STATIC_MAX_LEVEL
                                                && lvl <= ::log::max_level()
                                            {
                                                ::log::__private_api::log(
                                                    { ::log::__private_api::GlobalLogger },
                                                    format_args!("Training interrupted."),
                                                    lvl,
                                                    &(
                                                        "burn_train::learner::supervised::strategies::ddp::epoch",
                                                        "burn_train::learner::supervised::strategies::ddp::epoch",
                                                        ::log::__private_api::loc(),
                                                    ),
                                                    (),
                                                );
                                            }
                                        }
                                    };
                                    break;
                                }
                            }
                        }
                    }
                }
                mod strategy {
                    use core::panic;
                    use std::sync::{Arc, Mutex};
                    use crate::ddp::worker::DdpWorker;
                    use crate::metric::store::EventStoreClient;
                    use crate::{
                        EarlyStoppingStrategyRef, Interrupter, Learner,
                        LearningComponentsTypes, SupervisedLearningStrategy,
                        SupervisedTrainingEventProcessor, TrainLoader,
                        TrainingComponents, TrainingModel, ValidLoader,
                    };
                    use burn_core::data::dataloader::split::split_dataloader;
                    use burn_core::tensor::Device;
                    use burn_core::tensor::distributed::DistributedContext;
                    use typing_rules::*;
                    pub(crate) struct WorkerComponents {
                        /// The total number of epochs
                        pub num_epochs: usize,
                        /// Enables gradients accumulation.
                        pub grad_accumulation: Option<usize>,
                        /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
                        pub interrupter: Interrupter,
                        /// Cloneable reference to an early stopping strategy.
                        pub early_stopping: Option<EarlyStoppingStrategyRef>,
                        /// A reference to an [EventStoreClient](EventStoreClient).
                        pub event_store: Arc<EventStoreClient>,
                        /// The total number of items in the training dataset.
                        pub train_total_items: usize,
                        /// The total number of items in the validation dataset.
                        pub valid_total_items: usize,
                    }
                    #[automatically_derived]
                    impl ::core::clone::Clone for WorkerComponents {
                        #[inline]
                        fn clone(&self) -> WorkerComponents {
                            WorkerComponents {
                                num_epochs: ::core::clone::Clone::clone(&self.num_epochs),
                                grad_accumulation: ::core::clone::Clone::clone(
                                    &self.grad_accumulation,
                                ),
                                interrupter: ::core::clone::Clone::clone(&self.interrupter),
                                early_stopping: ::core::clone::Clone::clone(
                                    &self.early_stopping,
                                ),
                                event_store: ::core::clone::Clone::clone(&self.event_store),
                                train_total_items: ::core::clone::Clone::clone(
                                    &self.train_total_items,
                                ),
                                valid_total_items: ::core::clone::Clone::clone(
                                    &self.valid_total_items,
                                ),
                            }
                        }
                    }
                    /// A training strategy for Distributed Data Parallel (DDP) training.
                    ///
                    /// This strategy manages multiple workers and coordinates cross-device
                    /// gradient synchronization using the provided [`DistributedContext`].
                    pub struct DdpTrainingStrategy {
                        devices: Vec<Device>,
                        /// Kept alive to anchor the lifetime of the underlying distributed server.
                        /// Spawns communication servers on creation, automatically tears them down on drop.
                        _context: DistributedContext,
                    }
                    impl DdpTrainingStrategy {
                        pub fn new(
                            devices: Vec<Device>,
                            context: DistributedContext,
                        ) -> Self {
                            Self { devices, _context: context }
                        }
                    }
                    impl<LC, L> SupervisedLearningStrategy<LC, L> for DdpTrainingStrategy
                    where
                        LC: LearningComponentsTypes + Send + 'static,
                        L: Label,
                    {
                        fn fit(
                            &self,
                            training_components: TrainingComponents<LC>,
                            learner: Learner<LC>,
                            dataloader_train: TrainLoader<LC, L>,
                            dataloader_valid: ValidLoader<LC, L>,
                            starting_epoch: usize,
                        ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>) {
                            let main_device = self.devices.first().unwrap();
                            let train_total_items = dataloader_train.num_items();
                            let valid_total_items = dataloader_valid.num_items();
                            let mut dataloaders_train = split_dataloader(
                                dataloader_train,
                                &self.devices,
                            );
                            let dataloader_valid = dataloader_valid
                                .to_device(&main_device.clone().inner());
                            let main_device = self.devices[0].clone();
                            let peer_count = self.devices.len();
                            let event_processor = Arc::new(
                                Mutex::new(training_components.event_processor),
                            );
                            let interrupter = training_components.interrupter;
                            let worker_components = WorkerComponents {
                                num_epochs: training_components.num_epochs,
                                grad_accumulation: training_components.grad_accumulation,
                                interrupter: interrupter.clone(),
                                early_stopping: training_components.early_stopping,
                                event_store: training_components.event_store,
                                train_total_items,
                                valid_total_items,
                            };
                            let main_handle = DdpWorker::<
                                LC,
                                L,
                            >::start(
                                main_device.clone(),
                                learner.clone(),
                                event_processor.clone(),
                                worker_components.clone(),
                                training_components.checkpointer,
                                dataloaders_train.remove(0),
                                Some(dataloader_valid),
                                starting_epoch,
                                peer_count,
                                true,
                            );
                            let mut secondary_workers = ::alloc::vec::Vec::new();
                            for device in &self.devices[1..] {
                                let handle = DdpWorker::<
                                    LC,
                                    L,
                                >::start(
                                    device.clone(),
                                    learner.clone(),
                                    event_processor.clone(),
                                    worker_components.clone(),
                                    None,
                                    dataloaders_train.remove(0),
                                    None,
                                    starting_epoch,
                                    peer_count,
                                    false,
                                );
                                secondary_workers.push(handle);
                            }
                            for worker in secondary_workers {
                                worker
                                    .join()
                                    .expect("Distributed data parallel worker failed");
                            }
                            let model = main_handle
                                .join()
                                .expect("Distributed data parallel main worker failed");
                            if interrupter.should_stop() {
                                let reason = interrupter
                                    .get_message()
                                    .unwrap_or(String::from("Reason unknown"));
                                {
                                    {
                                        let lvl = ::log::Level::Info;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api::log(
                                                { ::log::__private_api::GlobalLogger },
                                                format_args!("Training interrupted: {0}", reason),
                                                lvl,
                                                &(
                                                    "burn_train::learner::supervised::strategies::ddp::strategy",
                                                    "burn_train::learner::supervised::strategies::ddp::strategy",
                                                    ::log::__private_api::loc(),
                                                ),
                                                (),
                                            );
                                        }
                                    }
                                };
                            }
                            let Ok(event_processor) = Arc::try_unwrap(event_processor)
                            else {
                                {
                                    ::core::panicking::panic_fmt(
                                        format_args!("Event processor still held!"),
                                    );
                                };
                            };
                            let Ok(event_processor) = event_processor.into_inner() else {
                                {
                                    ::core::panicking::panic_fmt(
                                        format_args!("Event processor lock poisoned"),
                                    );
                                };
                            };
                            (model, event_processor)
                        }
                    }
                }
                mod worker {
                    use crate::ddp::epoch::{DdpTrainEpoch, DdpValidEpoch};
                    use crate::ddp::strategy::WorkerComponents;
                    use crate::metric::processor::{EventProcessorTraining, LearnerEvent};
                    use crate::single::TrainingLoop;
                    use crate::{
                        Learner, LearningCheckpointer, LearningComponentsTypes,
                        SupervisedTrainingEventProcessor, TrainLoader, ValidLoader,
                    };
                    use burn_core::tensor::Device;
                    use std::sync::{Arc, Mutex};
                    use std::thread::JoinHandle;
                    use typing_rules::*;
                    /// A worker runs the model, syncing gradients using collective operations.
                    /// Event processing and validation is optional too.
                    pub(crate) struct DdpWorker<LC, L>
                    where
                        LC: LearningComponentsTypes + Send + 'static,
                        L: Label,
                    {
                        device: Device,
                        learner: Learner<LC>,
                        event_processor: Arc<
                            Mutex<SupervisedTrainingEventProcessor<LC>>,
                        >,
                        components: WorkerComponents,
                        checkpointer: Option<LearningCheckpointer<LC>>,
                        dataloader_train: TrainLoader<LC, L>,
                        dataloader_valid: Option<ValidLoader<LC, L>>,
                        starting_epoch: usize,
                        peer_count: usize,
                        is_main: bool,
                    }
                    impl<LC, L> DdpWorker<LC, L>
                    where
                        LC: LearningComponentsTypes + Send + 'static,
                        L: Label,
                    {
                        /// Starts a worker that runs the model in a data distributed parallel
                        #[allow(clippy::too_many_arguments)]
                        pub fn start(
                            device: Device,
                            learner: Learner<LC>,
                            event_processor: Arc<
                                Mutex<SupervisedTrainingEventProcessor<LC>>,
                            >,
                            components: WorkerComponents,
                            checkpointer: Option<LearningCheckpointer<LC>>,
                            dataloader_train: TrainLoader<LC, L>,
                            dataloader_valid: Option<ValidLoader<LC, L>>,
                            starting_epoch: usize,
                            peer_count: usize,
                            is_main: bool,
                        ) -> JoinHandle<<LC as LearningComponentsTypes>::Model> {
                            let worker = Self {
                                device,
                                learner,
                                event_processor,
                                components,
                                checkpointer,
                                dataloader_train,
                                dataloader_valid,
                                starting_epoch,
                                peer_count,
                                is_main,
                            };
                            std::thread::spawn(|| worker.fit())
                        }
                        /// Fits the model,
                        pub fn fit(mut self) -> <LC as LearningComponentsTypes>::Model {
                            let num_epochs = self.components.num_epochs;
                            let interrupter = self.components.interrupter;
                            let epoch_train = DdpTrainEpoch::<
                                LC,
                                L,
                            >::new(
                                self.dataloader_train.clone(),
                                self.components.grad_accumulation,
                            );
                            let epoch_valid = self
                                .dataloader_valid
                                .map(|dataloader| DdpValidEpoch::<LC, L>::new(dataloader));
                            self.learner.fork(&self.device);
                            self.learner.grad_sharded();
                            for training_progress in TrainingLoop::new(
                                self.starting_epoch,
                                num_epochs,
                            ) {
                                let epoch = training_progress.items_processed;
                                if self.is_main {
                                    self.event_processor
                                        .lock()
                                        .unwrap()
                                        .process_train(
                                            LearnerEvent::StartSplit(self.components.train_total_items),
                                        );
                                }
                                epoch_train
                                    .run(
                                        &mut self.learner,
                                        &training_progress,
                                        self.event_processor.clone(),
                                        &interrupter,
                                        self.peer_count,
                                    );
                                if self.is_main {
                                    self.event_processor
                                        .lock()
                                        .unwrap()
                                        .process_train(LearnerEvent::EndSplit(epoch));
                                }
                                if interrupter.should_stop() {
                                    break;
                                }
                                if let Some(runner) = &epoch_valid {
                                    {
                                        self.event_processor
                                            .lock()
                                            .unwrap()
                                            .process_valid(
                                                LearnerEvent::StartSplit(self.components.valid_total_items),
                                            );
                                    }
                                    let mut event_processor = self
                                        .event_processor
                                        .lock()
                                        .unwrap();
                                    runner
                                        .run(
                                            &self.learner.model(),
                                            &training_progress,
                                            &mut event_processor,
                                            &interrupter,
                                        );
                                    event_processor
                                        .process_valid(LearnerEvent::EndSplit(epoch));
                                    event_processor
                                        .process_train(LearnerEvent::EndEpoch(epoch));
                                }
                                if let Some(checkpointer) = &mut self.checkpointer {
                                    checkpointer
                                        .checkpoint(
                                            &self.learner,
                                            epoch,
                                            &self.components.event_store,
                                        );
                                }
                                if let Some(early_stopping) = &mut self
                                    .components
                                    .early_stopping
                                    && early_stopping
                                        .should_stop(epoch, &self.components.event_store)
                                {
                                    break;
                                }
                            }
                            self.learner.model()
                        }
                    }
                }
                pub use strategy::*;
            }
            pub(crate) mod multi {
                pub(crate) mod epoch {
                    use crate::learner::base::Interrupter;
                    use crate::metric::processor::{
                        EventProcessorTraining, LearnerEvent, TrainingItem,
                    };
                    use crate::train::MultiDevicesTrainStep;
                    use crate::{
                        Learner, LearningComponentsTypes, MultiDeviceOptim,
                        SupervisedTrainingEventProcessor, TrainLoader,
                    };
                    use burn_core::data::dataloader::Progress;
                    use burn_core::tensor::Device;
                    use burn_optim::GradientsAccumulator;
                    use burn_optim::MultiGradientsParams;
                    use typing_rules::*;
                    /// A training epoch.
                    pub struct MultiDeviceTrainEpoch<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > {
                        dataloaders: Vec<TrainLoader<LC, L>>,
                        grad_accumulation: Option<usize>,
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > MultiDeviceTrainEpoch<LC, L> {
                        ///Constructs a new `MultiDeviceTrainEpoch`.
                        pub fn new(
                            dataloaders: Vec<TrainLoader<LC, L>>,
                            grad_accumulation: Option<usize>,
                        ) -> Self {
                            MultiDeviceTrainEpoch {
                                dataloaders: dataloaders,
                                grad_accumulation: grad_accumulation,
                            }
                        }
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > MultiDeviceTrainEpoch<LC, L> {
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
                            learner: &mut Learner<LC>,
                            global_progress: &Progress,
                            event_processor: &mut SupervisedTrainingEventProcessor<LC>,
                            interrupter: &Interrupter,
                            devices: Vec<Device>,
                            strategy: MultiDeviceOptim,
                        ) {
                            match strategy {
                                MultiDeviceOptim::OptimMainDevice => {
                                    self.run_optim_main(
                                        learner,
                                        global_progress,
                                        event_processor,
                                        interrupter,
                                        devices,
                                    )
                                }
                                MultiDeviceOptim::OptimSharded => {
                                    self.run_optim_distr(
                                        learner,
                                        global_progress,
                                        event_processor,
                                        interrupter,
                                        devices,
                                    )
                                }
                            }
                        }
                        fn run_optim_main(
                            &self,
                            learner: &mut Learner<LC>,
                            global_progress: &Progress,
                            event_processor: &mut SupervisedTrainingEventProcessor<LC>,
                            interrupter: &Interrupter,
                            devices: Vec<Device>,
                        ) {
                            let epoch = global_progress.items_processed;
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Executing training step for epoch {0} on devices {1:?}",
                                                epoch,
                                                devices,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::learner::supervised::strategies::multi::epoch",
                                                "burn_train::learner::supervised::strategies::multi::epoch",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            let mut iterators = self
                                .dataloaders
                                .iter()
                                .map(|d| d.iter())
                                .collect::<Vec<_>>();
                            let mut iteration = 0;
                            let mut accumulator = GradientsAccumulator::new();
                            let mut accumulation_current = 0;
                            let accumulation = self.grad_accumulation.unwrap_or(1);
                            let step = MultiDevicesTrainStep::<LC, L>::new(&devices);
                            let device_main = devices
                                .first()
                                .expect("A minimum of one device.")
                                .clone();
                            loop {
                                let (items, progress) = step
                                    .step(iterators.as_mut_slice(), &learner.model());
                                if items.is_empty() {
                                    break;
                                }
                                learner.lr_step();
                                let mut progress_items = Vec::with_capacity(items.len());
                                for item in items.into_iter() {
                                    let grads = item
                                        .output
                                        .grads
                                        .to_device(&device_main, &learner.model());
                                    accumulator.accumulate(&learner.model(), grads);
                                    progress_items.push(item.output.item);
                                }
                                accumulation_current += 1;
                                if accumulation <= accumulation_current {
                                    let grads = accumulator.grads();
                                    learner.optimizer_step(grads);
                                    accumulation_current = 0;
                                }
                                for item in progress_items {
                                    iteration += 1;
                                    let item = TrainingItem::new(
                                        item,
                                        progress.clone(),
                                        Some(iteration),
                                        Some(learner.lr_current()),
                                    );
                                    event_processor
                                        .process_train(LearnerEvent::ProcessedItem(item));
                                }
                                if interrupter.should_stop() {
                                    break;
                                }
                            }
                        }
                        fn run_optim_distr(
                            &self,
                            learner: &mut Learner<LC>,
                            global_progress: &Progress,
                            event_processor: &mut SupervisedTrainingEventProcessor<LC>,
                            interrupter: &Interrupter,
                            devices: Vec<Device>,
                        ) {
                            let epoch = global_progress.items_processed;
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Executing training step for epoch {0} on devices {1:?}",
                                                epoch,
                                                devices,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::learner::supervised::strategies::multi::epoch",
                                                "burn_train::learner::supervised::strategies::multi::epoch",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            let mut iterators = self
                                .dataloaders
                                .iter()
                                .map(|d| d.iter())
                                .collect::<Vec<_>>();
                            let mut iteration = 0;
                            let mut accumulators: Vec<GradientsAccumulator<_>> = (0..devices
                                .len())
                                .map(|_| GradientsAccumulator::new())
                                .collect();
                            let mut accumulation_current = 0;
                            let accumulation = self.grad_accumulation.unwrap_or(1);
                            let step = MultiDevicesTrainStep::<LC, L>::new(&devices);
                            loop {
                                let (items, progress) = step
                                    .step(iterators.as_mut_slice(), &learner.model());
                                if items.is_empty() {
                                    break;
                                }
                                learner.lr_step();
                                let mut progress_items = Vec::with_capacity(items.len());
                                for item in items.into_iter() {
                                    let accumulator = &mut accumulators[item.device_id];
                                    accumulator.accumulate(&learner.model(), item.output.grads);
                                    progress_items.push(item.output.item);
                                }
                                accumulation_current += 1;
                                if accumulation <= accumulation_current {
                                    let mut grads = MultiGradientsParams::default();
                                    for (device_id, accumulator) in accumulators
                                        .iter_mut()
                                        .enumerate()
                                    {
                                        let grad = accumulator.grads();
                                        grads.grads.push((grad, devices[device_id].clone()));
                                    }
                                    learner.optimizer_step_multi(grads);
                                    accumulation_current = 0;
                                }
                                for item in progress_items {
                                    iteration += 1;
                                    let item = TrainingItem::new(
                                        item,
                                        progress.clone(),
                                        Some(iteration),
                                        Some(learner.lr_current()),
                                    );
                                    event_processor
                                        .process_train(LearnerEvent::ProcessedItem(item));
                                }
                                if interrupter.should_stop() {
                                    break;
                                }
                            }
                        }
                    }
                }
                mod strategy {
                    use crate::{
                        Learner, LearnerEvent, LearningComponentsTypes, MultiDeviceOptim,
                        SupervisedLearningStrategy, SupervisedTrainingEventProcessor,
                        TrainLoader, TrainingComponents, TrainingModel, ValidLoader,
                        metric::processor::EventProcessorTraining,
                        multi::epoch::MultiDeviceTrainEpoch,
                        single::{TrainingLoop, epoch::SingleDeviceValidEpoch},
                    };
                    use burn_core::{
                        data::dataloader::split::split_dataloader, tensor::Device,
                    };
                    use typing_rules::*;
                    pub struct MultiDeviceLearningStrategy {
                        devices: Vec<Device>,
                        optim: MultiDeviceOptim,
                    }
                    impl MultiDeviceLearningStrategy {
                        pub fn new(
                            devices: Vec<Device>,
                            optim: MultiDeviceOptim,
                        ) -> Self {
                            Self { devices, optim }
                        }
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > SupervisedLearningStrategy<LC, L> for MultiDeviceLearningStrategy {
                        fn fit(
                            &self,
                            training_components: TrainingComponents<LC>,
                            mut learner: Learner<LC>,
                            dataloader_train: TrainLoader<LC, L>,
                            dataloader_valid: ValidLoader<LC, L>,
                            starting_epoch: usize,
                        ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>) {
                            let main_device = self.devices.first().unwrap();
                            let train_total_items = dataloader_train.num_items();
                            let dataloader_train = split_dataloader(
                                dataloader_train,
                                &self.devices,
                            );
                            let dataloader_valid = dataloader_valid
                                .to_device(&main_device.clone().inner());
                            let valid_total_items = dataloader_valid.num_items();
                            learner.fork(main_device);
                            let mut event_processor = training_components
                                .event_processor;
                            let mut checkpointer = training_components.checkpointer;
                            let mut early_stopping = training_components.early_stopping;
                            let epoch_train = MultiDeviceTrainEpoch::<
                                LC,
                                L,
                            >::new(
                                dataloader_train.clone(),
                                training_components.grad_accumulation,
                            );
                            let epoch_valid: SingleDeviceValidEpoch<LC, L> = SingleDeviceValidEpoch::new(
                                dataloader_valid.clone(),
                            );
                            for training_progress in TrainingLoop::new(
                                starting_epoch,
                                training_components.num_epochs,
                            ) {
                                let epoch = training_progress.items_processed;
                                event_processor
                                    .process_train(LearnerEvent::StartSplit(train_total_items));
                                epoch_train
                                    .run(
                                        &mut learner,
                                        &training_progress,
                                        &mut event_processor,
                                        &training_components.interrupter,
                                        self.devices.to_vec(),
                                        self.optim,
                                    );
                                event_processor
                                    .process_train(LearnerEvent::EndSplit(epoch));
                                if training_components.interrupter.should_stop() {
                                    let reason = training_components
                                        .interrupter
                                        .get_message()
                                        .unwrap_or(String::from("Reason unknown"));
                                    {
                                        {
                                            let lvl = ::log::Level::Info;
                                            if lvl <= ::log::STATIC_MAX_LEVEL
                                                && lvl <= ::log::max_level()
                                            {
                                                ::log::__private_api::log(
                                                    { ::log::__private_api::GlobalLogger },
                                                    format_args!("Training interrupted: {0}", reason),
                                                    lvl,
                                                    &(
                                                        "burn_train::learner::supervised::strategies::multi::strategy",
                                                        "burn_train::learner::supervised::strategies::multi::strategy",
                                                        ::log::__private_api::loc(),
                                                    ),
                                                    (),
                                                );
                                            }
                                        }
                                    };
                                    break;
                                }
                                if #[allow(non_exhaustive_omitted_patterns)]
                                match self.optim {
                                    MultiDeviceOptim::OptimSharded => true,
                                    _ => false,
                                } {
                                    learner.fork(main_device);
                                }
                                event_processor
                                    .process_valid(LearnerEvent::StartSplit(valid_total_items));
                                epoch_valid
                                    .run(
                                        &learner,
                                        &training_progress,
                                        &mut event_processor,
                                        &training_components.interrupter,
                                    );
                                event_processor
                                    .process_valid(LearnerEvent::EndSplit(epoch));
                                event_processor
                                    .process_train(LearnerEvent::EndEpoch(epoch));
                                if let Some(checkpointer) = &mut checkpointer {
                                    checkpointer
                                        .checkpoint(
                                            &learner,
                                            epoch,
                                            &training_components.event_store,
                                        );
                                }
                                if let Some(early_stopping) = &mut early_stopping
                                    && early_stopping
                                        .should_stop(epoch, &training_components.event_store)
                                {
                                    break;
                                }
                            }
                            (learner.model(), event_processor)
                        }
                    }
                }
                pub use strategy::*;
            }
            pub(crate) mod single {
                pub(crate) mod epoch {
                    use crate::learner::base::Interrupter;
                    use crate::metric::processor::{
                        EventProcessorTraining, LearnerEvent, TrainingItem,
                    };
                    use crate::{
                        InferenceStep, Learner, LearningComponentsTypes,
                        SupervisedTrainingEventProcessor, TrainLoader, ValidLoader,
                    };
                    use burn_core::data::dataloader::Progress;
                    use burn_core::module::AutodiffModule;
                    use burn_optim::GradientsAccumulator;
                    use typing_rules::*;
                    /// A validation epoch.
                    pub struct SingleDeviceValidEpoch<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > {
                        dataloader: ValidLoader<LC, L>,
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > SingleDeviceValidEpoch<LC, L> {
                        ///Constructs a new `SingleDeviceValidEpoch`.
                        pub fn new(dataloader: ValidLoader<LC, L>) -> Self {
                            SingleDeviceValidEpoch {
                                dataloader: dataloader,
                            }
                        }
                    }
                    /// A training epoch.
                    pub struct SingleDeviceTrainEpoch<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > {
                        dataloader: TrainLoader<LC, L>,
                        grad_accumulation: Option<usize>,
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > SingleDeviceTrainEpoch<LC, L> {
                        ///Constructs a new `SingleDeviceTrainEpoch`.
                        pub fn new(
                            dataloader: TrainLoader<LC, L>,
                            grad_accumulation: Option<usize>,
                        ) -> Self {
                            SingleDeviceTrainEpoch {
                                dataloader: dataloader,
                                grad_accumulation: grad_accumulation,
                            }
                        }
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > SingleDeviceValidEpoch<LC, L> {
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
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Executing validation step for epoch {0}",
                                                epoch,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::learner::supervised::strategies::single::epoch",
                                                "burn_train::learner::supervised::strategies::single::epoch",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            let model = learner.model().valid();
                            let mut iterator = self.dataloader.iter();
                            let mut iteration = 0;
                            while let Some(item) = iterator.next() {
                                let progress = iterator.progress();
                                iteration += 1;
                                let item = model.step(item);
                                let item = TrainingItem::new(
                                    item,
                                    progress,
                                    Some(iteration),
                                    None,
                                );
                                processor.process_valid(LearnerEvent::ProcessedItem(item));
                                if interrupter.should_stop() {
                                    break;
                                }
                            }
                        }
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > SingleDeviceTrainEpoch<LC, L> {
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
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!(
                                                "Executing training step for epoch {0}",
                                                epoch,
                                            ),
                                            lvl,
                                            &(
                                                "burn_train::learner::supervised::strategies::single::epoch",
                                                "burn_train::learner::supervised::strategies::single::epoch",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            let mut iterator = self.dataloader.iter();
                            let mut iteration = 0;
                            let mut accumulator = GradientsAccumulator::new();
                            let mut accumulation_current = 0;
                            while let Some(item) = iterator.next() {
                                iteration += 1;
                                learner.lr_step();
                                {
                                    {
                                        let lvl = ::log::Level::Info;
                                        if lvl <= ::log::STATIC_MAX_LEVEL
                                            && lvl <= ::log::max_level()
                                        {
                                            ::log::__private_api::log(
                                                { ::log::__private_api::GlobalLogger },
                                                format_args!("Iteration {0}", iteration),
                                                lvl,
                                                &(
                                                    "burn_train::learner::supervised::strategies::single::epoch",
                                                    "burn_train::learner::supervised::strategies::single::epoch",
                                                    ::log::__private_api::loc(),
                                                ),
                                                (),
                                            );
                                        }
                                    }
                                };
                                let progress = iterator.progress();
                                let item = learner.train_step(item);
                                match self.grad_accumulation {
                                    Some(accumulation) => {
                                        accumulator.accumulate(&learner.model(), item.grads);
                                        accumulation_current += 1;
                                        if accumulation <= accumulation_current {
                                            let grads = accumulator.grads();
                                            learner.optimizer_step(grads);
                                            accumulation_current = 0;
                                        }
                                    }
                                    None => learner.optimizer_step(item.grads),
                                }
                                let item = TrainingItem::new(
                                    item.item,
                                    progress,
                                    Some(iteration),
                                    Some(learner.lr_current()),
                                );
                                processor.process_train(LearnerEvent::ProcessedItem(item));
                                if interrupter.should_stop() {
                                    break;
                                }
                            }
                        }
                    }
                }
                mod strategy {
                    use crate::{
                        EventProcessorTraining, Learner, LearnerEvent,
                        LearningComponentsTypes, SupervisedLearningStrategy,
                        SupervisedTrainingEventProcessor, TrainLoader,
                        TrainingComponents, TrainingModel, ValidLoader,
                        single::epoch::{SingleDeviceTrainEpoch, SingleDeviceValidEpoch},
                    };
                    use burn_core::{data::dataloader::Progress, tensor::Device};
                    use typing_rules::*;
                    /// Simplest learning strategy possible, with only a single devices doing both the training and
                    /// validation.
                    pub struct SingleDeviceTrainingStrategy {
                        device: Device,
                    }
                    impl SingleDeviceTrainingStrategy {
                        pub fn new(device: Device) -> Self {
                            Self { device }
                        }
                    }
                    pub(crate) struct TrainingLoop {
                        next_iteration: usize,
                        total_iteration: usize,
                    }
                    impl TrainingLoop {
                        ///Constructs a new `TrainingLoop`.
                        pub fn new(
                            next_iteration: usize,
                            total_iteration: usize,
                        ) -> Self {
                            TrainingLoop {
                                next_iteration: next_iteration,
                                total_iteration: total_iteration,
                            }
                        }
                    }
                    /// An iterator that returns the progress of the training.
                    impl Iterator for TrainingLoop {
                        type Item = Progress;
                        fn next(&mut self) -> Option<Self::Item> {
                            if self.next_iteration > self.total_iteration {
                                return None;
                            }
                            let progress = Progress {
                                items_processed: self.next_iteration,
                                items_total: self.total_iteration,
                                unit: Some("epochs".to_string()),
                            };
                            self.next_iteration += 1;
                            Some(progress)
                        }
                    }
                    impl<
                        LC: LearningComponentsTypes,
                        L: Label,
                    > SupervisedLearningStrategy<LC, L>
                    for SingleDeviceTrainingStrategy {
                        fn fit(
                            &self,
                            training_components: TrainingComponents<LC>,
                            mut learner: Learner<LC>,
                            dataloader_train: TrainLoader<LC, L>,
                            dataloader_valid: ValidLoader<LC, L>,
                            starting_epoch: usize,
                        ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>) {
                            let dataloader_train = dataloader_train
                                .to_device(&self.device);
                            let train_total_items = dataloader_train.num_items();
                            let dataloader_valid = dataloader_valid
                                .to_device(&self.device.clone().inner());
                            let valid_total_items = dataloader_valid.num_items();
                            learner.fork(&self.device);
                            let mut event_processor = training_components
                                .event_processor;
                            let mut checkpointer = training_components.checkpointer;
                            let mut early_stopping = training_components.early_stopping;
                            let epoch_train: SingleDeviceTrainEpoch<LC, L> = SingleDeviceTrainEpoch::new(
                                dataloader_train,
                                training_components.grad_accumulation,
                            );
                            let epoch_valid: SingleDeviceValidEpoch<LC, L> = SingleDeviceValidEpoch::new(
                                dataloader_valid.clone(),
                            );
                            for training_progress in TrainingLoop::new(
                                starting_epoch,
                                training_components.num_epochs,
                            ) {
                                let epoch = training_progress.items_processed;
                                event_processor
                                    .process_train(LearnerEvent::StartSplit(train_total_items));
                                epoch_train
                                    .run(
                                        &mut learner,
                                        &training_progress,
                                        &mut event_processor,
                                        &training_components.interrupter,
                                    );
                                event_processor
                                    .process_train(LearnerEvent::EndSplit(epoch));
                                if training_components.interrupter.should_stop() {
                                    let reason = training_components
                                        .interrupter
                                        .get_message()
                                        .unwrap_or(String::from("Reason unknown"));
                                    {
                                        {
                                            let lvl = ::log::Level::Info;
                                            if lvl <= ::log::STATIC_MAX_LEVEL
                                                && lvl <= ::log::max_level()
                                            {
                                                ::log::__private_api::log(
                                                    { ::log::__private_api::GlobalLogger },
                                                    format_args!("Training interrupted: {0}", reason),
                                                    lvl,
                                                    &(
                                                        "burn_train::learner::supervised::strategies::single::strategy",
                                                        "burn_train::learner::supervised::strategies::single::strategy",
                                                        ::log::__private_api::loc(),
                                                    ),
                                                    (),
                                                );
                                            }
                                        }
                                    };
                                    break;
                                }
                                event_processor
                                    .process_valid(LearnerEvent::StartSplit(valid_total_items));
                                epoch_valid
                                    .run(
                                        &learner,
                                        &training_progress,
                                        &mut event_processor,
                                        &training_components.interrupter,
                                    );
                                event_processor
                                    .process_valid(LearnerEvent::EndSplit(epoch));
                                event_processor
                                    .process_train(LearnerEvent::EndEpoch(epoch));
                                if let Some(checkpointer) = &mut checkpointer {
                                    checkpointer
                                        .checkpoint(
                                            &learner,
                                            epoch,
                                            &training_components.event_store,
                                        );
                                }
                                if let Some(early_stopping) = &mut early_stopping
                                    && early_stopping
                                        .should_stop(epoch, &training_components.event_store)
                                {
                                    break;
                                }
                            }
                            (learner.model(), event_processor)
                        }
                    }
                }
                pub use strategy::*;
            }
            pub use base::*;
        }
        pub use paradigm::*;
        pub use step::*;
        pub use strategies::*;
    }
    mod train_val {
        use crate::{ItemLazy, renderer::MetricsRenderer};
        use burn_core::{module::AutodiffModule, tensor::Gradients};
        use burn_optim::{GradientsParams, MultiGradientsParams, Optimizer};
        /// A training output.
        pub struct TrainOutput<TO> {
            /// The gradients.
            pub grads: GradientsParams,
            /// The item.
            pub item: TO,
        }
        impl<TO> TrainOutput<TO> {
            /// Creates a new training output.
            ///
            /// # Arguments
            ///
            /// * `module` - The module.
            /// * `grads` - The gradients.
            /// * `item` - The item.
            ///
            /// # Returns
            ///
            /// A new training output.
            pub fn new<M: AutodiffModule>(
                module: &M,
                grads: Gradients,
                item: TO,
            ) -> Self {
                let grads = GradientsParams::from_grads(grads, module);
                Self { grads, item }
            }
        }
        /// Trait to be implemented for models to be able to be trained.
        ///
        /// The [step](TrainStep::step) method needs to be manually implemented for all structs.
        ///
        /// The [optimize](TrainStep::optimize) method can be overridden if you want to control how the
        /// optimizer is used to update the model. This can be useful if you want to call custom mutable
        /// functions on your model (e.g., clipping the weights) before or after the optimizer is used.
        ///
        /// # Notes
        ///
        /// To be used with the [Learner](crate::Learner) struct, the struct which implements this trait must
        /// also implement the [AutodiffModule] trait, which is done automatically with the
        /// [Module](burn_core::module::Module) derive.
        pub trait TrainStep {
            /// Type of input for a step of the training stage.
            type Input: Send + 'static;
            /// Type of output for a step of the training stage.
            type Output: ItemLazy + 'static;
            /// Runs a step for training, which executes the forward and backward passes.
            ///
            /// # Arguments
            ///
            /// * `item` - The input for the model.
            ///
            /// # Returns
            ///
            /// The output containing the model output and the gradients.
            fn step(&self, item: Self::Input) -> TrainOutput<Self::Output>;
            /// Optimize the current module with the provided gradients and learning rate.
            ///
            /// # Arguments
            ///
            /// * `optim`: Optimizer used for learning.
            /// * `lr`: The learning rate used for this step.
            /// * `grads`: The gradients of each parameter in the current model.
            ///
            /// # Returns
            ///
            /// The updated model.
            fn optimize<O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
            where
                O: Optimizer<Self>,
                Self: AutodiffModule,
            {
                optim.step(lr, self, grads)
            }
            /// Optimize the current module with the provided gradients and learning rate.
            ///
            /// # Arguments
            ///
            /// * `optim`: Optimizer used for learning.
            /// * `lr`: The learning rate used for this step.
            /// * `grads`: Multiple gradients associated to each parameter in the current model.
            ///
            /// # Returns
            ///
            /// The updated model.
            fn optimize_multi<O>(
                self,
                optim: &mut O,
                lr: f64,
                grads: MultiGradientsParams,
            ) -> Self
            where
                O: Optimizer<Self>,
                Self: AutodiffModule,
            {
                optim.step_multi(lr, self, grads)
            }
        }
        /// Trait to be implemented for validating models.
        pub trait InferenceStep {
            /// Type of input for an inference step.
            type Input: Send + 'static;
            /// Type of output for an inference step.
            type Output: ItemLazy + 'static;
            /// Runs a validation step.
            ///
            /// # Arguments
            ///
            /// * `item` - The item to validate on.
            ///
            /// # Returns
            ///
            /// The validation output.
            fn step(&self, item: Self::Input) -> Self::Output;
        }
        /// The result of a training, containing the model along with the [renderer](MetricsRenderer).
        pub struct LearningResult<M> {
            /// The model with the learned weights.
            pub model: M,
            /// The renderer that can be used for follow up training and evaluation.
            pub renderer: Box<dyn MetricsRenderer>,
        }
    }
    pub use application_logger::*;
    pub use base::*;
    pub use classification::*;
    pub use early_stopping::*;
    pub use regression::*;
    pub use sequence::*;
    pub use sharder::*;
    pub use summary::*;
    pub use supervised::*;
    pub use train_val::*;
}
pub use learner::*;
mod evaluator {
    mod base {
        use crate::{
            AsyncProcessorEvaluation, EvaluationItem, FullEventProcessorEvaluation,
            InferenceStep, Interrupter, LearnerSummaryConfig,
            evaluator::components::EvaluatorComponentTypes,
            metric::processor::{EvaluatorEvent, EventProcessorEvaluation},
            renderer::{EvaluationName, MetricsRenderer},
        };
        use burn_core::{data::dataloader::DataLoader, module::Module};
        use std::sync::Arc;
        use typing_rules::*;
        pub(crate) type TestInput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Input;
        pub(crate) type TestOutput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Output;
        pub(crate) type TestLoader<EC, L: Label> = Arc<dyn DataLoader<TestInput<EC>, L>>;
        /// Evaluates a model on a specific dataset.
        pub struct Evaluator<EC: EvaluatorComponentTypes, L: Label> {
            pub(crate) model: EC::Model,
            pub(crate) interrupter: Interrupter,
            pub(crate) event_processor: AsyncProcessorEvaluation<
                FullEventProcessorEvaluation<TestOutput<EC>>,
            >,
            /// Config for creating a summary of the evaluation
            pub summary: Option<LearnerSummaryConfig>,
        }
        impl<EC: EvaluatorComponentTypes, L: Label> Evaluator<EC, L> {
            /// Run the evaluation on the given dataset.
            ///
            /// The data will be stored and displayed under the provided name.
            pub fn eval<S: core::fmt::Display>(
                self,
                name: S,
                dataloader: TestLoader<EC, L>,
            ) -> Box<dyn MetricsRenderer> {
                self.eval_all([(name, dataloader)])
            }
            /// Run the evaluation on multiple named datasets sequentially.
            ///
            /// Prefer this over calling [`eval`](Self::eval) in a loop — the progress logger
            /// receives the correct `total_tests` count and `end_test` is called between splits.
            pub fn eval_all<S: core::fmt::Display>(
                mut self,
                splits: impl IntoIterator<Item = (S, TestLoader<EC, L>)>,
            ) -> Box<dyn MetricsRenderer> {
                let splits: Vec<_> = splits.into_iter().collect();
                let total_tests = splits.len();
                self.event_processor
                    .process_test(EvaluatorEvent::Start {
                        total_tests,
                    });
                for (name, dataloader) in splits {
                    let dataloader = dataloader
                        .to_device(self.model.devices().first().unwrap());
                    let name = EvaluationName::new(name);
                    let total_items = dataloader.num_items();
                    let mut iterator = dataloader.iter();
                    let mut iteration = 0;
                    self.event_processor
                        .process_test(
                            EvaluatorEvent::StartTest(name.clone(), total_items),
                        );
                    while let Some(item) = iterator.next() {
                        let progress = iterator.progress();
                        iteration += 1;
                        let item = self.model.step(item);
                        let item = EvaluationItem::new(item, progress, Some(iteration));
                        self.event_processor
                            .process_test(
                                EvaluatorEvent::ProcessedItem(name.clone(), item),
                            );
                        if self.interrupter.should_stop() {
                            {
                                {
                                    let lvl = ::log::Level::Info;
                                    if lvl <= ::log::STATIC_MAX_LEVEL
                                        && lvl <= ::log::max_level()
                                    {
                                        ::log::__private_api::log(
                                            { ::log::__private_api::GlobalLogger },
                                            format_args!("Testing interrupted."),
                                            lvl,
                                            &(
                                                "burn_train::evaluator::base",
                                                "burn_train::evaluator::base",
                                                ::log::__private_api::loc(),
                                            ),
                                            (),
                                        );
                                    }
                                }
                            };
                            break;
                        }
                    }
                    self.event_processor.process_test(EvaluatorEvent::EndTest);
                }
                let summary = self
                    .summary
                    .and_then(|summary| {
                        summary
                            .init()
                            .map(|summary| summary.with_model(self.model.to_string()))
                            .ok()
                    });
                self.event_processor.process_test(EvaluatorEvent::End(summary));
                self.event_processor.renderer()
            }
        }
    }
    mod builder {
        use crate::{
            ApplicationLoggerInstaller, Evaluator, FileApplicationLoggerInstaller,
            InferenceStep, Interrupter, LearnerSummaryConfig, TestOutput,
            evaluator::components::{
                EvaluatorComponentTypes, EvaluatorComponentTypesMarker,
            },
            logger::{EvaluationProgressLogger, FileMetricLogger},
            metric::{
                Adaptor, Metric, Numeric,
                processor::{
                    AsyncProcessorEvaluation, FullEventProcessorEvaluation,
                    MetricsEvaluation,
                },
                store::{EventStoreClient, LogEventStore},
            },
            renderer::{MetricsRenderer, default_renderer},
        };
        use burn_core::module::Module;
        use std::{
            collections::BTreeSet, path::{Path, PathBuf},
            sync::Arc,
        };
        use typing_rules::*;
        /// Struct to configure and create an [evaluator](Evaluator).
        ///
        /// The generics components of the builder should probably not be set manually, as they are
        /// optimized for Rust type inference.
        pub struct EvaluatorBuilder<EC: EvaluatorComponentTypes, L: Label> {
            tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
            event_store: LogEventStore,
            summary_metrics: BTreeSet<String>,
            renderer: Option<Box<dyn MetricsRenderer + 'static>>,
            interrupter: Interrupter,
            metrics: MetricsEvaluation<TestOutput<EC>>,
            directory: PathBuf,
            summary: bool,
            progress_logger: Option<Box<dyn EvaluationProgressLogger>>,
        }
        impl<M, L> EvaluatorBuilder<EvaluatorComponentTypesMarker<M>, L>
        where
            M: Module + InferenceStep + core::fmt::Display + 'static,
            L: Label,
        {
            /// Creates a new evaluator builder.
            ///
            /// # Arguments
            ///
            /// * `directory` - The directory to save the checkpoints.
            pub fn new(directory: impl AsRef<Path>) -> Self {
                let directory = directory.as_ref().to_path_buf();
                let log_file = directory.join("evaluation.log");
                Self {
                    tracing_logger: Some(
                        Box::new(FileApplicationLoggerInstaller::new(log_file)),
                    ),
                    event_store: LogEventStore::default(),
                    summary_metrics: Default::default(),
                    renderer: None,
                    interrupter: Interrupter::new(),
                    summary: false,
                    metrics: MetricsEvaluation::default(),
                    directory,
                    progress_logger: None,
                }
            }
        }
        impl<EC: EvaluatorComponentTypes, L: Label> EvaluatorBuilder<EC, L> {
            /// Registers [numeric](crate::metric::Numeric) test [metrics](Metric).
            pub fn metrics<Me: EvalMetricRegistration<EC, L>>(
                self,
                metrics: Me,
            ) -> Self {
                metrics.register(self)
            }
            /// Registers text [metrics](Metric).
            pub fn metrics_text<Me: EvalTextMetricRegistration<EC, L>>(
                self,
                metrics: Me,
            ) -> Self {
                metrics.register(self)
            }
            /// By default, Rust logs are captured and written into
            /// `evaluation.log`. If disabled, standard Rust log handling
            /// will apply.
            pub fn with_application_logger(
                mut self,
                logger: Option<Box<dyn ApplicationLoggerInstaller>>,
            ) -> Self {
                self.tracing_logger = logger;
                self
            }
            /// Register a [numeric](crate::metric::Numeric) test [metric](Metric).
            pub fn metric_numeric<Me>(mut self, metric: Me) -> Self
            where
                Me: Metric + Numeric + 'static,
                TestOutput<EC>: Adaptor<Me::Input>,
            {
                self.summary_metrics.insert(metric.name().to_string());
                self.metrics.register_test_metric_numeric(metric);
                self
            }
            /// Register a text test [metric](Metric).
            pub fn metric<Me>(mut self, metric: Me) -> Self
            where
                Me: Metric + 'static,
                TestOutput<EC>: Adaptor<Me::Input>,
            {
                self.summary_metrics.insert(metric.name().to_string());
                self.metrics.register_test_metric(metric);
                self
            }
            /// Replace the default CLI renderer with a custom one.
            ///
            /// # Arguments
            ///
            /// * `renderer` - The custom renderer.
            pub fn renderer(
                mut self,
                renderer: Box<dyn MetricsRenderer + 'static>,
            ) -> Self {
                self.renderer = Some(renderer);
                self
            }
            /// Enable the evaluation summary report.
            ///
            /// The summary will be displayed at the end of `.eval()`.
            pub fn summary(mut self) -> Self {
                self.summary = true;
                self
            }
            /// Register a progress logger to track and store evaluation progress.
            ///
            /// # Example
            /// ```ignore
            /// // `MyEvaluationProgressLogger` is a user-defined type that implements
            /// // `burn_train::logger::EvaluationProgressLogger`.
            /// let evaluator = EvaluatorBuilder::new(...)
            ///     .with_progress_logger(MyEvaluationProgressLogger);
            /// ```
            pub fn with_progress_logger<PL>(mut self, logger: PL) -> Self
            where
                PL: EvaluationProgressLogger + 'static,
            {
                self.progress_logger = Some(Box::new(logger));
                self
            }
            /// Builds the evaluator.
            #[allow(clippy::type_complexity)]
            pub fn build(mut self, model: EC::Model) -> Evaluator<EC, L> {
                let renderer = self
                    .renderer
                    .unwrap_or_else(|| default_renderer(self.interrupter.clone(), None));
                self.event_store
                    .register_logger(FileMetricLogger::new_eval(self.directory.clone()));
                let event_store = Arc::new(EventStoreClient::new(self.event_store));
                let full_processor = FullEventProcessorEvaluation::new(
                    self.metrics,
                    renderer,
                    event_store,
                );
                let full_processor = match self.progress_logger {
                    Some(logger) => full_processor.with_progress_logger(logger),
                    None => full_processor,
                };
                let event_processor = AsyncProcessorEvaluation::new(full_processor);
                let summary = if self.summary {
                    Some(LearnerSummaryConfig {
                        directory: self.directory,
                        metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
                    })
                } else {
                    None
                };
                Evaluator::<EC, L> {
                    model,
                    interrupter: self.interrupter,
                    event_processor,
                    summary,
                }
            }
        }
        /// Trait to fake variadic generics.
        pub trait EvalMetricRegistration<EC: EvaluatorComponentTypes, L: Label>: Sized {
            /// Register the metrics.
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L>;
        }
        /// Trait to fake variadic generics.
        pub trait EvalTextMetricRegistration<
            EC: EvaluatorComponentTypes,
            L: Label,
        >: Sized {
            /// Register the metrics.
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L>;
        }
        impl<M1, EC: EvaluatorComponentTypes, L: Label> EvalTextMetricRegistration<EC, L>
        for (M1,)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            M1: Metric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1,) = self;
                let builder = Labeled::<_, L>::new(builder.metric(M1));
                builder
            }
        }
        impl<M1, EC: EvaluatorComponentTypes, L: Label> EvalMetricRegistration<EC, L>
        for (M1,)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            M1: Metric + crate::metric::Numeric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1,) = self;
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M1));
                builder
            }
        }
        impl<
            M1,
            M2,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalTextMetricRegistration<EC, L> for (M1, M2)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            M1: Metric + 'static,
            M2: Metric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2) = self;
                let builder = Labeled::<_, L>::new(builder.metric(M1));
                let builder = Labeled::<_, L>::new(builder.metric(M2));
                builder
            }
        }
        impl<M1, M2, EC: EvaluatorComponentTypes, L: Label> EvalMetricRegistration<EC, L>
        for (M1, M2)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            M1: Metric + crate::metric::Numeric + 'static,
            M2: Metric + crate::metric::Numeric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2) = self;
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M1));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M2));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalTextMetricRegistration<EC, L> for (M1, M2, M3)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            M1: Metric + 'static,
            M2: Metric + 'static,
            M3: Metric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3) = self;
                let builder = Labeled::<_, L>::new(builder.metric(M1));
                let builder = Labeled::<_, L>::new(builder.metric(M2));
                let builder = Labeled::<_, L>::new(builder.metric(M3));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalMetricRegistration<EC, L> for (M1, M2, M3)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            M1: Metric + crate::metric::Numeric + 'static,
            M2: Metric + crate::metric::Numeric + 'static,
            M3: Metric + crate::metric::Numeric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3) = self;
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M1));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M2));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M3));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            M4,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalTextMetricRegistration<EC, L> for (M1, M2, M3, M4)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            TestOutput<EC>: Adaptor<M4::Input>,
            M1: Metric + 'static,
            M2: Metric + 'static,
            M3: Metric + 'static,
            M4: Metric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3, M4) = self;
                let builder = Labeled::<_, L>::new(builder.metric(M1));
                let builder = Labeled::<_, L>::new(builder.metric(M2));
                let builder = Labeled::<_, L>::new(builder.metric(M3));
                let builder = Labeled::<_, L>::new(builder.metric(M4));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            M4,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalMetricRegistration<EC, L> for (M1, M2, M3, M4)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            TestOutput<EC>: Adaptor<M4::Input>,
            M1: Metric + crate::metric::Numeric + 'static,
            M2: Metric + crate::metric::Numeric + 'static,
            M3: Metric + crate::metric::Numeric + 'static,
            M4: Metric + crate::metric::Numeric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3, M4) = self;
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M1));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M2));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M3));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M4));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            M4,
            M5,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalTextMetricRegistration<EC, L> for (M1, M2, M3, M4, M5)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            TestOutput<EC>: Adaptor<M4::Input>,
            TestOutput<EC>: Adaptor<M5::Input>,
            M1: Metric + 'static,
            M2: Metric + 'static,
            M3: Metric + 'static,
            M4: Metric + 'static,
            M5: Metric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3, M4, M5) = self;
                let builder = Labeled::<_, L>::new(builder.metric(M1));
                let builder = Labeled::<_, L>::new(builder.metric(M2));
                let builder = Labeled::<_, L>::new(builder.metric(M3));
                let builder = Labeled::<_, L>::new(builder.metric(M4));
                let builder = Labeled::<_, L>::new(builder.metric(M5));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            M4,
            M5,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalMetricRegistration<EC, L> for (M1, M2, M3, M4, M5)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            TestOutput<EC>: Adaptor<M4::Input>,
            TestOutput<EC>: Adaptor<M5::Input>,
            M1: Metric + crate::metric::Numeric + 'static,
            M2: Metric + crate::metric::Numeric + 'static,
            M3: Metric + crate::metric::Numeric + 'static,
            M4: Metric + crate::metric::Numeric + 'static,
            M5: Metric + crate::metric::Numeric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3, M4, M5) = self;
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M1));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M2));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M3));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M4));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M5));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            M4,
            M5,
            M6,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalTextMetricRegistration<EC, L> for (M1, M2, M3, M4, M5, M6)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            TestOutput<EC>: Adaptor<M4::Input>,
            TestOutput<EC>: Adaptor<M5::Input>,
            TestOutput<EC>: Adaptor<M6::Input>,
            M1: Metric + 'static,
            M2: Metric + 'static,
            M3: Metric + 'static,
            M4: Metric + 'static,
            M5: Metric + 'static,
            M6: Metric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3, M4, M5, M6) = self;
                let builder = Labeled::<_, L>::new(builder.metric(M1));
                let builder = Labeled::<_, L>::new(builder.metric(M2));
                let builder = Labeled::<_, L>::new(builder.metric(M3));
                let builder = Labeled::<_, L>::new(builder.metric(M4));
                let builder = Labeled::<_, L>::new(builder.metric(M5));
                let builder = Labeled::<_, L>::new(builder.metric(M6));
                builder
            }
        }
        impl<
            M1,
            M2,
            M3,
            M4,
            M5,
            M6,
            EC: EvaluatorComponentTypes,
            L: Label,
        > EvalMetricRegistration<EC, L> for (M1, M2, M3, M4, M5, M6)
        where
            TestOutput<EC>: Adaptor<M1::Input>,
            TestOutput<EC>: Adaptor<M2::Input>,
            TestOutput<EC>: Adaptor<M3::Input>,
            TestOutput<EC>: Adaptor<M4::Input>,
            TestOutput<EC>: Adaptor<M5::Input>,
            TestOutput<EC>: Adaptor<M6::Input>,
            M1: Metric + crate::metric::Numeric + 'static,
            M2: Metric + crate::metric::Numeric + 'static,
            M3: Metric + crate::metric::Numeric + 'static,
            M4: Metric + crate::metric::Numeric + 'static,
            M5: Metric + crate::metric::Numeric + 'static,
            M6: Metric + crate::metric::Numeric + 'static,
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC, L>,
            ) -> EvaluatorBuilder<EC, L> {
                let (M1, M2, M3, M4, M5, M6) = self;
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M1));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M2));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M3));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M4));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M5));
                let builder = Labeled::<_, L>::new(builder.metric_numeric(M6));
                builder
            }
        }
    }
    pub(crate) mod components {
        use crate::InferenceStep;
        use burn_core::module::Module;
        use std::marker::PhantomData;
        /// All components necessary to evaluate a model grouped in one trait.
        pub trait EvaluatorComponentTypes {
            /// The model to evaluate.
            type Model: Module + InferenceStep + core::fmt::Display + 'static;
        }
        /// A marker type used to provide [evaluation components](EvaluatorComponentTypes).
        pub struct EvaluatorComponentTypesMarker<M> {
            _p: PhantomData<M>,
        }
        impl<M> EvaluatorComponentTypes for EvaluatorComponentTypesMarker<M>
        where
            M: Module + InferenceStep + core::fmt::Display + 'static,
        {
            type Model = M;
        }
    }
    pub use base::*;
    pub use builder::*;
}
pub use evaluator::*;
pub use components::*;
