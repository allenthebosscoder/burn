use crate::{
    AsyncProcessorEvaluation, EvaluationItem, FullEventProcessorEvaluation, InferenceStep,
    Interrupter, LearnerSummaryConfig,
    evaluator::components::EvaluatorComponentTypes,
    metric::processor::{EvaluatorEvent, EventProcessorEvaluation},
    renderer::{EvaluationName, MetricsRenderer},
};
use burn_core::{data::dataloader::DataLoader, module::Module};
use macros::fcall; // import ifc macros
use std::sync::Arc;
use typing_rules::*; // import filament ifc

pub(crate) type TestInput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Input;
pub(crate) type TestOutput<EC> = <<EC as EvaluatorComponentTypes>::Model as InferenceStep>::Output;

pub(crate) type TestLoader<EC, L: Label> = Arc<dyn DataLoader<TestInput<EC>, L>>;

/// Evaluates a model on a specific dataset.
pub struct Evaluator<EC: EvaluatorComponentTypes, L: Label> {
    pub(crate) model: EC::Model,
    pub(crate) interrupter: Interrupter,
    pub(crate) event_processor:
        AsyncProcessorEvaluation<FullEventProcessorEvaluation<TestOutput<EC>>>,
    /// Config for creating a summary of the evaluation
    pub summary: Option<LearnerSummaryConfig>,
    pub(crate) _label: std::marker::PhantomData<L>,
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
            .process_test(EvaluatorEvent::Start { total_tests });

        for (name, dataloader) in splits {
            let dataloader = dataloader.to_device(self.model.devices().first().unwrap());
            let name = EvaluationName::new(name);
            let total_items = dataloader.num_items();
            let mut iterator = dataloader.iter();
            let mut iteration = 0;

            self.event_processor
                .process_test(EvaluatorEvent::StartTest(name.clone(), total_items));

            while let Some(item) = iterator.next() {
                let progress = iterator.progress();
                iteration += 1;

                let item = fcall!(InferenceStep::step(&self.model, item));
                let labeled_event = item.map(|o| EvaluatorEvent::ProcessedItem(name.clone(), EvaluationItem::new(o, progress, Some(iteration))));
                fcall!(EventProcessorEvaluation::process_test(&mut self.event_processor, labeled_event));

                if self.interrupter.should_stop() {
                    log::info!("Testing interrupted.");
                    break;
                }
            }

            self.event_processor.process_test(EvaluatorEvent::EndTest);
        }

        let summary = self.summary.and_then(|summary| {
            summary
                .init()
                .map(|summary| summary.with_model(self.model.to_string()))
                .ok()
        });

        self.event_processor
            .process_test(EvaluatorEvent::End(summary));

        self.event_processor.renderer()
    }
}
