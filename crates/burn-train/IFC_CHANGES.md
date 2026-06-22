# IFC Type Labeling Changes

This document describes the changes made to add Filament IFC (Information Flow Control) type
labeling to the burn-train library.

## Background

Filament IFC tracks data sensitivity through the type system using `Labeled<T, L>` wrappers.
When a model processes sensitive input `Labeled<Input, L>`, the output should carry the same
label `Labeled<Output, L>` — proving at compile time that the output's sensitivity matches the
input's. Two macros drive this:

- **`mcall!(receiver.method(arg))`** — method calls where the receiver or an argument is labeled
- **`fcall!(SomeType::function(&receiver, arg))`** — free function calls (UFCS form) where an argument is labeled

Both macros unwrap labeled arguments, call the function with raw values, and rewrap the result
in `Labeled<Result, L>`.

Three other primitives are used at IFC boundaries:

- **`.map(|inner| ...)`** — consumes a `Labeled<T, L>`, gives the closure an **owned** `T`,
  and returns `Labeled<R, L>`. Used when you need to move fields out of a labeled struct while
  keeping the label attached to the result.
- **`.split()`** — defined on `Labeled<(A, B), L>`, splits into `(Labeled<A, L>, Labeled<B, L>)`.
  Used together with `map` to destructure a labeled struct into separately usable labeled values.
- **`.__private_into_value()`** — the explicit declassification step. Extracts the raw `T` from
  `Labeled<T, L>`, consuming the label entirely. Used at the boundary where IFC-tracked code
  hands off to code that doesn't know about labels (e.g., the optimizer).

---

## Changes

### `fg_ifc_library/macros/src/lib.rs`

**What changed:** Rewrote the plain method-call branch of `mcall!` to use a two-layer
`__chain_ref` / `__chain` pattern, replacing the old `__mcall_preserve_label` helper.

**Why:** The original `mcall!` only worked when the receiver itself was `Labeled<T, L>`. In
`train.rs`, the model (`model`) is a plain unlabeled value, but the input (`input`) is
`Labeled<TrainingModelInput, L>`. The new expansion handles both cases uniformly.

**Before** (`fg_ifc_library/macros/src/lib.rs` — the old method-call branch, now replaced):

```rust
// The helper that only worked for labeled receivers
fn __mcall_preserve_label<__T, __U, __L: ::typing_rules::Label>(
    wrapper: &::typing_rules::lattice::Labeled<__T, __L>,
    func: impl FnOnce(&__T) -> __U
) -> ::typing_rules::lattice::Labeled<__U, __L> {
    ::typing_rules::lattice::Labeled::<__U, __L>::new(func(wrapper.__private_value()))
}
// ...
// All args folded into a single closure body, no per-arg label chaining
let closure_body = chain.iter().fold(quote! { inner }, |acc, (method, turbofish, args)| { ... });
quote! { { #helper  __mcall_preserve_label(&(#base), |inner| #closure_body) } }
```

**After** ([`fg_ifc_library/macros/src/lib.rs:436-492`](fg_ifc_library/macros/src/lib.rs#L436-L492)):

```rust
// Split the chain so only the last method's args are individually chained
let (last_entry, intermediates) = chain.split_last()          // line 436
    .expect("mcall!: method call must have at least one method");

// Generate a name per arg of the last method (__av0, __av1, ...)
let arg_names: Vec<_> = (0..arg_count)                        // line 452
    .map(|i| format_ident!("__av{}", i)).collect();

// Innermost: call the method with unwrapped args, wrap result in Labeled<_, Public>
let inner_call = quote! {                                      // line 456-468
    ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
        (#intermediate_recv).#last_method(#(#arg_names),*)
    )
};

// Wrap each arg in its own __chain (inside-out)
for (arg, name) in last_args.iter().zip(arg_names.iter()).rev() {   // line 472-474
    body = quote! { (#arg).__chain(|#name| { #body }) };
}

// Outer: chain the receiver via __chain_ref
quote! {                                                       // line 484-492
    {
        use ::typing_rules::function_rewrite::SecureChain;
        use ::typing_rules::function_rewrite::SecureChainRef;
        (#base).__chain_ref(|__recv| { #body })
    }
}
```

So `mcall!(model.step(input))` expands to:

```rust
{
    use ::typing_rules::function_rewrite::SecureChain;
    use ::typing_rules::function_rewrite::SecureChainRef;
    (model).__chain_ref(|__recv| {
        (input).__chain(|__av0| {
            Labeled::<_, Public>::new(__recv.step(__av0))
        })
    })
}
```

- **`__chain_ref` on the receiver** — borrows it (closure gets `&model`). If `model` is
  `Labeled<T, L>`, the inherent `__chain_ref` propagates label `L`. If `model` is plain `T`,
  the `SecureChainRef` blanket impl treats it as `Public`.
- **`__chain` on each argument** — consumes it. If `input` is `Labeled<T, L>`, the inherent
  `__chain` propagates `L`. If it is a raw value, the `SecureChain` blanket impl treats it as
  `Public`.
- The outermost chains `Join` all labels together, so the final type is the LUB of every input.

For chained calls like `mcall!(key.chars().all(f))`, intermediate methods have their args passed
raw; only the final method's arguments are individually chained.

---

### `fg_ifc_library/typing_rules/src/lattice.rs`

**What changed:** Added `Join<Public, Out = Self>` as a supertrait of `Label`. Removed `: Label`
from `Join`'s own supertrait bound.

**Before/After** ([`fg_ifc_library/typing_rules/src/lattice.rs:13`](fg_ifc_library/typing_rules/src/lattice.rs#L13) and [`lattice.rs:32`](fg_ifc_library/typing_rules/src/lattice.rs#L32)):

```rust
// BEFORE
pub trait Label: Clone + Copy + Default + Send + Sync + 'static {}
// ...
pub trait Join<Other: Label>: Label {   // ← had `: Label` supertrait
    type Out: Label;
}

// AFTER
pub trait Label: Clone + Copy + Default + Send + Sync + Join<Public, Out = Self> + 'static {}
// ...
pub trait Join<Other: Label> {          // ← `: Label` removed
    type Out: Label;
}
```

**Why (the supertrait addition):** When `mcall!(model.step(input))` is used with a plain
`model` and a labeled `input: Labeled<TrainingModelInput, L>`, `SecureChainRef` treats the
receiver as `Public` so the chain produces `<Public as Join<L>>::Out`. The type checker can only
resolve this to `L` if it knows `L: Join<Public, Out = L>`. Making that a supertrait of `Label`
([`lattice.rs:13`](fg_ifc_library/typing_rules/src/lattice.rs#L13)) means every `L: Label`
satisfies it automatically — no per-call-site where-clause needed.

**Why (removing `: Label` from `Join`):** The old declaration was
`pub trait Join<Other: Label>: Label`. With the new `Label: Join<Public, Out = Self>` supertrait,
Rust 1.83 sees a coinductive cycle — `Label` requires `Join<Public>`, and `Join` (as it was)
required `Label` as its own supertrait. Rust 1.83 rejects this with `E0391`. Dropping `: Label`
from `Join` ([`lattice.rs:32`](fg_ifc_library/typing_rules/src/lattice.rs#L32)) breaks the
cycle. Nothing is lost in practice because every type we implement `Join` for (`Public`, `A`,
`B`, `AB`) is already a `Label`.

---

### `src/learner/supervised/step/train.rs`

**What changed:**

1. `mcall!(model.step(input))` replaces the raw `model.step(input)` call in the worker thread.
2. `MultiTrainOutput<TO, L>` gained a `L: Label` type parameter.
3. The `output` field changed from `TrainOutput<TO>` to `Labeled<TrainOutput<TO>, L>`.
4. All channel/method signatures updated to carry `L`.

**Why:** The worker thread receives `Labeled<TrainingModelInput<LC>, L>` from the dataloader.
The result of `model.step(input)` should carry the same label `L` to prove that the output's
sensitivity derives from the input's. `mcall!` performs this label propagation automatically.

**Before/After** ([`src/learner/supervised/step/train.rs:55-58`](crates/burn-train/src/learner/supervised/step/train.rs#L55-L58)):

```rust
// Before
let output = model.step(item.item);
let item = MultiTrainOutput { output, device_id };

// After (lines 55-58)
let input = item.item;                                   // move the labeled input out first
let output: Labeled<_, _> = mcall!(model.step(input));  // label propagates: input L → output L
let item = MultiTrainOutput { output, device_id };
```

**Before/After** ([`src/learner/supervised/step/train.rs:73-76`](crates/burn-train/src/learner/supervised/step/train.rs#L73-L76)):

```rust
// Before
pub struct MultiTrainOutput<TO> {
    pub output: TrainOutput<TO>,
    pub(crate) device_id: usize,
}

// After (lines 73-76)
pub struct MultiTrainOutput<TO, L: Label> {
    pub output: Labeled<TrainOutput<TO>, L>,  // entire TrainOutput wrapped with the input's label
    pub(crate) device_id: usize,
}
```

---

### `src/learner/supervised/strategies/multi/epoch.rs`

**What changed:** Both loops over worker outputs (`run_optim_main` and `run_optim_distr`) now
use `.map(...).split()` to extract `grads` and `item` from the labeled output, then pass each
to a `fcall!` call rather than accessing them as raw fields.

**Why ownership is needed here:** `item.output` is `Labeled<TrainOutput<TO>, L>`.
`TrainOutput<TO>` has two fields:

- `grads: GradientsParams` — a collection of GPU gradient tensors. Not `Clone`. The
  `accumulate` function takes it **by value** (consumes it) to merge it into the accumulator
  and free the GPU memory. You cannot borrow it.
- `item: TO` — the metric output (e.g. loss, accuracy). Needs to be moved into `progress_items`
  so it can be processed for events after the gradient step.

Both fields must be moved out of the inner `TrainOutput`. That requires consuming the outer
`Labeled<TrainOutput<TO>, L>`, which only `.map()` provides — it gives the closure an **owned**
`T`, unlike `__chain_ref` (which only borrows) or `__private_into_value()` (which strips the
label entirely).

**Why `map + split` and not `mcall!`:** `mcall!` is for method calls and always borrows the
receiver. Even if it consumed it, a single macro call produces one output. Here we need to
split one labeled value into **two separately labeled values** (`Labeled<GradientsParams, L>`
and `Labeled<TO, L>`) so each can be passed to its own `fcall!` call with the label intact.
That is structural destructuring, not a method call.

**Before/After** ([`src/learner/supervised/strategies/multi/epoch.rs:104-108`](crates/burn-train/src/learner/supervised/strategies/multi/epoch.rs#L104-L108) and [`epoch.rs:177-179`](crates/burn-train/src/learner/supervised/strategies/multi/epoch.rs#L177-L179)):

```rust
// Before (no IFC)
for item in items.into_iter() {
    let grads = item.output.grads.to_device(&device_main, &learner.model());
    accumulator.accumulate(&learner.model(), grads);
    progress_items.push(item.output.item);
}

// After — run_optim_main (lines 103-108)
for item in items.into_iter() {
    let (labeled_grads, labeled_item) = item.output   // Labeled<TrainOutput<TO>, L>
        .map(|o| (o.grads.to_device(&device_main, &learner.model()), o.item))
        //   ↑ consumes Labeled, gives owned TrainOutput — moves both fields into a tuple
        .split();
        //   ↑ Labeled<(GradientsParams, TO), L>  →  (Labeled<GradientsParams, L>, Labeled<TO, L>)
    fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));
    fcall!(Vec::push(&mut progress_items, labeled_item));
}

// After — run_optim_distr, per-device accumulator variant (lines 177-179)
for item in items.into_iter() {
    let accumulator = &mut accumulators[item.device_id];
    let (labeled_grads, labeled_item) = item.output.map(|o| (o.grads, o.item)).split();
    fcall!(GradientsAccumulator::accumulate(accumulator, &learner.model(), labeled_grads));
    fcall!(Vec::push(&mut progress_items, labeled_item));
}
```

The label `L` flows intact from the input data through `mcall!(model.step(input))` →
`MultiTrainOutput.output: Labeled<TrainOutput<TO>, L>` → `.map().split()` →
`labeled_grads` / `labeled_item` → `fcall!(accumulate(...))` / `fcall!(Vec::push(...))`.
`progress_items` therefore holds `Vec<Labeled<TO, L>>`, keeping the label alive until the
event processor boundary.

---

### `src/learner/supervised/strategies/ddp/epoch.rs`

**What changed:** Both `DdpTrainEpoch::run` and `DdpValidEpoch::run` now use `fcall!` for the
train/valid step and then use `.map().split()` (train) or `.map()` (valid) to keep labeled values
flowing through the rest of the function rather than declassifying immediately.

**Before/After — training loop** ([`src/learner/supervised/strategies/ddp/epoch.rs:104-130`](crates/burn-train/src/learner/supervised/strategies/ddp/epoch.rs#L104-L130)):

```rust
// Before
let item = learner.train_step(item);   // raw TrainOutput<...>
// item.grads / item.item accessed directly

// After (lines 104-130)
let item = fcall!(Learner::train_step(&learner, item));            // Labeled<TrainOutput<...>, L>
let (labeled_grads, labeled_item) = item.map(|o| (o.grads, o.item)).split();
// labeled_grads: Labeled<GradientsParams, L>
// labeled_item:  Labeled<TrainingModelOutput, L>

match self.grad_accumulation {
    Some(_) => {
        fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));
        // ...
    }
    None => {
        fcall!(Learner::optimizer_step(&mut learner, labeled_grads));  // label consumed here
    }
}

// labeled_item stays labeled all the way to the event processor
let labeled_event = labeled_item
    .map(|o| TrainingItem::new(o, progress, Some(iteration), Some(learner.lr_current())))
    .map(LearnerEvent::ProcessedItem);
fcall!(EventProcessorTraining::process_train(&mut *processor, labeled_event));
```

**Before/After — validation loop** ([`src/learner/supervised/strategies/ddp/epoch.rs:51-53`](crates/burn-train/src/learner/supervised/strategies/ddp/epoch.rs#L51-L53)):

```rust
// Before
let item = model.step(item);   // raw output

// After (lines 51-53)
let item = fcall!(InferenceStep::step(&model, item));   // Labeled<Output, L>
let labeled_event = item
    .map(|o| LearnerEvent::ProcessedItem(TrainingItem::new(o, progress, Some(iteration), None)));
fcall!(EventProcessorTraining::process_valid(processor, labeled_event));
```

In both loops the label is never stripped with `__private_into_value()` — it flows all the way
from the input through the step, through gradient accumulation or the optimizer, and into the
event processor. The `fcall!` at the event processor boundary is the final consumer.

---

### `src/learner/supervised/strategies/single/epoch.rs`

**Burn context — what is an "epoch"?**

Training a neural network is an iterative process. One *epoch* is one complete pass through the
entire training dataset. For each batch pulled from the dataloader:

1. Run the model forward (compute predictions)
2. Compute the loss (how wrong the predictions are)
3. Run backwards (compute gradients — how to nudge each weight to reduce the loss)
4. Apply the optimizer to update the weights using those gradients
5. Repeat for the next batch

A *validation epoch* does the same forward pass on a separate held-out dataset but skips steps
3 and 4 — it only measures quality without changing the weights.

`single/epoch.rs` handles the case where there is exactly one GPU and one dataloader. This is
the simplest training path. Both `SingleDeviceTrainEpoch` and `SingleDeviceValidEpoch` hold the
dataloader and drive the loop — they don't own the model; they borrow the `Learner` which does.

**What changed:** Added `use macros::fcall;` and rewrote both the train and validation loops to
use `fcall!` / `map` / `split` instead of calling `model.step` and `learner.train_step` directly.

**Why labeling is needed:** The dataloader yields `Labeled<Input, L>` items. Calling
`model.step(item)` directly fails to compile because `step` expects raw `Input`. The IFC
label must be threaded through every call — the model doesn't declassify the data, it just
processes it, so the output carries the same sensitivity as the input.

**Before/After — validation loop** ([`src/learner/supervised/strategies/single/epoch.rs:51-53`](crates/burn-train/src/learner/supervised/strategies/single/epoch.rs#L51-L53)):

```rust
// Before
let item = model.step(item);  // ❌ item: Labeled<Input, L>, step expects raw Input
let item = TrainingItem::new(item, progress, Some(iteration), None);
processor.process_valid(LearnerEvent::ProcessedItem(item));

// After (lines 51-53)
let item = fcall!(InferenceStep::step(&model, item));  // Labeled<Output, L>
let labeled_event = item
    .map(|o| LearnerEvent::ProcessedItem(TrainingItem::new(o, progress, Some(iteration), None)));
fcall!(EventProcessorTraining::process_valid(processor, labeled_event));
```

**Before/After — training loop** ([`src/learner/supervised/strategies/single/epoch.rs:97-117`](crates/burn-train/src/learner/supervised/strategies/single/epoch.rs#L97-L117)):

```rust
// Before
let item = learner.train_step(item);  // ❌ same problem: labeled input, raw-expecting method
accumulator.accumulate(&learner.model(), item.grads);  // field access on raw TrainOutput
// ...
None => learner.optimizer_step(item.grads),
let item = TrainingItem::new(item.item, progress, ...);
processor.process_train(LearnerEvent::ProcessedItem(item));

// After (lines 97-117)
let item = fcall!(Learner::train_step(&learner, item));        // Labeled<TrainOutput, L>
let (labeled_grads, labeled_item) = item.map(|o| (o.grads, o.item)).split();

match self.grad_accumulation {
    Some(accumulation) => {
        // labeled_grads consumed by accumulate, raw grads later come from accumulator.grads()
        fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));
        if accumulation <= accumulation_current {
            let grads = accumulator.grads();       // raw — came from accumulator, not input data
            learner.optimizer_step(grads);         // raw call is fine here
        }
    }
    None => { fcall!(Learner::optimizer_step(&mut *learner, labeled_grads)); }
    //        block {} needed: Some arm returns (), fcall returns Labeled<(), L> — must match
}

let labeled_event = labeled_item
    .map(|o| LearnerEvent::ProcessedItem(TrainingItem::new(o, progress, ...)));
fcall!(EventProcessorTraining::process_train(processor, labeled_event));
```

Note the `&mut *learner` reborrow: `learner` is already `&mut Learner<LC>`, so `&mut learner`
would be `&mut &mut Learner<LC>` (double reference). `&mut *learner` dereferences first and
then takes a mutable reference, giving back the correct `&mut Learner<LC>`.

**Why the `Some` accumulation arm stays unlabeled:** When gradient accumulation is enabled, the
optimizer only runs every `N` batches. The gradients from each batch are merged into an
`accumulator` buffer with `fcall!(accumulate(..., labeled_grads))`. When it's time to step,
`accumulator.grads()` returns a combined `GradientsParams` that is the average of many batches
— it has no single source label. Calling `learner.optimizer_step(grads)` directly is correct
because the label was already consumed by `fcall!(accumulate(...))`.

---

### `src/learner/supervised/paradigm.rs`

**Burn context — what is a "paradigm"?**

`SupervisedTraining` is the top-level training coordinator. It is what user code actually
constructs and calls. It holds configuration (number of epochs, checkpointing strategy,
which metrics to track, which learning rate scheduler to use) and the dataloaders. When `.fit()`
is called, it assembles a `Learner` (which wraps the model + optimizer + scheduler) and delegates
the per-epoch work to whichever epoch strategy is configured (single/multi/DDP).

It's called a "paradigm" because it encodes the full training paradigm: what to train, how many
times, what to checkpoint, when to stop early.

**What changed:** Removed the `O` (optimizer) type parameter from the `impl` block and the
`O: Optimizer<M>` where-clause ([`src/learner/supervised/paradigm.rs:78-83`](crates/burn-train/src/learner/supervised/paradigm.rs#L78-L83)).

**Why:** `LearningComponentsMarker` was refactored to only carry `<LR, M>` (the lr scheduler and
model). The optimizer is no longer a generic parameter because burn always uses the concrete
`ModuleOptimizer` type — there is only ever one optimizer implementation. Making it generic would
add a type parameter that buys nothing. The `paradigm.rs` impl was still using the old three-arg
form `LearningComponentsMarker<LR, M, O>`, which had already been removed from `components.rs`.

**Before/After** ([`src/learner/supervised/paradigm.rs:78-83`](crates/burn-train/src/learner/supervised/paradigm.rs#L78-L83)):

```rust
// Before
impl<LR, M, O, L> SupervisedTraining<LearningComponentsMarker<LR, M, O>, L>
where
    LR: LrScheduler + 'static,
    M: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static,
    O: Optimizer<M> + 'static,   // ❌ Optimizer trait no longer exists here; LearningComponentsMarker<LR, M, O> has 3 args
    L: Label

// After (lines 78-83)
impl<LR, M, L> SupervisedTraining<LearningComponentsMarker<LR, M>, L>
where
    LR: LrScheduler + 'static,
    M: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static,
    L: Label
```

This is not an IFC change — it is a stale reference to the old `LearningComponentsMarker` API
that was only caught once the prior IFC fixes made the crate compile far enough to reach this
impl block.

---

### `src/evaluator/base.rs` and `src/evaluator/builder.rs`

**Burn context — what is an evaluator?**

After training is complete (or at any point), you may want to measure the model's quality on a
test dataset that was never seen during training. The `Evaluator` does exactly that: it runs the
model over each batch of test data, collects metrics (accuracy, loss, etc.), and produces a
summary report. Unlike the training epoch, there are no gradients, no optimizer steps, and no
weight updates. It is read-only from the model's perspective.

**What is a builder?**

`EvaluatorBuilder` is a fluent configuration API. Instead of one large constructor call, you
chain method calls to configure the evaluator step by step:
```rust
EvaluatorBuilder::new("./output")
    .metrics((AccuracyMetric::new(),))
    .build(model)
```
Each method returns a new (or mutated) builder, and `.build(model)` produces the final
`Evaluator`. The `gen_tuple!` macro generates `EvalMetricRegistration` and
`EvalTextMetricRegistration` trait implementations for tuples of metrics so you can register
multiple metrics at once with `.metrics((M1, M2, M3))`.

**Why `L: Label` is on both structs:** The evaluator's dataloader also yields `Labeled<Input, L>`
items — the test data is sensitive in the same way the training data is. So the model call
`model.step(item)` faces the same labeled-input problem as the training loops.

**`evaluator/base.rs` — What changed and why:**
Added `PhantomData<L>` to hold the unused type parameter, and updated the eval loop to use
`fcall!` + `map` ([`src/evaluator/base.rs:18`](crates/burn-train/src/evaluator/base.rs#L18) and [`base.rs:68-72`](crates/burn-train/src/evaluator/base.rs#L68-L72)):

```rust
// Before — struct
pub struct Evaluator<EC: EvaluatorComponentTypes, L: Label> {
    pub(crate) model: EC::Model,
    pub(crate) interrupter: Interrupter,
    pub(crate) event_processor: ...,
    pub summary: Option<LearnerSummaryConfig>,
    // ❌ L is declared but never stored — rustc E0392 "type parameter never used"
}

// After (line 18-26) — PhantomData anchors the L parameter
pub struct Evaluator<EC: EvaluatorComponentTypes, L: Label> {
    pub(crate) model: EC::Model,
    pub(crate) interrupter: Interrupter,
    pub(crate) event_processor: ...,
    pub summary: Option<LearnerSummaryConfig>,
    pub(crate) _label: std::marker::PhantomData<L>,
}

// Before — eval loop body
let item = self.model.step(item);   // ❌ item: Labeled<Input, L>, step expects raw Input
let item = EvaluationItem::new(item, progress, Some(iteration));
self.event_processor.process_test(EvaluatorEvent::ProcessedItem(name.clone(), item));

// After (lines 68-72)
let item = fcall!(InferenceStep::step(&self.model, item));  // Labeled<Output, L>
let labeled_event = item.map(|o|
    EvaluatorEvent::ProcessedItem(name.clone(), EvaluationItem::new(o, progress, Some(iteration)))
);
fcall!(EventProcessorEvaluation::process_test(&mut self.event_processor, labeled_event));
```

**`evaluator/builder.rs` — What changed and why:**
Added `PhantomData<L>` to the struct and initialized it in `new()` and `build()`. Also removed
the incorrect `Labeled::new(...)` wraps in `gen_tuple!`
([`src/evaluator/builder.rs:25`](crates/burn-train/src/evaluator/builder.rs#L25),
[`builder.rs:52-62`](crates/burn-train/src/evaluator/builder.rs#L52-L62),
[`builder.rs:173-178`](crates/burn-train/src/evaluator/builder.rs#L173-L178),
[`builder.rs:208`](crates/burn-train/src/evaluator/builder.rs#L208) and [`builder.rs:225`](crates/burn-train/src/evaluator/builder.rs#L225)):

```rust
// Before — struct (L unused → E0392)
pub struct EvaluatorBuilder<EC: EvaluatorComponentTypes, L: Label> {
    tracing_logger: ...,
    event_store: ...,
    // ... no field holds L
}

// After (line 25-35)
pub struct EvaluatorBuilder<EC: EvaluatorComponentTypes, L: Label> {
    tracing_logger: ...,
    event_store: ...,
    // ...
    _label: std::marker::PhantomData<L>,  // anchors the L parameter
}

// Before — new() and build() (lines 52 and 173)
Self { tracing_logger: ..., ... }           // ❌ missing _label
Evaluator::<EC, L> { model, ..., summary }  // ❌ missing _label

// After
Self { tracing_logger: ..., ..., _label: std::marker::PhantomData }
Evaluator::<EC, L> { model, ..., summary, _label: std::marker::PhantomData }

// Before — gen_tuple! metric registration (lines 208, 225)
$(let builder = Labeled::<_, L>::new(builder.metric($M));)*
// ❌ wraps EvaluatorBuilder in Labeled — subsequent iterations then try to call
//    .metric() on Labeled<EvaluatorBuilder, L> which has no such method.
//    The builder itself is not sensitive data; only the input items are.

// After (lines 208, 225)
$(let builder = builder.metric($M);)*
$(let builder = builder.metric_numeric($M);)*
// The builder is not IFC-tracked — registering a metric is a configuration step,
// not a data-flow step. No labeling needed here.
```

---

### `src/learner/supervised/strategies/ddp/epoch.rs` (follow-up fix)

**What changed:** Fixed a double-reference bug introduced when `optimizer_step` was changed to use
`fcall!` ([`src/learner/supervised/strategies/ddp/epoch.rs:120`](crates/burn-train/src/learner/supervised/strategies/ddp/epoch.rs#L120)):

```rust
// Before (broken)
fcall!(Learner::optimizer_step(&mut learner, labeled_grads));
// ❌ learner: &mut Learner<LC>
//    &mut learner = &mut (&mut Learner<LC>) = &mut &mut Learner<LC>  ← double ref, won't compile

// After (line 120)
fcall!(Learner::optimizer_step(&mut *learner, labeled_grads));
//                              ^^^^^^^^
// &mut *learner: dereference first (→ Learner<LC>), then take mutable reference (→ &mut Learner<LC>)
// This is a "reborrow" — standard Rust idiom when you have &mut T and need to pass &mut T
// to something expecting it as an owned value (here: the UFCS first argument).
```

---

## The Core Pattern

```
Labeled<Input, L>
       |
       | fcall!() or mcall!()
       |   — unwraps label, calls function/method with raw values, rewraps result
       v
Labeled<Output, L>
       |
       +------ need multiple owned fields? --------------------------------+
       |                                                                   |
       | .__private_into_value()              .map(|o| (o.a, o.b)).split()
       |   — strip label, get raw T              — stay labeled, get two
       |   — use when downstream code             Labeled<A, L> and Labeled<B, L>
       |     doesn't know about labels            — use when each piece needs
       |     (optimizer, event processor)          its own fcall!() downstream
       v                                                                   v
Output  (raw, used by optimizer etc.)         (Labeled<A, L>, Labeled<B, L>)
                                                    |
                                                    | fcall!() each piece separately
                                                    v
                                              (eventually .__private_into_value()
                                               or another fcall!/mcall! chain)
```

The compiler enforces that every `Labeled<T, L>` is either propagated through a macro or
explicitly consumed. It cannot be silently dropped or accessed without going through the
IFC layer.
