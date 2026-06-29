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

### How `__chain` and `__chain_ref` work

You will see `__chain` and `__chain_ref` in expanded macro output and error messages — they are
the internal building blocks that `fcall!` and `mcall!` generate. Think of them as "suspend the
label outside, pass the raw value into a closure, then reattach the label to whatever the closure
produces":

```rust
// __chain (owned — takes value BY MOVE, consumes the Labeled wrapper):
// Used when fcall! sees a plain argument  →  fcall!(Func::f(arg))
let result: Labeled<R, L_joined> = labeled_x.__chain(|x: T| {
    // x is the raw T — the label is "waiting outside the closure"
    Labeled::new(something(x))  // inner call produces Labeled<R, Public>
});
// The __chain impl then joins: L_from_labeled_x ∨ Public = L_from_labeled_x

// __chain_ref (borrowed — closure gets &T, labeled_x is NOT consumed):
// Used when fcall! sees &arg  →  fcall!(Func::f(&arg))  or mcall! borrows the receiver
let result: Labeled<R, L_joined> = labeled_x.__chain_ref(|x: &T| {
    // x is &T — borrow only, label preserved outside
    Labeled::new(x.some_method())
});
```

When `fcall!` is given multiple arguments, it nests the chain calls — one per argument,
inside-out — so every argument's label is joined into the final result:

```rust
// fcall!(Func::f(&a, b))  expands to (roughly):
(a).__chain_ref(|__v0: &TypeA| {  // a is borrowed (&a)
    (b).__chain(|__v1: TypeB| {   // b is consumed (no &)
        Labeled::new(Func::f(__v0, __v1))  // call with raw values
    })
    // result of inner __chain is Labeled<R, Label_of_b>
})
// result of outer __chain_ref is Labeled<R, Label_of_a ∨ Label_of_b>
```

If either `a` or `b` is not a `Labeled<T, L>` but a plain raw value, the `SecureChain` /
`SecureChainRef` **blanket traits** handle the call and treat the label as `Public`. A raw arg
contributes nothing to the join — only the actually-labeled args matter.

The `Join` trait computes the label join (`∨`). For the `Secret`/`Public` lattice used in this
project: `Secret ∨ Public = Secret`. So one secret input makes the entire output `Secret`,
which is exactly the property IFC enforces.

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

**After** ([`fg_ifc_library/macros/src/lib.rs:634-689`](fg_ifc_library/macros/src/lib.rs#L634-L689)):

```rust
// Split the chain so only the last method's args are individually chained
let (last_entry, intermediates) = chain.split_last()          // line 634
    .expect("mcall!: method call must have at least one method");

// Generate a name per arg of the last method (__av0, __av1, ...)
let arg_names: Vec<_> = (0..arg_count)                        // line 650
    .map(|i| format_ident!("__av{}", i)).collect();

// Innermost: call the method with unwrapped args, wrap result in Labeled<_, Public>
let inner_call = quote! {                                      // line 654-666
    ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
        (#intermediate_recv).#last_method(#(#arg_names),*)
    )
};

// Wrap each arg in its own __chain (inside-out)
for (arg, name) in last_args.iter().zip(arg_names.iter()).rev() {   // line 670-672
    body = quote! { (#arg).__chain(|#name| { #body }) };
}

// Outer: chain the receiver via __chain_ref
quote! {                                                       // line 682-689
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
use `.map(...).split()` to extract `grads` and `item` from the labeled output. `labeled_item`
is pushed directly into `progress_items` (preserving its label), and the second loop uses
nested `fcall!` calls to build the event and send it to the processor.

**Why ownership is needed here (for `.map().split()`):** `item.output` is `Labeled<TrainOutput<TO>, L>`.
`TrainOutput<TO>` has two fields:

- `grads: GradientsParams` — a collection of GPU gradient tensors. Not `Clone`. The
  `accumulate` function takes it **by value** (consumes it) to merge it into the accumulator
  and free the GPU memory. You cannot borrow it.
- `item: TO` — the metric output (e.g. loss, accuracy). Needs to be collected and processed
  for events after the gradient step.

Both fields must be moved out of the inner `TrainOutput`. That requires consuming the outer
`Labeled<TrainOutput<TO>, L>`, which only `.map()` provides — it gives the closure an **owned**
`T`, unlike `__chain_ref` (which only borrows) or `__private_into_value()` (which strips the
label entirely).

**Why `map + split` and not `mcall!`:** `mcall!` is for method calls and always borrows the
receiver. Even if it consumed it, a single macro call produces one output. Here we need to
split one labeled value into **two separately labeled values** (`Labeled<GradientsParams, L>`
and `Labeled<TO, L>`) so each can be passed to its own `fcall!` call with the label intact.
That is structural destructuring, not a method call.

**Why `progress_items.push(labeled_item)` and not `fcall!(Vec::push(..., labeled_item))`:**
`fcall!` unwraps its arguments before calling the function. Using `fcall!(Vec::push(&mut
progress_items, labeled_item))` would push the raw `TO` (unwrapped from `labeled_item`) into
`progress_items`, making it `Vec<TO>` — silently dropping the label. Pushing the labeled value
directly keeps `progress_items: Vec<Labeled<TO, L>>` so the label flows through to the event
processor.

**Before/After** ([`src/learner/supervised/strategies/multi/epoch.rs:102-129`](crates/burn-train/src/learner/supervised/strategies/multi/epoch.rs#L102-L129) and [`epoch.rs:169-199`](crates/burn-train/src/learner/supervised/strategies/multi/epoch.rs#L169-L199)):

```rust
// Before (no IFC)
for item in items.into_iter() {
    let grads = item.output.grads.to_device(&device_main, &learner.model());
    accumulator.accumulate(&learner.model(), grads);
    progress_items.push(item.output.item);
}
// ...
for item in progress_items {   // item: TO (raw)
    event_processor.process_train(LearnerEvent::ProcessedItem(TrainingItem::new(item, ...)));
}

// After — run_optim_main (lines 102-129)
for item in items.into_iter() {
    // item.output: Labeled<TrainOutput<TO>, L>
    // .map() consumes the Labeled wrapper, gives the closure an owned TrainOutput<TO>
    //        so we can move out both fields (grads and item) at once.
    // .split() then takes the Labeled<(GradientsParams, TO), L> tuple and turns it into
    //          two separate Labeled values — each carrying label L independently.
    let (labeled_grads, labeled_item) = item.output
        .map(|o| (o.grads.to_device(&device_main, &learner.model()), o.item))
        .split();
    // labeled_grads: Labeled<GradientsParams, L>
    // labeled_item:  Labeled<TO, L>

    // fcall! unwraps labeled_grads to pass raw GradientsParams to accumulate().
    // The label is "consumed" here — once gradients enter the accumulator they
    // are blended across batches and no longer track a single source.
    fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));

    // Push the *labeled* item directly — NOT via fcall!, which would unwrap it first.
    // This keeps progress_items as Vec<Labeled<TO, L>>, not Vec<TO>.
    progress_items.push(labeled_item);
}
// ... optimizer step (accumulator.grads() is raw — OK since label was consumed above) ...
for item in progress_items {   // item: Labeled<TO, L> — label intact across the whole loop
    // Build TrainingItem (wraps the metric output + progress info) — still labeled
    let labeled_training_item = fcall!(TrainingItem::new(item, progress.clone(), Some(iteration), Some(learner.lr_current())));
    // Wrap in the event enum variant — still labeled
    let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
    // Deliver to the event processor — label enforced at this API boundary
    fcall!(EventProcessorTraining::process_train(event_processor, labeled_event));
}

// After — run_optim_distr, per-device accumulator variant (lines 169-199)
// (Same pattern; grads stay on their originating device rather than being moved to main)
for item in items.into_iter() {
    let accumulator = &mut accumulators[item.device_id];
    let (labeled_grads, labeled_item) = item.output.map(|o| (o.grads, o.item)).split();
    fcall!(GradientsAccumulator::accumulate(accumulator, &learner.model(), labeled_grads));
    progress_items.push(labeled_item);  // direct push — preserves label
}
// ...
for item in progress_items {   // item: Labeled<TO, L>
    let labeled_training_item = fcall!(TrainingItem::new(item, progress.clone(), Some(iteration), Some(learner.lr_current())));
    let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
    fcall!(EventProcessorTraining::process_train(event_processor, labeled_event));
}
```

The label `L` flows intact: `mcall!(model.step(input))` → `MultiTrainOutput.output:
Labeled<TrainOutput<TO>, L>` → `.map().split()` → `labeled_grads` / `labeled_item` →
`fcall!(accumulate(...))` / `progress_items.push(labeled_item)` →
`fcall!(TrainingItem::new(...))` → `fcall!(LearnerEvent::ProcessedItem(...))` →
`fcall!(process_train(...))`. The label survives all the way to the event processor.

---

### `src/learner/supervised/strategies/ddp/epoch.rs`

**What changed:** Both `DdpTrainEpoch::run` and `DdpValidEpoch::run` now use `fcall!` for the
train/valid step and then use `.map().split()` (train) or nested `fcall!` (both) to keep labeled
values flowing through the rest of the function rather than declassifying immediately.

**Before/After — training loop** ([`src/learner/supervised/strategies/ddp/epoch.rs:105-131`](crates/burn-train/src/learner/supervised/strategies/ddp/epoch.rs#L105-L131)):

```rust
// Before
let item = learner.train_step(item);   // raw TrainOutput<...>
// item.grads / item.item accessed directly

// After (lines 105-131)
// fcall! borrows &learner (via __chain_ref) and consumes item (via __chain),
// then calls Learner::train_step with raw values and rewraps the result.
let item = fcall!(Learner::train_step(&learner, item));  // Labeled<TrainOutput<...>, L>

// .map() consumes the Labeled wrapper so we can move both grads and item out.
// .split() gives us two independently-labeled values from the tuple.
let (labeled_grads, labeled_item) = item.map(|o| (o.grads, o.item)).split();
// labeled_grads: Labeled<GradientsParams, L>
// labeled_item:  Labeled<TrainingModelOutput, L>

match self.grad_accumulation {
    Some(_) => {
        // fcall! unwraps labeled_grads to pass raw GradientsParams to accumulate().
        fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));
        // grads from accumulator.grads() are raw — their label was consumed above, which is fine
        // ...
    }
    None => {
        // &mut *learner is a "reborrow": learner is &mut Learner<LC>, so &mut learner would be
        // &mut &mut Learner<LC> (double ref). &mut *learner dereferences first → &mut Learner<LC>.
        fcall!(Learner::optimizer_step(&mut *learner, labeled_grads));  // label consumed here
    }
}

// labeled_item still carries label L — it was not touched by the optimizer path above.
// Build the event chain: item → TrainingItem → LearnerEvent, each step still labeled.
let labeled_training_item = fcall!(TrainingItem::new(labeled_item, progress, Some(iteration), Some(learner.lr_current())));
let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
// &mut *processor is the same reborrow trick: processor is Arc<Mutex<...>>, then lock().unwrap()
// gives &mut SupervisedTrainingEventProcessor<LC> — need &mut *processor to avoid double-ref.
fcall!(EventProcessorTraining::process_train(&mut *processor, labeled_event));
```

**Before/After — validation loop** ([`src/learner/supervised/strategies/ddp/epoch.rs:51-54`](crates/burn-train/src/learner/supervised/strategies/ddp/epoch.rs#L51-L54)):

```rust
// Before
let item = model.step(item);   // raw output

// After (lines 51-54)
// Same pattern as single/epoch.rs: plain model receiver (Public) + labeled item (L) → Labeled<Output, L>
let item = fcall!(InferenceStep::step(&model, item));
let labeled_training_item = fcall!(TrainingItem::new(item, progress, Some(iteration), None));
let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
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

**Before/After — validation loop** ([`src/learner/supervised/strategies/single/epoch.rs:51-54`](crates/burn-train/src/learner/supervised/strategies/single/epoch.rs#L51-L54)):

```rust
// Before
let item = model.step(item);  // ❌ item: Labeled<Input, L>, step expects raw Input
let item = TrainingItem::new(item, progress, Some(iteration), None);
processor.process_valid(LearnerEvent::ProcessedItem(item));

// After (lines 51-54)
// &model is a plain unlabeled value → __chain_ref treats it as Public.
// item: Labeled<Input, L>            → __chain propagates L.
// Result label = Public ∨ L = L      → output is Labeled<Output, L>
let item = fcall!(InferenceStep::step(&model, item));

// progress and Some(iteration) are plain values (Public), None is plain too.
// item carries label L → result is still Labeled<TrainingItem, L>
let labeled_training_item = fcall!(TrainingItem::new(item, progress, Some(iteration), None));

// Wrap in the enum variant — plain enum constructor, label flows through
let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));

// Final consumer: fcall! enforces that process_valid receives a Labeled event
fcall!(EventProcessorTraining::process_valid(processor, labeled_event));
```

**Before/After — training loop** ([`src/learner/supervised/strategies/single/epoch.rs:98-118`](crates/burn-train/src/learner/supervised/strategies/single/epoch.rs#L98-L118)):

```rust
// Before
let item = learner.train_step(item);  // ❌ same problem: labeled input, raw-expecting method
accumulator.accumulate(&learner.model(), item.grads);  // field access on raw TrainOutput
// ...
None => learner.optimizer_step(item.grads),
let item = TrainingItem::new(item.item, progress, ...);
processor.process_train(LearnerEvent::ProcessedItem(item));

// After (lines 98-118)
// fcall! generates: (&learner).__chain_ref(|__v0| (item).__chain(|__v1| Labeled::new(Learner::train_step(__v0, __v1))))
// &learner borrows the learner (label Public); item is consumed (label L) → result is Labeled<TrainOutput, L>
let item = fcall!(Learner::train_step(&learner, item));  // Labeled<TrainOutput, L>

// .map() opens the Labeled wrapper so we can move both fields out at once (grads is not Clone).
// .split() splits the Labeled<(GradientsParams, TO), L> into two separate Labeled values.
let (labeled_grads, labeled_item) = item.map(|o| (o.grads, o.item)).split();
// labeled_grads: Labeled<GradientsParams, L>   ← goes to optimizer (label consumed there)
// labeled_item:  Labeled<TO, L>                ← goes to event processor (label lives on)

match self.grad_accumulation {
    Some(accumulation) => {
        // fcall! unwraps labeled_grads to give accumulate() the raw GradientsParams.
        // After this call, labeled_grads is consumed; its label is "done".
        fcall!(GradientsAccumulator::accumulate(&mut accumulator, &learner.model(), labeled_grads));
        if accumulation <= accumulation_current {
            let grads = accumulator.grads();   // raw GradientsParams — blended across many batches,
            learner.optimizer_step(grads);     // no single source label → raw call is correct
        }
    }
    None => {
        // block {} required: Some arm's body returns () (plain unit), but fcall! returns
        // Labeled<(), L>. Both arms of a match must have the same type, so wrap in a block
        // that also evaluates to () by dropping the Labeled result.
        fcall!(Learner::optimizer_step(&mut *learner, labeled_grads));
    }
}

// labeled_item has been untouched through the whole optimizer section.
// Chain it through the event structs — each fcall! propagates the label.
let labeled_training_item = fcall!(TrainingItem::new(labeled_item, progress, Some(iteration), Some(learner.lr_current())));
let labeled_event = fcall!(LearnerEvent::ProcessedItem(labeled_training_item));
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
nested `fcall!` ([`src/evaluator/base.rs:19`](crates/burn-train/src/evaluator/base.rs#L19) and [`base.rs:69-72`](crates/burn-train/src/evaluator/base.rs#L69-L72)):

```rust
// Before — struct
pub struct Evaluator<EC: EvaluatorComponentTypes, L: Label> {
    pub(crate) model: EC::Model,
    pub(crate) interrupter: Interrupter,
    pub(crate) event_processor: ...,
    pub summary: Option<LearnerSummaryConfig>,
    // ❌ L is declared but never stored — rustc E0392 "type parameter never used"
}

// After (lines 19-27) — PhantomData anchors the L parameter
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

// After (lines 69-72)
// &self.model is plain (unlabeled) — __chain_ref treats it as Public.
// item: Labeled<Input, L> — __chain propagates L.
// Result: Labeled<Output, Public ∨ L> = Labeled<Output, L>
let item = fcall!(InferenceStep::step(&self.model, item));

// Wrap the output in EvaluationItem (adds progress info) — still labeled
let labeled_eval_item = fcall!(EvaluationItem::new(item, progress, Some(iteration)));

// name.clone() is a plain String (Public label) — does not change the label of labeled_eval_item
// Result: Labeled<EvaluatorEvent, L>
let labeled_event = fcall!(EvaluatorEvent::ProcessedItem(name.clone(), labeled_eval_item));

// Deliver to the event processor — the label is enforced at this API boundary
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
