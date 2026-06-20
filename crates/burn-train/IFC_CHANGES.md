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

`.__private_into_value()` is the explicit declassification step that extracts the raw `T` from
a `Labeled<T, L>`, consuming the label.

---

## Changes

### `fg_ifc_library/macros/src/lib.rs`

**What changed:** Extended the `mcall!` macro to handle the case where the receiver is
*unlabeled* but an argument is labeled.

**Why:** The original `mcall!` only worked when the receiver itself was a `Labeled<T, L>` value.
In `train.rs`, the model (`model`) is a plain unlabeled value, but the input (`input`) is
`Labeled<TrainingModelInput, L>`. The macro now handles both cases:

- Labeled receiver → uses inherent `__chain_ref` on the `Labeled` receiver
- Unlabeled receiver with labeled arg(s) → uses the `SecureChainRef` blanket impl (treats receiver
  as `Public`-labeled) and chains on each labeled argument

---

### `fg_ifc_library/typing_rules/src/lattice.rs`

**What changed:** Added `Join<Public, Out = Self>` as a supertrait of `Label`. Removed `: Label`
from `Join`'s trait bound.

**Why:** Generic code with `L: Label` needed to call `__chain` on labeled arguments, which
requires `L: Join<L2>`. By making `Label` imply `Join<Public, Out = Self>`, any `L: Label`
automatically satisfies the bound without extra annotations at every call site.

The `: Label` bound on `Join` was removed to fix a compile cycle error (`E0391`) in Rust 1.83:
`Label: Join<Public, ...>` combined with `Join<Other: Label>: Label` created a coinductive cycle
that Rust 1.83 cannot resolve (Rust 1.95+ handles it, but burn targets 1.83).

---

### `src/learner/supervised/step/train.rs`

**What changed:**

1. `mcall!(model.step(input))` replaces the raw `model.step(input)` call in the worker thread.
2. `MultiTrainOutput<TO, L>` gained a `L: Label` type parameter.
3. The `output` field changed from `TrainOutput<TO>` to `Labeled<TrainOutput<TO>, L>`.
4. All channel/method signatures updated to carry `L`.

**Why:** The worker thread receives `Labeled<TrainingModelInput<LC>, L>` from the dataloader.
The result of `model.step(input)` should carry the same label `L` to record that the output's
sensitivity derives from the input's. `mcall!` performs this label propagation automatically.

```rust
// Before
let output = model.step(item.item);
let item = MultiTrainOutput { output, device_id };

// After
let input = item.item;
let output: Labeled<_, _> = mcall!(model.step(input));
let item = MultiTrainOutput { output, device_id };
```

```rust
// Before
pub struct MultiTrainOutput<TO> {
    pub output: TrainOutput<TO>,
    pub(crate) device_id: usize,
}

// After
pub struct MultiTrainOutput<TO, L: Label> {
    pub output: Labeled<TrainOutput<TO>, L>,
    pub(crate) device_id: usize,
}
```

---

### `src/learner/supervised/strategies/multi/epoch.rs`

**What changed:** Both loops over worker outputs (`run_optim_main` and `run_optim_distr`) now
call `item.output.__private_into_value()` to extract the raw `TrainOutput` before accessing
`.grads` and `.item`.

**Why:** After the `MultiTrainOutput` change above, `item.output` became
`Labeled<TrainOutput<TO>, L>`. Direct field access (`item.output.grads`, `item.output.item`)
no longer compiles. `__private_into_value()` is the IFC declassification step — it consumes
the label and returns the raw `TrainOutput`.

```rust
// Before
for item in items.into_iter() {
    let grads = item.output.grads.to_device(&device_main, &learner.model());
    accumulator.accumulate(&learner.model(), grads);
    progress_items.push(item.output.item);
}

// After
for item in items.into_iter() {
    let raw_output = item.output.__private_into_value();
    let grads = raw_output.grads.to_device(&device_main, &learner.model());
    accumulator.accumulate(&learner.model(), grads);
    progress_items.push(raw_output.item);
}
```

---

### `src/learner/supervised/strategies/ddp/epoch.rs`

**What changed:** In `DdpTrainEpoch::run`, added `let item = item.__private_into_value();`
immediately after the `fcall!` train step call. In `DdpValidEpoch::run`, added
`item.__private_into_value()` when constructing `TrainingItem`.

**Why:** `fcall!(Learner::train_step(&learner, item))` returns `Labeled<TrainOutput<...>, L>`.
Downstream code (`optimizer_step(item.grads)`, `TrainingItem::new(item.item, ...)`) operates on
raw `TrainOutput`, so the label must be explicitly consumed at the boundary.

```rust
// Before
let item = learner.train_step(item);   // raw TrainOutput

// After
let item = fcall!(Learner::train_step(&learner, item));  // Labeled<TrainOutput, L>
let item = item.__private_into_value();                   // raw TrainOutput — label consumed here
// rest of function unchanged
```

---

## The Core Pattern

```
Labeled<Input, L>
       |
       | fcall!() or mcall!()
       |   — unwraps label, calls function with raw input, rewraps result
       v
Labeled<Output, L>
       |
       | .__private_into_value()
       |   — explicit declassification at IFC boundary
       v
Output  (used normally by optimizer, event processor, etc.)
```

Label propagation (`fcall!`/`mcall!`) and declassification (`.__private_into_value()`) are the
two halves of the IFC boundary. The compiler enforces that every labeled value is either
propagated or explicitly declassified — it cannot be silently dropped.
