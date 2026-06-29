# IFC Type Labeling Changes — simple-regression

This document describes the changes made to add Filament IFC (Information Flow Control) type
labeling to the `simple-regression` example in burn.

## Background

The California housing dataset contains sensitive information about housing districts — median
income, population, location, and house value. IFC labels track this sensitivity through the
entire ML pipeline at the type level, so the compiler proves that sensitive data cannot leak
to a public output without an explicit `declassify()` call.

The key types and tools used here:

- **`Labeled<T, L>`** — wraps a value `T` with a sensitivity label `L`; produced by the
  dataset and consumed by the batcher
- **`Secret`** — the label used in this example; sits above `Public` in the lattice, meaning
  Secret data cannot flow to a Public output without declassification
- **`fcall!(F::func(&receiver, arg, ...))`** — UFCS-style macro for calling existing library
  functions that don't know about IFC. Each labeled arg is unwrapped via `__chain`/`__chain_ref`,
  the function is called with raw values, and the result is rewrapped with the joined label.
- **`declassify()`** — consumes a `Labeled<T, L>` and returns the raw `T`, stripping the label
  entirely. Used only at public output boundaries (e.g., displaying predictions in a chart)

### Why tensors are not labeled

In burn, tensor operations (matmul, relu, loss, backward) are executed by the backend (WGPU,
ndarray, etc.) and have no knowledge of the IFC type system. Labeling individual tensors would
require rewriting every tensor operation in burn. Instead, IFC tracks sensitivity at the
**batch level**: the batcher receives `Vec<Labeled<Item, L>>` and returns `Labeled<Batch, L>`,
where the batch contains plain tensors. The explicit crossing from labeled items into the tensor
world happens inside helper methods called via `fcall!`.

### Why `Labeled::map()` is backend-only

`Labeled::map()` and `Labeled::__map()` are implementation details of the IFC library and
burn-train internals. User code (dataset.rs, inference.rs) should not call them directly,
because doing so bypasses the macro's controlled unwrap/rewrap discipline. Instead, user code
uses `fcall!` to call existing library functions, letting the macro handle the label
propagation.

---

## Changes

### `fg_ifc_library/typing_rules/src/lattice.rs`

**What changed:** Added `Join<Self, Out = Self>` to the `Label` supertrait.

**Why:** When `fcall!` chains two labeled owned args of the same generic type `L`, the
resulting type involves `Join<L, L>`. This is idempotency of join (`L ∨ L = L`) — always
true in any join-semilattice, but not provable from the original `Label` supertrait which
only guaranteed `Join<Public, Out = Self>`. Adding `Join<Self, Out = Self>` to the supertrait
makes this available to any generic `L: Label` without needing special-case macro logic.

All existing concrete labels (`Public`, `Secret`, `A`, `B`, `AB`) already had `impl Join<Self>`
defined (e.g., `impl Join<Secret> for Secret { type Out = Secret; }`), so this is purely a
bound tightening — no new implementations needed.

**Before** (`lattice.rs:13`):
```rust
pub trait Label: Clone + Copy + Default + Send + Sync + Join<Public, Out = Self> + 'static {}
```

**After** ([`lattice.rs:13`](../../fg_ifc_library/typing_rules/src/lattice.rs#L13)):
```rust
pub trait Label: Clone + Copy + Default + Send + Sync + Join<Public, Out = Self> + Join<Self, Out = Self> + 'static {}
```

---

### `Cargo.toml`

**What changed:** Added `ndarray` as a declared feature.

**Burn context:** Burn supports multiple compute backends selected at compile time via feature
flags (`wgpu`, `tch-cpu`, `ndarray`, etc.). The example's `Cargo.toml` only declared `wgpu`,
`tch-cpu`, `tch-gpu`, `flex`, and `remote`. Without `ndarray` declared, `#[cfg(feature = "ndarray")]`
blocks are silently ignored by the compiler, meaning `--features ndarray` compiles but does
nothing at runtime.

**Before** (`Cargo.toml:12-18`):
```toml
[features]
default = ["burn/dataset", "burn/sqlite-bundled"]
flex = ["burn/flex"]
tch-cpu = ["burn/tch"]
tch-gpu = ["burn/tch"]
wgpu = ["burn/wgpu"]
remote = ["burn/remote"]
```

**After** ([`Cargo.toml:12-19`](Cargo.toml#L12-L19)):
```toml
[features]
default = ["burn/dataset", "burn/sqlite-bundled"]
flex = ["burn/flex"]
ndarray = ["burn/ndarray"]
tch-cpu = ["burn/tch"]
tch-gpu = ["burn/tch"]
wgpu = ["burn/wgpu"]
remote = ["burn/remote"]
```

---

### `examples/regression.rs`

**What changed:** Added an `ndarray` feature module and branch in `main()`.

**Burn context:** The `main()` function in the example dispatches to the correct device
constructor based on which backend feature is active. Each backend exposes its device via a
different constructor (`Device::wgpu(...)`, `Device::ndarray()`, etc.). Without the `ndarray`
branch, running with `--features ndarray` would compile successfully but `main()` would return
immediately with no output — every branch was gated on a feature that was not active.

**Before** (`examples/regression.rs:62-73`):
```rust
fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "remote")]
    remote::run();
}
```

**After** ([`examples/regression.rs:56-81`](examples/regression.rs#L56-L81)):
```rust
#[cfg(feature = "ndarray")]
mod ndarray {
    use burn::tensor::Device;

    pub fn run() {
        super::run(Device::ndarray());
    }
}

fn main() {
    #[cfg(feature = "flex")]
    flex::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "ndarray")]
    ndarray::run();
    #[cfg(feature = "remote")]
    remote::run();
}
```

---

### `fg_ifc_library/macros/src/lib.rs`

**What changed:** Extended `fcall!` with four new argument/expression shapes, so that user
code can call burn library functions directly without writing any helper wrapper functions.

**Background:** `fcall!` is a proc macro — it runs at compile time and rewrites the code you
write into a chain of `.__chain()` / `.__chain_ref()` calls. The rewriting works by:
1. Parsing the expression you wrote (the function call and its arguments)
2. For each argument that might be a labeled value, wrapping the rest of the expression in a
   closure that receives the unwrapped inner value
3. At the innermost level, calling the actual function with all-raw (unwrapped) values and
   wrapping the result in `Labeled::new(...)`

Before these extensions, `fcall!` could only handle simple argument shapes: plain variables
(`item`), references (`&self`), and the like. Anything more complex — an array literal, a
`vec!`, a trailing method call, a struct literal — required writing a helper function whose
only job was to translate those shapes into something `fcall!` could handle. These extensions
eliminate that need.

---

#### Extension 1 — Array literal args: `[item.field, ...]`

**Code location:** [`macros/src/lib.rs:221-271`](../../fg_ifc_library/macros/src/lib.rs#L221-L271)

**What it handles:**
```rust
fcall!(Tensor::<1>::from_floats(
    [item.median_income, item.house_age, item.avg_rooms, ...],
    device
))
```
where `item: Labeled<HousingDistrictItem, L>`.

**How it works:** When an argument is an array literal `[...]`, the macro scans every element
looking for `base.field` patterns. Each unique `base` variable (e.g. `item` appears 8 times
but is still one variable) gets one chain entry. The array is then reconstructed with the base
replaced by the unwrapped variable:

```rust
// Generated code:
(item).__chain(|__v0| {
    Labeled::new(Tensor::<1>::from_floats(
        [__v0.median_income, __v0.house_age, __v0.avg_rooms, ...],
        device
    ))
})
```

`__v0` here is `HousingDistrictItem` (the unwrapped inner type), so all the field accesses
work normally and the array of `f32` is what `Tensor::from_floats` actually receives.

---

#### Extension 2 — Trailing method chain: `func(...).method()`

**Code location:** [`macros/src/lib.rs:149-188`](../../fg_ifc_library/macros/src/lib.rs#L149-L188)

**What it handles:**
```rust
fcall!(Tensor::<1>::from_floats([...], device).unsqueeze())
```

**How it works:** The macro peels trailing method calls off the expression one at a time until
it finds the base function call. The peeled methods are saved as a "suffix" and appended to
the raw function call inside `Labeled::new(...)`:

```rust
// Generated code:
(item).__chain(|__v0| {
    Labeled::new(
        Tensor::<1>::from_floats([__v0.median_income, ...], device).unsqueeze()
    )
})
```

**Why this is needed:** `.unsqueeze()` takes `self` by value — it consumes the tensor.
`mcall!` only works for `&self` methods because it uses `__chain_ref` internally (giving you
a `&T`, not an owned `T`). The method-chain extension runs `.unsqueeze()` on the raw unwrapped
result before the label is put back, so ownership is never an issue.

---

#### Extension 3 — `vec![...]` args

**Code location:** [`macros/src/lib.rs:273-332`](../../fg_ifc_library/macros/src/lib.rs#L273-L332)

**What it handles:**
```rust
fcall!(Tensor::cat(vec![a, b], 0))
```
where `a, b: Labeled<Tensor<2>, L>`.

**How it works:** Same logic as the array extension, but for the `vec![...]` macro. Elements
can be plain path variables (`a`, `b`) or field accesses (`a.inputs`, `b.inputs`). Each unique
base variable gets a chain entry; the inner arg is reconstructed as `vec![__v0, __v1]`:

```rust
// Generated code:
(a).__chain(|__v0| {
    (b).__chain(|__v1| {
        Labeled::new(Tensor::cat(vec![__v0, __v1], 0))
    })
})
```

`__v0` and `__v1` are raw `Tensor<2>` values, which is exactly what `Tensor::cat` expects.

---

#### Extension 4 — Struct literal: `fcall!(Path { field: val, ... })`

**Code location:** [`macros/src/lib.rs:93-147`](../../fg_ifc_library/macros/src/lib.rs#L93-L147)

**What it handles:**
```rust
fcall!(HousingBatch { inputs: labeled_inputs, targets: labeled_targets })
```

**How it works:** This is a separate early-return path — it never enters the function-call
handling at all. For each field, the macro creates a chain entry for the field's value and
builds the inner expression as the struct literal using the unwrapped variable names:

```rust
// Generated code:
(labeled_inputs).__chain(|__v0| {
    (labeled_targets).__chain(|__v1| {
        Labeled::new(HousingBatch { inputs: __v0, targets: __v1 })
    })
})
```

This lets you construct a struct from labeled fields without any constructor helper function.

---

### `src/dataset.rs`

**What changed:** Added `L: Label` type parameter to `HousingDataset` and the `Batcher`
implementation; rewrote `batch()` to call burn library functions directly via `fcall!` and
`mcall!`, eliminating all helper functions.

**Burn context:** In burn, a `Dataset<Item>` is a collection of typed records loaded from a
source (here, a SQLite-backed HuggingFace dataset). A `Batcher<Item, Batch>` collects a
`Vec<Item>` and assembles it into a `Batch` containing stacked tensors — one row per item.
With IFC, items come out of the dataset already labeled (`Labeled<HousingDistrictItem, L>`),
so the batcher signature changes to `Vec<Labeled<Item, L>> -> Labeled<Batch, L>`.

**Why `fcall!` and not `declassify_ref()`:** `declassify_ref()` strips the label immediately
inside the batch loop — earlier than necessary and without any compiler-enforced discipline.
`fcall!` is the correct boundary: it unwraps each labeled item only at the call site of a
specific library function (`Tensor::from_floats`, `Tensor::cat`, `Normalizer::normalize`),
and the macro automatically rewraps the result with the propagated label.

**Why no helper functions:** Earlier versions of `fcall!` could only handle simple argument
shapes, so helper functions like `item_to_tensors` and `cat_pairs` were needed as wrappers.
After the four macro extensions above, `fcall!` can handle array literals, trailing method
calls, `vec![...]`, and struct literals directly. The helper functions are no longer needed
and have been removed — the IFC version of `batch()` now looks almost identical to the
original non-IFC version.

**Why `HousingDistrictItem` derives `Copy`:** `batch()` iterates over `items` twice — once
to build all the input tensors and once to build all the target tensors. When the items are
`Labeled<HousingDistrictItem, L>`, both the outer `Labeled` and the inner struct need to be
`Copy` so that `.copied()` on the iterator can cheaply duplicate each item. All fields of
`HousingDistrictItem` are `f32`, so `Copy` is safe to derive.

**Before** (`src/dataset.rs:58-70`, `src/dataset.rs:143-178`):
```rust
// Dataset had no label parameter
pub struct HousingDataset {
    dataset: SqliteDataset<HousingDistrictItem>,
}

impl Dataset<HousingDistrictItem> for HousingDataset { ... }

// Batcher used declassify_ref() to access fields
impl Batcher<HousingDistrictItem, HousingBatch> for HousingBatcher {
    fn batch(&self, items: Vec<HousingDistrictItem>, device: &Device) -> HousingBatch {
        for item in items.iter() {
            let d = item.declassify_ref(); // stripped label here
            let input_tensor = Tensor::<1>::from_floats(
                [d.median_income, d.house_age, ...],
                device,
            );
            // ...
        }
        HousingBatch { inputs, targets }
    }
}
```

**After** ([`src/dataset.rs:20-164`](src/dataset.rs#L20-L164)):
```rust
// HousingDistrictItem now derives Copy (all fields are f32)
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct HousingDistrictItem { ... }

// Dataset is now generic over the label
pub struct HousingDataset<L: Label> {
    dataset: SqliteDataset<HousingDistrictItem, L>,
}

impl<L: Label> Dataset<HousingDistrictItem, L> for HousingDataset<L> { ... }

// No helper functions — batch() calls burn library functions directly via fcall!/mcall!
impl<L: Label> Batcher<HousingDistrictItem, HousingBatch, L> for HousingBatcher {
    fn batch(&self, items: Vec<Labeled<HousingDistrictItem, L>>, device: &Device) -> Labeled<HousingBatch, L> {
        let inputs = items.iter().copied()
            .map(|item| fcall!(Tensor::<1>::from_floats(
                [item.median_income, item.house_age, item.avg_rooms, item.avg_bedrooms,
                 item.population, item.avg_occupancy, item.latitude, item.longitude],
                device
            ).unsqueeze()))
            .reduce(|a, b| fcall!(Tensor::cat(vec![a, b], 0)))
            .unwrap();
        let normalizer = self.normalizer.to_device(device);
        let inputs = mcall!(normalizer.normalize(inputs));

        let targets = items.iter().copied()
            .map(|item| fcall!(Tensor::<1>::from_floats([item.median_house_value], device)))
            .reduce(|a, b| fcall!(Tensor::cat(vec![a, b], 0)))
            .unwrap();

        fcall!(HousingBatch { inputs: inputs, targets: targets })
    }
}
```

Each `fcall!` call here lands directly on a burn library function — there are no user-written
intermediaries. The macro extensions handle all the labeled-to-raw translation:
- `[item.median_income, ...]` — array literal extension unwraps `item` once, passes raw fields
- `.unsqueeze()` — method chain extension applies the consuming call on the raw tensor
- `vec![a, b]` — vec extension unwraps each labeled tensor, reconstructs `vec![__v0, __v1]`
- `HousingBatch { inputs: ..., targets: ... }` — struct literal extension chains both fields

---

### `src/training.rs`

**What changed:** Applied the `Secret` label when constructing train and validation datasets.

**Burn context:** In burn, training is driven by a `SupervisedTraining` coordinator that owns
a train dataloader and a valid dataloader. These dataloaders pull batches from the dataset
and feed them to the model's train step (forward + loss + backward + optimizer) and valid step
(forward + loss only, no gradients) respectively. By annotating the datasets with `Secret` at
construction time, every batch that flows through training and validation carries the `Secret`
label — the type system enforces that gradients and loss values derived from Secret data cannot
escape without declassification.

**Before** (`src/training.rs:49-50`):
```rust
let train_dataset = HousingDataset::train();
let valid_dataset = HousingDataset::validation();
```

**After** ([`src/training.rs:49-50`](src/training.rs#L49-L50)):
```rust
let train_dataset = HousingDataset::<Secret>::train();
let valid_dataset = HousingDataset::<Secret>::validation();
```

---

### `src/inference.rs`

**What changed:** Applied `Secret` label to the test dataset; replaced `declassify(batch)` at
the batch boundary with `fcall!(run_forward(batch, &model)).split()` and moved
`declassify()` to the true public output — the terminal chart.

**Burn context:** Inference is the phase after training where the saved model is loaded and
run on test data. In this example, 1000 test items are loaded, batched, passed through the
model's forward pass, and the predicted vs. expected house values are displayed in a terminal
scatter chart.

**Why `fcall!` instead of `Labeled::map()`:** `Labeled::map()` is an IFC library primitive
reserved for internal use (the backend). User code should not call it directly. `fcall!`
is the correct boundary for calling the model's `forward` method (a burn library function)
with a labeled batch: it unwraps the `Labeled<HousingBatch, Secret>`, passes the raw batch
to `run_forward`, and rewraps the result.

**Why a `run_forward` helper:** `fcall!` needs a named function path (UFCS style). The helper
captures the two operations that must happen at the forward-pass boundary: running
`model.forward(batch.inputs)` and extracting `batch.targets`. Both are burn library operations
on raw tensors. `fcall!` bridges from the labeled `batch` into these raw operations.

**Why `.split()`:** `run_forward` returns a raw tuple `(Tensor<1>, Tensor<1>)`. After `fcall!`
wraps it, the result is `Labeled<(Tensor<1>, Tensor<1>), Secret>`. `.split()` destructures
this into two separately labeled values: `(Labeled<Tensor<1>, Secret>, Labeled<Tensor<1>, Secret>)`.
This lets each tensor be declassified independently at the output boundary.

**Why `declassify()` at the chart, not the batch:** The batch is still an input — the `Secret`
label should flow through the forward pass until the very last moment. `declassify()` at the
chart display line is the explicit, auditable decision: *"we are intentionally making these
Secret predictions public."* Declassifying earlier (at the batch level) was incorrect because
it removed the label before the forward pass, hiding the information flow from the type system.

**Before** (`src/inference.rs:26-32`):
```rust
let dataset = HousingDataset::test();
let items: Vec<HousingDistrictItem> = dataset.iter().take(1000).collect();

let batcher = HousingBatcher::new(&device);
let batch = declassify(batcher.batch(items.clone(), &device)); // too early
let predicted = model.forward(batch.inputs);
let targets = batch.targets;
```

**After** ([`src/inference.rs:18-42`](src/inference.rs#L18-L42)):
```rust
// Helper: takes raw HousingBatch and model ref, returns raw output tensors.
// Called via fcall! to bridge from Labeled<HousingBatch, Secret> into burn.
fn run_forward(batch: HousingBatch, model: &RegressionModel) -> (Tensor<1>, Tensor<1>) {
    (model.forward(batch.inputs).squeeze_dim::<1>(1), batch.targets)
}

// In infer():
let dataset = HousingDataset::<Secret>::test();
let items: Vec<Labeled<HousingDistrictItem, Secret>> = dataset.iter().take(1000).collect();

let batcher = HousingBatcher::new(&device);
let batch = batcher.batch(items, &device); // Labeled<HousingBatch, Secret>

// fcall! unwraps batch, calls run_forward with raw args, rewraps result
let (labeled_predicted, labeled_targets) = fcall!(run_forward(batch, &model)).split();

// Declassify at the actual public output boundary
let predicted = declassify(labeled_predicted).into_data();
let expected = declassify(labeled_targets).into_data();
```

---

## Core Pattern

```
HuggingFace SQLite
       │
       ▼
HousingDataset<Secret>                ← data is labeled Secret at source
       │  Vec<Labeled<Item, Secret>>
       ▼
HousingBatcher::batch()
  │
  ├── items.iter().copied().map(|item|
  │     fcall!(Tensor::from_floats([item.field, ...], device).unsqueeze())
  │   )                               ← array extension unwraps item; method-chain
  │                                      extension applies .unsqueeze() on raw tensor
  │   .reduce(|a, b|
  │     fcall!(Tensor::cat(vec![a, b], 0))
  │   )                               ← vec extension unwraps a and b (labeled Tensor<2>s)
  │   .unwrap()
  │   → Labeled<Tensor<2>, Secret>    (stacked input tensor, still labeled)
  │
  ├── mcall!(normalizer.normalize(inputs))
  │                                   ← mcall! borrows inputs via __chain_ref, calls
  │                                      normalize on the raw tensor, preserves label
  │   → Labeled<Tensor<2>, Secret>    (normalized inputs)
  │
  ├── items.iter().copied().map(|item|
  │     fcall!(Tensor::from_floats([item.median_house_value], device))
  │   ).reduce(|a, b| fcall!(Tensor::cat(vec![a, b], 0))).unwrap()
  │   → Labeled<Tensor<1>, Secret>    (stacked target tensor)
  │
  └── fcall!(HousingBatch { inputs: inputs, targets: targets })
                                      ← struct extension chains both labeled tensors,
                                         constructs HousingBatch with raw values inside
       │  Labeled<HousingBatch, Secret>
       ▼
Training loop (burn-train)            ← Secret flows through forward/loss/backward
  └── gradients → optimizer           ← label consumed inside burn-train (never public)

       │  Labeled<HousingBatch, Secret>   (inference path)
       ▼
fcall!(run_forward(batch, &model))    ← batch unwrapped, forward + squeeze, rewrap
  └── .split()                        ← Labeled<(T1,T1), S> → (Labeled<T1,S>, Labeled<T1,S>)
       │  Labeled<Tensor<1>, Secret>  (predicted)
       │  Labeled<Tensor<1>, Secret>  (targets)
       ▼
declassify(labeled_predicted)         ← explicit public boundary (chart display)
declassify(labeled_targets)
       │
       ▼
Terminal chart                        ← public output
```

### `fcall!` extension summary

| Shape written in user code | Extension | What the macro generates |
|---|---|---|
| `[item.field1, item.field2, ...]` | Array literal | chains `item` once, rebuilds array as `[__v0.field1, __v0.field2, ...]` |
| `func(...).method()` | Method chain | peels `.method()`, appends it inside `Labeled::new(func(...).method())` |
| `vec![a, b]` | vec! macro | chains `a` and `b`, rebuilds as `vec![__v0, __v1]` |
| `Struct { field: val, ... }` | Struct literal | chains each field value, builds `Struct { field: __v0, ... }` inside `Labeled::new(...)` |
