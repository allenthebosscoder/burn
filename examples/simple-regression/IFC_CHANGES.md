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

### `src/dataset.rs`

**What changed:** Added `L: Label` type parameter to `HousingDataset` and to the `Batcher`
implementation; replaced `declassify_ref()` with three helper methods called via `fcall!`.

**Burn context:** In burn, a `Dataset<Item>` is a collection of typed records loaded from a
source (here, a SQLite-backed HuggingFace dataset). A `Batcher<Item, Batch>` collects a
`Vec<Item>` and assembles it into a `Batch` containing stacked tensors — one row per item.
With IFC, items come out of the dataset already labeled (`Labeled<HousingDistrictItem, L>`),
so the batcher signature changes to `Vec<Labeled<Item, L>> -> Labeled<Batch, L>`.

**Why `fcall!` and not `declassify_ref()`:** `declassify_ref()` strips the label immediately
inside the batch loop — earlier than necessary and without any compiler-enforced discipline.
`fcall!` is the correct boundary: it unwraps each labeled item only at the call site of a
specific library function (`Tensor::from_floats`, `Tensor::cat`, `normalizer.normalize`),
and the macro automatically rewraps the result with the propagated label.

**Why helper methods:** `fcall!` calls existing functions that don't know about IFC. The three
helpers (`item_to_tensors`, `cat_pairs`, `build_batch`) each wrap a natural step in the
batching process that calls into burn library functions. They take and return raw (unlabeled)
types; `fcall!` is what bridges them into the labeled world.

- `item_to_tensors`: converts one `HousingDistrictItem` into a `(Tensor<2>, Tensor<1>)` pair
  (inputs as a `[1 × 8]` matrix, target as a 1-element vector)
- `cat_pairs`: stacks two `(Tensor<2>, Tensor<1>)` pairs along axis 0, building up a batch
  row by row via `Tensor::cat`
- `build_batch`: normalizes the stacked input tensor (min–max normalization) and packages the
  pair into a `HousingBatch`

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

**After** ([`src/dataset.rs:59-184`](src/dataset.rs#L59-L184)):
```rust
// Dataset is now generic over the label
pub struct HousingDataset<L: Label> {
    dataset: SqliteDataset<HousingDistrictItem, L>,
}

impl<L: Label> Dataset<HousingDistrictItem, L> for HousingDataset<L> { ... }

// Three helpers that operate on raw (unlabeled) types
impl HousingBatcher {
    fn item_to_tensors(&self, item: HousingDistrictItem, device: &Device) -> (Tensor<2>, Tensor<1>) {
        let input = Tensor::<1>::from_floats(
            [item.median_income, item.house_age, item.avg_rooms, item.avg_bedrooms,
             item.population, item.avg_occupancy, item.latitude, item.longitude],
            device,
        ).unsqueeze();
        let target = Tensor::<1>::from_floats([item.median_house_value], device);
        (input, target)
    }

    fn cat_pairs(&self, a: (Tensor<2>, Tensor<1>), b: (Tensor<2>, Tensor<1>)) -> (Tensor<2>, Tensor<1>) {
        (Tensor::cat(vec![a.0, b.0], 0), Tensor::cat(vec![a.1, b.1], 0))
    }

    fn build_batch(&self, pair: (Tensor<2>, Tensor<1>), device: &Device) -> HousingBatch {
        HousingBatch {
            inputs: self.normalizer.to_device(device).normalize(pair.0),
            targets: pair.1,
        }
    }
}

// Batcher signature threaded with L; uses fcall! instead of declassify_ref
impl<L: Label> Batcher<HousingDistrictItem, HousingBatch, L> for HousingBatcher {
    fn batch(&self, items: Vec<Labeled<HousingDistrictItem, L>>, device: &Device) -> Labeled<HousingBatch, L> {
        let labeled_pair = items
            .into_iter()
            .map(|item| fcall!(HousingBatcher::item_to_tensors(&self, item, device)))
            .reduce(|a, b| fcall!(HousingBatcher::cat_pairs(&self, a, b)))
            .unwrap();

        fcall!(HousingBatcher::build_batch(&self, labeled_pair, device))
    }
}
```

Note: `.map(|item| ...)` here is `Iterator::map` (standard Rust iterator), NOT `Labeled::map`.
The result of each `fcall!` is a `Labeled<(Tensor<2>, Tensor<1>), L>`, and `reduce` folds
them together by calling `fcall!(cat_pairs(...))` with two labeled owned args. This works
because the `Label` supertrait now requires `Join<Self, Out = Self>`, making `Join<L, L> = L`
available for any generic `L`.

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
  ├── fcall!(item_to_tensors(&self, item, device))   ← fcall! crosses into tensor world
  │     item unwrapped → raw fields → Tensor::from_floats → rewrap Labeled<pair, Secret>
  ├── fcall!(cat_pairs(&self, a, b))                 ← two labeled args, same label L
  │     both a and b unwrapped → Tensor::cat → rewrap Labeled<pair, Secret>
  │     (works because Label: Join<Self, Out = Self> → Join<L,L> = L for generic L)
  └── fcall!(build_batch(&self, labeled_pair, device))
        pair unwrapped → normalize → HousingBatch → rewrap Labeled<HousingBatch, Secret>
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
