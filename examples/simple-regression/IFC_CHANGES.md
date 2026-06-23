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
- **`declassify_ref()`** — borrows the inner `T` from `Labeled<T, L>` without consuming it or
  the label. Used at the tensor boundary inside the batcher — tensors are not labeled, so this
  is the controlled crossing point into the tensor world
- **`declassify()`** — consumes a `Labeled<T, L>` and returns the raw `T`, stripping the label
  entirely. Used at public output boundaries (e.g., displaying predictions in a chart)

### Why tensors are not labeled

In burn, tensor operations (matmul, relu, loss, backward) are executed by the backend (WGPU,
ndarray, etc.) and have no knowledge of the IFC type system. Labeling individual tensors would
require rewriting every tensor operation in burn. Instead, IFC tracks sensitivity at the
**batch level**: the batcher receives `Vec<Labeled<Item, L>>` and returns `Labeled<Batch, L>`,
where the batch contains plain tensors. `declassify_ref()` is the explicit, auditable point
where each item's fields cross from the labeled world into the tensor world.

---

## Changes

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
implementation; used `declassify_ref()` inside `batch()` to cross from labeled items into
the tensor world.

**Burn context:** In burn, a `Dataset<Item>` is a collection of typed records loaded from a
source (here, a SQLite-backed HuggingFace dataset). A `Batcher<Item, Batch>` collects a
`Vec<Item>` and assembles it into a `Batch` containing stacked tensors — one row per item.
With IFC, items come out of the dataset already labeled (`Labeled<HousingDistrictItem, L>`),
so the batcher signature changes to `Vec<Labeled<Item, L>> -> Labeled<Batch, L>`.

Inside `batch()`, each item's fields must be passed to `Tensor::from_floats([...])`, which
takes raw `f32` values. `mcall!(item.field)` would give `Labeled<f32, L>`, which cannot go
directly into a tensor constructor. `declassify_ref()` is the correct tool: it borrows the
raw `HousingDistrictItem` from the labeled wrapper, reads its fields, and lets the tensor
constructor proceed. After all tensors are assembled, `Labeled::new(HousingBatch { ... })`
rewraps the batch with the label, re-entering the labeled world.

**Before** (`src/dataset.rs:58-70`, `src/dataset.rs:143-178`):
```rust
// Dataset had no label parameter
pub struct HousingDataset {
    dataset: SqliteDataset<HousingDistrictItem>,
}

impl Dataset<HousingDistrictItem> for HousingDataset { ... }

// Batcher signature was unlabeled
impl Batcher<HousingDistrictItem, HousingBatch> for HousingBatcher {
    fn batch(&self, items: Vec<HousingDistrictItem>, device: &Device) -> HousingBatch {
        for item in items.iter() {
            let input_tensor = Tensor::<1>::from_floats(
                [item.median_income, item.house_age, ...],
                device,
            );
            // ...
        }
        HousingBatch { inputs, targets }
    }
}
```

**After** ([`src/dataset.rs:58-70`](src/dataset.rs#L58-L70), [`src/dataset.rs:143-178`](src/dataset.rs#L143-L178)):
```rust
// Dataset is now generic over the label
pub struct HousingDataset<L: Label> {
    dataset: SqliteDataset<HousingDistrictItem, L>,
}

impl<L: Label> Dataset<HousingDistrictItem, L> for HousingDataset<L> { ... }

// Batcher signature now threaded with L
impl<L: Label> Batcher<HousingDistrictItem, HousingBatch, L> for HousingBatcher {
    fn batch(&self, items: Vec<Labeled<HousingDistrictItem, L>>, device: &Device) -> Labeled<HousingBatch, L> {
        for item in items.iter() {
            let d = item.declassify_ref(); // cross into tensor world
            let input_tensor = Tensor::<1>::from_floats(
                [d.median_income, d.house_age, d.avg_rooms, d.avg_bedrooms,
                 d.population, d.avg_occupancy, d.latitude, d.longitude],
                device,
            );
            // ...
        }
        let targets = items
            .iter()
            .map(|item| Tensor::<1>::from_floats([item.declassify_ref().median_house_value], device))
            .collect();
        // ...
        Labeled::new(HousingBatch { inputs, targets }) // rewrap with label
    }
}
```

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

**What changed:** Applied `Secret` label to the test dataset and items; added explicit
`declassify()` at the public output boundary before passing the batch to the model.

**Burn context:** Inference is the phase after training where the saved model is loaded and
run on test data to evaluate real-world performance. In this example, the test split of the
California housing dataset is loaded, batched, passed through `model.forward()`, and the
predictions are displayed in a terminal scatter chart.

The predictions are derived from `Secret` input data (house values, income, location), but the
chart itself is a public output — anyone can see it. `declassify()` at line 30 is the explicit,
auditable point where the team has decided: *"we are intentionally making this Secret batch
public."* Without it, `batch` would be `Labeled<HousingBatch, Secret>`, and `model.forward(batch.inputs)`
would fail to compile because `model.forward` expects a raw (unlabeled) `Tensor`, not a
labeled one.

This is different from training: during training, the Secret label stays on the batch the
entire time it flows through the train step, gradient computation, and optimizer. It never
needs to be declassified because those outputs (weight updates) stay inside the model.
In inference, the final predictions are displayed externally, which requires an explicit
declassification.

**Before** (`src/inference.rs:26-32`):
```rust
let dataset = HousingDataset::test();
let items: Vec<HousingDistrictItem> = dataset.iter().take(1000).collect();

let batcher = HousingBatcher::new(&device);
let batch = batcher.batch(items.clone(), &device);
let predicted = model.forward(batch.inputs);
let targets = batch.targets;
```

**After** ([`src/inference.rs:26-32`](src/inference.rs#L26-L32)):
```rust
let dataset = HousingDataset::<Secret>::test();
let items: Vec<Labeled<HousingDistrictItem, Secret>> = dataset.iter().take(1000).collect();

let batcher = HousingBatcher::new(&device);
let batch = declassify(batcher.batch(items.clone(), &device)); // explicit public boundary
let predicted = model.forward(batch.inputs);
let targets = batch.targets;
```

---

## Core Pattern

```
HuggingFace SQLite
       │
       ▼
HousingDataset<Secret>          ← data is labeled Secret at source
       │  Vec<Labeled<Item, Secret>>
       ▼
HousingBatcher::batch()
  ├── item.declassify_ref()     ← controlled crossing into tensor world
  ├── Tensor::from_floats(...)  ← raw tensors, no label
  └── Labeled::new(HousingBatch { inputs, targets })  ← rewrap with label
       │  Labeled<HousingBatch, Secret>
       ▼
Training loop (burn-train)      ← Secret flows through forward/loss/backward
  └── gradients → optimizer     ← label consumed inside burn-train (never public)

       │  Labeled<HousingBatch, Secret>   (inference path)
       ▼
declassify(batch)               ← explicit public boundary (predictions will be displayed)
       │  HousingBatch (raw)
       ▼
model.forward(batch.inputs)     ← predictions
       │
       ▼
Terminal chart                  ← public output
```
