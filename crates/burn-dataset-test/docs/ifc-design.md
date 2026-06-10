# IFC Integration into Burn Dataset

## Goal

Integrate Filament-style Information Flow Control (IFC) into Burn.

The objective is to track sensitive training data as it flows through Burn's machine learning pipeline.

The security lattice consist of only two labels:

```text
Public
Secret
```

where:

```text
Public ⊑ Secret
```

Any computation derived from Secret data should also be considered Secret unless explicitly declassified.

---

# Design Philosophy

The primary goal is to identify where user-provided data enters the Burn framework and introduce IFC labels at that point.

Labels should then propagate naturally through the system.

---

# Burn Data Flow

User training data enters Burn through the Dataset abstraction.

Current Burn pipeline:

```text
Dataset
    ↓
Iterator
    ↓
DataLoader
    ↓
Batcher
    ↓
Tensor
    ↓
Model
    ↓
Loss
    ↓
Gradients
    ↓
Parameters
```

Therefore the Dataset abstraction is treated as the IFC entry point.

---

# Dataset Labeling

## Original Burn Design

Burn datasets are parameterized only by their item type:

```rust
Dataset<I>
```

Example:

```rust
Dataset<MnistItem>
```

The Dataset trait originally returned:

```rust
Option<I>
```

from:

```rust
get()
```

---

## IFC Dataset Design

Datasets are now parameterized by both:

```rust
Dataset<I, L>
```

where:

- `I` = dataset item type
- `L` = security label

Example:

```rust
Dataset<MnistItem, Secret>
```

This represents a dataset whose contents are Secret.

---

# Dataset Retrieval

The Dataset trait was modified from:

```rust
fn get(&self, index: usize) -> Option<I>
```

to:

```rust
fn get(&self, index: usize)
    -> Option<Labeled<I, L>>
```

Meaning:

```text
Data retrieved from a dataset
inherits the dataset label.
```

---

# Labeling Strategy: Immediate Labeling

Store:

```rust
Vec<Labeled<I, L>>
```

inside the dataset.

Example:

```text
User Data
    ↓
Labeled<I, L>
    ↓
Stored
    ↓
Retrieved
```

The project requirement is that data should become labeled immediately when it enters Burn.

This prevents unlabeled user data from existing within Burn's internal storage structures.

---

# InMemDataset Changes

Original:

```rust
pub struct InMemDataset<I> {
    items: Vec<I>,
}
```

Modified:

```rust
pub struct InMemDataset<I, L: Label> {
    items: Vec<Labeled<I, L>>,
}
```

All items stored by the dataset are now labeled.

---

# Dataset Iterator

Original iterator design:

```rust
Iterator<Item = I>
```

IFC design:

```rust
Iterator<Item = Labeled<I, L>>
```

Reason:

Labels must be preserved while iterating through a dataset.

The iterator should never strip labels.

---

# Dataset Transforms

The following Burn components consume Dataset:

```text
DatasetIterator
SelectionDataset
PartialDataset
ShuffleDataset
WindowDataset
SamplerDataset
MapperDataset
```

These transforms are expected to preserve labels.

For example:

```text
Dataset<I, Secret>
    ↓
ShuffleDataset
    ↓
Dataset<I, Secret>
```

No declassification should occur during dataset transformations.

---

# IFC Library Changes

The IFC implementation is based on:

```text
typing_rules/src/lattice.rs
```

Core IFC types:

```rust
Label
Labeled<T, L>
FlowsTo
Join
```

The current implementation uses phantom types to associate security labels with values:

```rust
pub struct Labeled<T, L: Label> {
    value: T,
    _marker: PhantomData<L>,
}
```

---

# Outstanding Questions

## Send + Sync

Burn datasets require:

```rust
Send + Sync
```

because datasets may be shared across worker threads.

The IFC labels currently do not explicitly implement these bounds.

Need to determine whether:

```rust
pub trait Label
```

should be updated to include:

```rust
Send + Sync
```

or whether those bounds should be introduced elsewhere.

---

## DataLoader

Not yet analyzed.

Need to determine how dataset labels propagate through:

```text
Dataset
    ↓
DataLoader
```

---

## Batcher

Not yet analyzed.

Need to determine how labeled dataset items become labeled tensors.

---

## Tensor

Not yet analyzed.

Need to determine whether tensor labels should be stored:

- on tensors themselves
- on tensor data
- through wrapper types

---

# Current Status

Successfully prototyped:

```rust
Dataset<I, L>
```

and:

```rust
InMemDataset<I, L>
```

in a separate test crate.

Current propagation path under investigation:

```text
Dataset
    ↓
DatasetIterator
```

Next step is ensuring iterator semantics correctly preserve:

```rust
Labeled<I, L>
```

through iteration.