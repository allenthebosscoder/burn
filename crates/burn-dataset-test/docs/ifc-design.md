# IFC Integration into Burn Dataset

## Goal

Integrate Filament-style Information Flow Control (IFC) into Burn.

The objective is to track sensitive training data as it flows through Burn's machine learning pipeline.

The security lattice currently consists of:

```text
Public
Secret
```

where:

```text
Public ⊑ Secret
```

Any value derived from Secret data should also be considered Secret unless explicitly declassified.

The goal is to identify where user-provided data enters Burn and introduce IFC labels at those boundaries.

Once data becomes labeled, labels should propagate naturally through the rest of the framework.

---

# Burn Data Flow

User training data enters Burn through the Dataset abstraction.

Current Burn pipeline:

```text
Dataset
    ↓
Dataset Transforms
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

Therefore the Dataset abstraction is treated as the primary IFC entry point.

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

The Dataset trait returned:

```rust
Option<I>
```

from:

```rust
get()
```

---

## IFC Dataset Design

Datasets are now parameterized by:

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

This represents a dataset containing Secret information.

---

## Dataset Retrieval

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

# Dataset Transforms

Several Burn abstractions operate on top of an existing dataset:

```text
DatasetIterator
SelectionDataset
PartialDataset
ShuffleDataset
WindowDataset
SamplerDataset
MapperDataset
```

These consume one dataset and produce another.

Example:

```text
Dataset<I, Secret>
    ↓
ShuffleDataset
    ↓
Dataset<I, Secret>
```

---

## IFC Design

Dataset transforms must preserve labels.

Transforms are not allowed to:

- remove labels
- downgrade labels
- declassify data

They only modify how data is accessed.

Example:

```text
Dataset<I, Secret>
    ↓
SelectionDataset
    ↓
Dataset<I, Secret>
```

```text
Dataset<I, Secret>
    ↓
WindowDataset
    ↓
Dataset<I, Secret>
```

The security level of the dataset should remain unchanged.

---

# InMemDataset

InMemDataset stores records directly in memory.

Because user data already exists inside Burn memory, labels must be attached immediately.

---

## Original Design

```rust
pub struct InMemDataset<I> {
    items: Vec<I>,
}
```

Records were stored as raw values.

---

## IFC Design

```rust
pub struct InMemDataset<I, L: Label> {
    items: Vec<Labeled<I, L>>,
}
```

Records are stored as labeled values.

Example:

```text
CSV / JSON
    ↓
    I
    ↓
Labeled<I, L>
    ↓
Stored
```

---

## Immediate Labeling

The project requirement is:

```text
Data should be labeled immediately
when it enters Burn.
```

Therefore records loaded from:

```rust
from_csv(...)
from_json_rows(...)
```

are labeled before storage.

This prevents unlabeled user data from existing inside Burn memory.

---

## Iterator Propagation

DatasetIterator was updated to preserve labels.

Original:

```rust
Iterator<Item = I>
```

Modified:

```rust
Iterator<Item = Labeled<I, L>>
```

The iterator performs no IFC checks.

Its sole responsibility is preserving labels established by the underlying dataset.

Conceptually:

```text
Dataset<I, Secret>
    ↓
get()
    ↓
Labeled<I, Secret>
    ↓
Iterator
    ↓
Labeled<I, Secret>
```

---

# SqliteDataset

SqliteDataset differs from InMemDataset because it does not store dataset records in memory.

Instead it stores:

```text
Database metadata
Connection pool
Table information
SQL queries
```

The actual records remain inside SQLite.

---

## Original Design

```text
SQLite Database
    ↓
query_row(...)
    ↓
I
    ↓
return
```

The dataset returned:

```rust
Option<I>
```

---

## IFC Design

The project requirement is:

```text
Data should be labeled immediately
when it enters Burn.
```

For SqliteDataset, data enters Burn when a SQLite row is converted into a Rust value.

Therefore the IFC boundary is:

```rust
fn get(...)
```

rather than the SqliteDataset struct itself.

---

## Data Flow

```text
SQLite Database
    ↓
Deserialize Row
    ↓
    I
    ↓
Labeled<I, L>
    ↓
Return
```

Conceptually:

```rust
let item: I = deserialize(...);

Labeled::<I, L>::new(item)
```

---

## Why SqliteDataset Differs From InMemDataset

InMemDataset stores records directly:

```rust
Vec<Labeled<I, L>>
```

because the data already exists inside Burn memory.

SqliteDataset does not store records.

Instead, it retrieves records from an external database.

Therefore records are labeled at retrieval time rather than storage time.

Both implementations follow the same rule:

```text
Label data when it enters Burn.
```

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

---

## Label Trait

Original:

```rust
pub trait Label:
    Clone + Copy + Default + 'static
{}
```

Modified:

```rust
pub trait Label:
    Clone
    + Copy
    + Default
    + Send
    + Sync
    + 'static
{}
```

Burn datasets require:

```rust
Send + Sync
```

because datasets may be shared across worker threads.

Adding these bounds to the Label trait propagates thread-safety requirements throughout the IFC system and avoids repeatedly adding:

```rust
L: Label + Send + Sync
```

to dataset implementations.

---

## Labeled Values

Security labels are implemented using phantom types:

```rust
pub struct Labeled<T, L: Label> {
    value: T,
    _marker: PhantomData<L>,
}
```

Labels exist entirely at compile time.

No runtime label information is stored.

Example:

```rust
Labeled<MnistItem, Secret>
```

changes the Rust type but does not change the runtime representation.

This design follows the Filament approach and introduces effectively zero runtime overhead.

---

# Current Status

Successfully prototyped:

```rust
Dataset<I, L>
```

```rust
InMemDataset<I, L>
```

```rust
DatasetIterator<I, L>
```

Current propagation path:

```text
Dataset
    ↓
Dataset Transforms
    ↓
Iterator
```

Current work focuses on propagating labels through:

```text
SqliteDataset
    ↓
DataLoader
    ↓
Batcher
    ↓
Tensor
```

while maintaining the invariant:

```text
Data is labeled immediately when it enters Burn.
```