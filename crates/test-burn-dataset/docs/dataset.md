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

Several Burn dataset abstractions operate on top of an existing dataset:

```text
DatasetIterator
SelectionDataset
PartialDataset
ShuffleDataset
SamplerDataset
ComposedDataset
MapperDataset
WindowsDataset
```

These transforms consume an existing dataset and produce a new dataset.

Example:

```text
Dataset<I, Secret>
    ↓
Transform
    ↓
Dataset<..., Secret>
```

---

## IFC Design Principle

Dataset transforms must preserve labels.

Transforms are not allowed to:

- remove labels
- downgrade labels
- declassify data

Transforms may:

- change ordering
- change indexing
- change grouping
- change item representation

while preserving security labels.

---

# SelectionDataset

SelectionDataset allows selecting arbitrary indices from an existing dataset.

Example:

```text
Dataset<I, Secret>
    ↓
Select [4, 7, 10]
    ↓
Dataset<I, Secret>
```

### IFC Changes

Original:

```rust
SelectionDataset<D, I>
```

Modified:

```rust
SelectionDataset<D, I, L>
```

with:

```rust
D: Dataset<I, L>
```

and:

```rust
impl<D, I, L> Dataset<I, L>
```

The selected dataset preserves the original label.

Selection only changes which records are visible.

It does not change information sensitivity.

---

# PartialDataset

PartialDataset exposes only a subset of an existing dataset.

Example:

```text
Dataset<I, Secret>
    ↓
Take first 100 samples
    ↓
Dataset<I, Secret>
```

### IFC Changes

Original:

```rust
PartialDataset<D, I>
```

Modified:

```rust
PartialDataset<D, I, L>
```

with:

```rust
D: Dataset<I, L>
```

and:

```rust
impl<D, I, L> Dataset<I, L>
```

The resulting dataset preserves the label of the original dataset.

Only the visible range changes.

---

# ShuffledDataset

ShuffledDataset randomly reorders dataset items.

Example:

```text
Dataset<I, Secret>
    ↓
Shuffle
    ↓
Dataset<I, Secret>
```

### IFC Changes

Original:

```rust
ShuffledDataset<D, I>
```

Modified:

```rust
ShuffledDataset<D, I, L>
```

with:

```rust
D: Dataset<I, L>
```

and:

```rust
impl<D, I, L> Dataset<I, L>
```

Shuffling changes only item order.

No labels are modified.

The implementation simply forwards calls to an underlying:

```rust
SelectionDataset<D, I, L>
```

which already preserves labels.

---

# SamplerDataset

SamplerDataset randomly samples dataset elements according to a probability distribution.

Example:

```text
Dataset<I, Secret>
    ↓
Sampling
    ↓
Dataset<I, Secret>
```

### IFC Changes

Original:

```rust
SamplerDataset<D, I>
```

Modified:

```rust
SamplerDataset<D, I, L>
```

with:

```rust
D: Dataset<I, L>
```

and:

```rust
impl<D, I, L> Dataset<I, L>
```

Sampling changes only which index is selected.

Labels propagate unchanged from the underlying dataset.

---

# ComposedDataset

ComposedDataset concatenates multiple datasets together.

Example:

```text
Dataset<I, Secret>
Dataset<I, Secret>
          ↓
      Compose
          ↓
Dataset<I, Secret>
```

### IFC Changes

Original:

```rust
ComposedDataset<D>
```

implemented:

```rust
Dataset<I>
```

Modified:

```rust
impl<D, I, L> Dataset<I, L>
```

with:

```rust
D: Dataset<I, L>
```

### Design Decision

All component datasets must share the same label:

```rust
D: Dataset<I, L>
```

Mixed-label composition is currently unsupported.

Example:

```text
Secret + Secret -> Secret
```

Supported.

```text
Public + Secret
```

Not currently supported.

The composed dataset simply forwards labeled values from the underlying datasets.

---

# MapperDataset

MapperDataset transforms dataset items from one type into another.

Example:

```text
Dataset<I, Secret>
        ↓
      Mapper
        ↓
Dataset<O, Secret>
```

This is the first transform that changes the dataset item type.

### IFC Changes

Original:

```rust
Mapper<I, O>
```

Modified:

```rust
Mapper<I, O, L>
```

with:

```rust
fn map(
    &self,
    item: &Labeled<I, L>
) -> Labeled<O, L>;
```

### Design Decision

The mapper receives labeled input:

```rust
Labeled<I, L>
```

and produces labeled output:

```rust
Labeled<O, L>
```

Labels are preserved through transformations.

Example:

```text
Labeled<String, Secret>
            ↓
       truncate
            ↓
Labeled<String, Secret>
```

The mapper may change the inner value type:

```text
I
↓
O
```

but it must not change the security label.

### Label-Preserving Transformations

The IFC library helper:

```rust
__map_ref(...)
```

is used to transform inner values while preserving labels.

Conceptually:

```text
Labeled<I, Secret>
        ↓
   __map_ref
        ↓
Labeled<O, Secret>
```

No declassification occurs.

---

# WindowsDataset

WindowsDataset groups adjacent dataset elements into overlapping windows.

Example:

Original dataset:

```text
1
2
3
4
5
```

Window size:

```text
3
```

Produces:

```text
[1,2,3]
[2,3,4]
[3,4,5]
```

### IFC Changes

Original:

```rust
Dataset<I>
```

Window output:

```rust
Dataset<Vec<I>>
```

Modified:

```rust
Dataset<I, L>
```

Window output:

```rust
Dataset<Vec<I>, L>
```

### Design Decision

Each individual dataset element is already labeled:

```rust
Labeled<I, L>
```

A window combines multiple labeled elements into a new dataset item.

Example:

```text
Labeled<I, Secret>
Labeled<I, Secret>
Labeled<I, Secret>
            ↓
       Window
            ↓
Labeled<Vec<I>, Secret>
```

The window itself becomes the labeled value.

The output is:

```rust
Labeled<Vec<I>, L>
```

rather than:

```rust
Vec<Labeled<I, L>>
```

because a window is treated as a new dataset item.

### Window Iterator

Original:

```rust
Iterator<Item = Vec<I>>
```

Modified:

```rust
Iterator<Item = Labeled<Vec<I>, L>>
```

The iterator preserves labels established during window creation.

---

# Dataset Transform Summary

All dataset transforms now preserve IFC labels.

Transforms fall into two categories:

### Index-Based Transforms

These preserve both item type and label:

```text
SelectionDataset
PartialDataset
ShuffledDataset
SamplerDataset
ComposedDataset
```

Conceptually:

```text
Dataset<I, Secret>
        ↓
     Transform
        ↓
Dataset<I, Secret>
```

### Value-Transforming Transforms

These change the item representation while preserving labels:

```text
MapperDataset
WindowsDataset
```

Conceptually:

```text
Dataset<I, Secret>
        ↓
     Transform
        ↓
Dataset<O, Secret>
```

or:

```text
Dataset<I, Secret>
        ↓
     Window
        ↓
Dataset<Vec<I>, Secret>
```

In all cases, security labels are preserved and no transform performs declassification.

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

# SqliteDataset

SqliteDataset represents a labeled dataset whose records are stored externally in SQLite.

Unlike InMemDataset, SqliteDataset does not store all records directly in Burn memory. The SQLite database stores raw external rows.

Instead it stores:

```text
Database metadata
Connection pool
Table information
SQL queries
```

The actual records remain inside SQLite.

However, the dataset abstraction itself is still labeled:
```rust
SqliteDataset<I, L>
```

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