# IFC Integration into Burn Dataset

## Purpose

This document describes the Information Flow Control (IFC) modifications made to Burn's dataset layer.

The dataset layer is the first point where user-provided training data enters Burn. Because of this, it serves as the primary IFC entry point.

The guiding principle is:

```text
Data should be labeled when it enters Burn.

Once labeled, labels should propagate naturally
through the remainder of the pipeline.
```

---

# IFC Model

The current security lattice consists of:

```text
Public
Secret
```

with:

```text
Public ⊑ Secret
```

Meaning:

```text
Public data may flow into Secret contexts.

Secret data may not flow into Public contexts
unless explicitly declassified.
```

Any value derived from Secret data should also be considered Secret.

---

# Burn Data Flow

The relevant portion of Burn's training pipeline is:

```text
Dataset
    ↓
Dataset Transforms
    ↓
Iterator
    ↓
DataLoader
    ↓
Tensor
```

This document focuses on:

```text
Dataset
    ↓
Dataset Transforms
    ↓
Iterator
```

---

# Dataset Trait

## Original Burn Design

Burn datasets were originally parameterized only by the dataset item type:

```rust
Dataset<I>
```

Example:

```rust
Dataset<MnistItem>
```

The Dataset trait returned raw values:

```rust
fn get(&self, index: usize) -> Option<I>;
```

No IFC labels were present.

---

## IFC Dataset Design

Datasets are now parameterized by:

```rust
Dataset<I, L>
```

where:

```text
I = dataset item type
L = security label
```

Example:

```rust
Dataset<MnistItem, Secret>
```

This represents a dataset that produces Secret MnistItem values.

The Dataset trait was modified to return labeled values:

```rust
fn get(&self, index: usize)
    -> Option<Labeled<I, L>>;
```

Conceptually:

```text
Dataset<I, Secret>
        ↓
      get()
        ↓
Labeled<I, Secret>
```

The dataset establishes the label that will be used throughout the remainder of the pipeline.

---

# Dataset Labels vs Item Labels

The IFC design introduces two related concepts:

```rust
Dataset<I, L>
```

and:

```rust
Labeled<I, L>
```

These represent different things.

---

## Dataset Label

The dataset label identifies the security level of the dataset source.

Example:

```rust
Dataset<MnistItem, Secret>
```

This means:

```text
The dataset is a source of Secret MnistItem values.
```

---

## Item Label

The item label identifies the security level of a specific dataset record.

Example:

```rust
Labeled<MnistItem, Secret>
```

This means:

```text
This particular MnistItem is Secret.
```

---

## Why Both Exist

A dataset is a source of labeled values.

Therefore:

```rust
Dataset<I, Secret>
```

produces:

```rust
Labeled<I, Secret>
```

The dataset label identifies the security level of the source.

The item label allows that security level to continue propagating after the item leaves the dataset.

Conceptually:

```text
Dataset<I, Secret>
        ↓ get()
Labeled<I, Secret>
        ↓
Transforms
        ↓
Iterator
        ↓
DataLoader
```

The dataset establishes the label.

The item carries the label through the remainder of the pipeline.

---

# InMemDataset

InMemDataset stores dataset records directly inside Burn memory.

Because the records already exist inside Burn, they are labeled before being stored.

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
CSV Row
    ↓
Deserialize into I
    ↓
Labeled<I, Secret>
    ↓
Store in memory
```

This prevents unlabeled user data from existing inside Burn memory.

---

## Stored Representation

An IFC-enabled in-memory dataset contains:

```rust
InMemDataset<I, L>
```

and internally stores:

```rust
Vec<Labeled<I, L>>
```

Example:

```text
InMemDataset<MnistItem, Secret>

contains

Vec<Labeled<MnistItem, Secret>>
```

The dataset is a Secret source and each stored record is also Secret.

---

## Dataset Construction

The primary entry points are:

```rust
from_csv(...)
from_json_rows(...)
from_dataset(...)
```

The general flow is:

```text
External Data
    ↓
Deserialize into I
    ↓
Labeled<I, L>
    ↓
Store in Vec<Labeled<I, L>>
```

Once data enters Burn memory, it is immediately labeled and remains labeled while stored.

---

# SqliteDataset

SqliteDataset represents a labeled dataset whose records are stored externally in SQLite.

Unlike InMemDataset, it does not store all records directly inside Burn memory.

Instead it stores metadata such as:

```text
database path
connection pool
table information
split information
query configuration
```

The actual records remain inside SQLite until they are requested.

---

## IFC Design

The dataset itself remains labeled:

```rust
SqliteDataset<I, L>
```

and implements:

```rust
Dataset<I, L>
```

Example:

```text
SqliteDataset<MnistItem, Secret>
```

represents a source of Secret MnistItem values.

---

## Where Data Enters Burn

The dataset object itself does not contain the records.

Records enter Burn when a SQLite row is read and converted into a Rust value.

Therefore the IFC boundary is:

```rust
get(...)
```

rather than the SqliteDataset struct itself.

---

## SQLite Row Formats

SqliteDataset supports two storage formats.

### Serialized Rows

Rows are stored as serialized blobs.

Conceptually:

```text
SQLite Blob
    ↓
Deserialize
    ↓
I
```

Internally this uses:

```rust
rmp_serde::from_slice(...)
```

to reconstruct the Rust value.

---

### Column-Based Rows

Rows are stored across multiple SQLite columns.

Conceptually:

```text
SQLite Columns
    ↓
Deserialize
    ↓
I
```

Internally this uses:

```rust
serde_rusqlite::from_row_with_columns(...)
```

to reconstruct the Rust value.

---

## Labeling at Retrieval

Both row formats eventually produce:

```rust
Option<I>
```

after deserialization.

The retrieved value is then labeled using:

```rust
.map(|item| Labeled::<I, L>::new(item))
```

This converts:

```rust
Option<I>
```

into:

```rust
Option<Labeled<I, L>>
```

Conceptually:

```text
Some(I)
    ↓
Some(Labeled<I, L>)

None
    ↓
None
```

This is the point where external SQLite data becomes IFC-tracked Burn data.

---

## SQLite Data Flow

The full retrieval path is:

```text
SQLite Row
    ↓
Query
    ↓
Deserialize
    ↓
I
    ↓
Option<I>
    ↓
.map(...)
    ↓
Option<Labeled<I, L>>
    ↓
Return
```

This matches the Dataset trait:

```rust
fn get(...)
    -> Option<Labeled<I, L>>
```

---

## Comparison with InMemDataset

InMemDataset stores records directly inside Burn memory:

```rust
Vec<Labeled<I, L>>
```

because the records already exist inside Burn.

SqliteDataset stores records externally.

The records become labeled only when they cross into Burn memory.

Both implementations follow the same policy:

```text
Data becomes labeled
when it enters Burn.
```

The difference is simply where the data resides before it enters Burn.

---

# DatasetIterator

DatasetIterator provides sequential access to dataset items.

The iterator does not create labels.

It only preserves labels established by the dataset.

---

## Original Design

```rust
Iterator<Item = I>
```

---

## IFC Design

```rust
Iterator<Item = Labeled<I, L>>
```

The iterator retrieves labeled values from:

```rust
dataset.get(...)
```

and forwards them unchanged.

Conceptually:

```text
Dataset<I, Secret>
    ↓ get()
Labeled<I, Secret>
    ↓ iterator
Labeled<I, Secret>
```

No IFC decisions occur inside the iterator itself.

---

# Dataset Transforms

Dataset transforms wrap an existing dataset and produce another dataset.

Examples include:

```text
SelectionDataset
PartialDataset
ShuffledDataset
SamplerDataset
ComposedDataset
MapperDataset
WindowsDataset
```

The guiding rule is:

```text
Dataset transforms preserve labels.
```

Transforms may:

```text
change ordering
change indexing
change grouping
change representation
```

but they may not:

```text
remove labels
downgrade labels
declassify data
```

---

## Transform Categories

The transforms fall into two categories.

### Index-Based Transforms

These preserve both the item type and the label.

Examples:

```text
SelectionDataset
PartialDataset
ShuffledDataset
SamplerDataset
ComposedDataset
```

General form:

```text
Dataset<I, L>
    ↓
Transform
    ↓
Dataset<I, L>
```

---

### Value-Transforming Transforms

These change the item representation while preserving the label.

Examples:

```text
MapperDataset
WindowsDataset
```

General form:

```text
Dataset<I, L>
    ↓
Transform
    ↓
Dataset<O, L>
```

or:

```text
Dataset<I, L>
    ↓
Window
    ↓
Dataset<Vec<I>, L>
```

---

# SelectionDataset

SelectionDataset exposes specific indices from an existing dataset.

Example:

```text
Dataset<I, Secret>
    ↓
Select [4, 7, 10]
    ↓
Dataset<I, Secret>
```

Only the visible indices change.

The security label remains unchanged.

---

## IFC Changes

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

SelectionDataset simply forwards labeled values from the underlying dataset.

No new IFC logic is introduced.

---

# PartialDataset

PartialDataset exposes a contiguous subset of an existing dataset.

Example:

```text
Dataset<I, Secret>
    ↓
Take first 100 records
    ↓
Dataset<I, Secret>
```

---

## IFC Changes

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

Only the visible range changes.

The label remains unchanged.

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

---

## IFC Changes

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

Shuffling changes only item ordering.

The label remains unchanged.

Internally, ShuffledDataset is implemented as a thin wrapper around:

```rust
SelectionDataset<D, I, L>
```

which already preserves labels.

---

# SamplerDataset

SamplerDataset selects records according to a sampling strategy.

Example:

```text
Dataset<I, Secret>
    ↓
Sampling
    ↓
Dataset<I, Secret>
```

---

## IFC Changes

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

Sampling changes which item is selected.

It does not change the item's label.

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

---

## IFC Changes

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

---

## Design Decision

All component datasets must currently share the same label:

```rust
D: Dataset<I, L>
```

Supported:

```text
Secret + Secret
```

Result:

```text
Secret
```

Not currently supported:

```text
Public + Secret
```

Supporting mixed-label composition would require introducing label joins such as:

```text
Public ⊔ Secret = Secret
```

This is left as future work.

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

MapperDataset is the first transform that changes the dataset item type.

---

## IFC Changes

Original mapper trait:

```rust
Mapper<I, O>
```

Modified mapper trait:

```rust
Mapper<I, O, L>
```

The mapper now receives:

```rust
Labeled<I, L>
```

and returns:

```rust
Labeled<O, L>
```

---

## Design Decision

The mapper may change:

```text
I → O
```

but it may not change:

```text
L
```

Example:

```text
Labeled<String, Secret>
        ↓
truncate
        ↓
Labeled<String, Secret>
```

The transformed value remains Secret.

---

## Label-Preserving Transformations

The IFC helper:

```rust
__map_ref(...)
```

can be used to transform the inner value while preserving the label.

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

Input:

```text
1
2
3
4
5
```

Window Size:

```text
3
```

Output:

```text
[1,2,3]
[2,3,4]
[3,4,5]
```

---

## IFC Changes

Original:

```rust
Dataset<Vec<I>>
```

Modified:

```rust
Dataset<Vec<I>, L>
```

The resulting dataset item is:

```rust
Labeled<Vec<I>, L>
```

---

## Design Decision

Each input item is already labeled:

```rust
Labeled<I, L>
```

A window combines multiple labeled items into a new dataset item.

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

The resulting window inherits the dataset label.

The output is:

```rust
Labeled<Vec<I>, L>
```

rather than:

```rust
Vec<Labeled<I, L>>
```

because a window is treated as a new dataset item.

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

# Label Trait

The Label trait was extended with thread-safety requirements.

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

This was required because Burn datasets may be shared across worker threads.

Adding:

```rust
Send + Sync
```

to the Label trait propagates these requirements automatically throughout the IFC system.

---

# Labeled Values

Labels are implemented using phantom types:

```rust
pub struct Labeled<T, L: Label> {
    value: T,
    _marker: PhantomData<L>,
}
```

The label exists entirely at the type level.

Example:

```rust
Labeled<MnistItem, Secret>
```

changes the Rust type but does not change the runtime representation.

Because labels are represented using:

```rust
PhantomData<L>
```

the IFC implementation introduces effectively zero runtime overhead.

---

# Dataset Layer Summary

The dataset layer introduces IFC labels into Burn.

The overall flow is:

```text
Dataset<I, L>
        ↓
Labeled<I, L>
        ↓
Transforms
        ↓
Iterator
        ↓
DataLoader
```

Key properties:

- Data becomes labeled when it enters Burn.
- Labels propagate through dataset retrieval.
- Labels propagate through dataset transforms.
- Labels propagate through iteration.
- No dataset operation performs declassification.

The dataset layer establishes the IFC foundation used by the remainder of the Burn pipeline.