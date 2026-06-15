# DataLoader Layer

The DataLoader layer sits between the Dataset layer and the Tensor layer.

```text
Dataset
    ↓
DataLoader
    ↓
Tensor
```

Internally, BatchDataLoader is composed of:

```text
BatchDataLoader
    ├── BatchStrategy
    └── Batcher
```

Data flows through these components as:

```text
Dataset
    ↓
Labeled<I, L>
    ↓
BatchStrategy
    ↓
Vec<Labeled<I, L>>
    ↓
Batcher
    ↓
Labeled<O, L>
    ↓
DataLoader Iterator
    ↓
Labeled<O, L>
```

---

# Original DataLoader Design

Original Burn design:

```text
Dataset<I>
    ↓
BatchStrategy<I>
    ↓
Batcher<I, O>
    ↓
DataLoader<O>
```

Dataset items were represented as:

```rust
I
```

Batches were represented as:

```rust
O
```

No IFC labels were present.

---

# IFC DataLoader Design

Dataset items are now:

```rust
Labeled<I, L>
```

where:

```text
I = item type
L = security label
```

All DataLoader components were modified to preserve IFC labels.

---

# DataLoader Trait

Original:

```rust
pub trait DataLoader<O>
```

Modified:

```rust
pub trait DataLoader<O, L: Label>
```

---

## Iterator Output

Original:

```rust
Iterator<Item = O>
```

Modified:

```rust
Iterator<Item = Labeled<O, L>>
```

This ensures that batches produced by the dataloader remain labeled.

Example:

```text
Dataset<I, Secret>
        ↓
DataLoader
        ↓
Labeled<Batch, Secret>
```

---

# BatchDataLoader

BatchDataLoader is the primary implementation of the DataLoader trait.

Original:

```rust
BatchDataLoader<I, O>
```

Modified:

```rust
BatchDataLoader<I, O, L>
```

Dataset storage changed from:

```rust
Arc<dyn Dataset<I>>
```

to:

```rust
Arc<dyn Dataset<I, L>>
```

Batcher storage changed from:

```rust
Arc<dyn Batcher<I, O>>
```

to:

```rust
Arc<dyn Batcher<I, O, L>>
```

The dataloader now operates entirely on labeled data.

---

# BatchDataloaderIterator

Original:

```rust
Iterator<Item = O>
```

Modified:

```rust
Iterator<Item = Labeled<O, L>>
```

Example:

```text
Labeled<MnistBatch, Secret>
```

instead of:

```text
MnistBatch
```

---

# BatchStrategy

BatchStrategy accumulates dataset items until a batch is ready.

Original:

```rust
pub trait BatchStrategy<I>
```

Modified:

```rust
pub trait BatchStrategy<I, L: Label>
```

---

## Item Storage

Original:

```rust
Vec<I>
```

Modified:

```rust
Vec<Labeled<I, L>>
```

The strategy now stores labeled dataset items directly.

Example:

```text
Labeled<I, Secret>
Labeled<I, Secret>
Labeled<I, Secret>
```

The labels are preserved while the batch is being assembled.

---

## Add Operation

Original:

```rust
fn add(&mut self, item: I)
```

Modified:

```rust
fn add(&mut self, item: Labeled<I, L>)
```

Incoming dataset items remain labeled during accumulation.

---

## Batch Operation

Original:

```rust
fn batch(&mut self, force: bool)
    -> Option<Vec<I>>
```

Modified:

```rust
fn batch(&mut self, force: bool)
    -> Option<Vec<Labeled<I, L>>>
```

The strategy returns a collection of labeled items.

Aggregation into a labeled batch is deferred to the batcher.

---

# FixBatchStrategy

Original:

```rust
FixBatchStrategy<I>
```

Modified:

```rust
FixBatchStrategy<I, L>
```

Storage changed from:

```rust
Vec<I>
```

to:

```rust
Vec<Labeled<I, L>>
```

This mirrors the Dataset design:

```text
Dataset
    stores
Vec<Labeled<I, L>>

BatchStrategy
    stores
Vec<Labeled<I, L>>
```

Labels remain attached to individual items throughout accumulation.

---

# Batcher

Batcher converts a collection of items into a batch object.

Original:

```rust
pub trait Batcher<I, O>
```

Modified:

```rust
pub trait Batcher<I, O, L: Label>
```

---

## Input

Original:

```rust
Vec<I>
```

Modified:

```rust
Vec<Labeled<I, L>>
```

The batcher receives labeled dataset items.

---

## Output

Original:

```rust
O
```

Modified:

```rust
Labeled<O, L>
```

The batch inherits the label of the dataset items used to construct it.

Example:

```text
Labeled<I, Secret>
Labeled<I, Secret>
Labeled<I, Secret>
        ↓
Batcher
        ↓
Labeled<Batch, Secret>
```

---

# MultiThreadDataLoader

MultiThreadDataLoader required updates to preserve labels when communicating between worker threads.

Original:

```rust
Message<O>
```

Modified:

```rust
Message<O, L>
```

Batch messages now contain:

```rust
Labeled<O, L>
```

instead of:

```rust
O
```

This preserves labels while batches move between worker threads.

Example:

```text
Worker Thread
        ↓
Labeled<Batch, Secret>
        ↓
Channel
        ↓
Main Thread
```

Labels are preserved throughout the transfer.

---

# Builder and Split Support

Builder and split utilities were updated to propagate:

```rust
L
```

through all DataLoader-related abstractions.

These changes were primarily generic parameter propagation and did not introduce new IFC behavior.

---

# Design Decision

A key design decision was how batches should be represented.

Two possible representations were considered:

```rust
Vec<Labeled<I, L>>
```

or:

```rust
Labeled<Vec<I>, L>
```

The final design preserves labels on individual items during batch accumulation:

```rust
Vec<Labeled<I, L>>
```

and constructs:

```rust
Labeled<O, L>
```

only when the batch object is created.

This mirrors the Dataset implementation, where individual dataset records remain labeled while stored.

---

# DataLoader Layer Summary

The DataLoader layer now preserves IFC labels through:

```text
Dataset
        ↓
Labeled<I, L>

BatchStrategy
        ↓
Vec<Labeled<I, L>>

Batcher
        ↓
Labeled<O, L>

DataLoader
        ↓
Labeled<O, L>
```

All dataset items remain labeled throughout loading, batching, and multi-threaded transfer.

No DataLoader operation performs declassification.