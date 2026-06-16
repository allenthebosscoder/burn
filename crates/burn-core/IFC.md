# IFC Integration into Burn DataLoader

## Purpose

This document describes the Information Flow Control (IFC) modifications made to Burn's DataLoader layer.

The Dataset layer introduces labeled records into Burn:

```rust
Labeled<I, L>
```

The DataLoader layer collects those labeled records into batches while preserving their labels.

The primary rule is:

```text
Dataset items enter the dataloader as Labeled<I, L>.

Batches leave the dataloader as Labeled<O, L>.
```

---

# DataLoader Position in the Pipeline

The DataLoader layer sits between the Dataset layer and tensor/model execution.

```text
Dataset
    ↓
DataLoader
    ↓
Tensor / Model
```

Internally, BatchDataLoader consists of two major components:

```text
BatchDataLoader
    ├── BatchStrategy
    └── Batcher
```

The complete flow is:

```text
Dataset<I, L>
    ↓ get()
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
DataLoaderIterator
    ↓
Labeled<O, L>
```

where:

```text
I = dataset item type
O = batch output type
L = security label
```

---

# Original Burn Design

Originally, dataloaders operated on unlabeled values.

```text
Dataset<I>
    ↓
I
    ↓
BatchStrategy<I>
    ↓
Vec<I>
    ↓
Batcher<I, O>
    ↓
O
    ↓
DataLoader<O>
```

The iterator yielded:

```rust
O
```

No IFC labels were present.

---

# IFC DataLoader Design

The IFC version preserves the labels introduced by the Dataset layer.

```text
Dataset<I, L>
    ↓
Labeled<I, L>
    ↓
BatchStrategy<I, L>
    ↓
Vec<Labeled<I, L>>
    ↓
Batcher<I, O, L>
    ↓
Labeled<O, L>
    ↓
DataLoader<O, L>
```

The central invariant is:

```text
A batch derived from labeled dataset items
must also be labeled.
```

---

# DataLoader Trait

## Original Design

```rust
pub trait DataLoader<O>
```

The dataloader produced batches of type:

```rust
O
```

---

## IFC Design

```rust
pub trait DataLoader<O, L: Label>
```

The dataloader is now parameterized by:

```text
O = batch output type
L = security label
```

The iterator output changed from:

```rust
Iterator<Item = O>
```

to:

```rust
Iterator<Item = Labeled<O, L>>
```

Conceptually:

```text
DataLoader<MnistBatch, Secret>
        ↓ iter()
Labeled<MnistBatch, Secret>
```

---

# BatchDataLoader

BatchDataLoader is the primary implementation of the DataLoader trait.

## Original Design

```rust
BatchDataLoader<I, O>
```

It stored:

```rust
Arc<dyn Dataset<I>>
Arc<dyn Batcher<I, O>>
```

---

## IFC Design

```rust
BatchDataLoader<I, O, L>
```

It now stores:

```rust
Arc<dyn Dataset<I, L>>
Arc<dyn Batcher<I, O, L>>
```

This means the dataloader receives labeled items from the dataset and produces labeled batches through the batcher.

---

## BatchDataLoader Flow

Inside the iterator:

```rust
dataset.get(...)
```

returns:

```rust
Labeled<I, L>
```

The item is passed into the batch strategy:

```rust
strategy.add(item)
```

When enough items are collected, the strategy returns:

```rust
Vec<Labeled<I, L>>
```

That collection is passed into the batcher:

```rust
batcher.batch(items, device)
```

The batcher returns:

```rust
Labeled<O, L>
```

The complete flow is:

```text
Labeled<I, L>
    ↓
BatchStrategy
    ↓
Vec<Labeled<I, L>>
    ↓
Batcher
    ↓
Labeled<O, L>
```

---

# BatchDataLoaderIterator

The iterator now yields labeled batches.

## Original Design

```rust
Iterator<Item = O>
```

---

## IFC Design

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

The iterator performs no IFC operations itself.

It simply returns the labeled batch produced by the batcher.

---

# BatchStrategy

BatchStrategy decides when enough items have been collected to form a batch.

It does not construct the final batch object.

Its only responsibility is accumulating dataset items.

---

## Original Design

```rust
pub trait BatchStrategy<I>
```

with:

```rust
fn add(&mut self, item: I);

fn batch(&mut self, force: bool)
    -> Option<Vec<I>>;
```

---

## IFC Design

```rust
pub trait BatchStrategy<I, L: Label>
```

with:

```rust
fn add(&mut self, item: Labeled<I, L>);

fn batch(&mut self, force: bool)
    -> Option<Vec<Labeled<I, L>>>;
```

The strategy receives labeled items and returns labeled items.

Labels remain attached to individual records during accumulation.

---

# FixBatchStrategy

FixBatchStrategy is the fixed-size implementation of BatchStrategy.

## Original Design

```rust
FixBatchStrategy<I>
```

with storage:

```rust
Vec<I>
```

---

## IFC Design

```rust
FixBatchStrategy<I, L>
```

with storage:

```rust
Vec<Labeled<I, L>>
```

This mirrors the Dataset layer.

```text
Dataset
    stores
Vec<Labeled<I, L>>

BatchStrategy
    stores
Vec<Labeled<I, L>>
```

Labels remain attached to individual items while the batch is being assembled.

---

# Batch Representation Decision

A major design decision was determining how batches should be represented while they are being assembled.

Two possible representations were considered.

---

## Option 1

```rust
Vec<Labeled<I, L>>
```

Each item retains its individual label.

Example:

```text
[
    Labeled<I, Secret>,
    Labeled<I, Secret>,
    Labeled<I, Secret>
]
```

---

## Option 2

```rust
Labeled<Vec<I>, L>
```

The collection itself carries the label.

Example:

```text
Labeled<
    [I, I, I],
    Secret
>
```

---

## Final Decision

The final implementation uses:

```rust
Vec<Labeled<I, L>>
```

inside `BatchStrategy`.

Reason:

```text
BatchStrategy is an accumulator.

It receives one labeled item at a time.

Therefore it stores the labeled items it receives.
```

This mirrors the Dataset layer.

```text
Dataset
    stores
Vec<Labeled<I, L>>

BatchStrategy
    stores
Vec<Labeled<I, L>>
```

The batch is not complete until the batcher constructs the final output type.

Therefore the conversion into:

```rust
Labeled<O, L>
```

occurs inside the batcher rather than inside the strategy.

---

# Batcher

Batcher converts collected dataset items into the batch object consumed by training code.

## Original Design

```rust
pub trait Batcher<I, O>
```

with:

```rust
fn batch(
    &self,
    items: Vec<I>,
    device: &Device,
) -> O;
```

---

## IFC Design

```rust
pub trait Batcher<I, O, L: Label>
```

with:

```rust
fn batch(
    &self,
    items: Vec<Labeled<I, L>>,
    device: &Device,
) -> Labeled<O, L>;
```

---

## Batcher Input

The batcher receives:

```rust
Vec<Labeled<I, L>>
```

These are the labeled items collected by the batch strategy.

---

## Batcher Output

The batcher returns:

```rust
Labeled<O, L>
```

where:

```text
O = final batch type
```

Example:

```text
Labeled<I, Secret>
Labeled<I, Secret>
Labeled<I, Secret>
        ↓
Batcher
        ↓
Labeled<O, Secret>
```

For a training dataset:

```rust
O = MnistBatch
```

so the output becomes:

```rust
Labeled<MnistBatch, Secret>
```

---

## Label Aggregation

The batcher is the point where individual labeled records become a labeled batch.

Conceptually:

```text
Vec<Labeled<I, Secret>>
        ↓
Batcher
        ↓
Labeled<O, Secret>
```

This is the transition from:

```text
individual records
```

to:

```text
training batch
```

while preserving the security label.

---

# MultiThreadDataLoader

MultiThreadDataLoader moves batches between worker threads and the main iterator using channels.

Labels must be preserved while batches cross thread boundaries.

---

## Original Design

Messages contained:

```rust
O
```

through:

```rust
Message<O>
```

---

## IFC Design

Messages now contain:

```rust
Labeled<O, L>
```

through:

```rust
Message<O, L>
```

Conceptually:

```text
Worker Thread
        ↓
Labeled<O, L>
        ↓
Message<O, L>
        ↓
Channel
        ↓
Main Thread
```

The label remains attached to the batch throughout the transfer.

---

## Debug Derive Change

Originally:

```rust
#[derive(Debug)]
```

was applied to:

```rust
Message<O>
```

After introducing:

```rust
Labeled<O, L>
```

the compiler required:

```rust
Labeled<O, L>: Debug
```

Since the IFC wrapper did not implement Debug, the derive was removed.

This change affected debugging output only.

It did not change IFC behavior.

---

# Builder

The DataLoader builder was updated to propagate the label type.

Original:

```rust
DataLoader<O>
```

Modified:

```rust
DataLoader<O, L>
```

The builder introduces no new IFC behavior.

Its responsibility is simply ensuring that:

```rust
L
```

is propagated through dataloader construction.

---

# Split

The split utilities were updated to preserve labels.

Splitting a dataloader changes how batches are distributed but does not change their sensitivity.

Conceptually:

```text
DataLoader<O, Secret>
        ↓
      split
        ↓
DataLoader<O, Secret>
```

The label remains unchanged.

---

# DataLoader Layer Summary

The DataLoader layer extends IFC propagation from individual dataset records to training batches.

The complete flow is:

```text
Dataset<I, L>
        ↓
Labeled<I, L>
        ↓
BatchStrategy<I, L>
        ↓
Vec<Labeled<I, L>>
        ↓
Batcher<I, O, L>
        ↓
Labeled<O, L>
        ↓
DataLoader<O, L>
        ↓
Labeled<O, L>
```

Key properties:

- Dataset items enter the dataloader as `Labeled<I, L>`.
- BatchStrategy accumulates `Vec<Labeled<I, L>>`.
- Batcher constructs `Labeled<O, L>`.
- DataLoader iterators yield `Labeled<O, L>`.
- MultiThreadDataLoader preserves labels across worker threads.
- No dataloader operation performs declassification.

The DataLoader layer establishes IFC propagation for batch construction and serves as the bridge between datasets and tensor creation.