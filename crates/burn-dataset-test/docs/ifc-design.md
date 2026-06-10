# IFC Integration into Burn

## Goal

Integrate Filament-style IFC into Burn.

Security lattice:

Public
Secret

## Entry Point

User data enters Burn through Dataset.

Dataset -> Iterator -> DataLoader -> Batcher -> Tensor

## Design Decisions

### Dataset Labels

Dataset is parameterized by:

Dataset<I, L>

where:

- I = item type
- L = security label

### Immediate Labeling

Data is labeled when entering Burn.

Stored representation:

Vec<Labeled<I, L>>

not:

Vec<I>

### Dataset Retrieval

get() returns:

Option<Labeled<I, L>>

### Iterator

Iterator preserves labels:

Iterator<Item = Labeled<I, L>>