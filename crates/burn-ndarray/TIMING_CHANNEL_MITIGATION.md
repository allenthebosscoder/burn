# Mitigating Timing Channels in `sign_op` — Mission Writeup

Original mission: `sign_op` (`src/ops/base.rs`) computes the sign of every element in a
tensor using data-dependent `if`/`match` branches. If the tensor holds secret data, the
branches taken (and thus the execution time) vary with the secret values, creating a
timing side channel. This document answers the three-step mission using the `fg_ifc_library`
(`typing_rules` + `macros`) IFC framework already used elsewhere in `burn-core`/`burn-train`.

## Step 1 — Apply Security Labels

`sign_op_safe` labels the tensor **per element**, not as a single label on the whole
container:

```rust
pub(crate) fn sign_op_safe<L>(tensor: SharedArray<Labeled<E, L>>) -> SharedArray<Labeled<E, L>>
where
    E: Signed + PartialOrd,
    L: Label,
```

This was not the first design tried. The original attempt labeled the *whole* tensor —
`Labeled<SharedArray<E>, L>` — mirroring how `Batcher<I, O, L>` labels whole batches
elsewhere in the codebase. That design has a structural blind spot (see Step 2): `ndarray`'s
`.mapv()` always hands its closure the raw, unlabeled element type, so a container-level
label can never reach the place where the actual per-element branch happens. Moving the
label onto the element type (`SharedArray<Labeled<E, L>>`) fixes this: the value inside the
`.mapv()` closure is genuinely `Labeled<E, L>`, so a real per-element security label exists
exactly where the branch does. `.mapv()` itself can then be called completely bare — no
`fcall!`/`mcall!`/`.map()` needed around the call — because the array type itself is no
longer wrapped in `Labeled<...>`, only its elements are.

## Step 2 — Error Analysis and `pc_block!` Exploration

**Naively porting the original branching body produces *no* compiler error at all**, and
the reason changes depending on how the tensor is labeled:

- **Container-level label** (`Labeled<SharedArray<E>, L>`): `.mapv()`'s closure receives
  raw `E`, with zero label information — there is no `Labeled` value anywhere inside the
  closure for any checker to inspect. Wrapping the `.mapv()` call in `mcall!`/`fcall!`
  doesn't help either — those macros only check the *call boundary* (the receiver and
  return value), not the body of a closure passed in as an argument. Verified directly by
  compiling both variants against the real crate.
- **Element-level label** (`SharedArray<Labeled<E, L>>`): now `x` genuinely is
  `Labeled<E, L>` inside the closure, but `if x == zero { .. }` *still* compiles silently,
  because `Labeled<T, L>` implements `PartialEq<T>` (and `PartialOrd<T>`) returning a raw
  `bool`, not `Labeled<bool, L>` — the label is silently discarded by the comparison
  operator itself, one line before the `if` ever runs. Same gap applies to `x > zero`,
  `x < zero`.

**The real, catchable error** only appears once you use the library's actual
label-preserving comparison, `.labeled_eq()` (returns `Labeled<bool, L>`), and try to use
that in a bare `if`:

```
error[E0308]: mismatched types
   expected `bool`, found `Labeled<bool, Secret>`
```

This is because Rust's `if` requires a literal `bool`, and `Labeled<bool, L>` is a distinct
type. `is_positive()`/`is_negative()` (needed for the `>`/`<` case) don't exist on
`Labeled<E, L>` at all (`Labeled` never implements `Signed`); calling them via `mcall!`
(e.g. `mcall!(x.is_positive())`) correctly produces `Labeled<bool, L>` (verified — the
receiver's label is preserved through the call via `Labeled::__chain_ref`'s `Join<Public>`
law), and hits the identical `E0308` when used in a bare `if`.

**Does `pc_block!` need to be used, and does it help?** Wrapping the corrected
`Labeled<bool, L>` condition in `pc_block! { { if cond { .. } else { .. } } }` does compile
— `pc_block!` knows how to unwrap a `Labeled<bool, L>` condition and thread a PC (program
counter) label through the branch, checking that any assignment inside obeys the `FlowsTo`
lattice (no writing a secret-conditioned result into a less-secret variable). But two
things limit its relevance here:

1. It's not solving the target problem. Read directly from `pc_block!`'s own generated
   code: even for a fully-labeled condition, the *executed* code is still a literal
   `if __cond_val { .. } else { .. }` — a real branch, same timing exposure as before.
   `pc_block!` proves the branch doesn't leak information through *value flow*
   (declassification); it says nothing about, and does nothing to fix, *timing*.
2. It's not even doing useful work in this specific function. `pc_block!`'s FlowsTo check
   only matters when a branch could write into something *less secret* than its condition.
   In `sign_op_safe`, every possible destination (the mask values, the function's own
   return type) carries the same label `L` as the condition — `L` trivially flows to `L`,
   so the check is vacuously satisfied and catches nothing.

**Conclusion: `pc_block!` is unnecessary for this problem.** It answers a real but
different question (illegal declassification) than the one this mission is about (timing).
The final `sign_op_safe` uses no `pc_block!` at all — because it uses no branch at all
(Step 3).

## Step 3 — Constant-Time Rewrite

```rust
pub(crate) fn sign_op_safe<L>(tensor: SharedArray<Labeled<E, L>>) -> SharedArray<Labeled<E, L>>
where
    E: Signed + PartialOrd,
    L: Label,
{
    let one: Labeled<E, L> = Labeled::new(1.elem::<E>());

    tensor.mapv(|x| {
        let pos_mask = mcall!(x.is_positive()).map(|b| (b as i32).elem::<E>());
        let neg_mask = mcall!(x.is_negative()).map(|b| (b as i32).elem::<E>());
        pos_mask * one - neg_mask * one
    }).into_shared()
}
```

No `if`, no `match`, no `pc_block!`. Every element runs the exact same sequence of
operations regardless of its value:

- `mcall!(x.is_positive())` / `mcall!(x.is_negative())` call the one primitive that has no
  labeled equivalent in `typing_rules`, while correctly preserving the label as
  `Labeled<bool, L>` (rather than fully declassifying `x`, which would drop label
  protection for the rest of the computation).
- `.map(|b| (b as i32).elem::<E>())` is a branch-free value transform (`bool` → `0`/`1` →
  `E`), still under the same label — safe to wrap in `.map()`/`mcall!` specifically
  *because* the wrapped logic contains no branch to hide, unlike wrapping the original
  `if`/`match` body would.
- `pos_mask * one - neg_mask * one` combines the two 0/1 masks arithmetically instead of
  branching: multiplying by `1` or `0` acts as an unconditional on/off switch (`Add`/`Sub`/
  `Mul`/`Div` are implemented directly on `Labeled<T, L>` and correctly propagate the label
  through the join lattice, unlike the comparison operators from Step 2).

This differs from the mentor's other suggested option — a `select`-style
`if cond { a } else { b }` — deliberately. Rust gives no guarantee that such an expression
compiles to a branchless `select`/`cmov`; that's an optimizer judgment call (available to
both `if`/`else` and C-style ternaries identically, since both lower to the same LLVM IR),
not a language guarantee. A dedicated tool like the one in the referenced paper
(control-flow linearization at the LLVM IR level) could force that guarantee, but building
or integrating such a tool is disproportionate to fixing one function — and wouldn't even
have anything to key off of here, since `Labeled<T, L>` is `#[repr(transparent)]` and
erased by the time code reaches LLVM IR, giving an IR-level pass no way to know which
values were ever secret. The arithmetic-mask version above sidesteps all of that: it is
unconditionally branch-free by construction, verified against the real compiler, and
requires no new tooling.

### Known remaining gap

`is_positive()`/`is_negative()` are sign-bit tests for floats (not value tests), so
`+0.0`/`-0.0` currently produce `+1`/`-1` instead of the original `sign_op`'s `0`. Fixing
this (without reintroducing a branch) requires one more mask:

```rust
let zero_mask = mcall!(x.eq(&zero)).map(|b| (b as i32).elem::<E>()); // needs zero: Labeled<E,L>
let keep = one - zero_mask;
(pos_mask * keep) - (neg_mask * keep)
```

`pos_mask`/`neg_mask` computed and multiplied by `keep` individually (not the whole
expression multiplied by `keep` at the end) to avoid IEEE-754 producing `-0.0` instead of
canonical `+0.0` when `x` was originally `-0.0` — verified numerically. Not yet applied to
the checked-in version.
