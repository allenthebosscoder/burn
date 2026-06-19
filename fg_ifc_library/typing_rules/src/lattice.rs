use std::cmp::PartialEq;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, BitOr, Div, Index, IndexMut, Mul, Not, Rem, Sub};
use std::path::{Path, PathBuf};

// LABEL TRAITS
// The core security labels used to tag data and execution contexts.
// The supertrait Join<Public, Out = Self> encodes the law that joining any label
// with Public (the bottom element) returns the label unchanged. This lets the
// type-checker resolve <L as Join<Public>>::Out = L for any generic L: Label,
// which is required by __chain when a labeled arg is passed to an unlabeled receiver.
pub trait Label: Clone + Copy + Default + Send + Sync + Join<Public, Out = Self> + 'static {}

#[derive(Clone, Copy, Default)]
pub struct Public;
#[derive(Clone, Copy, Default)]
pub struct A;
#[derive(Clone, Copy, Default)]
pub struct B;
#[derive(Clone, Copy, Default)]
pub struct AB; // Represents the Join of A and B (Top Secret / Shared)

impl Label for Public {}
impl Label for A {}
impl Label for B {}
impl Label for AB {}

// JOIN OPERATION
// This trait calculates the Least Upper Bound (LUB) of two labels.
// It answers: "If I combine data from L1 and L2, what is the new security level?"
pub trait Join<Other: Label> {
    type Out: Label;
}

impl<L: Label> Join<L> for Public {
    type Out = L;
}

// A Joins (A + Public = A, A + B = AB)
impl Join<Public> for A {
    type Out = A;
}
impl Join<A> for A {
    type Out = A;
}
impl Join<B> for A {
    type Out = AB;
}
impl Join<AB> for A {
    type Out = AB;
}

// B Joins (B + Public = B, B + A = AB)
impl Join<Public> for B {
    type Out = B;
}
impl Join<A> for B {
    type Out = AB;
}
impl Join<B> for B {
    type Out = B;
}
impl Join<AB> for B {
    type Out = AB;
}

// AB Joins (Top level dominates everything)
impl Join<Public> for AB {
    type Out = AB;
}
impl Join<A> for AB {
    type Out = AB;
}
impl Join<B> for AB {
    type Out = AB;
}
impl Join<AB> for AB {
    type Out = AB;
}

// --- [FLOWSTO OPERATION] ---
// Logic: "Can information flow from Self to Target?" (Self <= Target)
// This is used for Write Control (No Write Down).
pub trait FlowsTo<Target: Label>: Label {}

impl<L: Label> FlowsTo<L> for L {}

impl FlowsTo<A> for Public {} // Public flows to A
impl FlowsTo<B> for Public {} // Public flows to B
impl FlowsTo<AB> for Public {} // Public flows to AB
impl FlowsTo<AB> for A {} // A flows to AB
impl FlowsTo<AB> for B {} // B flows to AB
                          // impl FlowsTo<B> for B {} // B flows to
                          // RELABEL HELPER
                          // Used by relabel! macro to enforce FlowsTo when upgrading labels.
                          // Raw values are first wrapped as Labeled<T, Public> by the macro,
                          // then this function checks OldLabel: FlowsTo<NewLabel>.
#[doc(hidden)]
#[inline(always)]
pub fn __relabel_checked<T, OldL: Label + FlowsTo<NewL>, NewL: Label>(v: Labeled<T, OldL>) -> Labeled<T, NewL> {
    Labeled { value: v.value, _marker: PhantomData }
}

// SECURE CONTAINER
// Wraps data 'T' with a security label 'L'.
// The data inside is inaccessible unless properly unwrapped via 'read'.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Labeled<T, L: Label> {
    pub(crate) value: T,
    pub(crate) _marker: PhantomData<L>,
}

impl<T, L: Label> Labeled<T, L> {
    /// Internal accessor for macro-generated code. Not part of the public API.
    #[doc(hidden)]
    #[inline(always)]
    pub fn __private_value(&self) -> &T {
        &self.value
    }

    /// Internal mutable accessor for macro-generated code. Not part of the public API.
    #[doc(hidden)]
    #[inline(always)]
    pub fn __private_value_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Internal consuming accessor for macro-generated code. Not part of the public API.
    #[doc(hidden)]
    #[inline(always)]
    pub fn __private_into_value(self) -> T {
        self.value
    }
}

// PC label
// Represents the security level of the current program execution (Program Counter).
// This is passed around to track "implicit flows" (like if/else branches on secrets).
#[derive(Clone)]
pub struct PcContext<P: Label> {
    pub _marker: PhantomData<P>,
}

impl<P: Label> PcContext<P> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

// Implement PartialEq for Labeled types
impl<T: PartialEq, L: Label> PartialEq for Labeled<T, L> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

// Implement Eq for Labeled types (needed for HashMap keys, etc.)
impl<T: Eq, L: Label> Eq for Labeled<T, L> {}

// Implement Hash for Labeled types (needed for HashMap keys, etc.)
impl<T: Hash, L: Label> Hash for Labeled<T, L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

// // Implement Display for Labeled types (allows use in format! macros)
// impl<T: fmt::Display, L: Label> fmt::Display for Labeled<T, L> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         self.value.fmt(f)
//     }
// }

impl<T, L: Label> Labeled<T, L> {
    // Standard constructor
    pub fn new(value: T) -> Self {
        Self { value, _marker: PhantomData }
    }

    // Helper for references (Essential for fs::write, etc.)
    pub fn as_ref(&self) -> Labeled<&T, L> {
        Labeled {
            value: &self.value,
            _marker: PhantomData,
        }
    }

    /// For macro use only: consume and transform the inner value while preserving label L.
    #[doc(hidden)]
    pub fn __map<R, F: FnOnce(T) -> R>(self, f: F) -> Labeled<R, L> {
        Labeled::new(f(self.value))
    }

    /// For macro use only: transform the inner value while preserving the exact label L.
    #[doc(hidden)]
    pub fn __map_ref<R, F: FnOnce(&T) -> R>(&self, f: F) -> Labeled<R, L> {
        Labeled::new(f(&self.value))
    }

    /// Declassify by reference — strips the label and returns &T.
    pub fn declassify_ref(&self) -> &T {
        &self.value
    }

    /// Declassify by mutable reference — strips the label and returns &mut T.
    pub fn declassify_ref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

// Helper for Labeled<Option<T>, L> — enables pattern matching without .value
impl<T, L: Label> Labeled<Option<T>, L> {
    pub fn as_option_ref(&self) -> Option<Labeled<&T, L>> {
        self.value.as_ref().map(|v| Labeled::new(v))
    }
}

// Implement Default for Labeled<T, L> where T implements Default
impl<T: Default, L: Label> Default for Labeled<T, L> {
    fn default() -> Self {
        Self {
            value: T::default(),
            _marker: PhantomData,
        }
    }
}

// Declassification Function
// Turn a secret to public by stripping the label entirely and returning the raw value.
pub fn declassify<T, L: Label>(secret: Labeled<T, L>) -> T {
    secret.value
}

/// Convert `Labeled<Option<T>, L>` into `Option<Labeled<T, L>>`.
/// Preserves the label through an Option without declassifying.
pub fn labeled_transpose<T, L: Label>(lo: Labeled<Option<T>, L>) -> Option<Labeled<T, L>> {
    lo.value.map(|v| Labeled::new(v))
}

// --- [ 1. ARITHMETIC OPERATORS ] ---
// Logic: Labeled<T, L1> op Labeled<T, L2> -> Labeled<T, Join<L1, L2>>

// ADDITION (+)
impl<T, L1, L2> Add<Labeled<T, L2>> for Labeled<T, L1>
where
    T: Add<Output = T>,
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<T, <L1 as Join<L2>>::Out>;

    fn add(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled {
            value: self.value + rhs.value,
            _marker: PhantomData,
        }
    }
}

impl<T, L> Add<T> for Labeled<T, L>
where
    T: Add<Output = T>, // The raw types must be addable
    L: Label,
{
    type Output = Labeled<T, L>; // The label stays the same

    fn add(self, rhs: T) -> Self::Output {
        Labeled {
            value: self.value + rhs, // Add raw value directly
            _marker: PhantomData,
        }
    }
}

// SUBTRACTION (-)
impl<T, L1, L2> Sub<Labeled<T, L2>> for Labeled<T, L1>
where
    T: Sub<Output = T>,
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<T, <L1 as Join<L2>>::Out>;

    fn sub(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled {
            value: self.value - rhs.value,
            _marker: PhantomData,
        }
    }
}

impl<T, L> Sub<T> for Labeled<T, L>
where
    T: Sub<Output = T>,
    L: Label,
{
    type Output = Labeled<T, L>;

    fn sub(self, rhs: T) -> Self::Output {
        Labeled {
            value: self.value - rhs,
            _marker: PhantomData,
        }
    }
}

// MULTIPLICATION (*)
impl<T, L1, L2> Mul<Labeled<T, L2>> for Labeled<T, L1>
where
    T: Mul<Output = T>,
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<T, <L1 as Join<L2>>::Out>;

    fn mul(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled {
            value: self.value * rhs.value,
            _marker: PhantomData,
        }
    }
}

impl<T, L> Mul<T> for Labeled<T, L>
where
    T: Mul<Output = T>,
    L: Label,
{
    type Output = Labeled<T, L>;

    fn mul(self, rhs: T) -> Self::Output {
        Labeled {
            value: self.value * rhs,
            _marker: PhantomData,
        }
    }
}

// DIVISION (/)
impl<T, L1, L2> Div<Labeled<T, L2>> for Labeled<T, L1>
where
    T: Div<Output = T>,
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<T, <L1 as Join<L2>>::Out>;

    fn div(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled {
            value: self.value / rhs.value,
            _marker: PhantomData,
        }
    }
}

impl<T, L> Div<T> for Labeled<T, L>
where
    T: Div<Output = T>,
    L: Label,
{
    type Output = Labeled<T, L>;

    fn div(self, rhs: T) -> Self::Output {
        Labeled {
            value: self.value / rhs,
            _marker: PhantomData,
        }
    }
}

// REMAINDER (%)
impl<T, L1, L2> Rem<Labeled<T, L2>> for Labeled<T, L1>
where
    T: Rem<Output = T>,
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<T, <L1 as Join<L2>>::Out>;

    fn rem(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled {
            value: self.value % rhs.value,
            _marker: PhantomData,
        }
    }
}

impl<T, L> Rem<T> for Labeled<T, L>
where
    T: Rem<Output = T>,
    L: Label,
{
    type Output = Labeled<T, L>;

    fn rem(self, rhs: T) -> Self::Output {
        Labeled {
            value: self.value % rhs,
            _marker: PhantomData,
        }
    }
}

impl<T, L, const N: usize> Index<usize> for Labeled<[T; N], L>
where
    L: Label,
{
    type Output = Labeled<T, L>;

    fn index(&self, index: usize) -> &Self::Output {
        // 1. Get raw reference to the item inside the array
        let inner_ref = &self.value[index];

        // 2. Cast it to Labeled (Safe due to repr(transparent))
        unsafe { &*(inner_ref as *const T as *const Labeled<T, L>) }
    }
}

// Support indexing with Labeled<usize, L2> — preserves labels through indexing
impl<T, L, L2, const N: usize> Index<Labeled<usize, L2>> for Labeled<[T; N], L>
where
    L: Label,
    L2: Label,
{
    type Output = Labeled<T, L>;

    fn index(&self, index: Labeled<usize, L2>) -> &Self::Output {
        let inner_ref = &self.value[index.value];
        unsafe { &*(inner_ref as *const T as *const Labeled<T, L>) }
    }
}

// Support for Mutable Indexing on Arrays
// This allows: labeled_array[0] = new_labeled_val;
impl<T, L, const N: usize> IndexMut<usize> for Labeled<[T; N], L>
where
    L: Label,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let inner_mut = &mut self.value[index];
        unsafe { &mut *(inner_mut as *mut T as *mut Labeled<T, L>) }
    }
}

// Support mutable indexing with Labeled<usize, L2>
impl<T, L, L2, const N: usize> IndexMut<Labeled<usize, L2>> for Labeled<[T; N], L>
where
    L: Label,
    L2: Label,
{
    fn index_mut(&mut self, index: Labeled<usize, L2>) -> &mut Self::Output {
        let inner_mut = &mut self.value[index.value];
        unsafe { &mut *(inner_mut as *mut T as *mut Labeled<T, L>) }
    }
}

// BITWISE OR (|)
impl<T, L1, L2> BitOr<Labeled<T, L2>> for Labeled<T, L1>
where
    T: BitOr<Output = T>,
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<T, <L1 as Join<L2>>::Out>;

    fn bitor(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled {
            value: self.value | rhs.value,
            _marker: PhantomData,
        }
    }
}

impl<L: Label> AsRef<Path> for Labeled<PathBuf, L> {
    fn as_ref(&self) -> &Path {
        self.value.as_path()
    }
}

impl<T, E, L: Label> Labeled<Result<T, E>, L> {
    /// Swaps Labeled<Result<T, E>> -> Result<Labeled<T>>
    /// This allows you to use '?' on a Labeled Result.
    pub fn transpose(self) -> Result<Labeled<T, L>, E> {
        match self.value {
            Ok(v) => Ok(Labeled::new(v)), // Re-wrap Success in Labeled
            Err(e) => Err(e),             // Propagate Error raw
        }
    }
}

impl<T: PartialEq, L: Label> PartialEq<T> for Labeled<T, L> {
    fn eq(&self, other: &T) -> bool {
        self.value == *other
    }
}

// --- LABELED COMPARISON TRAITS ---
// These return Labeled<bool, L> instead of plain bool, preserving security labels
// through comparisons. Used by pc_block! macro to track implicit flows.

pub trait LabeledCmp<Rhs> {
    type Output;
    fn labeled_eq(self, rhs: Rhs) -> Self::Output;
    fn labeled_ne(self, rhs: Rhs) -> Self::Output;
}

// Labeled<T, L1> cmp Labeled<T, L2> → Labeled<bool, Join<L1, L2>>
impl<T: PartialEq, L1, L2> LabeledCmp<Labeled<T, L2>> for Labeled<T, L1>
where
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<bool, <L1 as Join<L2>>::Out>;
    fn labeled_eq(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled::new(self.value == rhs.value)
    }
    fn labeled_ne(self, rhs: Labeled<T, L2>) -> Self::Output {
        Labeled::new(self.value != rhs.value)
    }
}

// Labeled<T, L> cmp T → Labeled<bool, L>
impl<T: PartialEq, L: Label> LabeledCmp<T> for Labeled<T, L> {
    type Output = Labeled<bool, L>;
    fn labeled_eq(self, rhs: T) -> Self::Output {
        Labeled::new(self.value == rhs)
    }
    fn labeled_ne(self, rhs: T) -> Self::Output {
        Labeled::new(self.value != rhs)
    }
}

// --- LABELED LOGICAL OPERATORS ---
// These preserve labels through && and || operations inside pc_block!

pub trait LabeledAnd<Rhs> {
    type Output;
    fn labeled_and(self, rhs: Rhs) -> Self::Output;
}

pub trait LabeledOr<Rhs> {
    type Output;
    fn labeled_or(self, rhs: Rhs) -> Self::Output;
}

// Labeled<bool, L1> && Labeled<bool, L2> → Labeled<bool, Join<L1, L2>>
impl<L1, L2> LabeledAnd<Labeled<bool, L2>> for Labeled<bool, L1>
where
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<bool, <L1 as Join<L2>>::Out>;
    fn labeled_and(self, rhs: Labeled<bool, L2>) -> Self::Output {
        Labeled::new(self.value && rhs.value)
    }
}

// Labeled<bool, L> && bool → Labeled<bool, L>
impl<L: Label> LabeledAnd<bool> for Labeled<bool, L> {
    type Output = Labeled<bool, L>;
    fn labeled_and(self, rhs: bool) -> Self::Output {
        Labeled::new(self.value && rhs)
    }
}

// bool && bool → bool
impl LabeledAnd<bool> for bool {
    type Output = bool;
    fn labeled_and(self, rhs: bool) -> bool {
        self && rhs
    }
}

// Labeled<bool, L1> || Labeled<bool, L2> → Labeled<bool, Join<L1, L2>>
impl<L1, L2> LabeledOr<Labeled<bool, L2>> for Labeled<bool, L1>
where
    L1: Label + Join<L2>,
    L2: Label,
    <L1 as Join<L2>>::Out: Label,
{
    type Output = Labeled<bool, <L1 as Join<L2>>::Out>;
    fn labeled_or(self, rhs: Labeled<bool, L2>) -> Self::Output {
        Labeled::new(self.value || rhs.value)
    }
}

// Labeled<bool, L> || bool → Labeled<bool, L>
impl<L: Label> LabeledOr<bool> for Labeled<bool, L> {
    type Output = Labeled<bool, L>;
    fn labeled_or(self, rhs: bool) -> Self::Output {
        Labeled::new(self.value || rhs)
    }
}

// bool || bool → bool
impl LabeledOr<bool> for bool {
    type Output = bool;
    fn labeled_or(self, rhs: bool) -> bool {
        self || rhs
    }
}

// NEGATION (!)
// !Labeled<bool, L> → Labeled<bool, L>
impl<L: Label> Not for Labeled<bool, L> {
    type Output = Labeled<bool, L>;
    fn not(self) -> Self::Output {
        Labeled::new(!self.value)
    }
}

// PartialOrd: Labeled<T, L> compared with Labeled<T, L>
impl<T: PartialOrd, L: Label> PartialOrd for Labeled<T, L> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

// PartialOrd: Labeled<T, L> compared with raw T
impl<T: PartialOrd, L: Label> PartialOrd<T> for Labeled<T, L> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(other)
    }
}
