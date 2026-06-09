use crate::lattice::{FlowsTo, Join, Label, Labeled, Public};

// =========================================================================
// 1. SIDE EFFECT TRAITS (Ported from Cocoon)
// =========================================================================

/// Marker trait for types that are safe to use in implicit flow blocks.
/// They must not have visible side effects (like I/O) when dropped or cloned.
/// This prevents leaking secret conditions via side effects.
pub unsafe trait InvisibleSideEffectFree {
    unsafe fn check_all_types() {}
}
pub struct Vetted<T>
where
    T: InvisibleSideEffectFree,
{
    item: T,
}

impl<T> Vetted<T>
where
    T: InvisibleSideEffectFree,
{
    // Marks a return value as side-effect free.
    pub unsafe fn wrap(item: T) -> Self {
        Vetted::<T> { item }
    }

    // Extracts the return value.
    pub fn unwrap(self) -> T {
        self.item
    }
}

unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for Vetted<T> {}
// --- HELPER FUNCTIONS CALLED BY MACRO ---

/// Checks if a value is safe to handle (no side effects).
/// The macro wraps expressions in this. If T doesn't implement the trait, it fails to compile.
#[inline(always)]
pub fn check_isef<T: InvisibleSideEffectFree>(x: T) -> T {
    x
}

// =========================================================================
// 2. CONDITION HANDLING (Extract Label from Boolean)
// =========================================================================

/// Trait to inspect a condition and retrieve its security label.
pub trait ConditionInspect {
    type Label: Label;
    fn inspect(self) -> (bool, Self::Label);
}

// Case A: Standard boolean (Public condition)
impl ConditionInspect for bool {
    type Label = Public;
    fn inspect(self) -> (bool, Self::Label) {
        (self, Public)
    }
}

// Case B: Labeled boolean (Secret condition)
impl<L: Label + Default> ConditionInspect for Labeled<bool, L> {
    type Label = L;
    fn inspect(self) -> (bool, Self::Label) {
        // "Unwrap" the value for the 'if', but return the Label token for PC tracking.
        (self.value, L::default())
    }
}

/// Helper called by the macro to inspect 'if' conditions.
#[inline(always)]
pub fn inspect_condition<C: ConditionInspect>(cond: C) -> (bool, C::Label) {
    cond.inspect()
}

/// Cocoon Escape Hatch: Allows operations that are not vetted by the IFC macro.
pub fn unchecked_operation<T>(val: T) -> T {
    val
}

// =========================================================================
// 3. PC TRACKING (Label Join)
// =========================================================================

/// Joins the current PC label with a new condition's label.
/// Returns a phantom token representing the new security context (PC).
#[inline(always)]
pub fn join_labels<L1: Label, L2: Label>(_current_pc: L1, _new_label: L2) -> <L1 as Join<L2>>::Out
where
    L1: Join<L2>,
    <L1 as Join<L2>>::Out: Label,
{
    // Returns a zeroed-initialized value of the joined label type
    // This is safe because Labels are Copy and don't contain significant data
    unsafe { std::mem::zeroed() }
}

// =========================================================================
// 4. SECURE ASSIGNMENT (Implicit Flow Check)
// =========================================================================

/// PC-only guard for assignments where the source label equals the destination
/// (e.g. `x = Labeled::new(val)` where `L` is inferred from `x`'s type).
/// Only enforces the implicit-flow rule: PC must flow to the destination label.
#[inline(always)]
pub fn pc_guard_assign<T, Dest, PC>(_dest: &mut Labeled<T, Dest>, _pc: PC)
where
    Dest: Label,
    PC: Label + FlowsTo<Dest>, // PC guard only
{
}

/// Performs a secure assignment enforcing both Explicit and Implicit flow.
///
/// Security Rules:
/// 1. **Explicit Flow:** Source Label (`Src`) must flow to Destination Label (`Dest`).
/// 2. **Implicit Flow:** Current PC (`PC`) must flow to Destination Label (`Dest`).
///
/// This prevents writing to a Public variable while inside a Secret 'if' block.
#[inline(always)]
pub fn secure_assign_with_pc<T, Dest, Src, PC>(dest: &mut Labeled<T, Dest>, src: Labeled<T, Src>, _pc: PC)
where
    Dest: Label,
    Src: Label + FlowsTo<Dest>, // Check 1: Value Flow
    PC: Label + FlowsTo<Dest>,  // Check 2: Context Flow (PC guard)
{
    dest.value = src.value;
}

// =========================================================================
// 5. SAFE TRAIT IMPLEMENTATIONS
// =========================================================================

// Primitives are safe (no side effects on drop/clone)
unsafe impl InvisibleSideEffectFree for i8 {}
unsafe impl InvisibleSideEffectFree for i16 {}
unsafe impl InvisibleSideEffectFree for i32 {}
unsafe impl InvisibleSideEffectFree for i64 {}
unsafe impl InvisibleSideEffectFree for isize {}
unsafe impl InvisibleSideEffectFree for u8 {}
unsafe impl InvisibleSideEffectFree for u16 {}
unsafe impl InvisibleSideEffectFree for u32 {}
unsafe impl InvisibleSideEffectFree for u64 {}
unsafe impl InvisibleSideEffectFree for usize {}
unsafe impl InvisibleSideEffectFree for f32 {}
unsafe impl InvisibleSideEffectFree for f64 {}
unsafe impl InvisibleSideEffectFree for bool {}
unsafe impl InvisibleSideEffectFree for char {}
unsafe impl InvisibleSideEffectFree for () {}
// Allow Slices [T] (if not already present)
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for [T] {}

// Allow References to Slices &[T]
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for &[T] {}

// Allow Mutable References to Slices &mut [T]
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for &mut [T] {}
unsafe impl<T: InvisibleSideEffectFree, const N: usize> InvisibleSideEffectFree for [T; N] {}

// Standard Types
unsafe impl InvisibleSideEffectFree for String {}
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for Option<T> {}
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for Vec<T> {}
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for Box<T> {}
unsafe impl<'a, T: InvisibleSideEffectFree> InvisibleSideEffectFree for std::slice::Iter<'a, T> {}

// References
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for &T {}
unsafe impl<T: InvisibleSideEffectFree> InvisibleSideEffectFree for &mut T {}
unsafe impl InvisibleSideEffectFree for &str {}

// Your Labeled Types
// Labeled data is safe to handle inside the block
unsafe impl<T: InvisibleSideEffectFree, L: Label> InvisibleSideEffectFree for Labeled<T, L> {}

unsafe impl InvisibleSideEffectFree for std::time::Instant {}
// unsafe impl InvisibleSideEffectFree for decla

// =========================================================================
// 6. PC BLOCK FUNCTION CALL RESULT HANDLING (Autoref Specialization)
// =========================================================================

/// Autoref specialization for handling function call results inside pc_block.
/// - For functions with #[side_effect_free_attr] that return Vetted<T>: unwraps Vetted, returns T
/// - For other functions returning raw T: wraps in Labeled<T, Public>
pub struct PcCallResult;

// High priority (inherent method): matches Vetted<T> → unwraps to T
impl PcCallResult {
    pub fn wrap_result<T: InvisibleSideEffectFree>(&self, x: Vetted<T>) -> T {
        x.unwrap()
    }
}

// Low priority (trait method): matches any T → wraps in Labeled<T, Public>
pub trait PcCallResultFallback {
    fn wrap_result<T: InvisibleSideEffectFree>(&self, x: T) -> Labeled<T, Public>;
}

impl PcCallResultFallback for PcCallResult {
    fn wrap_result<T: InvisibleSideEffectFree>(&self, x: T) -> Labeled<T, Public> {
        Labeled::new(x)
    }
}

// =========================================================================
// 7. PC-AWARE ISEF CHECK (Autoref Specialization)
// =========================================================================

/// PC-aware side-effect checker.
/// When PC is Public, no InvisibleSideEffectFree check is required
/// (no information leak risk in a Public context).
/// When PC is Secret (A, B, AB, etc.), InvisibleSideEffectFree is enforced.
///
/// Uses the same autoref specialization pattern as PcCallResult:
/// - PcIsef<Public> has an inherent `check<T>` (no ISEF bound) → higher priority
/// - PcIsefFallback trait has `check<T: ISEF>` → lower priority, used for non-Public PCs
pub struct PcIsef<PC: Label>(std::marker::PhantomData<PC>);

impl<PC: Label> PcIsef<PC> {
    #[inline(always)]
    pub fn new(_pc: &PC) -> Self {
        PcIsef(std::marker::PhantomData)
    }
}

// Public PC: no ISEF check needed (inherent method → higher priority)
impl PcIsef<Public> {
    #[inline(always)]
    pub fn check<T>(&self, x: T) -> T {
        x
    }
    #[inline(always)]
    pub fn reject_side_effecting_macro<T>(&self, x: T) -> T {
        x
    }
}

// Any PC: requires ISEF (trait method → lower priority, used for non-Public)
pub trait PcIsefFallback {
    fn check<T: InvisibleSideEffectFree>(&self, x: T) -> T;
    fn reject_side_effecting_macro<T: MacroSideEffectFree>(&self, x: T) -> T;
}

/// Marker trait that nothing implements. Used to reject side-effecting macros
/// (like println!) under a non-Public PC. The non-Public fallback trait requires
/// this bound, which always fails, producing a compile error. The Public inherent
/// method has no such bound, so it compiles fine.
pub trait MacroSideEffectFree {}

impl<PC: Label> PcIsefFallback for PcIsef<PC> {
    #[inline(always)]
    fn check<T: InvisibleSideEffectFree>(&self, x: T) -> T {
        x
    }
    #[inline(always)]
    fn reject_side_effecting_macro<T: MacroSideEffectFree>(&self, x: T) -> T {
        x
    }
}
