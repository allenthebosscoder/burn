use crate::lattice::*;
use std::marker::PhantomData;

impl<T, L: Label> Labeled<T, L> {
    #[doc(hidden)]
    pub fn __chain<R, L2, F>(self, f: F) -> Labeled<R, <L as Join<L2>>::Out>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(T) -> Labeled<R, L2>,
    {
        let inner_res = f(self.value);
        Labeled {
            value: inner_res.value,
            _marker: PhantomData,
        }
    }

    /// Like `__chain` but borrows `self`, giving the closure `&T`.
    /// Used by `fcall!` for `&expr` arguments so the label `L` is propagated
    #[doc(hidden)]
    pub fn __chain_ref<'a, R, L2, F>(&'a self, f: F) -> Labeled<R, <L as Join<L2>>::Out>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(&'a T) -> Labeled<R, L2>,
    {
        let inner_res = f(&self.value);
        Labeled {
            value: inner_res.value,
            _marker: PhantomData,
        }
    }

    /// Like `__chain` but gives the closure `&mut T` instead of consuming `T`.
    /// Consumes `self`, mutates the inner value via the closure, then packs the
    /// (possibly mutated) `T` and the closure's return `R` together as
    /// `Labeled<(T, R), JoinedLabel>`.  The caller splits this to recover the
    /// updated labeled `T` and the labeled return value separately.
    /// Used by `fcall!` for `&mut ident` arguments so the label on the mutated
    /// value grows to include every label that flowed into the call.
    #[doc(hidden)]
    pub fn __chain_mut<R, L2, F>(mut self, f: F) -> Labeled<(T, R), <L as Join<L2>>::Out>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(&mut T) -> Labeled<R, L2>,
    {
        let ret = f(&mut self.value);
        Labeled {
            value: (self.value, ret.value),
            _marker: PhantomData,
        }
    }
}

// =========================================================================
//  THE TRAIT: FOR RAW VALUES (Priority #2)
// =========================================================================
// This is needed so you can pass raw '5' or '"filename"' to the macro.
#[doc(hidden)]
pub trait SecureChain<T, L: Label> {
    fn __chain<R, L2, F>(self, f: F) -> Labeled<R, <L as Join<L2>>::Out>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(T) -> Labeled<R, L2>;
}

// Blanket implementation for ANY type T that isn't caught by the inherent impl above.
// Treats the value as 'Public'.
impl<T> SecureChain<T, Public> for T
where
    T: Sized,
{
    fn __chain<R, L2, F>(self, f: F) -> Labeled<R, <Public as Join<L2>>::Out>
    where
        L2: Label,
        Public: Join<L2>,
        F: FnOnce(T) -> Labeled<R, L2>,
    {
        // Passthrough: Just run the function on the raw value
        let inner_res = f(self);
        Labeled {
            value: inner_res.value,
            _marker: PhantomData,
        }
    }
}

// =========================================================================
//  CHAIN_REF TRAIT: FOR PLAIN (non-Labeled) REFERENCE ARGUMENTS
// =========================================================================
// `fcall!(func(&plain_val))` strips the `&` and calls `plain_val.chain_ref(...)`.
// For Labeled<T, L>: the inherent `chain_ref` above takes priority → label propagates.
// For any other T:   this blanket trait kicks in → treats the value as Public.
#[doc(hidden)]
pub trait SecureChainRef<T, L: Label> {
    fn __chain_ref<R, L2, F>(&self, f: F) -> Labeled<R, <L as Join<L2>>::Out>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(&T) -> Labeled<R, L2>;
}

impl<T> SecureChainRef<T, Public> for T
where
    T: Sized,
{
    fn __chain_ref<R, L2, F>(&self, f: F) -> Labeled<R, <Public as Join<L2>>::Out>
    where
        L2: Label,
        Public: Join<L2>,
        F: FnOnce(&T) -> Labeled<R, L2>,
    {
        let inner_res = f(self);
        Labeled {
            value: inner_res.value,
            _marker: PhantomData,
        }
    }
}

// =========================================================================
//  CHAIN_MUT TRAIT: FOR PLAIN (non-Labeled) MUTABLE ARGUMENTS
// =========================================================================
// `fcall!(func(&mut plain_val))` strips the `&mut`, moves the plain value,
// gives `&mut T` to the closure, and re-packs `(T, R)` as a labeled tuple.
// For Labeled<T, L>: the inherent `__chain_mut` above takes priority.
// For any other T:   this blanket trait kicks in → treats value as Public.
#[doc(hidden)]
pub trait SecureChainMut<T, L: Label> {
    fn __chain_mut<R, L2, F>(self, f: F) -> Labeled<(T, R), <L as Join<L2>>::Out>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(&mut T) -> Labeled<R, L2>;
}

impl<T> SecureChainMut<T, Public> for T
where
    T: Sized,
{
    fn __chain_mut<R, L2, F>(mut self, f: F) -> Labeled<(T, R), <Public as Join<L2>>::Out>
    where
        L2: Label,
        Public: Join<L2>,
        F: FnOnce(&mut T) -> Labeled<R, L2>,
    {
        let ret = f(&mut self);
        Labeled {
            value: (self, ret.value),
            _marker: PhantomData,
        }
    }
}

use std::future::Future;
use std::pin::Pin;

/// Async version of SecureChain: returns a boxed Future so chains can compose over async calls.
pub trait SecureAsyncChain<T, L: Label> {
    fn async_chain<R, L2, F, Fut>(self, f: F) -> Pin<Box<dyn Future<Output = Labeled<R, <L as Join<L2>>::Out>>>>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(T) -> Fut + 'static,
        Fut: Future<Output = Labeled<R, L2>> + 'static;
}

// Async chain inherent impl for owned Labeled
impl<T, L: Label> SecureAsyncChain<T, L> for Labeled<T, L>
where
    T: 'static,
    L: 'static,
{
    fn async_chain<R, L2, F, Fut>(self, f: F) -> Pin<Box<dyn Future<Output = Labeled<R, <L as Join<L2>>::Out>>>>
    where
        L2: Label,
        L: Join<L2>,
        F: FnOnce(T) -> Fut + 'static,
        Fut: Future<Output = Labeled<R, L2>> + 'static,
    {
        Box::pin(async move {
            let inner_res = f(self.value).await;
            Labeled {
                value: inner_res.value,
                _marker: PhantomData,
            }
        })
    }
}

// Async chain for raw/public values
impl<T> SecureAsyncChain<T, Public> for T
where
    T: 'static,
{
    fn async_chain<R, L2, F, Fut>(self, f: F) -> Pin<Box<dyn Future<Output = Labeled<R, <Public as Join<L2>>::Out>>>>
    where
        L2: Label,
        Public: Join<L2>,
        F: FnOnce(T) -> Fut + 'static,
        Fut: Future<Output = Labeled<R, L2>> + 'static,
    {
        Box::pin(async move {
            let inner_res = f(self).await;
            Labeled {
                value: inner_res.value,
                _marker: PhantomData,
            }
        })
    }
}
