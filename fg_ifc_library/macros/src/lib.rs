use proc_macro::TokenStream;
use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{format_ident, quote, ToTokens};
use std::collections::HashSet;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Expr, ExprCall, ExprMethodCall, Token, Type,
};

// =========================================================================
// Helper Functions
// =========================================================================

/// Custom parser for the relabel! syntax:
///   relabel!(expr, Label) → wraps raw value with label
struct RelabelInput {
    var: Expr,
    label: Type,
}

impl Parse for RelabelInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let var: Expr = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let label: Type = input.parse()?;
        Ok(RelabelInput { var, label })
    }
}

// =========================================================================
// THE FCALL MACRO (Function Calls)
// =========================================================================

#[proc_macro]
pub fn fcall(input: TokenStream) -> TokenStream {
    // 1. Parse as a generic Expression first
    let expr = parse_macro_input!(input as Expr);

    // 2. Check if it ends with '?' (ExprTry), or is an awaited call (Expr::Await), or is a plain call
    let mut has_question_mark = false;
    let mut has_await = false;
    // Work with a mutable ownership of the expression so we can peel layers
    let mut expr_to_check = expr;

    // Unwrap await: fcall!( ... .await ) -> treat inner expression as the call but remember await
    if let syn::Expr::Await(await_expr) = expr_to_check {
        has_await = true;
        expr_to_check = *await_expr.base;
    }

    // Special case: fcall!(format!("...", arg1, arg2))
    // Chains labeled arguments through __chain(), calls format! with unwrapped values,
    // wraps result in Labeled<String, Public>.
    if let syn::Expr::Macro(ref mac) = expr_to_check {
        let is_format = mac.mac.path.segments.last().map(|s| s.ident == "format").unwrap_or(false);
        if is_format {
            let tokens = mac.mac.tokens.clone();
            let parsed = syn::parse::Parser::parse2(syn::punctuated::Punctuated::<Expr, Token![,]>::parse_terminated, tokens).expect("format! should contain comma-separated expressions");
            let mut items = parsed.iter();
            let fmt_str = items.next().expect("format! needs a format string");
            let args: Vec<&Expr> = items.collect();

            if args.is_empty() {
                return TokenStream::from(quote! {
                    ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                        format!(#fmt_str)
                    )
                });
            }

            let unwrapped_names: Vec<_> = (0..args.len()).map(|i| format_ident!("__v{}", i)).collect();

            let mut expanded = quote! {
                ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                    format!(#fmt_str, #(#unwrapped_names),*)
                )
            };

            for (arg, name) in args.iter().zip(unwrapped_names.iter()).rev() {
                expanded = quote! { (#arg).__chain(|#name| { #expanded }) };
            }

            return TokenStream::from(quote! {
                {
                    use ::typing_rules::function_rewrite::SecureChain;
                    use ::typing_rules::function_rewrite::SecureChainRef;
                    #expanded
                }
            });
        }
    }

    // Handle struct literals: fcall!(Path { field1: val1, field2: val2 })
    // Chain through each field's value and reconstruct the struct inside Labeled::new(...).
    // This allows writing e.g. fcall!(HousingBatch { inputs: labeled_a, targets: labeled_b })
    // without any helper constructor function.
    let expr_to_check = match expr_to_check {
        syn::Expr::Struct(s) => {
            let path = &s.path;
            let mut struct_chains: Vec<(Ident, TokenStream2)> = Vec::new();
            let mut field_tokens: Vec<TokenStream2> = Vec::new();
            for (i, field) in s.fields.iter().enumerate() {
                let member = &field.member;
                let val = &field.expr;
                let name = format_ident!("__v{}", i);
                struct_chains.push((name.clone(), quote! { (#val) }));
                field_tokens.push(quote! { #member: #name });
            }
            let mut expanded = quote! {
                ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                    #path { #(#field_tokens),* }
                )
            };
            for (name, target) in struct_chains.iter().rev() {
                expanded = quote! { (#target).__chain(|#name| { #expanded }) };
            }
            let panic_guard = quote! {
                let __fcall_prev_hook = ::std::panic::take_hook();
                ::std::panic::set_hook(::std::boxed::Box::new(|_| {}));
                struct __FcallPanicGuard(::std::option::Option<::std::boxed::Box<dyn ::std::ops::FnMut()>>);
                impl ::std::ops::Drop for __FcallPanicGuard {
                    fn drop(&mut self) {
                        if !::std::thread::panicking() {
                            if let ::std::option::Option::Some(mut f) = self.0.take() { f(); }
                        }
                    }
                }
                let mut __fcall_hook_opt = ::std::option::Option::Some(__fcall_prev_hook);
                let __fcall_panic_guard = __FcallPanicGuard(::std::option::Option::Some(::std::boxed::Box::new(move || {
                    if let ::std::option::Option::Some(hook) = __fcall_hook_opt.take() {
                        ::std::panic::set_hook(hook);
                    }
                })));
            };
            return TokenStream::from(quote! {
                {
                    use ::typing_rules::function_rewrite::SecureChain;
                    use ::typing_rules::function_rewrite::SecureChainRef;
                    #panic_guard
                    let __fcall_result = { #expanded };
                    drop(__fcall_panic_guard);
                    __fcall_result
                }
            });
        }
        other => other,
    };

    // Extract the base function call and any trailing method chain.
    // fcall!(func(args).m1().m2()) calls func(args) with labeled arg unwrapping
    // and applies .m1().m2() on the raw result before wrapping in Labeled.
    // This handles e.g. Tensor::from_floats([item.field, ...], device).unsqueeze()
    // where unsqueeze(self) consumes its receiver and cannot be called via mcall!'s __chain_ref.
    let (func, args, method_suffix): (_, _, TokenStream2) = match expr_to_check {
        syn::Expr::Try(expr_try) => {
            has_question_mark = true;
            match *expr_try.expr {
                syn::Expr::Call(call) => (call.func, call.args, quote! {}),
                other => return syn::Error::new_spanned(other, "fcall! expects a function call").to_compile_error().into(),
            }
        }
        syn::Expr::Call(call) => (call.func, call.args, quote! {}),
        other => {
            // Peel trailing method chain to find the base Call.
            let mut current = other;
            let mut methods_rev: Vec<TokenStream2> = Vec::new();
            let base_call = loop {
                match current {
                    syn::Expr::MethodCall(mc) => {
                        let method = mc.method;
                        let margs = mc.args;
                        let turbofish = mc.turbofish;
                        let receiver = *mc.receiver;
                        let tok = if let Some(ref tf) = turbofish {
                            quote! { .#method::<#tf>(#margs) }
                        } else {
                            quote! { .#method(#margs) }
                        };
                        methods_rev.push(tok);
                        current = receiver;
                    }
                    syn::Expr::Call(c) => break c,
                    expr => return syn::Error::new_spanned(expr, "fcall! expects a function call or method chain on a function call").to_compile_error().into(),
                }
            };
            let suffix = methods_rev.into_iter().rev().fold(TokenStream2::new(), |acc, m| quote! { #acc #m });
            (base_call.func, base_call.args, suffix)
        }
    };

    // 3. Prepare chain entries and inner call args.
    // For regular and reference args: one chain entry per arg.
    // For Expr::Array args: scan elements for `base.field` patterns — each unique base
    // variable becomes its own chain entry, and the inner call arg is the reconstructed
    // array with base paths replaced by the chain vars. This lets labeled variables
    // embedded inside array literals (e.g. `[item.x, item.y]` where item: Labeled<T,L>)
    // be properly tracked without any user-written helper functions.
    enum ChainKind {
        Owned,
        Ref,
    }
    struct ChainEntry {
        kind: ChainKind,
        target: TokenStream2,
        name: Ident,
    }

    let mut chain_entries: Vec<ChainEntry> = Vec::new();
    let mut inner_call_args: Vec<TokenStream2> = Vec::new();
    let mut chain_idx: usize = 0;

    for arg in args.iter() {
        match arg {
            syn::Expr::Reference(r) if r.mutability.is_none() => {
                let name = format_ident!("__v{}", chain_idx);
                chain_idx += 1;
                let e = &r.expr;
                inner_call_args.push(quote! { #name });
                chain_entries.push(ChainEntry { kind: ChainKind::Ref, target: quote! { #e }, name });
            }
            syn::Expr::Array(arr) => {
                // Collect unique base variables from field-access elements (`base.field`
                // where base is a simple path). Each unique base becomes a chain entry.
                let mut seen = std::collections::HashSet::<String>::new();
                let mut bases: Vec<(String, Ident)> = Vec::new();

                for elem in &arr.elems {
                    if let syn::Expr::Field(f) = elem {
                        if let syn::Expr::Path(p) = f.base.as_ref() {
                            let key = quote!(#p).to_string();
                            if !seen.contains(&key) {
                                seen.insert(key.clone());
                                let name = format_ident!("__v{}", chain_idx);
                                chain_idx += 1;
                                chain_entries.push(ChainEntry {
                                    kind: ChainKind::Owned,
                                    target: quote! { (#p) },
                                    name: name.clone(),
                                });
                                bases.push((key, name));
                            }
                        }
                    }
                }

                if bases.is_empty() {
                    // No field-access patterns found; treat as a plain owned arg.
                    let name = format_ident!("__v{}", chain_idx);
                    chain_idx += 1;
                    inner_call_args.push(quote! { #name });
                    chain_entries.push(ChainEntry {
                        kind: ChainKind::Owned,
                        target: quote! { (#arr) },
                        name,
                    });
                } else {
                    // Reconstruct the array substituting base paths with their chain vars.
                    let elems: Vec<TokenStream2> = arr.elems.iter().map(|elem| {
                        if let syn::Expr::Field(f) = elem {
                            if let syn::Expr::Path(p) = f.base.as_ref() {
                                let key = quote!(#p).to_string();
                                if let Some((_, inner)) = bases.iter().find(|(k, _)| k == &key) {
                                    let member = &f.member;
                                    return quote! { #inner.#member };
                                }
                            }
                        }
                        quote! { #elem }
                    }).collect();
                    inner_call_args.push(quote! { [#(#elems),*] });
                }
            }
            // vec![a, b] or vec![a.field, b.field] where elements may be labeled variables.
            // Each unique base variable (plain path or field-access base) becomes a chain
            // entry; the inner arg is vec![__v0, __v1, ...] with unwrapped values.
            syn::Expr::Macro(mac) if mac.mac.path.segments.last().map(|s| s.ident == "vec").unwrap_or(false) => {
                let tokens = mac.mac.tokens.clone();
                let elems: syn::punctuated::Punctuated<Expr, syn::Token![,]> =
                    syn::parse::Parser::parse2(
                        syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated,
                        tokens,
                    ).unwrap_or_default();

                let mut seen = std::collections::HashSet::<String>::new();
                let mut bases: Vec<(String, Ident, TokenStream2)> = Vec::new();

                for elem in &elems {
                    let (key, base_target) = match elem {
                        syn::Expr::Field(f) if matches!(f.base.as_ref(), syn::Expr::Path(_)) => {
                            let p = if let syn::Expr::Path(p) = f.base.as_ref() { p } else { unreachable!() };
                            (Some(quote!(#p).to_string()), quote! { (#p) })
                        }
                        syn::Expr::Path(p) => (Some(quote!(#p).to_string()), quote! { (#p) }),
                        _ => (None, quote! {}),
                    };
                    if let Some(key) = key {
                        if !seen.contains(&key) {
                            seen.insert(key.clone());
                            let name = format_ident!("__v{}", chain_idx);
                            chain_idx += 1;
                            chain_entries.push(ChainEntry { kind: ChainKind::Owned, target: base_target.clone(), name: name.clone() });
                            bases.push((key, name, base_target));
                        }
                    }
                }

                if bases.is_empty() {
                    let name = format_ident!("__v{}", chain_idx);
                    chain_idx += 1;
                    inner_call_args.push(quote! { #name });
                    chain_entries.push(ChainEntry { kind: ChainKind::Owned, target: quote! { (#mac) }, name });
                } else {
                    let reconstructed: Vec<TokenStream2> = elems.iter().map(|elem| {
                        let key = match elem {
                            syn::Expr::Field(f) if matches!(f.base.as_ref(), syn::Expr::Path(_)) => {
                                if let syn::Expr::Path(p) = f.base.as_ref() { Some(quote!(#p).to_string()) } else { None }
                            }
                            syn::Expr::Path(p) => Some(quote!(#p).to_string()),
                            _ => None,
                        };
                        if let Some(key) = key {
                            if let Some((_, inner, _)) = bases.iter().find(|(k, _, _)| k == &key) {
                                return match elem {
                                    syn::Expr::Field(f) => { let m = &f.member; quote! { #inner.#m } }
                                    _ => quote! { #inner },
                                };
                            }
                        }
                        quote! { #elem }
                    }).collect();
                    inner_call_args.push(quote! { vec![#(#reconstructed),*] });
                }
            }
            other => {
                let name = format_ident!("__v{}", chain_idx);
                chain_idx += 1;
                inner_call_args.push(quote! { #name });
                chain_entries.push(ChainEntry {
                    kind: ChainKind::Owned,
                    target: quote! { (#other) },
                    name,
                });
            }
        }
    }

    // 4. Inner Execution Logic
    let mut expanded = if has_await {
        quote! {
            ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                #func( #(#inner_call_args),* ) #method_suffix .await
            )
        }
    } else {
        quote! {
            ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                #func( #(#inner_call_args),* ) #method_suffix
            )
        }
    };

    // 5. Wrap args in .chain() / .chain_ref() / .async_chain()
    // Label idempotency (L ∨ L = L) is guaranteed by the Label supertrait bound
    // Join<Self, Out = Self>, so chaining multiple args of the same label L produces
    // Join<L, Join<L, Public>> = Join<L, L> = L — no special handling needed.
    if has_await {
        for entry in chain_entries.iter().rev() {
            let target = &entry.target;
            let name = &entry.name;
            expanded = quote! {
                (#target).async_chain(|#name| async move {
                    #expanded
                })
            };
        }
    } else {
        for entry in chain_entries.iter().rev() {
            let target = &entry.target;
            let name = &entry.name;
            expanded = match entry.kind {
                ChainKind::Ref => quote! {
                    (#target).__chain_ref(|#name| {
                        #expanded
                    })
                },
                ChainKind::Owned => quote! {
                    (#target).__chain(|#name| {
                        #expanded
                    })
                },
            };
        }
    }

    // 6. Handle the '?' operator if present
    // If the user wrote fcall!(foo()?), we unwrap the Labeled Result,
    // propagate the error, and re-wrap the success value.
    if has_question_mark {
        expanded = quote! {
            (#expanded).transpose()?
        };
    }

    // 7. Panic hook suppression guard
    //    Suppress panic messages while the unwrapped secret values are
    //    in scope.  This prevents secret data from leaking through panic
    //    payloads (e.g., `format!("{}", secret_value)` inside a panicking
    //    function).
    let panic_guard = quote! {
        let __fcall_prev_hook = ::std::panic::take_hook();
        ::std::panic::set_hook(::std::boxed::Box::new(|_| {}));

        struct __FcallPanicGuard(
            ::std::option::Option<
                ::std::boxed::Box<dyn ::std::ops::FnMut()>
            >
        );
        impl ::std::ops::Drop for __FcallPanicGuard {
            fn drop(&mut self) {
                if !::std::thread::panicking() {
                    if let ::std::option::Option::Some(mut f) = self.0.take() {
                        f();
                    }
                }
            }
        }
        let mut __fcall_hook_opt = ::std::option::Option::Some(__fcall_prev_hook);
        let __fcall_panic_guard = __FcallPanicGuard(
            ::std::option::Option::Some(::std::boxed::Box::new(move || {
                if let ::std::option::Option::Some(hook) = __fcall_hook_opt.take() {
                    ::std::panic::set_hook(hook);
                }
            }))
        );
    };

    // 8. Final Output
    let final_output = if has_await {
        quote! {
            {
                use ::typing_rules::function_rewrite::SecureAsyncChain;
                #panic_guard
                let __fcall_result = { #expanded };
                drop(__fcall_panic_guard);
                __fcall_result
            }
        }
    } else {
        quote! {
            {
                use ::typing_rules::function_rewrite::SecureChain;
                use ::typing_rules::function_rewrite::SecureChainRef;
                #panic_guard
                let __fcall_result = { #expanded };
                drop(__fcall_panic_guard);
                __fcall_result
            }
        }
    };

    TokenStream::from(final_output)
}

// Macro for method calls AND field access on Labeled values.
//
//   mcall!(obj.method(args))  — method call:  preserves label, calls method on &inner
//   mcall!(obj.method(args)?) — fallible call: preserves label, propagates error via ?
//   mcall!(obj.field)         — field access:  preserves label, reads field from &inner
//   mcall!(obj.0)             — tuple index:   preserves label, reads .0 from &inner
//
// Both forms use the same internal helper so the label is preserved exactly
// (no join needed — the field/method result inherits the receiver's label L).
#[proc_macro]
pub fn mcall(input: TokenStream) -> TokenStream {
    // 1. Parse as a general expression first to prevent strict-parsing panics
    let expr = parse_macro_input!(input as Expr);

    // 2. The shared label-preserving helper (emitted inline in every expansion)
    let helper = quote! {
        fn __mcall_preserve_label<__T, __U, __L: ::typing_rules::Label>(
            wrapper: &::typing_rules::lattice::Labeled<__T, __L>,
            func: impl FnOnce(&__T) -> __U
        ) -> ::typing_rules::lattice::Labeled<__U, __L> {
            ::typing_rules::lattice::Labeled::<__U, __L>::new(func(wrapper.__private_value()))
        }
    };

    // 3. Match method call, field access, or awaited method call
    let expanded = match expr {
        // --- awaited method call: mcall!(obj.method(args).await) ---
        //     Unwraps the receiver via .value, extracts inner values from each
        //     argument via chain (works for both Labeled and raw args),
        //     calls the async method, awaits, and returns the raw result.
        Expr::Await(await_expr) => {
            match *await_expr.base {
                Expr::MethodCall(mc) => {
                    let receiver = &mc.receiver;
                    let method = &mc.method;
                    let args = &mc.args;

                    // Classify each argument:
                    //   &expr  → reference arg: pass through inline
                    //            to avoid temporary lifetime issues
                    //   other  → may be Labeled or raw: extract via chain into
                    //            a let binding so the label is checked
                    let mut extractions: Vec<TokenStream2> = Vec::new();
                    let mut call_args: Vec<TokenStream2> = Vec::new();
                    let mut needs_chain = false;

                    for (i, arg) in args.iter().enumerate() {
                        match arg {
                            Expr::Reference(_) => {
                                // Pass reference args directly — avoids
                                // dropping the temporary before the await.
                                call_args.push(quote! { #arg });
                            }
                            _ => {
                                let name = format_ident!("__mv{}", i);
                                extractions.push(quote! {
                                    let #name = (#arg).__chain(|__v| {
                                        ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(__v)
                                    }).__private_into_value();
                                });
                                call_args.push(quote! { #name });
                                needs_chain = true;
                            }
                        }
                    }

                    if needs_chain {
                        quote! {
                            {
                                use ::typing_rules::function_rewrite::SecureChain;
                                #(#extractions)*
                                (#receiver).__private_value_mut().#method(#(#call_args),*).await
                            }
                        }
                    } else {
                        // All args are references or no args — direct call
                        quote! {
                            {
                                (#receiver).__private_value_mut().#method(#(#call_args),*).await
                            }
                        }
                    }
                }
                _ => {
                    return syn::Error::new_spanned(await_expr, "mcall! with .await expects a method call `obj.method(args).await`")
                        .to_compile_error()
                        .into();
                }
            }
        }

        // --- fallible method call: mcall!(obj.method(args)?) ---
        // Same as the method call case, but wraps with .transpose()? to
        // propagate errors while preserving the label on the success value.
        // e.g. mcall!(file.read_to_string()?) where read_to_string returns Result<T, E>
        //   → __mcall_preserve_label(&file, |inner| inner.read_to_string()).transpose()?
        //   → Result<Labeled<T, L>, E> after transpose, then ? yields Labeled<T, L>
        Expr::Try(expr_try) => {
            if let Expr::MethodCall(mc) = *expr_try.expr {
                fn peel_try(
                    expr: &Expr,
                ) -> (
                    &Expr,
                    Vec<(&syn::Ident, Option<&syn::AngleBracketedGenericArguments>, &syn::punctuated::Punctuated<Expr, syn::token::Comma>)>,
                ) {
                    if let Expr::MethodCall(mc) = expr {
                        let (base, mut chain) = peel_try(&mc.receiver);
                        chain.push((&mc.method, mc.turbofish.as_ref(), &mc.args));
                        (base, chain)
                    } else {
                        (expr, vec![])
                    }
                }
                let mc_expr = Expr::MethodCall(mc);
                let (base, chain) = peel_try(&mc_expr);
                let closure_body = chain.iter().fold(quote! { inner }, |acc, (method, turbofish, args)| {
                    if let Some(tf) = turbofish {
                        quote! { #acc.#method::<#tf>(#args) }
                    } else {
                        quote! { #acc.#method(#args) }
                    }
                });
                quote! {
                    {
                        #helper
                        __mcall_preserve_label(&(#base), |inner| #closure_body).transpose()?
                    }
                }
            } else {
                return syn::Error::new_spanned(expr_try, "mcall! with ? expects a method call `obj.method(args)?`").to_compile_error().into();
            }
        }

        // --- method call: mcall!(obj.method(args)) or mcall!(obj.m1().m2().m3(args)) ---
        //
        // Handles two cases via the same expansion:
        //   1. Labeled receiver, unlabeled args:
        //      e.g. mcall!(key.chars().all(f))
        //      → (key).__chain_ref(|__recv| (f).__chain(|__av0| Labeled::new(__recv.chars().all(__av0))))
        //      Inherent __chain_ref on Labeled<T,L> gives &T and preserves label L.
        //
        //   2. Unlabeled receiver, labeled args (the train.rs case):
        //      e.g. mcall!(model.step(item.item))  where model: TrainingModel<LC>, item.item: Labeled<_, L>
        //      → (model).__chain_ref(|__recv| (item.item).__chain(|__av0| Labeled::new(__recv.step(__av0))))
        //      SecureChainRef blanket impl handles the raw receiver (treats it as Public),
        //      while the inherent __chain on the labeled arg propagates its label L.
        //
        // The base receiver is always accessed via __chain_ref (borrow, not move).
        // The last method's args are each chained individually; intermediate method
        // args (in a chained call) are passed raw (assumed non-labeled).
        Expr::MethodCall(mc) => {
            fn peel(
                expr: &Expr,
            ) -> (
                &Expr,
                Vec<(&syn::Ident, Option<&syn::AngleBracketedGenericArguments>, &syn::punctuated::Punctuated<Expr, syn::token::Comma>)>,
            ) {
                if let Expr::MethodCall(mc) = expr {
                    let (base, mut chain) = peel(&mc.receiver);
                    chain.push((&mc.method, mc.turbofish.as_ref(), &mc.args));
                    (base, chain)
                } else {
                    (expr, vec![])
                }
            }
            let mc_expr = Expr::MethodCall(mc);
            let (base, chain) = peel(&mc_expr);

            // Split the last method from the chain so we can chain its args individually.
            // Intermediate methods (all but last) have their args passed raw.
            let (last_entry, intermediates) = chain.split_last()
                .expect("mcall!: method call must have at least one method");
            let (last_method, last_tf, last_args) = last_entry;

            // Build the receiver expression after applying any intermediate methods.
            // For a simple single-method call this is just `__recv`.
            let intermediate_recv = intermediates.iter().fold(quote! { __recv }, |acc, (method, turbofish, args)| {
                if let Some(tf) = turbofish {
                    quote! { #acc.#method::<#tf>(#args) }
                } else {
                    quote! { #acc.#method(#args) }
                }
            });

            // Generate names for the last method's unwrapped args.
            let arg_count = last_args.len();
            let arg_names: Vec<_> = (0..arg_count).map(|i| format_ident!("__av{}", i)).collect();

            // Innermost expression: call the last method and wrap result in Labeled<_, Public>.
            // The label joins with whatever labels are contributed by the receiver and arg chains.
            let inner_call = if let Some(tf) = last_tf {
                quote! {
                    ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                        (#intermediate_recv).#last_method::<#tf>(#(#arg_names),*)
                    )
                }
            } else {
                quote! {
                    ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                        (#intermediate_recv).#last_method(#(#arg_names),*)
                    )
                }
            };

            // Build arg chains from the inside out (last arg wraps innermost).
            let mut body = inner_call;
            for (arg, name) in last_args.iter().zip(arg_names.iter()).rev() {
                body = quote! { (#arg).__chain(|#name| { #body }) };
            }

            // Chain the base receiver via __chain_ref (called on the value, not a reference):
            //   - Labeled<T, L>: inherent __chain_ref is found first; auto-refs to &Labeled<T,L>,
            //     closure receives &T, label L propagates via Join
            //   - Raw T: SecureChainRef blanket impl (T: Sized); auto-refs to &T,
            //     closure receives &T, contributes Public label
            // Using `(#base)` (not `(&(#base))`) ensures the blanket impl resolves
            // to T = TrainingModel<LC> (giving __recv: &T), not T = &TrainingModel<LC>
            // (which would give __recv: &&T via double-ref).
            quote! {
                {
                    use ::typing_rules::function_rewrite::SecureChain;
                    use ::typing_rules::function_rewrite::SecureChainRef;
                    (#base).__chain_ref(|__recv| {
                        #body
                    })
                }
            }
        }

        // --- field access: mcall!(obj.field) or mcall!(obj.0) ---
        Expr::Field(f) => {
            let base = &f.base;
            let member = &f.member;
            quote! {
                {
                    #helper
                    __mcall_preserve_label(&(#base), |inner| inner.#member)
                }
            }
        }

        _ => {
            return syn::Error::new_spanned(expr, "mcall! expects a method call `obj.method(args)` or field access `obj.field`")
                .to_compile_error()
                .into();
        }
    };

    expanded.into()
}

// =========================================================================
// 3. THE RELABEL MACRO (Updating Labels)
// =========================================================================

#[proc_macro]
pub fn relabel(input: TokenStream) -> TokenStream {
    let RelabelInput { var, label } = parse_macro_input!(input as RelabelInput);

    // Reject mutable references: relabel!(&mut x, Label) is not allowed.
    if let syn::Expr::Reference(ref_expr) = &var {
        if ref_expr.mutability.is_some() {
            return syn::Error::new_spanned(&var, "relabel! cannot be used on mutable references (`&mut`)").to_compile_error().into();
        }
    }

    let expanded = quote! {
        {
            // Step 1: Normalize input via autoref specialization.
            // Labeled<T, L> → kept as Labeled<T, L>.
            // Raw T → wrapped as Labeled<T, Public>.
            struct __Wrap<V>(V);

            // Inherent: Labeled values pass through unchanged
            impl<T, L: typing_rules::lattice::Label> __Wrap<typing_rules::lattice::Labeled<T, L>> {
                fn __to_labeled(self) -> typing_rules::lattice::Labeled<T, L> {
                    self.0
                }
            }

            // Trait fallback: raw values get wrapped as Labeled<T, Public>
            trait __AsPublic {
                type Inner;
                fn __to_labeled(self) -> typing_rules::lattice::Labeled<Self::Inner, typing_rules::lattice::Public>;
            }
            impl<T> __AsPublic for __Wrap<T> {
                type Inner = T;
                fn __to_labeled(self) -> typing_rules::lattice::Labeled<T, typing_rules::lattice::Public> {
                    typing_rules::lattice::Labeled::new(self.0)
                }
            }

            // Step 2: Check FlowsTo and relabel.
            typing_rules::__relabel_checked::<_, _, #label>(__Wrap(#var).__to_labeled())
        }
    };
    TokenStream::from(expanded)
}

// =========================================================================
// // =========================================================================
// // 1. MACRO: pc_block!
// // Usage: pc_block! { stmt; stmt; ... }
// // =========================================================================

// =========================================================================
// 1. PARSING INPUT
// =========================================================================

struct PcBlockInput {
    start_label: syn::Type,
    block: syn::Block,
}

impl Parse for PcBlockInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse optional label: (Label)
        let start_label = if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            let ty: syn::Type = content.parse()?;
            ty
        } else {
            // Default to Public
            syn::parse_quote!(::typing_rules::lattice::Public)
        };

        let block: syn::Block = input.parse()?;
        Ok(PcBlockInput { start_label, block })
    }
}

#[proc_macro]
pub fn pc_block(tokens: TokenStream) -> TokenStream {
    let PcBlockInput { start_label, block } = parse_macro_input!(tokens as PcBlockInput);

    // 1. Generate EXECUTED Code (Runtime)
    //    - Rewrites assignments to 'secure_assign_with_pc'
    //    - Rewrites 'if' to track PC
    //    - Calls allowlisted functions normally
    let executed_code: TokenStream2 = expand_block(&block).into();

    // 2. Generate CHECKING Code (Compile-Time Safety)
    //    - Enforces Allowlist (errors on unknown functions)
    //    - Enforces InvisibleSideEffectFree (ISEF) on method calls
    //    - Checks Implicit Flow
    let checking_code: TokenStream2 = check_block(&block).into();

    let expanded = quote! {
        // ==========================================================
        // MACRO TRUST BOUNDARY:
        // The macro provides the `unsafe` context for `.unwrap()`
        // because it has verified the Information Flow statically!
        // In order to use vetted
        // ==========================================================
        unsafe {
            if true {
                // ── Panic message suppression ─────────────────────
                // Suppress panic messages inside pc_block! to prevent
                // secret data from leaking through panic payloads.
                //
                // Strategy:
                //   1. Save the current hook and install a silent one.
                //   2. A drop guard restores the hook on early `?`
                //      return (normal unwinding, no panic in progress).
                //   3. During panic unwinding, the guard skips
                //      restoration (set_hook is not safe to call while
                //      panicking); the silent hook stays and the panic
                //      message remains suppressed.
                //   4. On normal block completion, the guard drops
                //      and restores the original hook.
                let __pc_prev_hook = ::std::panic::take_hook();
                ::std::panic::set_hook(::std::boxed::Box::new(|_| {}));

                struct __PcPanicGuard(
                    ::std::option::Option<
                        ::std::boxed::Box<dyn ::std::ops::FnMut()>
                    >
                );
                impl ::std::ops::Drop for __PcPanicGuard {
                    fn drop(&mut self) {
                        // Only restore the hook on normal cleanup
                        // (e.g. early ? return). During panic unwinding
                        // set_hook is not safe to call, and the silent
                        // hook should remain active anyway.
                        if !::std::thread::panicking() {
                            if let ::std::option::Option::Some(mut f) = self.0.take() {
                                f();
                            }
                        }
                    }
                }
                let mut __pc_hook_opt = ::std::option::Option::Some(__pc_prev_hook);
                let __pc_panic_guard = __PcPanicGuard(
                    ::std::option::Option::Some(::std::boxed::Box::new(move || {
                        if let ::std::option::Option::Some(hook) = __pc_hook_opt.take() {
                            ::std::panic::set_hook(hook);
                        }
                    }))
                );

                // ── PC initialization and user code ───────────────
                let __pc_temp: #start_label = ::std::mem::zeroed();
                let __pc = __pc_temp;
                #executed_code

                // Normal exit — guard drops here, restoring the hook.
                // (Also drops on early ? return or panic unwinding.)
                drop(__pc_panic_guard);
            } else {
                // Initialize PC for Checking
                let __pc_temp: #start_label = ::std::mem::zeroed();
                let __pc = __pc_temp;
                let __pc_checker = ::typing_rules::implicit::PcIsef::new(&__pc);
                #checking_code
            }
        }
    };

    TokenStream::from(expanded)
}

// =========================================================================
// 2. EXECUTION LOGIC (Runtime Rewriter)
// =========================================================================

fn expand_expr(expr: &syn::Expr) -> TokenStream2 {
    if let syn::Expr::Call(call) = expr {
        let func_str = quote!(#call.func).to_string();
        if func_str.contains("unchecked_operation") {
            let inner = call.args.first().expect("unchecked_operation needs an argument");
            // Return the raw, un-transformed tokens of the argument
            return quote!(#inner);
        }
    }
    match expr {
        syn::Expr::If(i) => {
            // Check if this is `if let` pattern (e.g., if let Some(x) = expr)
            if let syn::Expr::Let(let_expr) = i.cond.as_ref() {
                let pat = &let_expr.pat;
                let scrutinee = check_expr(&let_expr.expr);
                let then_block = check_block(&i.then_branch);
                let else_block = match &i.else_branch {
                    Some((_, e)) => {
                        let e_trans = check_expr(e);
                        quote! { else { #e_trans } }
                    }
                    None => quote! {},
                };
                quote! {
                    if let #pat = #scrutinee {
                        #then_block
                    }
                    #else_block
                }
            } else {
                let cond_expr = check_expr(&i.cond);
                let then_block = check_block(&i.then_branch);

                // [CHANGE 3] Ensure Else block is explicitly generated
                let else_block = match &i.else_branch {
                    Some((_, e)) => {
                        let e_trans = check_expr(e);
                        // We must generate the 'else' block so types match the 'if' block
                        quote! { else {
                            let __pc = ::typing_rules::implicit::join_labels(__pc, __cond_label);
                            let __pc_checker = ::typing_rules::implicit::PcIsef::new(&__pc);
                            #e_trans
                        }}
                    }
                    None => quote! {}, // If user wrote no else, we generate no else
                };

                quote! {
                    {
                        // Inspect Condition
                        let (__cond_val, __cond_label) = ::typing_rules::implicit::inspect_condition(#cond_expr);
                        ::typing_rules::implicit::check_isef(__cond_val);

                        // If/Else structure mirrors the user's code exactly
                        if __cond_val {
                            let __pc = ::typing_rules::implicit::join_labels(__pc, __cond_label);
                            let __pc_checker = ::typing_rules::implicit::PcIsef::new(&__pc);
                            #then_block
                        }
                        #else_block
                    }
                }
            }
        }

        // [B] ASSIGNMENTS (Flow Check)
        syn::Expr::Assign(assign) => {
            let lhs = &assign.left;
            let rhs_expr = &assign.right;

            // Special case: Labeled::new(...) without turbofish — preserve type inference
            if let syn::Expr::Call(call) = rhs_expr.as_ref() {
                let func_str = quote!(#call.func).to_string();
                let is_labeled_new_no_turbofish = func_str.contains("Labeled") && func_str.contains("new") && !func_str.contains('<');
                if is_labeled_new_no_turbofish || is_call_to_allowlisted_function(call) {
                    let raw_func = &call.func;
                    let args: Vec<_> = call.args.iter().map(|a| expand_expr(a)).collect();
                    return quote! {
                        {
                            #lhs = #raw_func(#(#args),*);
                            ::typing_rules::implicit::pc_guard_assign(&mut #lhs, __pc);
                        }
                    };
                }
            }

            let rhs = expand_expr(rhs_expr);
            // Evaluate RHS into a temp first to avoid borrow conflict
            // (e.g. `row = row + 1` borrows row mutably and reads it simultaneously)
            quote! {
                {
                    let __temp_rhs = #rhs;
                    ::typing_rules::implicit::secure_assign_with_pc(&mut #lhs, __temp_rhs, __pc)
                }
            }
        }

        // [C] COMPOUND ASSIGNMENTS (x += y)
        syn::Expr::Binary(b) if is_compound_assign(&b.op) => {
            let lhs = &b.left;
            let rhs = expand_expr(&b.right);
            let op = &b.op;
            quote! {
                {
                    // Check implicit flow: PC <= LHS
                    // We use self-assignment as a trick to invoke the check
                    ::typing_rules::implicit::secure_assign_with_pc(&mut #lhs, #lhs, __pc);
                    #lhs #op #rhs
                }
            }
        }

        // [D] RECURSION
        syn::Expr::Block(b) => expand_block(&b.block),
        syn::Expr::While(w) => {
            let cond = expand_expr(&w.cond);
            let body = expand_block(&w.body);
            quote! { while { let (v, _) = ::typing_rules::implicit::inspect_condition(#cond); v } { #body } }
        }
        syn::Expr::ForLoop(f) => {
            let pat = &f.pat;
            let expr = expand_expr(&f.expr);
            let body = expand_block(&f.body);
            quote! { for #pat in #expr { #body } }
        }

        // [Expand] UNSAFE BLOCKS
        syn::Expr::Unsafe(expr_unsafe) => {
            // Wrap the syn::Block inside a syn::Expr::Block so our function can parse it
            let block_expr = syn::Expr::Block(syn::ExprBlock {
                attrs: expr_unsafe.attrs.clone(),
                label: None,
                block: expr_unsafe.block.clone(),
            });

            let inner = expand_expr(&block_expr);

            // Note: We use `unsafe #inner` instead of `unsafe { #inner }`
            // because a Block expression already provides its own curly braces!
            quote! { unsafe #inner }
        }

        // [E] FUNCTION CALLS (Pass-through for execution)
        // syn::Expr::Call(c) => {
        //     let args = comma_separate(c.args.iter().map(expand_expr));
        //     let func = &c.func;
        //     quote! { #func(#args) }
        // }
        // [Expand] FUNCTION CALLS
        syn::Expr::Call(call) => {
            let func = expand_expr(&call.func);
            let args: Vec<_> = call.args.iter().map(|arg| expand_expr(arg)).collect();
            let unwrapped_names: Vec<_> = (0..args.len()).map(|i| quote::format_ident!("__v{}", i)).collect();

            let inner_call = quote! { #func( #(#unwrapped_names),* ) };

            // FIX: Prevent double-wrapping Labeled::new
            let func_str = quote!(#func).to_string();
            let is_labeled_new = func_str.contains("Labeled") && func_str.contains("new");

            let mut expanded = if is_labeled_new || is_call_to_allowlisted_function(call) {
                quote! { #inner_call }
            } else {
                quote! { {
                    use ::typing_rules::implicit::PcCallResultFallback;
                    ::typing_rules::implicit::PcCallResult.wrap_result(#inner_call)
                } }
            };

            for (arg, name) in args.iter().zip(unwrapped_names.iter()).rev() {
                expanded = quote! { (#arg).__chain(|#name| { #expanded }) };
            }
            quote! { { use ::typing_rules::function_rewrite::SecureChain; #expanded } }
        }

        syn::Expr::MethodCall(m) => {
            let receiver = expand_expr(&m.receiver);
            let method_name = &m.method;
            let turbofish = &m.turbofish;
            let args: Vec<_> = m.args.iter().map(|arg| expand_expr(arg)).collect();
            // Pass method calls through directly without chain/wrap transformation.
            // Methods marked #[side_effect_free_attr] already handle Labeled types
            // and return Vetted<T> for safety verification.
            quote! { #receiver.#method_name #turbofish (#(#args),*) }
        }

        // [G] MACROS — format! is treated like fcall (chain args, wrap result)
        syn::Expr::Macro(m) => {
            let name = m.mac.path.segments.last().map(|s| s.ident.to_string());
            if name.as_deref() == Some("format") {
                // Parse format! arguments: format!("...", arg1, arg2, ...)
                let tokens = m.mac.tokens.clone();
                let parsed = syn::parse::Parser::parse2(syn::punctuated::Punctuated::<Expr, Token![,]>::parse_terminated, tokens).expect("format! should contain comma-separated expressions");
                let mut items = parsed.iter();
                let fmt_str = items.next().expect("format! needs a format string");
                let args: Vec<&Expr> = items.collect();

                if args.is_empty() {
                    // No args to chain — just wrap the result
                    quote! {
                        ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                            format!(#fmt_str)
                        )
                    }
                } else {
                    let unwrapped_names: Vec<_> = (0..args.len()).map(|i| format_ident!("__v{}", i)).collect();
                    let expanded_args: Vec<_> = args.iter().map(|a| expand_expr(a)).collect();

                    let mut expanded = quote! {
                        ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                            format!(#fmt_str, #(#unwrapped_names),*)
                        )
                    };

                    for (arg, name) in expanded_args.iter().zip(unwrapped_names.iter()).rev() {
                        expanded = quote! { (#arg).__chain(|#name| { #expanded }) };
                    }

                    quote! { { use ::typing_rules::function_rewrite::SecureChain; #expanded } }
                }
            } else {
                // Other macros: pass-through
                expr.to_token_stream()
            }
        }

        // [H] RETURN STATEMENTS (Pass-through)
        syn::Expr::Return(r) => {
            let val = r.expr.as_ref().map(|e| expand_expr(e));
            quote! { return #val; }
        }

        // UNARY OPERATORS (!, -) → recursively transform operand
        syn::Expr::Unary(u) => {
            let op = u.op;
            let expr = expand_expr(&u.expr);
            quote! { #op #expr }
        }

        // COMPARISON OPERATORS (==, !=) → labeled comparison preserving security labels
        syn::Expr::Binary(b) if is_comparison_op(&b.op) => {
            let lhs = expand_expr(&b.left);
            let rhs = expand_expr(&b.right);
            match &b.op {
                syn::BinOp::Eq(_) => quote! {
                    { use ::typing_rules::lattice::LabeledCmp; (#lhs).labeled_eq(#rhs) }
                },
                syn::BinOp::Ne(_) => quote! {
                    { use ::typing_rules::lattice::LabeledCmp; (#lhs).labeled_ne(#rhs) }
                },
                _ => unreachable!(),
            }
        }
        // LOGICAL OPERATORS (&&, ||) → labeled logical preserving security labels
        syn::Expr::Binary(b) if is_logical_op(&b.op) => {
            let lhs = expand_expr(&b.left);
            let rhs = expand_expr(&b.right);
            match &b.op {
                syn::BinOp::And(_) => quote! {
                    { use ::typing_rules::lattice::LabeledAnd; (#lhs).labeled_and(#rhs) }
                },
                syn::BinOp::Or(_) => quote! {
                    { use ::typing_rules::lattice::LabeledOr; (#lhs).labeled_or(#rhs) }
                },
                _ => unreachable!(),
            }
        }

        // [I] STRUCT LITERALS
        syn::Expr::Struct(s) => {
            let path = &s.path;
            let fields = s.fields.iter().map(|f| {
                let member = &f.member;
                let val = expand_expr(&f.expr);
                quote! { #member: #val }
            });
            let rest = s.rest.as_ref().map(|r| {
                let r = expand_expr(r);
                quote! { ..#r }
            });
            quote! { #path { #(#fields),* #rest } }
        }

        // [J] FALLBACK
        _ => expr.to_token_stream(),
    }
}

fn expand_block(input: &syn::Block) -> TokenStream2 {
    let stmts = input.stmts.iter().map(|stmt| match stmt {
        syn::Stmt::Expr(e, semi) => {
            let expanded = expand_expr(e);
            if semi.is_some() {
                quote! { #expanded; }
            } else {
                expanded
            }
        }
        syn::Stmt::Local(l) => {
            let pat = &l.pat;
            let init = l.init.as_ref().map(|init| {
                let ex = expand_expr(&init.expr);
                quote! { = #ex }
            });
            quote! { let #pat #init; }
        }
        syn::Stmt::Macro(m) => m.to_token_stream(),
        _ => stmt.to_token_stream(),
    });
    quote! { { #(#stmts)* } }
}

// =========================================================================
// 3. CHECKING
// =========================================================================

fn check_expr(expr: &syn::Expr) -> TokenStream2 {
    if let syn::Expr::Call(call) = expr {
        let func_str = quote!(#call.func).to_string();
        if func_str.contains("unchecked_operation") {
            let inner = call.args.first().expect("unchecked_operation needs an argument");
            // Return the raw, un-transformed tokens of the argument
            return quote!(#inner);
        }
    }

    match expr {
        // [A] IF STATEMENTS (Must track PC here too)
        syn::Expr::If(i) => {
            // Check if this is `if let` pattern (e.g., if let Some(x) = expr)
            if let syn::Expr::Let(let_expr) = i.cond.as_ref() {
                let pat = &let_expr.pat;
                let scrutinee = check_expr(&let_expr.expr);
                let then_block = check_block(&i.then_branch);
                let else_block = match &i.else_branch {
                    Some((_, e)) => {
                        let e_trans = check_expr(e);
                        quote! { else { #e_trans } }
                    }
                    None => quote! {},
                };
                quote! {
                    if let #pat = #scrutinee {
                        #then_block
                    }
                    #else_block
                }
            } else {
                let cond_expr = check_expr(&i.cond);
                let then_block = check_block(&i.then_branch);
                let else_block = match &i.else_branch {
                    Some((_, e)) => {
                        let e_trans = check_expr(e);
                        quote! { else {
                            let __pc = ::typing_rules::implicit::join_labels(__pc, __cond_label);
                            let __pc_checker = ::typing_rules::implicit::PcIsef::new(&__pc);
                            #e_trans
                        }}
                    }
                    None => quote! {},
                };

                quote! {
                    {
                        // Inspect Condition & Check Side Effects
                        let (__cond_val, __cond_label) = ::typing_rules::implicit::inspect_condition(#cond_expr);
                        ::typing_rules::implicit::check_isef(__cond_val);

                        if __cond_val {
                            let __pc = ::typing_rules::implicit::join_labels(__pc, __cond_label);
                            let __pc_checker = ::typing_rules::implicit::PcIsef::new(&__pc);
                            #then_block
                        }
                        #else_block
                    }
                }
            }
        }

        // [B] ASSIGNMENTS (Flow Check)
        syn::Expr::Assign(assign) => {
            let lhs = &assign.left;
            let rhs_expr = &assign.right;

            // Special case: if RHS is `Labeled::new(...)` WITHOUT a turbofish
            // label parameter, emit a raw assignment + PC-only guard.
            // This preserves type inference (L is inferred from the LHS),
            // while still enforcing PC ⊑ Dest.
            if let syn::Expr::Call(call) = rhs_expr.as_ref() {
                let func_str = quote!(#call.func).to_string();
                let is_labeled_new_no_turbofish = func_str.contains("Labeled") && func_str.contains("new") && !func_str.contains('<');
                if is_labeled_new_no_turbofish || is_call_to_allowlisted_function(call) {
                    let raw_func = &call.func;
                    let args: Vec<_> = call.args.iter().map(|a| check_expr(a)).collect();
                    return quote! {
                        {
                            #lhs = #raw_func(#(#args),*);
                            ::typing_rules::implicit::pc_guard_assign(&mut #lhs, __pc);
                        }
                    };
                }
            }

            let rhs = check_expr(rhs_expr);
            quote! {
                {
                    let __temp_rhs = #rhs;
                    ::typing_rules::implicit::secure_assign_with_pc(&mut #lhs, __temp_rhs, __pc)
                }
            }
        }

        // [C] COMPOUND ASSIGNMENTS
        syn::Expr::Binary(b) if is_compound_assign(&b.op) => {
            let lhs = &b.left;
            let rhs = check_expr(&b.right);
            let op = &b.op;
            quote! {
                {
                    let __temp_rhs = #rhs;
                    ::typing_rules::implicit::secure_assign_with_pc(&mut #lhs, __temp_rhs, __pc)
                }
            }
        }

        syn::Expr::Unsafe(expr_unsafe) => {
            // Wrap the syn::Block inside a syn::Expr::Block
            let block_expr = syn::Expr::Block(syn::ExprBlock {
                attrs: expr_unsafe.attrs.clone(),
                label: None,
                block: expr_unsafe.block.clone(),
            });

            let inner = check_expr(&block_expr);
            quote! { unsafe #inner }
        }

        // [D] FUNCTION CALLS (Verification Path)
        // [Check] FUNCTION CALLS
        syn::Expr::Call(call) => {
            let raw_func = &call.func;
            let func_str = quote!(#raw_func).to_string();
            let is_labeled_new = func_str.contains("Labeled") && func_str.contains("new");

            // The function's RESULT is already checked via PcCallResult.wrap_result(check_isef(...)).
            // Applying check_expr to the function path would wrap the function item
            // type in check_isef, which fails because fn items don't impl ISEF.
            let func = quote! { #raw_func };

            let args: Vec<_> = call.args.iter().map(|arg| check_expr(arg)).collect();
            let unwrapped_names: Vec<_> = (0..args.len()).map(|i| quote::format_ident!("__v{}", i)).collect();

            let raw_call = quote! { #func( #(#unwrapped_names),* ) };

            // Do not wrap the trusted execution in check_isef!
            let mut expanded = if is_labeled_new || is_call_to_allowlisted_function(call) {
                // Trust Labeled::new and allowlisted functions. Pass through!
                quote! { #raw_call }
            } else {
                // - Vetted<T> returns (from #[side_effect_free_attr]) → unwraps to T
                // - Raw T returns → wraps in Labeled<T, Public>
                let checked_call = quote! { ::typing_rules::implicit::check_isef(#raw_call) };
                quote! { {
                    use ::typing_rules::implicit::PcCallResultFallback;
                    ::typing_rules::implicit::PcCallResult.wrap_result(#checked_call)
                } }
            };

            for (arg, name) in args.iter().zip(unwrapped_names.iter()).rev() {
                expanded = quote! { (#arg).__chain(|#name| { #expanded }) };
            }
            quote! { { use ::typing_rules::function_rewrite::SecureChain; #expanded } }
        }

        // [E] METHOD CALLS (Side-Effect Check)
        syn::Expr::MethodCall(m) => {
            // Pass method calls through directly without transformation.
            // Safety is enforced by the type system: #[side_effect_free_attr]
            // methods return Vetted<T> which proves they are side-effect free.
            m.to_token_stream()
        }

        // [F] RECURSION
        syn::Expr::Block(b) => check_block(&b.block),
        syn::Expr::While(w) => {
            let cond = check_expr(&w.cond);
            let body = check_block(&w.body);
            quote! { while { let (v,_) = ::typing_rules::implicit::inspect_condition(#cond); v } { #body } }
        }
        syn::Expr::ForLoop(f) => {
            let pat = &f.pat;
            let expr = check_expr(&f.expr);
            let body = check_block(&f.body);
            quote! { for #pat in #expr { #body } }
        }

        // [G] BASIC EXPRESSIONS
        syn::Expr::Paren(p) => {
            let inner = check_expr(&p.expr);
            quote! { (#inner) }
        }
        // COMPARISON OPERATORS (==, !=) → labeled comparison preserving security labels
        syn::Expr::Binary(b) if is_comparison_op(&b.op) => {
            let lhs = check_expr(&b.left);
            let rhs = check_expr(&b.right);
            match &b.op {
                syn::BinOp::Eq(_) => quote! {
                    { use ::typing_rules::lattice::LabeledCmp; (#lhs).labeled_eq(#rhs) }
                },
                syn::BinOp::Ne(_) => quote! {
                    { use ::typing_rules::lattice::LabeledCmp; (#lhs).labeled_ne(#rhs) }
                },
                _ => unreachable!(),
            }
        }
        // LOGICAL OPERATORS (&&, ||) → labeled logical preserving security labels
        syn::Expr::Binary(b) if is_logical_op(&b.op) => {
            let lhs = check_expr(&b.left);
            let rhs = check_expr(&b.right);
            match &b.op {
                syn::BinOp::And(_) => quote! {
                    { use ::typing_rules::lattice::LabeledAnd; (#lhs).labeled_and(#rhs) }
                },
                syn::BinOp::Or(_) => quote! {
                    { use ::typing_rules::lattice::LabeledOr; (#lhs).labeled_or(#rhs) }
                },
                _ => unreachable!(),
            }
        }
        syn::Expr::Binary(b) => {
            let lhs = check_expr(&b.left);
            let rhs = check_expr(&b.right);
            let op = b.op;
            quote! { #lhs #op #rhs }
        }
        syn::Expr::Unary(u) => {
            let op = u.op;
            let expr = check_expr(&u.expr);
            quote! { #op #expr }
        }
        // Reading a variable has no side effect — ISEF check is for function calls, not reads.
        syn::Expr::Path(p) => p.to_token_stream(),
        syn::Expr::Lit(l) => l.into_token_stream(),
        syn::Expr::Field(f) => {
            let base = check_expr(&f.base);
            let member = &f.member;
            quote! { (#base).#member }
        }
        syn::Expr::Index(idx) => {
            let expr = check_expr(&idx.expr);
            let index = check_expr(&idx.index);
            quote! { #expr[#index] }
        }

        // [H] MACROS — format! is side-effect-free; reject others under non-Public PC
        syn::Expr::Macro(m) => {
            let name = m.mac.path.segments.last().map(|s| s.ident.to_string());
            let name_str = name.as_deref().unwrap_or("");
            match name_str {
                "fcall" | "mcall" | "relabel" | "pc_block" | "panic" => m.to_token_stream(),
                "format" => {
                    // Transform format! the same way as expand_expr: chain args, wrap result.
                    // Needed because the checking branch must still compile (Labeled has no Display).
                    let tokens = m.mac.tokens.clone();
                    let parsed = syn::parse::Parser::parse2(syn::punctuated::Punctuated::<Expr, Token![,]>::parse_terminated, tokens).expect("format! should contain comma-separated expressions");
                    let mut items = parsed.iter();
                    let fmt_str = items.next().expect("format! needs a format string");
                    let args: Vec<&Expr> = items.collect();

                    if args.is_empty() {
                        quote! {
                            ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                                format!(#fmt_str)
                            )
                        }
                    } else {
                        let unwrapped_names: Vec<_> = (0..args.len()).map(|i| format_ident!("__v{}", i)).collect();
                        let checked_args: Vec<_> = args.iter().map(|a| check_expr(a)).collect();

                        let mut expanded = quote! {
                            ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
                                format!(#fmt_str, #(#unwrapped_names),*)
                            )
                        };

                        for (arg, name) in checked_args.iter().zip(unwrapped_names.iter()).rev() {
                            expanded = quote! { (#arg).__chain(|#name| { #expanded }) };
                        }

                        quote! { { use ::typing_rules::function_rewrite::SecureChain; #expanded } }
                    }
                }
                _ => {
                    let mac = &m.mac;
                    quote! {
                        {
                            use ::typing_rules::implicit::PcIsefFallback;
                            __pc_checker.reject_side_effecting_macro(#mac)
                        }
                    }
                }
            }
        }

        // [I] RETURN STATEMENTS (Pass-through)
        syn::Expr::Return(r) => {
            let val = r.expr.as_ref().map(|e| check_expr(e));
            quote! { return #val; }
        }

        // [NEW 1] ARRAYS: [a, b, c]
        syn::Expr::Array(a) => {
            let elems = comma_separate(a.elems.iter().map(check_expr));
            quote! { [#elems] }
        }

        // [NEW 2] REFERENCES: &x or &mut x
        syn::Expr::Reference(r) => {
            let e = check_expr(&r.expr);
            if r.mutability.is_some() {
                quote! { &mut #e }
            } else {
                quote! { &#e }
            }
        }

        // (format!/cfg! etc. handled by the [H] MACROS arm above)

        // STRUCT LITERALS — pass through as-is; the Rust type system
        // enforces IFC constraints via the Labeled field types.
        syn::Expr::Struct(_) => expr.to_token_stream(),

        _ => {
            // If we don't recognize it, it might be unsafe.
            // We can emit a compile error or just try to pass it through.
            // For safety, let's error on unknown syntax in the checked block.
            let msg = format!("Syntax not supported in pc_block (checked path): {:?}", quote! {#expr}.to_string());
            quote! { compile_error!(#msg); }
        }
    }
}

fn check_block(input: &syn::Block) -> TokenStream2 {
    let stmts = input.stmts.iter().map(|stmt| match stmt {
        syn::Stmt::Expr(e, semi) => {
            let checked = check_expr(e);
            if semi.is_some() {
                quote! { #checked; }
            } else {
                checked
            }
        }
        syn::Stmt::Local(l) => {
            let pat = &l.pat;
            let init = l.init.as_ref().map(|init| {
                let ex = check_expr(&init.expr);
                quote! { = #ex }
            });
            quote! { let #pat #init; }
        }
        syn::Stmt::Macro(m) => {
            let name = m.mac.path.segments.last().map(|s| s.ident.to_string());
            let name_str = name.as_deref().unwrap_or("");
            // Safe macros: cfg!, and our own IFC macros which enforce their own checking.
            match name_str {
                "fcall" | "mcall" | "relabel" | "pc_block" | "panic" | "format" => m.to_token_stream(),
                _ => {
                    // Side-effecting macros (println!, panic!, etc.) are rejected
                    // under non-Public PC via MacroSideEffectFree bound.
                    let mac = &m.mac;
                    let semi = &m.semi_token;
                    quote! {
                        {
                            use ::typing_rules::implicit::PcIsefFallback;
                            __pc_checker.reject_side_effecting_macro(#mac);
                        } #semi
                    }
                }
            }
        }
        _ => stmt.to_token_stream(),
    });
    quote! { { #(#stmts)* } }
}

// =========================================================================
// 2. SIDE EFFECT FREE ATTRIBUTE (The New Addition)
// =========================================================================

#[proc_macro_attribute]
pub fn side_effect_free_attr(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::Item);

    match input {
        // [A] Mark a Function as Safe
        syn::Item::Fn(mut func) => {
            // 1. Extract the original return type
            let orig_return_type = match &func.sig.output {
                syn::ReturnType::Default => quote! { () },
                syn::ReturnType::Type(_, ty) => quote! { #ty },
            };

            // 2. Change the signature to return Vetted<T>
            func.sig.output = syn::parse_quote! {
                -> ::typing_rules::implicit::Vetted<#orig_return_type>
            };

            let orig_block = &func.block;

            // 3. THE CLOSURE TRAP:
            // Wrap the body in a closure so early `return;` statements
            // exit the closure instead of skipping the Vetted wrapper!
            func.block = syn::parse_quote! {
                {
                    let mut __cocoon_inner = || -> #orig_return_type #orig_block;

                    unsafe {
                        ::typing_rules::implicit::Vetted::wrap( __cocoon_inner() )
                    }
                }
            };

            quote! { #func }.into()
        }

        // [B] Mark a Struct as Safe (Auto-Derive InvisibleSideEffectFree)
        // Usage: #[side_effect_free_attr] struct MySafeData { ... }
        syn::Item::Struct(s) => {
            let name = &s.ident;
            let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();

            // Generate the safety trait implementation
            let expanded = quote! {
                #s

                unsafe impl #impl_generics ::typing_rules::implicit::InvisibleSideEffectFree for #name #ty_generics #where_clause {
                     // We could optionally add checks for fields here
                }
            };
            expanded.into()
        }

        _ => {
            // Pass through other items
            let item = input.to_token_stream();
            quote! { #item }.into()
        }
    }
}

// =========================================================================
// 4. HELPERS & ALLOWLIST
// =========================================================================

fn make_check_safe(e: TokenStream2) -> TokenStream2 {
    quote! {
        // { ::typing_rules::implicit::check_isef(#e) }
        ::typing_rules::implicit::check_isef(#e)
    }
}

fn is_compound_assign(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

fn is_comparison_op(op: &syn::BinOp) -> bool {
    matches!(op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_))
}

fn is_logical_op(op: &syn::BinOp) -> bool {
    matches!(op, syn::BinOp::And(_) | syn::BinOp::Or(_))
}

fn comma_separate<T: Iterator<Item = TokenStream2>>(ts: T) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for (i, t) in ts.enumerate() {
        if i > 0 {
            tokens.extend(quote! {,});
        }
        tokens.extend(t);
    }
    tokens
}

// THE ALLOWLIST (Ported from Cocoon's lib.rs)
fn is_call_to_allowlisted_function(call: &syn::ExprCall) -> bool {
    let allowed_functions = HashSet::from([
        // [Cocoon Standard Primitives]
        "char::is_digit".to_string(),
        "core::primitive::str::len".to_string(),
        "std::clone::Clone::clone".to_string(),
        "std::cmp::min".to_string(),
        "std::cmp::max".to_string(),
        "std::fs::File::open".to_string(),
        "std::iter::Iterator::next".to_string(),
        "std::iter::Iterator::take".to_string(),
        "std::iter::zip".to_string(),
        "std::option::Option::Some".to_string(),
        "std::option::Option::unwrap".to_string(),
        "std::string::String::clear".to_string(),
        "std::string::String::from".to_string(),
        "std::string::String::len".to_string(),
        "std::time::Instant::now".to_string(),
        "std::vec::Vec::new".to_string(),
        "std::vec::Vec::push".to_string(),
        "std::vec::Vec::len".to_string(),
        "std::vec::Vec::with_capacity".to_string(),
        "std::collections::HashMap::get".to_string(),
        "std::collections::HashMap::insert".to_string(),
        "std::collections::HashSet::insert".to_string(),
        "str::to_string".to_string(),
        "usize::to_string".to_string(),
        // [Safe Ops from Lattice]
        "typing_rules::lattice::safe_add".to_string(),
        "typing_rules::lattice::safe_sub".to_string(),
        "Labeled::new".to_string(),
        "typing_rules::lattice::Labeled::new".to_string(),
        // Add others as needed...
    ]);

    if let syn::Expr::Path(path_expr) = &*call.func {
        let mut path_str = quote! {#path_expr}.to_string();
        path_str.retain(|c| !c.is_whitespace());
        allowed_functions.contains(&path_str)
    } else {
        false
    }
}
