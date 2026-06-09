// pub extern crate typing_rules;
// #[macro_export]
// macro_rules! fcall {
//     // =========================================================================
//     // 1. ENTRY POINT
//     // User writes: fcall!( std::fs::write(a, b) )
//     // We initialize the "Done" list as empty: ()
//     // =========================================================================
//     ( $($func:ident)::+ ( $($args:expr),* ) ) => {
//         $crate::fcall!( @recurse ($($func)::+) () ($($args),*) )
//     };

//     // =========================================================================
//     // 2. RECURSIVE STEP (Process one argument)
//     // Matches: @recurse (func) (done...) (head, tail...)
//     // Logic: Unwrap 'head' -> Move to 'done' -> Recurse on 'tail'
//     // =========================================================================
//     ( @recurse $path:tt ($($done:expr),*) ($head:expr, $($tail:expr),+) ) => {
//         {
//             // "Use crate on top": We import the trait strictly inside this block
//             // so it works anywhere this macro is called.
//             use $crate::typing_rules::function_rewrite::SecureChain;

//             $head.chain(|val| {
//                 $crate::fcall!( @recurse $path ($($done,)* val) ($($tail),+) )
//             })
//         }
//     };

//     // =========================================================================
//     // 3. BASE STEP (Last argument)
//     // Matches: @recurse (func) (done...) (head)
//     // Logic: Unwrap 'head' -> Move to 'done' -> EXECUTE
//     // =========================================================================
//     ( @recurse $path:tt ($($done:expr),*) ($head:expr) ) => {
//         {
//             use $crate::typing_rules::function_rewrite::SecureChain;

//             $head.chain(|val| {
//                 $crate::fcall!( @exec $path ($($done,)* val) )
//             })
//         }
//     };

//     // =========================================================================
//     // 4. EDGE CASE (No arguments)
//     // Matches: fcall!( func() )
//     // =========================================================================
//     ( @recurse $path:tt () () ) => {
//         $crate::fcall!( @exec $path () )
//     };

//     // =========================================================================
//     // 5. EXECUTION STEP
//     // Matches: @exec (func) (args...)
//     // Logic: Call the raw function and wrap the result in Public Label
//     // =========================================================================
//     ( @exec ($($func:ident)::+) ($($args:expr),*) ) => {
//         $crate::typing_rules::Labeled::<_, $crate::typing_rules::Public>::new(
//             $($func)::+ ( $($args),* )
//         )
//     };
// }

// #[macro_export]
// macro_rules! relabel {
//     // Usage: relabel!( variable, TargetLabelType )
//     ( $var:expr, $dest_label:ty ) => {{
//         // We specify the generic type <$dest_label> explicitly
//         $var.relabel::<$dest_label>()
//     }};
// }

// #[proc_macro]
// pub fn fcall(input: TokenStream) -> TokenStream {
//     let call = parse_macro_input!(input as ExprCall);
//     let func = call.func;
//     let args = call.args;

//     let arg_count = args.len();
//     let unwrapped_names: Vec<_> = (0..arg_count).map(|i| format_ident!("__v{}", i)).collect();

//     let mut expanded = quote! {
//         ::typing_rules::lattice::Labeled::<_, ::typing_rules::lattice::Public>::new(
//             #func( #(#unwrapped_names),* )
//         )
//     };

//     // [FIX] Add parentheses around #arg
//     for (arg, name) in args.iter().zip(unwrapped_names.iter()).rev() {
//         expanded = quote! {
//             (#arg).chain(|#name| {
//                 #expanded
//             })
//         };
//     }
//     // let final_output = quote! {
//     //     {
//     //         use ::typing_rules::function_rewrite::SecureChain;

//     //         // We generated the Labeled wrapper...
//     //         let labeled_result = #expanded;

//     //         // ...now we automatically unwrap it because it's Public!
//     //         labeled_result.value
//     //     }
//     // };

//     // TokenStream::from(final_output)
//     let final_output = quote! {
//         {
//             use ::typing_rules::function_rewrite::SecureChain;
//             #expanded
//         }
//     };

//     TokenStream::from(final_output)
// }

// #[proc_macro]
// pub fn pc_block(tokens: TokenStream) -> TokenStream {
//     // 1. Parse input as a standard Block { ... }
//     let blk = parse_macro_input!(tokens as syn::Block);

//     // 2. Generate EXECUTED code (Runs at runtime)
//     //    We just expand it normally.
//     let executed_code: TokenStream2 = expand_block(&blk).into();

//     // 3. Generate CHECKING code (Runs at compile-time via 'dead code')
//     //    This recurses into every expression to enforce security rules.
//     let checking_code: TokenStream2 = check_block(&blk).into();

//     // 4. Combine them
//     //    'if true' runs the real code. 'else' allows the compiler to check the security constraints.
//     quote! {
//         if true {
//             #executed_code
//         } else {
//             #checking_code
//         }
//     }
//     .into()
// }

// // =========================================================================
// // 2. CHECKING LOGIC (The Security Enforcer)
// // =========================================================================

// fn check_expr(expr: &syn::Expr, do_check: bool) -> TokenStream2 {
//     match expr {
//         // --- A. ASSIGNMENTS (Fine-Grained Flow Check) ---
//         // Catch: x = y
//         // Action: Enforce that we are allowed to write to 'x' in this context.
//         syn::Expr::Assign(assign) => {
//             let lhs = &assign.left;
//             // Recursively check the RHS for side effects
//             let rhs = check_expr(&assign.right, true);

//             // Inject the write check:
//             quote! {
//                 {
//                     ::typing_rules::implicit::check_write_allowed_generic(&mut #lhs);
//                     #rhs
//                 }
//             }
//         }

//         // --- B. BINARY OPERATIONS & COMPOUND ASSIGNMENTS (Syn 2.0) ---
//         // Catch: x += y, x + y, etc.
//         syn::Expr::Binary(b) => {
//             match b.op {
//                 // 1. Compound Assignments (Write Check Needed)
//                 syn::BinOp::AddAssign(_)
//                 | syn::BinOp::SubAssign(_)
//                 | syn::BinOp::MulAssign(_)
//                 | syn::BinOp::DivAssign(_)
//                 | syn::BinOp::RemAssign(_)
//                 | syn::BinOp::BitXorAssign(_)
//                 | syn::BinOp::BitAndAssign(_)
//                 | syn::BinOp::BitOrAssign(_)
//                 | syn::BinOp::ShlAssign(_)
//                 | syn::BinOp::ShrAssign(_) => {
//                     let lhs = &b.left;
//                     let rhs = check_expr(&b.right, true);
//                     let op = &b.op;

//                     quote! {
//                         {
//                             ::typing_rules::implicit::check_write_allowed_generic(&mut #lhs);
//                             #lhs #op #rhs
//                         }
//                     }
//                 }

//                 // 2. Standard Binary Ops (Read Only)
//                 _ => {
//                     let lhs = check_expr(&b.left, false);
//                     let rhs = check_expr(&b.right, false);
//                     let op = b.op;
//                     quote! { #lhs #op #rhs }
//                 }
//             }
//         }

//         // --- C. FUNCTION CALLS (Allowlist) ---
//         syn::Expr::Call(call) => {
//             let args = comma_separate(call.args.iter().map(|arg| check_expr(arg, true)));
//             let func = &call.func;

//             if is_call_to_allowlisted_function(call) {
//                 // If allowed, ensure result is Side-Effect Free
//                 make_check_safe(quote! { #func(#args) }, do_check)
//             } else {
//                 // If NOT allowed, block it.
//                 // Note: You can change this to a warning or rely on the trait check failing later.
//                 quote! {
//                     compile_error!("Function call not in allowlist. Unsafe for implicit flow.");
//                 }
//             }
//         }

//         // --- D. METHOD CALLS (Side-Effect Checks) ---
//         syn::Expr::MethodCall(method) => {
//             let receiver = check_expr(&method.receiver, true);
//             let args = comma_separate(method.args.iter().map(|arg| check_expr(arg, true)));
//             let name = &method.method;
//             let turbofish = &method.turbofish;

//             // Wrap method calls in `check_ISEF` to ensure no side effects (e.g. I/O).
//             make_check_safe(quote! { (#receiver).#name #turbofish(#args) }, do_check)
//         }

//         // --- E. CONTROL FLOW (Recursion) ---
//         syn::Expr::Block(b) => check_block(&b.block),
//         syn::Expr::If(i) => {
//             let cond = check_expr(&i.cond, true);
//             let then_block = check_block(&i.then_branch);
//             let else_block = match &i.else_branch {
//                 Some((_, e)) => check_expr(e, true),
//                 None => quote! {},
//             };
//             quote! { if #cond { #then_block } else { #else_block } }
//         }
//         syn::Expr::While(w) => {
//             let cond = check_expr(&w.cond, true);
//             let body = check_block(&w.body);
//             quote! { while #cond { #body } }
//         }
//         syn::Expr::ForLoop(f) => {
//             let pat = &f.pat;
//             let expr = check_expr(&f.expr, true);
//             let body = check_block(&f.body);
//             quote! { for #pat in #expr { #body } }
//         }

//         // --- F. BASIC EXPRESSIONS ---
//         syn::Expr::Paren(p) => {
//             let inner = check_expr(&p.expr, do_check);
//             quote! { (#inner) }
//         }
//         syn::Expr::Unary(u) => {
//             let op = u.op;
//             let expr = check_expr(&u.expr, false);
//             quote! { #op #expr }
//         }
//         syn::Expr::Lit(l) => l.into_token_stream(),
//         syn::Expr::Path(p) => {
//             // Variable Read: Check if type is safe (Side-Effect Free)
//             let p_tok = p.to_token_stream();
//             make_check_safe(p_tok, do_check)
//         }
//         syn::Expr::Field(f) => {
//             let base = check_expr(&f.base, true);
//             let member = &f.member;
//             quote! { (#base).#member }
//         }
//         syn::Expr::Index(idx) => {
//             let expr = check_expr(&idx.expr, false);
//             let index = check_expr(&idx.index, true);
//             quote! { #expr[#index] }
//         }
//         syn::Expr::Array(a) => {
//             let elems = comma_separate(a.elems.iter().map(|e| check_expr(e, true)));
//             quote! { [#elems] }
//         }
//         syn::Expr::Reference(r) => {
//             let expr = check_expr(&r.expr, false);
//             // We don't check the reference creation itself, but we check what it points to above
//             if r.mutability.is_some() {
//                 quote! { &mut #expr }
//             } else {
//                 quote! { &#expr }
//             }
//         }

//         // --- G. UNSUPPORTED ---
//         _ => {
//             let msg = format!(
//                 "Syntax not supported in pc_block: {:?}",
//                 quote! {#expr}.to_string()
//             );
//             quote! { compile_error!(#msg); }
//         }
//     }
// }

// fn check_block(input: &syn::Block) -> TokenStream2 {
//     let stmts = input.stmts.iter().map(|stmt| match stmt {
//         syn::Stmt::Local(l) => {
//             // let x = ...
//             let pat = &l.pat;
//             let init = l.init.as_ref().map(|init| {
//                 let e = check_expr(&init.expr, true);
//                 quote! { = #e }
//             });
//             quote! { let #pat #init; }
//         }
//         syn::Stmt::Expr(e, _) => check_expr(e, true),
//         syn::Stmt::Macro(m) => m.to_token_stream(),
//         _ => stmt.to_token_stream(),
//     });
//     quote! { { #(#stmts)* } }
// }

// // =========================================================================
// // 3. EXECUTION LOGIC (Pass-through)
// // =========================================================================

// fn expand_block(input: &syn::Block) -> TokenStream2 {
//     // Just run the code as-is. Checks are done in the dead-code branch.
//     input.to_token_stream()
// }

// // =========================================================================
// // 4. HELPERS
// // =========================================================================

// fn make_check_safe(e: TokenStream2, do_check: bool) -> TokenStream2 {
//     if do_check {
//         // Enforce InvisibleSideEffectFree trait from your implicit module
//         quote! {
//             { ::typing_rules::implicit::check_ISEF(#e) }
//         }
//     } else {
//         e
//     }
// }

// fn is_call_to_allowlisted_function(call: &syn::ExprCall) -> bool {
//     // You can expand this list based on what functions you trust
//     let allowed_functions = HashSet::from([
//         // Standard Primitives
//         "std::cmp::min".to_string(),
//         "std::cmp::max".to_string(),
//         "std::clone::Clone::clone".to_string(),
//         "std::option::Option::unwrap".to_string(),
//         "std::option::Option::Some".to_string(),
//         "std::vec::Vec::new".to_string(),
//         "std::vec::Vec::push".to_string(),
//         "std::vec::Vec::len".to_string(),
//         // YOUR SAFE OPS (Example)
//         "typing_rules::lattice::safe_add".to_string(),
//     ]);

//     if let syn::Expr::Path(path_expr) = &*call.func {
//         let mut path_str = quote! {#path_expr}.to_string();
//         path_str.retain(|c| !c.is_whitespace());
//         allowed_functions.contains(&path_str)
//     } else {
//         false
//     }
// }

// fn comma_separate<T: Iterator<Item = TokenStream2>>(ts: T) -> TokenStream2 {
//     let mut tokens = TokenStream2::new();
//     for (i, t) in ts.enumerate() {
//         if i > 0 {
//             tokens.extend(quote! {,});
//         }
//         tokens.extend(t);
//     }
//     tokens
// }

// #[proc_macro]
// pub fn pc_block(tokens: TokenStream) -> TokenStream {
//     let PcBlockInput { start_label, block } = parse_macro_input!(tokens as PcBlockInput);

//     let executed_code: TokenStream2 = expand_block(&block).into();
//     let checking_code: TokenStream2 = check_block(&block).into();

//     quote! {
//         if true {
//             // EXECUTION PATH
//             // Initialize PC using the generic type P as a type argument
//             let __pc = ::typing_rules::lattice::PcContext::<#start_label>::new();
//             #executed_code
//         } else {
//             // CHECKING PATH (Dead Code)
//             let __pc = ::typing_rules::lattice::PcContext::<#start_label>::new();
//             #checking_code
//         }
//     }
//     .into()
// }
