//! IFC-aware I/O wrappers for the standard library.
//!
//! Provides labeled handles for all external system boundaries that
//! enforce FlowsTo lattice bounds at compile time.
//!
//! - [`SecureFile`] — wraps `std::fs` (read/write files)
//! - [`SecureStream`] — wraps `std::net::TcpStream` (network I/O)
//! - [`SecureCommand`] — wraps `std::process::Command` (subprocess I/O)
//! - [`secure_print`] / [`secure_eprint`] — wraps stdout/stderr output

use crate::lattice::{FlowsTo, Label, Labeled};
use std::marker::PhantomData;
use std::path::PathBuf;

/// A file handle labeled with security level `L`.
///
/// Wraps `std::fs` free functions (`read_to_string`, `read`, `write`)
/// with FlowsTo enforcement on writes.
pub struct SecureFile<L: Label> {
    path: PathBuf,
    _label: PhantomData<L>,
}

impl<L: Label> SecureFile<L> {
    /// Create a labeled file handle. Returns `Labeled<Self, L>`.
    pub fn open(path: PathBuf) -> Labeled<Self, L> {
        Labeled::new(SecureFile {
            path,
            _label: PhantomData,
        })
    }

    /// Wraps `std::fs::read_to_string`.
    pub fn read_to_string(&self) -> std::io::Result<String> {
        std::fs::read_to_string(&self.path)
    }

    /// Wraps `std::fs::read`.
    pub fn read(&self) -> std::io::Result<Vec<u8>> {
        std::fs::read(&self.path)
    }

    /// Wraps `std::fs::write` with `Src: FlowsTo<L>` (no-write-down).
    pub fn write<Src: Label + FlowsTo<L>>(
        &self,
        data: &Labeled<String, Src>,
    ) -> std::io::Result<()> {
        std::fs::write(&self.path, data.declassify_ref())
    }

    /// Wraps `std::fs::write` for byte data with `Src: FlowsTo<L>`.
    pub fn write_bytes<Src: Label + FlowsTo<L>>(
        &self,
        data: &Labeled<Vec<u8>, Src>,
    ) -> std::io::Result<()> {
        std::fs::write(&self.path, data.declassify_ref())
    }
}

// =========================================================================
// SecureStream — labeled network stream
// =========================================================================

/// A TCP stream labeled with security level `L`.
///
/// Wraps `std::net::TcpStream` with FlowsTo enforcement on writes.
pub struct SecureStream<L: Label> {
    stream: std::net::TcpStream,
    _label: PhantomData<L>,
}

impl<L: Label> SecureStream<L> {
    /// Wraps `TcpStream::connect`. Returns `Labeled<Self, L>`.
    pub fn connect(addr: &str) -> std::io::Result<Labeled<Self, L>> {
        let stream = std::net::TcpStream::connect(addr)?;
        Ok(Labeled::new(SecureStream {
            stream,
            _label: PhantomData,
        }))
    }

    /// Wraps `TcpStream::read` via `std::io::Read`. Data inherits label L via mcall.
    pub fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        use std::io::Read;
        self.stream.read(buf)
    }

    /// Wraps `std::io::read_to_string` on the stream.
    pub fn read_to_string(&mut self) -> std::io::Result<String> {
        use std::io::Read;
        let mut s = String::new();
        self.stream.read_to_string(&mut s)?;
        Ok(s)
    }

    /// Wraps `TcpStream::write_all` with `Src: FlowsTo<L>` (no-write-down).
    pub fn write<Src: Label + FlowsTo<L>>(
        &mut self,
        data: &Labeled<String, Src>,
    ) -> std::io::Result<()> {
        use std::io::Write;
        self.stream.write_all(data.declassify_ref().as_bytes())
    }

    /// Wraps `TcpStream::write_all` for byte data with `Src: FlowsTo<L>`.
    pub fn write_bytes<Src: Label + FlowsTo<L>>(
        &mut self,
        data: &Labeled<Vec<u8>, Src>,
    ) -> std::io::Result<()> {
        use std::io::Write;
        self.stream.write_all(data.declassify_ref())
    }
}

// =========================================================================
// SecureCommand — labeled process spawning
// =========================================================================

/// A process command labeled with security level `L`.
///
/// Wraps `std::process::Command`. Arguments carrying labeled data
/// must satisfy `Src: FlowsTo<L>` — prevents exfiltrating secrets
/// via subprocess arguments or stdin.
pub struct SecureCommand<L: Label> {
    cmd: std::process::Command,
    _label: PhantomData<L>,
}

impl<L: Label> SecureCommand<L> {
    /// Wraps `Command::new`.
    pub fn new(program: &str) -> Labeled<Self, L> {
        Labeled::new(SecureCommand {
            cmd: std::process::Command::new(program),
            _label: PhantomData,
        })
    }

    /// Wraps `Command::arg` with `Src: FlowsTo<L>`.
    pub fn arg<Src: Label + FlowsTo<L>>(
        &mut self,
        arg: &Labeled<String, Src>,
    ) -> &mut Self {
        self.cmd.arg(arg.declassify_ref());
        self
    }

    /// Add a public (unlabeled) argument.
    pub fn arg_public(&mut self, arg: &str) -> &mut Self {
        self.cmd.arg(arg);
        self
    }

    /// Wraps `Command::output`. Output inherits label L via mcall.
    pub fn output(&mut self) -> std::io::Result<std::process::Output> {
        self.cmd.output()
    }

    /// Wraps `Command::status`.
    pub fn status(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.cmd.status()
    }
}

// =========================================================================
// Stdout / Stderr — labeled console output
// =========================================================================

/// Write labeled data to stdout. Requires `Src: FlowsTo<Public>` —
/// only public data can be printed to the console.
pub fn secure_print<Src: Label + FlowsTo<crate::lattice::Public>>(
    data: &Labeled<String, Src>,
) {
    print!("{}", data.declassify_ref());
}

/// Write labeled data to stdout with newline. Requires `Src: FlowsTo<Public>`.
pub fn secure_println<Src: Label + FlowsTo<crate::lattice::Public>>(
    data: &Labeled<String, Src>,
) {
    println!("{}", data.declassify_ref());
}

/// Write labeled data to stderr. Requires `Src: FlowsTo<Public>`.
pub fn secure_eprintln<Src: Label + FlowsTo<crate::lattice::Public>>(
    data: &Labeled<String, Src>,
) {
    eprintln!("{}", data.declassify_ref());
}
