# Rust notes

## Ownership
Every value has exactly one owner. When the owner goes out of scope the value is dropped. Assigning a heap value to another variable moves it, so the original binding can no longer be used. Types that implement Copy, like integers, are copied instead of moved.

## Borrowing
You can have any number of shared references (&T) or exactly one mutable reference (&mut T), but not both at the same time. The borrow checker enforces this at compile time, which is how Rust prevents data races without a garbage collector.

## Lifetimes
Lifetimes are annotations that tell the compiler how long references are valid relative to each other. Most are elided. You write them explicitly when a function returns a reference derived from one of several inputs, e.g. fn longest<'a>(x: &'a str, y: &'a str) -> &'a str.

## Error handling
Rust has no exceptions. Recoverable errors use Result<T, E> and the ? operator propagates them. Unrecoverable bugs use panic!. I use thiserror for library error types and anyhow in binaries.

## Async
async fn returns a Future that does nothing until polled. You need a runtime such as tokio to drive futures. Holding a std Mutex guard across an .await point is a common bug; use tokio's Mutex or restructure the code.
