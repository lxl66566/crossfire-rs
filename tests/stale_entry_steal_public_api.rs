//! Minimal public-API repro of the lost wakeup in crossfire 3.1.22: a
//! receiver departing by timeout can win its abandon CAS inside `wake()`'s
//! read->store gap; the late unconditional `store(Woken)` stamps the
//! departed node, `fire()` reports the coin delivered and stops popping,
//! and waiters parked behind never wake — the item strands in the buffer
//! permanently for an indefinitely parked receiver.
//!
//! Detection uses the 1:1 wake economy: every push fires exactly one coin
//! that must reach a live waiter. The victim loops `recv_timeout` with a
//! tiny timeout to supply departing waiters; the driver pauses it right
//! after each push so a stranded item cannot be swept; the canary parks
//! with a long timeout. A canary timeout with the item still in the
//! buffer therefore has no cause other than a stolen coin.
//!
//! Probabilistic: the steal window is a few instructions wide, so the
//! driver sweeps the victim's cycle phase across rounds. Run:
//!   POC_SECS=120 cargo test --release --test stale_entry_steal_public_api -- --ignored --nocapture

use crossfire::mpmc::bounded_blocking;
use crossfire::RecvTimeoutError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const VICTIM_TIMEOUT: Duration = Duration::from_micros(30);
/// Must outlast the victim's cycle: a legitimately woken victim then
/// always consumes before the canary can report a false Timeout.
const CANARY_TIMEOUT: Duration = Duration::from_micros(500);

#[test]
#[ignore = "hardware race hunt: cargo test --release --test stale_entry_steal_public_api -- --ignored --nocapture"]
fn poc_public_api_fire_stolen_by_timeout_departure() {
    let secs: u64 = std::env::var("POC_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
    let (tx, rx) = bounded_blocking::<u64>(64);
    let stop = Arc::new(AtomicBool::new(false));
    // Set right after each push: a paused victim performs no recv, so a
    // stranded item survives until the canary observes it.
    let pause = Arc::new(AtomicBool::new(false));
    let (report_tx, report_rx) = mpsc::channel::<Option<u64>>();

    // The victim: free-running receiver with a tiny timeout. Every cycle
    // registers a live entry and abandons it at the deadline — the
    // departing waiter whose coin the fire can steal.
    std::thread::spawn({
        let (rx, stop, pause) = (rx.clone(), stop.clone(), pause.clone());
        move || {
            while !stop.load(Ordering::Relaxed) {
                if pause.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                    continue;
                }
                match rx.recv_timeout(VICTIM_TIMEOUT) {
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        }
    });

    // The canary: a plain parked receiver. Ok means the coin reached it;
    // a Timeout with the item still in the buffer means it reached nobody.
    std::thread::spawn({
        let rx = rx.clone();
        move || loop {
            match rx.recv_timeout(CANARY_TIMEOUT) {
                Ok(_) => {
                    let _ = report_tx.send(None);
                }
                Err(RecvTimeoutError::Timeout) => match rx.try_recv() {
                    Ok(item) => {
                        let _ = report_tx.send(Some(item));
                    }
                    Err(_) => {
                        let _ = report_tx.send(None);
                    }
                },
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    });

    // The driver: one item per round at a random phase sweeping the
    // victim's abandon moment, then wait for the canary to resolve.
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut rounds = 0u64;
    while Instant::now() < deadline {
        rounds += 1;
        // Golden-ratio hash of the round counter: equidistributed phase in
        // [50, 150) us, non-resonant with the victim's cycle.
        let phase = (rounds.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) % 100 + 50;
        let until = Instant::now() + Duration::from_micros(phase);
        while Instant::now() < until {
            std::hint::spin_loop();
        }
        tx.send(rounds).expect("send into a drained bounded channel");
        pause.store(true, Ordering::Relaxed);
        match report_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(None) => pause.store(false, Ordering::Relaxed),
            Ok(Some(item)) => {
                stop.store(true, Ordering::Relaxed);
                panic!("LOST WAKEUP at round {rounds}: canary timed out while item {item} sat in the buffer");
            }
            Err(_) => panic!("canary stopped reporting"),
        }
    }
    stop.store(true, Ordering::Relaxed);
    println!("no steal observed in {secs}s across {rounds} rounds");
}
