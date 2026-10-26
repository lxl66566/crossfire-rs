//! Public-API reproduction of the stale-entry fire steal on crossfire
//! 3.1.22 (master). Uses nothing but the crate's public channel surface —
//! no registry internals — so it doubles as the upstream issue repro.
//!
//! The mechanism: `fire()`'s `wake()` reads the node state and then
//! unconditionally stores `Woken`. A receiver parked with a timeout wakes
//! itself at the deadline and abandons (`Waiting -> Closed`) before
//! returning `Err(Timeout)`. `abandon_recv_waker` re-fires when the
//! abandon CAS *loses* (it observes `Woken`), but when the abandon lands
//! *inside* wake's read->store window the abandon returns `Ok`, the late
//! `store(Woken)` stamps a node nobody watches anymore, and `fire()`
//! reports the event as delivered. The coin is spent; a receiver parked
//! behind the departing one is never woken — permanently, when that fire
//! was the last event for the channel.
//!
//! The harness exploits the 1:1 wake economy for detection. Each round:
//! the canary parks on the channel, the driver pushes exactly one item
//! after a random phase offset, and the victim (a receiver looping
//! `recv_timeout` with a tiny timeout) provides the departing entry. If
//! the canary reports `Timeout` while the item is still in the buffer, a
//! fire was spent without waking any live waiter — a lost wakeup with
//! no other possible cause: every push fires, and every coin either
//! delivers to a live waiter, is handed back on Skip, or is stolen here.
//!
//! Probabilistic: the steal window is a handful of instructions wide, so
//! rounds sweep the victim's cycle phase uniformly and rely on volume
//! (multiple independent channel groups). The internal POCs
//! (`loom_committed_waiter_migrated_between_pop_and_wake_steals_fire`,
//! `poc_fire_stolen_by_waiter_migrated_after_pop`) are the deterministic
//! proofs; this file shows the same lost wakeup through the public API.
//!
//! Run with:
//!   cargo test --release --test stale_entry_steal_public_api -- --ignored --nocapture
//!
//! Knobs: POC_SECS (soak seconds per run, default 120), POC_GROUPS
//! (parallel independent channels, default 4).

use crossfire::mpmc::bounded_blocking;
use crossfire::RecvTimeoutError;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Base park window of the departing receiver. Smaller = more departures
/// per second, but the push phase must also sweep the cycle, so the
/// jitter below keeps the same order of magnitude.
const VICTIM_TIMEOUT_NANOS: u64 = 50_000;
/// Per-cycle jitter on top of the base.
const VICTIM_JITTER_NANOS: u64 = 50_000;
/// The canary's wait per round. Must outlast the victim's cycle so a
/// legitimate wake on the victim always resolves the round before the
/// canary reports Timeout.
const CANARY_TIMEOUT_NANOS: u64 = 500_000;
/// Floor/ceiling of the driver's phase offset before each push. The floor
/// keeps the push clear of the canary's re-registration gap after the
/// previous report; the ceiling bounds the round cost.
const PHASE_MIN_NANOS: u64 = 50_000;
const PHASE_MAX_NANOS: u64 = 250_000;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

enum RoundOutcome {
    /// The canary got the item (its entry was at the queue front).
    Consumed,
    /// The canary timed out and the buffer was empty: the coin went to
    /// the victim's live entry, no harm.
    TimedOutEmpty,
    /// The canary timed out with the item still in the buffer: some fire
    /// was spent without waking any live waiter.
    TimedOutWithItem(u64),
}

fn run_group(group: u64, secs: u64, evidence: std_mpsc::Sender<String>) {
    let (tx, rx) = bounded_blocking::<u64>(64);
    let stop = Arc::new(AtomicBool::new(false));
    // Freezes the victim between rounds: a paused victim performs no
    // try_recv sweep, so a stranded item cannot be quietly consumed
    // before the canary's timeout check observes it.
    let pause = Arc::new(AtomicBool::new(false));
    let victim_consumed = Arc::new(AtomicUsize::new(0));

    // The victim: parks on the channel with a tiny timeout, over and
    // over. Every cycle leaves a live queue entry whose episode dies at
    // the deadline — the departing waiter whose coin can be stolen.
    let victim_handle = {
        let rx = rx.clone();
        let stop = stop.clone();
        let pause = pause.clone();
        let victim_consumed = victim_consumed.clone();
        std::thread::spawn(move || {
            let mut rng = 0x9E37_79B9_7F4A_7C15 ^ group.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            while !stop.load(Ordering::Relaxed) {
                while pause.load(Ordering::Relaxed) {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::yield_now();
                }
                let t = VICTIM_TIMEOUT_NANOS + xorshift(&mut rng) % VICTIM_JITTER_NANOS;
                match rx.recv_timeout(Duration::from_nanos(t)) {
                    Ok(_) => {
                        victim_consumed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        })
    };

    // The canary: a plain parked receiver. It must be the one woken for
    // every round's item; a Timeout with the item still in the buffer is
    // the lost-wakeup evidence.
    let (report_tx, report_rx) = std_mpsc::channel::<RoundOutcome>();
    let canary_handle = {
        let rx = rx.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match rx.recv_timeout(Duration::from_nanos(CANARY_TIMEOUT_NANOS)) {
                    Ok(item) => {
                        let _ = report_tx.send(RoundOutcome::Consumed);
                        let _ = item;
                    }
                    Err(RecvTimeoutError::Timeout) => match rx.try_recv() {
                        Ok(item) => {
                            let _ = report_tx.send(RoundOutcome::TimedOutWithItem(item));
                        }
                        Err(_) => {
                            let _ = report_tx.send(RoundOutcome::TimedOutEmpty);
                        }
                    },
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        })
    };

    let mut rng = 0xA24B_AED4_963E_E407 ^ group.wrapping_mul(0x9FB2_1C0E_3D77_4A25);
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut rounds = 0usize;
    let mut consumed = 0usize;
    let mut inconclusive = 0usize;
    let mut hit = false;
    while Instant::now() < deadline {
        rounds += 1;
        // Random phase inside the victim's cycle: the push then samples
        // the read->store window uniformly across rounds.
        let d = PHASE_MIN_NANOS + xorshift(&mut rng) % (PHASE_MAX_NANOS - PHASE_MIN_NANOS);
        let until = Instant::now() + Duration::from_nanos(d);
        while Instant::now() < until {
            std::hint::spin_loop();
        }
        tx.send(rounds as u64).expect("send into an empty bounded channel");
        pause.store(true, Ordering::Relaxed);
        match report_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(RoundOutcome::Consumed) => consumed += 1,
            Ok(RoundOutcome::TimedOutEmpty) => inconclusive += 1,
            Ok(RoundOutcome::TimedOutWithItem(item)) => {
                evidence
                    .send(format!(
                        "group {group}: LOST WAKEUP at round {rounds} — canary timed out while \
                         item {item} sat in the buffer (rounds={rounds}, consumed={consumed}, \
                         inconclusive={inconclusive}, victim_consumed={})",
                        victim_consumed.load(Ordering::Relaxed)
                    ))
                    .ok();
                hit = true;
                break;
            }
            Err(_) => panic!("group {group}: canary stopped reporting"),
        }
        pause.store(false, Ordering::Relaxed);
    }
    stop.store(true, Ordering::Relaxed);
    let _ = victim_handle.join();
    let _ = canary_handle.join();
    if !hit {
        println!(
            "group {group}: {} rounds, no steal observed (consumed={consumed}, inconclusive={inconclusive})",
            rounds
        );
    }
}

#[test]
#[ignore = "hardware race hunt: cargo test --release --test stale_entry_steal_public_api -- --ignored --nocapture"]
fn poc_public_api_fire_stolen_by_timeout_departure() {
    let secs: u64 = std::env::var("POC_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
    let groups: u64 = std::env::var("POC_GROUPS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);

    let (evidence_tx, evidence_rx) = std_mpsc::channel::<String>();
    let mut handles = Vec::new();
    for group in 0..groups {
        let evidence_tx = evidence_tx.clone();
        handles.push(std::thread::spawn(move || {
            run_group(group, secs, evidence_tx);
        }));
    }
    drop(evidence_tx);

    // Wait for the soak to end, surfacing the first evidence as it comes.
    let soak = Instant::now() + Duration::from_secs(secs + 30);
    loop {
        match evidence_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(msg) => panic!("{msg}"),
            Err(std_mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() > soak {
                    break;
                }
            }
            Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    for h in handles {
        h.join().expect("group thread");
    }
    println!(
        "poc_public_api_fire_stolen_by_timeout_departure: no steal observed in {secs}s across {groups} groups"
    );
}
