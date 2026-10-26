/**
 * Prune renderer-stored session references whose sessions no longer exist on
 * the backend.
 *
 * Pins (`hermes.desktop.pinnedSessions`), composer drafts
 * (`hermes:composer-drafts:v3`) and queued prompts
 * (`hermes.desktop.composerQueue.v1`) are keyed by session id and re-asserted
 * or probed at boot. When a session is removed from state.db (retention purge,
 * manual delete, another surface pruning it), those keys go stale: nothing
 * ever drops them, so every boot re-requests the dead ids and the gateway
 * answers `404 {"detail":"Session not found"}` once per stale reference,
 * forever.
 *
 * This module sweeps for dead ids after the sidebar list has spoken. A session
 * is declared dead ONLY when a by-id probe 404s on every known profile of the
 * selected connection. A 404 proves nothing about another gateway. Pins are
 * already connection-scoped; globally keyed queued prompts and drafts require
 * an exact owner captured with the entry before they can be removed. Legacy
 * ownerless entries stay intact rather than guessing from a later session hint.
 * Any non-404 probe failure (gateway mid-restart, network) is inconclusive:
 * the id is deferred to a later pass instead of being dropped.
 */

import { getSession } from '@/hermes'
import { mapPool } from '@/lib/pool'
import { clearSessionDraftIfOwned, stashedDraftOwner, stashedDraftScopes } from '@/store/composer'
import { $queuedPromptsBySession, clearQueuedPromptsIfOwned } from '@/store/composer-queue'
import { $pinnedSessionIds, unpinSession } from '@/store/layout'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'

import { forgetPinSyncState } from './session-pin-sync'

const SWEEP_CONCURRENCY = 4
// After the first list payload, let boot settle (pin reconcile, resume probes)
// before probing; later list changes re-sweep debounced.
const FIRST_PASS_DELAY_MS = 2_000
const RERUN_DELAY_MS = 5_000
// How long an 'alive' verdict is trusted. Without expiry, a session deleted
// mid-session would stay cached alive until the next app restart — re-creating
// exactly the boot 404s this prune exists to remove. With expiry, a later
// sweep re-probes and heals it within one TTL.
const ALIVE_TTL_MS = 10 * 60_000

// Ids verified ALIVE this app session, id -> verdict expiry timestamp — later
// sweeps skip re-probing ids whose verdict is still fresh.
const alive = new Map<string, number>()
// Ids currently mid-probe, so overlapping sweeps don't double-probe.
const inFlight = new Map<string, number>()

let listLoaded = false
// Bumped by resetDeadSessionPrune: in-flight probes from a pre-reset sweep
// must not write verdicts (alive cache, drops) that belong to the OLD backend.
let epoch = 0
// Debounce deadline, clamped to the EARLIEST change since the last pass: a
// busy list that keeps changing must not starve the sweep indefinitely.
let sweepDueAt = 0
let sweepTimer: ReturnType<typeof setTimeout> | undefined

type Verdict = 'alive' | 'dead' | 'unknown'

function isNotFoundError(error: unknown): boolean {
  return /session not found/i.test(error instanceof Error ? error.message : String(error))
}

/**
 * A stored id is dead only if a by-id lookup 404s on the active profile AND on
 * every named profile of this connection. Pin the connection explicitly so an
 * unrelated same-id hint cannot redirect the probe to another backend.
 *
 * `$profiles` is the currently served backend's picker list. If no list has
 * loaded, or the active profile is absent, a missing profile is inconclusive.
 */
function currentConnectionId(): string | undefined {
  const connection = $connection.get()

  return connection?.connectionId?.trim() || (connection?.mode === 'local' ? 'local' : undefined)
}

/** A global queue/draft key is prunable only with one exact persisted owner. */
function ownedDraftProfile(id: string, connectionId: string): string | undefined {
  const owner = stashedDraftOwner(id)

  return owner?.connectionId === connectionId ? normalizeProfileKey(owner.profile) : undefined
}

function ownedQueueProfile(id: string, connectionId: string): string | undefined {
  const entries = $queuedPromptsBySession.get()[id]

  if (!entries?.length || entries.some(entry => entry.ownerRoute?.connectionId !== connectionId)) {
    return undefined
  }

  const profiles = new Set(entries.map(entry => normalizeProfileKey(entry.ownerRoute?.profile)))

  return profiles.size === 1 ? [...profiles][0] : undefined
}

async function sessionVerdict(id: string, profiles: readonly string[], connectionId: string): Promise<Verdict> {
  const activeKey = normalizeProfileKey($activeGatewayProfile.get())

  for (const profile of [activeKey, ...profiles.filter(key => key !== activeKey)]) {
    try {
      // Pin BOTH axes. A session owner hint must not reroute a pin's probe
      // to another gateway, and an ambient dial may change during the await.
      await getSession(id, { connectionId, profile })

      return 'alive'
    } catch (error) {
      if (!isNotFoundError(error)) {
        return 'unknown'
      }
    }
  }

  return 'dead'
}

/** Stored ids no loaded sidebar row covers — the only ids that can be dead. */
function collectCandidates(connectionId: string, profiles: readonly string[]): string[] {
  const rows = $sessions.get()
  const covered = new Set<string>()

  for (const row of rows) {
    if (row.id) {
      covered.add(row.id)
    }

    if (row._lineage_root_id) {
      covered.add(row._lineage_root_id)
    }
  }

  const candidates = new Set<string>()

  for (const id of $pinnedSessionIds.get()) {
    if (!covered.has(id)) {
      candidates.add(id)
    }
  }

  for (const scope of stashedDraftScopes()) {
    if (!covered.has(scope) && profiles.includes(ownedDraftProfile(scope, connectionId) ?? '')) {
      candidates.add(scope)
    }
  }

  for (const key of Object.keys($queuedPromptsBySession.get())) {
    if (!covered.has(key) && profiles.includes(ownedQueueProfile(key, connectionId) ?? '')) {
      candidates.add(key)
    }
  }

  return [...candidates].filter(id => {
    const expiresAt = alive.get(id)

    return !(expiresAt !== undefined && Date.now() < expiresAt) && !inFlight.has(id)
  })
}

function dropDeadId(id: string, connectionId: string, profiles: readonly string[]): void {
  // Cheap re-verification before destructive drops: if a row for this id
  // appeared while the probe was in flight (a session resurrected or a slow
  // page landed), the id is no longer dead — keep every entry.
  const covered = $sessions.get().some(row => row.id === id || row._lineage_root_id === id)

  if (covered) {
    return
  }

  // Forget the pin's sync bookkeeping BEFORE unpinning: the reconcile listener
  // fires synchronously on unpin, and with the id still tracked it would
  // re-PATCH the dead id (a fresh 404 on every boot — the noise we're removing).
  if ($pinnedSessionIds.get().includes(id)) {
    forgetPinSyncState(id)
    unpinSession(id)
  }

  // No-ops for surfaces that don't hold the id — the guards also skip the
  // localStorage write a no-op clear would still perform.
  if (profiles.includes(ownedDraftProfile(id, connectionId) ?? '') && stashedDraftScopes().includes(id)) {
    clearSessionDraftIfOwned(id, connectionId, profiles)
  }

  if (profiles.includes(ownedQueueProfile(id, connectionId) ?? '')) {
    clearQueuedPromptsIfOwned(id, connectionId, profiles)
  }
}

async function sweepOnce(requireLoaded = true): Promise<void> {
  if (requireLoaded && !listLoaded) {
    return
  }

  if (typeof window === 'undefined' || !window.hermesDesktop) {
    return
  }

  const sweepEpoch = epoch
  const connection = $connection.get()
  const connectionId = currentConnectionId()

  if (!connectionId) {
    return
  }

  const profiles = [...new Set($profiles.get().map(profile => normalizeProfileKey(profile.name)))]
  const activeProfile = normalizeProfileKey($activeGatewayProfile.get())

  if (profiles.length === 0 || !profiles.includes(activeProfile)) {
    return
  }

  const candidates = collectCandidates(connectionId, profiles)

  if (candidates.length === 0) {
    return
  }

  for (const id of candidates) {
    inFlight.set(id, sweepEpoch)
  }

  try {
    await mapPool(candidates, SWEEP_CONCURRENCY, async id => {
      if (epoch !== sweepEpoch) {
        return
      }

      const verdict = await sessionVerdict(id, profiles, connectionId)

      // A gateway switch (reset) while this probe was in flight invalidates
      // the verdict — it is about the backend we measured, not the live one.
      if (
        epoch !== sweepEpoch ||
        $connection.get() !== connection ||
        currentConnectionId() !== connectionId ||
        normalizeProfileKey($activeGatewayProfile.get()) !== activeProfile ||
        $profiles
          .get()
          .map(profile => normalizeProfileKey(profile.name))
          .join('\0') !== profiles.join('\0')
      ) {
        return
      }

      if (verdict === 'alive') {
        alive.set(id, Date.now() + ALIVE_TTL_MS)
      } else if (verdict === 'dead') {
        dropDeadId(id, connectionId, profiles)
      }
      // 'unknown' — neither cached nor dropped; a later sweep retries it.
    })
  } finally {
    for (const id of candidates) {
      if (inFlight.get(id) === sweepEpoch) {
        inFlight.delete(id)
      }
    }
  }
}

/** Run one prune pass now. Test-only hook — production never calls it. */
export function __runDeadSessionPrunePass(): Promise<void> {
  // Bypass the list-loaded gate WITHOUT mutating `listLoaded` (a mutation
  // here would leak into the watcher's first-payload timing).
  return sweepOnce(false)
}

/**
 * Sweep after the first sidebar payload, then re-sweep (debounced) whenever
 * the list changes — a purge while the app is running heals within one pass.
 * Call once per app, next to `watchSessionPins`. Returns the store unsubscribe
 * (used by tests).
 */
export function watchDeadSessionPrune(): () => void {
  return $sessions.listen(() => {
    const firstPayload = !listLoaded
    listLoaded = true

    const delay = firstPayload ? FIRST_PASS_DELAY_MS : RERUN_DELAY_MS

    // Clamp to the EARLIEST change since the last pass: constant list churn
    // must delay the sweep, not starve it.
    const dueAt = Date.now() + delay
    const nextDueAt = sweepTimer ? Math.min(sweepDueAt, dueAt) : dueAt

    if (sweepTimer && nextDueAt !== sweepDueAt) {
      clearTimeout(sweepTimer)
      sweepTimer = undefined
    }

    sweepDueAt = nextDueAt

    // Exactly one timer at a time: a churn that doesn't move the deadline
    // keeps the existing timer instead of stacking a second one.
    if (!sweepTimer) {
      sweepTimer = setTimeout(
        () => {
          sweepDueAt = 0
          sweepTimer = undefined
          void sweepOnce()
        },
        Math.max(0, nextDueAt - Date.now())
      )
    }
  })
}

/**
 * Forget probe results, because the backend they were measured against is
 * gone. `alive` means "alive on the gateway we probed"; a switched gateway has
 * its own state.db and must be re-probed from scratch.
 */
export function resetDeadSessionPrune(): void {
  epoch += 1
  alive.clear()
  inFlight.clear()
  listLoaded = false

  if (sweepTimer) {
    clearTimeout(sweepTimer)
    sweepTimer = undefined
  }

  sweepDueAt = 0
}
