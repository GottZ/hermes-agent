import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesModule from '@/hermes'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

const getSessionMock =
  vi.fn<(id: string, profile?: null | string | { connectionId: string; profile: string }) => Promise<SessionInfo>>()

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesModule>()

  return {
    ...actual,
    getSession: (id: string, profile?: null | string | { connectionId: string; profile: string }) =>
      getSessionMock(id, profile)
  }
})

import {
  clearSessionDraft,
  reloadPersistedDrafts,
  stashedDraftOwner,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  simulateComposerQueueReloadForTests
} from '@/store/composer-queue'
import { wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { $pinnedSessionIds } from '@/store/layout'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $sessions, _resetSessionOwnerHintsForTests, setConnection, setSessionOwnerHint } from '@/store/session'
import { $sessionTiles, openSessionTile } from '@/store/session-states'

import { __runDeadSessionPrunePass, resetDeadSessionPrune, watchDeadSessionPrune } from './dead-session-prune'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

const profile = (name: string): ProfileInfo =>
  ({
    has_env: false,
    is_default: name === 'default',
    model: null,
    name,
    path: name,
    provider: null,
    skill_count: 0
  }) as ProfileInfo

const notFound = () => new Error('404: {"detail":"Session not found"}')

beforeAll(() => {
  ;(globalThis as { window?: unknown }).window ??= {}
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}
})

beforeEach(() => {
  window.localStorage.clear()
  reloadPersistedDrafts()
  setConnection({ connectionId: 'local', mode: 'local', baseUrl: 'http://localhost' } as never)
  $sessions.set([])
  $pinnedSessionIds.set([])
  $queuedPromptsBySession.set({})
  $profiles.set([profile('default'), profile('work')])
  $activeGatewayProfile.set('default')
  getSessionMock.mockReset()
  _resetSessionOwnerHintsForTests({ storage: true })
  $sessionTiles.set([])
  resetDeadSessionPrune()
})

afterEach(() => {
  clearSessionDraft('dead-draft-session')
  clearSessionDraft('__new__')
  $sessions.set([])
  $pinnedSessionIds.set([])
  $queuedPromptsBySession.set({})
})

describe('__runDeadSessionPrunePass', () => {
  it('unpins a pin whose session 404s on every profile', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['dead-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
    // Active-scope probe first, then each other profile.
    expect(getSessionMock).toHaveBeenCalledWith('dead-pin', { connectionId: 'local', profile: 'default' })
    expect(getSessionMock).toHaveBeenCalledWith('dead-pin', { connectionId: 'local', profile: 'work' })
  })

  it('keeps a pin whose session lives on another profile, and caches it as alive', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['other-pin'])
    getSessionMock.mockImplementation(async (id, profileName) => {
      if (typeof profileName === 'object' && profileName?.profile === 'work') {
        return row(id, { profile: 'work' })
      }

      throw notFound()
    })

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual(['other-pin'])

    // Alive verdict is cached — a second pass must not re-probe.
    getSessionMock.mockClear()
    await __runDeadSessionPrunePass()
    expect(getSessionMock).not.toHaveBeenCalled()
  })

  it('defers on non-404 probe failures instead of dropping, and retries later', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['flaky-pin'])
    getSessionMock.mockRejectedValue(new Error('gateway restarting'))

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual(['flaky-pin'])

    // Not cached — the next pass probes again, and a definitive 404 prunes.
    getSessionMock.mockRejectedValue(notFound())
    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('discards drafts for dead sessions but never the new-chat draft', async () => {
    $sessions.set([row('unrelated')])
    setSessionOwnerHint('dead-draft-session', { connectionId: 'local', profile: 'default' })
    stashSessionDraft('dead-draft-session', 'stale text', [])
    stashSessionDraft('__new__', 'brand new', [])
    reloadPersistedDrafts()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(takeSessionDraft('dead-draft-session').text).toBe('')
    expect(takeSessionDraft('__new__').text).toBe('brand new')
  })

  it('keeps a remote tile draft when a stale local hint contradicts its authoritative route', async () => {
    setSessionOwnerHint('tile-draft', { connectionId: 'local', profile: 'default' })
    openSessionTile('tile-draft', 'right', undefined, undefined, {
      workspaceMode: 'sessions',
      ownerRoute: { connectionId: 'remote-a', profile: 'default' }
    })
    stashSessionDraft('tile-draft', 'remote unsent text', [])
    expect(stashedDraftOwner('tile-draft')).toEqual({ connectionId: 'remote-a', profile: 'default' })
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(takeSessionDraft('tile-draft').text).toBe('remote unsent text')
    expect(JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')['tile-draft']).toBe(
      'remote unsent text'
    )
  })

  it('drops queued prompts for dead sessions and keeps live ones', async () => {
    $sessions.set([row('unrelated')])
    setSessionOwnerHint('dead-queue-session', { connectionId: 'local', profile: 'default' })
    enqueueQueuedPrompt('dead-queue-session', { text: 'go', attachments: [] })
    enqueueQueuedPrompt('live-queue-session', { text: 'stay', attachments: [] })
    getSessionMock.mockImplementation(async (id, profileName) => {
      if (id === 'live-queue-session') {
        return row(id, { profile: typeof profileName === 'object' ? profileName?.profile : (profileName ?? 'default') })
      }

      throw notFound()
    })

    await __runDeadSessionPrunePass()

    expect(getQueuedPrompts('dead-queue-session')).toEqual([])
    expect(getQueuedPrompts('live-queue-session')).toHaveLength(1)
  })

  it('prunes a confirmed dead owned queue after reloading its persisted route', async () => {
    setSessionOwnerHint('persisted-dead', { connectionId: 'local', profile: 'work' })
    enqueueQueuedPrompt('persisted-dead', { text: 'stale pending', attachments: [] })
    simulateComposerQueueReloadForTests()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getQueuedPrompts('persisted-dead')).toEqual([])
    expect(getSessionMock).toHaveBeenCalledWith('persisted-dead', { connectionId: 'local', profile: 'work' })
  })

  it('preserves a legacy persisted queue even if a later same-id owner hint appears', async () => {
    window.localStorage.setItem(
      'hermes.desktop.composerQueue.v1',
      JSON.stringify({
        'legacy-session': [{ id: 'legacy-entry', text: 'do not lose me', attachments: [], queuedAt: 1 }]
      })
    )
    simulateComposerQueueReloadForTests()
    setSessionOwnerHint('legacy-session', { connectionId: 'local', profile: 'default' })
    $pinnedSessionIds.set(['legacy-session'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getQueuedPrompts('legacy-session').map(entry => entry.text)).toEqual(['do not lose me'])
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('preserves a legacy draft when a later hint cannot establish its original owner', async () => {
    window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ 'legacy-draft': 'keep unsent' }))
    // Reload through the public draft-sync path, not by writing module internals.
    reloadPersistedDrafts()
    setSessionOwnerHint('legacy-draft', { connectionId: 'local', profile: 'default' })
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(takeSessionDraft('legacy-draft').text).toBe('keep unsent')
  })

  it('keeps a draft when persisted owner metadata is malformed', async () => {
    window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ 'unknown-draft': 'keep me' }))
    window.localStorage.setItem('hermes:composer-drafts:v3.owners', 'null')
    reloadPersistedDrafts()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(takeSessionDraft('unknown-draft').text).toBe('keep me')
  })

  it('keeps a replacement draft written by another window during its dead probe', async () => {
    setSessionOwnerHint('race-draft', { connectionId: 'local', profile: 'default' })
    stashSessionDraft('race-draft', 'original', [])
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )

    const pass = __runDeadSessionPrunePass()
    window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ 'race-draft': 'replacement' }))
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    expect(JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')['race-draft']).toBe(
      'replacement'
    )
  })

  it('prunes one dead draft without overwriting an unrelated cross-window draft before its storage event', async () => {
    setSessionOwnerHint('dead-a', { connectionId: 'local', profile: 'default' })
    setSessionOwnerHint('other-b', { connectionId: 'local', profile: 'default' })
    stashSessionDraft('dead-a', 'delete me', [])
    stashSessionDraft('other-b', 'old B', [])
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )
    getSessionMock.mockRejectedValue(notFound())

    const pass = __runDeadSessionPrunePass()
    const texts = JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')
    window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ ...texts, 'other-b': 'NEW B' }))
    const owners = JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3.owners') ?? '{}')
    window.localStorage.setItem(
      'hermes:composer-drafts:v3.owners',
      JSON.stringify({ ...owners, 'other-b': { ...owners['other-b'], text: 'NEW B' } })
    )
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    const persisted = JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')
    expect(persisted['dead-a']).toBeUndefined()
    expect(persisted['other-b']).toBe('NEW B')
    expect(JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3.owners') ?? '{}')['other-b']?.text).toBe(
      'NEW B'
    )
  })

  it('skips ids already covered by loaded sidebar rows', async () => {
    $sessions.set([row('covered', { _lineage_root_id: 'root-covered' })])
    $pinnedSessionIds.set(['covered', 'root-covered', 'rowless-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getSessionMock).not.toHaveBeenCalledWith('covered', expect.anything())
    expect(getSessionMock).not.toHaveBeenCalledWith('root-covered', expect.anything())
    expect(getSessionMock).toHaveBeenCalledWith('rowless-pin', { connectionId: 'local', profile: 'default' })
    // Covered pins are alive (their rows exist) — only the rowless pin dies.
    expect($pinnedSessionIds.get()).toEqual(['covered', 'root-covered'])
  })

  it('defers when the profile list is not loaded yet', async () => {
    $sessions.set([row('unrelated')])
    $profiles.set([])
    $pinnedSessionIds.set(['mystery-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    // Can't rule out another profile we don't know about — keep the pin.
    expect($pinnedSessionIds.get()).toEqual(['mystery-pin'])
  })

  it('prunes on a genuinely empty backend (all sessions purged)', async () => {
    // An empty list is a real answer here — the wipe path is handled by the
    // gateway-switch reset cancelling the wipe-scheduled sweep, not by the
    // sweep refusing to run. A backend with zero sessions has no rows to
    // cover stored ids, so probes decide.
    $sessions.set([])
    $pinnedSessionIds.set(['orphan-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
    expect(getSessionMock).toHaveBeenCalledWith('orphan-pin', { connectionId: 'local', profile: 'default' })
  })

  it('re-probes an alive id once its verdict TTL expires', async () => {
    vi.useFakeTimers()

    try {
      $sessions.set([row('unrelated')])
      $pinnedSessionIds.set(['ttl-pin'])
      getSessionMock.mockImplementation(async (id, profileName) => {
        if (typeof profileName === 'object' && profileName?.profile === 'work') {
          return row(id, { profile: 'work' })
        }

        throw notFound()
      })

      // Alive on another profile — cached, not pruned, not re-probed.
      await __runDeadSessionPrunePass()
      expect($pinnedSessionIds.get()).toEqual(['ttl-pin'])
      getSessionMock.mockClear()
      await __runDeadSessionPrunePass()
      expect(getSessionMock).not.toHaveBeenCalled()

      // Session deleted mid-session; after the TTL the next pass re-probes
      // and a definitive all-profile 404 prunes.
      vi.setSystemTime(Date.now() + 10 * 60_000 + 1_000)
      getSessionMock.mockRejectedValue(notFound())
      await __runDeadSessionPrunePass()

      expect($pinnedSessionIds.get()).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  it('discards in-flight verdicts when a gateway reset happens mid-sweep', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['race-pin'])
    let resolveProbe!: (value: SessionInfo) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise(resolve => {
          resolveProbe = resolve
        })
    )

    const pass = __runDeadSessionPrunePass()
    // Gateway switch while the probe is in flight.
    resetDeadSessionPrune()
    resolveProbe(row('race-pin', { profile: 'default' }))
    await pass

    // The pre-reset verdict must not be applied: the pin survives, and the
    // discarded 'alive' verdict must not poison the cache (next pass re-probes).
    expect($pinnedSessionIds.get()).toEqual(['race-pin'])
    getSessionMock.mockClear()
    getSessionMock.mockRejectedValue(notFound())
    $sessions.set([row('unrelated')])

    await __runDeadSessionPrunePass()

    expect(getSessionMock).toHaveBeenCalledWith('race-pin', { connectionId: 'local', profile: 'default' })
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('does not clear a newly written unknown queue while a dead probe is in flight', async () => {
    setSessionOwnerHint('race-queue', { connectionId: 'local', profile: 'default' })
    enqueueQueuedPrompt('race-queue', { text: 'original', attachments: [] })
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )

    const pass = __runDeadSessionPrunePass()
    window.localStorage.setItem(
      'hermes.desktop.composerQueue.v1',
      JSON.stringify({
        'race-queue': [{ id: 'new-from-other-window', text: 'keep me', attachments: [], queuedAt: 2 }]
      })
    )
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    expect(
      JSON.parse(window.localStorage.getItem('hermes.desktop.composerQueue.v1') ?? '{}')['race-queue']?.[0]?.text
    ).toBe('keep me')
  })

  it('discards a stale 404 when a connection descriptor changes without a prune reset', async () => {
    setSessionOwnerHint('route-race', { connectionId: 'local', profile: 'default' })
    enqueueQueuedPrompt('route-race', { text: 'pending', attachments: [] })
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )

    const pass = __runDeadSessionPrunePass()
    setConnection({ connectionId: 'remote-race', mode: 'remote', baseUrl: 'https://race.example' } as never)
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    expect(getQueuedPrompts('route-race').map(entry => entry.text)).toEqual(['pending'])
  })

  it('resetDeadSessionPrune drops the alive cache so a switched gateway is re-probed', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['switched-pin'])
    getSessionMock.mockImplementation(async id => row(id, { profile: 'default' }))

    await __runDeadSessionPrunePass()
    expect($pinnedSessionIds.get()).toEqual(['switched-pin'])

    // The gateway switched; the new backend no longer has the session. Without
    // the reset the alive verdict would keep the dead pin forever.
    resetDeadSessionPrune()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
  })
})

describe('watchDeadSessionPrune', () => {
  it('keeps the original gateway queue and draft after a switch and real scheduled prune', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $sessions.set([row('local-session')])
      setSessionOwnerHint('local-session', { connectionId: 'local', profile: 'work' })
      enqueueQueuedPrompt('local-session', { text: 'unsent follow-up', attachments: [] })
      stashSessionDraft('local-session', 'unsent draft', [])
      $pinnedSessionIds.set(['local-session'])

      wipeSessionListsForGatewaySwitch()
      setConnection({ connectionId: 'remote-a', mode: 'remote', baseUrl: 'https://remote.example' } as never)
      $profiles.set([profile('default'), profile('work')])
      getSessionMock.mockRejectedValue(notFound())
      $sessions.set([row('remote-session')])

      await vi.advanceTimersByTimeAsync(2_000)

      expect(getQueuedPrompts('local-session').map(entry => entry.text)).toEqual(['unsent follow-up'])
      expect(takeSessionDraft('local-session').text).toBe('unsent draft')
      expect(getSessionMock).not.toHaveBeenCalledWith('local-session', expect.anything())
      setConnection({ connectionId: 'local', mode: 'local', baseUrl: 'http://localhost' } as never)
      expect($pinnedSessionIds.get()).toEqual(['local-session'])
      expect(getQueuedPrompts('local-session').map(entry => entry.text)).toEqual(['unsent follow-up'])
    } finally {
      unsubscribe()
      resetDeadSessionPrune()
      vi.useRealTimers()
    }
  })

  it('sweeps once after the first list payload', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $pinnedSessionIds.set(['timed-pin'])
      getSessionMock.mockRejectedValue(notFound())

      $sessions.set([row('boot-row')])

      vi.advanceTimersByTime(2_000)
      await vi.runOnlyPendingTimersAsync()

      expect($pinnedSessionIds.get()).toEqual([])
    } finally {
      unsubscribe()
      vi.useRealTimers()
    }
  })

  it('clamps the debounce so list churn cannot starve the sweep', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $pinnedSessionIds.set(['churn-pin'])
      getSessionMock.mockRejectedValue(notFound())

      // First payload schedules the pass at +2s; a change 1s later would
      // naively re-schedule to +6s. The clamp keeps the earlier deadline —
      // and must not stack a second timer for it.
      $sessions.set([row('a')])
      vi.advanceTimersByTime(1_000)
      $sessions.set([row('a'), row('b')])
      expect(vi.getTimerCount()).toBe(1)

      vi.advanceTimersByTime(1_000)
      await vi.runOnlyPendingTimersAsync()

      expect($pinnedSessionIds.get()).toEqual([])
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      unsubscribe()
      vi.useRealTimers()
    }
  })

  it('re-opens a fresh debounce window after a sweep fires', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      // First sweep fires at +2s and prunes the boot dead pin.
      $pinnedSessionIds.set(['first-pin'])
      getSessionMock.mockRejectedValue(notFound())
      $sessions.set([row('a')])
      vi.advanceTimersByTime(2_000)
      await vi.runOnlyPendingTimersAsync()
      expect($pinnedSessionIds.get()).toEqual([])

      // A change right after the sweep must NOT fire an immediate pass — the
      // next pass waits out a fresh RERUN_DELAY window.
      $pinnedSessionIds.set(['second-pin'])
      $sessions.set([row('a'), row('c')])

      vi.advanceTimersByTime(1_000)
      expect($pinnedSessionIds.get()).toEqual(['second-pin'])

      vi.advanceTimersByTime(4_000)
      await vi.runOnlyPendingTimersAsync()
      expect($pinnedSessionIds.get()).toEqual([])
    } finally {
      unsubscribe()
      vi.useRealTimers()
    }
  })
})
